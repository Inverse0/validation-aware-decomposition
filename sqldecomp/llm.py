"""The model client of the pipeline: ONE configuration (`ModelConfig`) for every model call the pipeline makes, and
ONE call (`call(prompt, template, config)`). The task writer, semantic judge and verifier use the same client with explicit
ModelConfig settings for each role.

THE SANDBOX. By default (`ModelConfig.sandbox = True`) every call runs the codex CLI inside a bwrap sandbox that holds
NOTHING but the call's own prompt:

    bwrap --unshare-all --share-net --die-with-parent --new-session --clearenv
          --ro-bind /usr /usr  (+ the lib / lib64 / bin / sbin symlinks)
          --ro-bind /etc/resolv.conf, /etc/hosts, /etc/nsswitch.conf, /etc/ssl, /etc/pki, /etc/crypto-policies
                                                                              TLS roots and name resolution for the API
          --ro-bind <codex binary> /opt/codex
          --proc /proc --dev /dev --tmpfs /tmp --tmpfs /home/sandbox
          --bind <fresh codex home: auth.json + config.toml> /home/sandbox/.codex   a copy per call, deleted afterwards
          --bind <work dir: prompt.txt> /work                                 the ONE item; answer.txt comes back here
          --chdir /work --setenv HOME /home/sandbox --setenv CODEX_HOME /home/sandbox/.codex --setenv PATH /usr/bin:/bin
          --setenv TERM dumb --setenv NO_COLOR 1
          /opt/codex exec --skip-git-repo-check -s read-only --ephemeral --json -o /work/answer.txt - < /work/prompt.txt

No repository, no dataset, no gold label, no cache of another item, no environment variable of the host. `audit()`
runs `find`, `env`, the mounts and `id` through the same wrapper and records what a process in there can see. When
bwrap is not installed, a call that needs the model REFUSES (`SandboxUnavailable`) unless the configuration says
`sandbox=False`, which only an explicit `--no-sandbox` sets and which every output record carries; the unsandboxed
call runs the same codex command in a scratch directory with a fresh codex home and its own HOME, but with the host's
file system and PATH visible to it: it is not isolated.

THE CACHE. Every call is cached under `cache_dir/<key>.json`, key = SHA-256 of (model, effort, template, prompt): a
rerun is free, and a cached call keeps the tokens, the latency and the estimated cost of the call that produced it
(`Call.cached` says it was not paid again). An unsandboxed call is keyed under the template plus `|no-sandbox`
(`key_template`), so that a sandboxed run never reuses an answer made where the host was visible; sandboxed keys
are stable across runs. Every call is limited to `ModelConfig.timeout_s` (900 s), a guard against a hung process: a
call that reaches it is an error of that call (exit code -1), never an answer; SQLDECOMP_CODEX_TIMEOUT=none lifts the
limit (`call_timeout`). Several processes may share one
credentials file and one cache: the copy of the credentials into a call's home and the write-back of a refreshed
token hold an inter-process lock (`_auth_lock`), and every file is written to a temporary name and renamed into
place (`atomic_write_text`), so that no reader sees half a file. The model and the reasoning effort are fixed in the per-call
`config.toml`. With SQLDECOMP_CODEX_STARTS set (`N/S`), the calls of every process that shares its lock file start
at most N per S seconds (`start_slot`): the
wait comes before the call, whose latency, prompt, command and cache key are unchanged.
Tokens come from codex's `turn.completed` event (`Tokens`: input, cached input, output, reasoning
where reported); `cost_usd(tokens, price)` estimates the cost from a price table (`Price`; the table itself is part
of the experiment configuration, `sqldecomp.config`), and gives None, never a guess, when no price is known.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

MODEL = "gpt-5.5"
EFFORT = "medium"
TEMPLATE_VERSION = "model-call-v1"  # the call template: the codex command line, the sandbox layout, the parsing of answer and usage
SANDBOX_HOME = "/home/sandbox"
CODEX_COMMAND = ("exec", "--skip-git-repo-check", "-s", "read-only", "--ephemeral", "--json")
_lock = threading.Lock()


class SandboxUnavailable(RuntimeError):
    """A model call was asked for in the sandbox, and bwrap is not installed."""


def sandbox_refusal() -> str:
    return ("bwrap (bubblewrap) is not installed on this machine, and every model call of the pipeline runs in the bwrap "
            "sandbox by default. Install bubblewrap, or run with --no-sandbox (recorded in every output record) to call "
            "the model without it.")


def sandbox_available() -> bool:
    return shutil.which("bwrap") is not None


def require_sandbox(config: "ModelConfig") -> None:
    """Refuse before any work when the configuration asks for the sandbox and bwrap is missing (a run that will call
    the model checks this first, so that it never stops halfway)."""
    if config.sandbox and not sandbox_available():
        raise SandboxUnavailable(sandbox_refusal())


def codex_binary() -> Path:
    return Path(os.environ.get("SQLDECOMP_CODEX", os.path.expanduser("~/.local/bin/codex"))).resolve()


def master_auth() -> Path:
    return Path(os.environ.get("SQLDECOMP_CODEX_AUTH", os.path.expanduser("~/.codex/auth.json")))


@dataclass(frozen=True)
class ModelConfig:
    """The one configuration of the pipeline's model calls (the task writer's and the verifier's)."""

    model: str = MODEL
    effort: str = EFFORT
    template_version: str = TEMPLATE_VERSION  # recorded with every call; not part of the cache key (the answer is the model's to the prompt)
    cache_dir: str | None = None  # raw calls keyed by prompt hash; None: no cache
    sandbox: bool = True  # False only through an explicit --no-sandbox, recorded in every output
    timeout_s: float = 900.0  # one call; a call that times out is an error of that call, never an answer

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**{k: d[k] for k in ("model", "effort", "template_version", "cache_dir", "sandbox", "timeout_s") if k in d})

    def with_(self, **changes: Any) -> "ModelConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class Tokens:
    input: int | None = None  # every input token, the cached ones included (OpenAI's usage semantics)
    cached_input: int | None = None
    output: int | None = None  # every output token, the reasoning ones included (billed as output)
    reasoning: int | None = None  # the reasoning part of `output`, where the client reports it

    def to_json(self) -> dict[str, int | None]:
        return asdict(self)


def tokens_of(usage: dict[str, Any] | None) -> Tokens:
    """The four counts of a codex usage dict; a count the client does not report is None, never 0."""
    usage = usage or {}

    def get(*names: str) -> int | None:
        for n in names:
            if usage.get(n) is not None:
                return int(usage[n])
        return None

    return Tokens(get("input_tokens"), get("cached_input_tokens"), get("output_tokens"), get("reasoning_output_tokens", "reasoning_tokens"))


@dataclass(frozen=True)
class Price:
    """USD per one million tokens of one model, from the provider's published price list."""

    input: float  # the input tokens that are not cached
    cached_input: float
    output: float  # reasoning tokens are billed as output tokens and counted inside output_tokens
    long_input_threshold: int | None = None  # a call with more input tokens than this is priced at the multipliers below
    long_input_multiplier: float = 1.0
    long_output_multiplier: float = 1.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Price":
        return cls(**d)


def cost_usd(tokens: Tokens, price: Price | None) -> float | None:
    """The estimated cost of one call: (input - cached) x input + cached x cached_input + output x output, per 1M,
    each input term and the output term raised by the long-input multipliers when the call's input exceeds the
    threshold. None when no price is known or the client reported no input or output count: unavailable, never a
    guess."""
    if price is None or tokens.input is None or tokens.output is None:
        return None
    cached = tokens.cached_input or 0
    fresh = max(0, tokens.input - cached)
    long = price.long_input_threshold is not None and tokens.input > price.long_input_threshold
    mi = price.long_input_multiplier if long else 1.0
    mo = price.long_output_multiplier if long else 1.0
    return (fresh * price.input * mi + cached * price.cached_input * mi + tokens.output * price.output * mo) / 1_000_000


@dataclass
class Call:
    """One model call, or its cached record."""

    answer: str
    usage: dict[str, int]  # codex's usage dict as reported
    seconds: float  # the latency of the call that produced the answer (for a cache hit: the original call's)
    exit_code: int
    cached: bool = False
    events: list[dict] = field(default_factory=list)
    stderr: str = ""
    key: str = ""
    model: str = MODEL
    effort: str = EFFORT
    template: str = ""
    sandbox: bool = True
    template_version: str = TEMPLATE_VERSION

    @property
    def tokens(self) -> Tokens:
        return tokens_of(self.usage)

    def record(self, price: Price | None = None) -> dict[str, Any]:
        """What an output record holds of the call: the keys, the tokens, the latency in ms, the estimated cost, the
        cache and sandbox flags (never the prompt or the answer: those are in the cache)."""
        out = {"key": self.key, "template": self.template, "model": self.model, "effort": self.effort,
               "template_version": self.template_version, "sandbox": self.sandbox, "cached": self.cached,
               "exit_code": self.exit_code, "latency_ms": round(self.seconds * 1000, 1), "tokens": self.tokens.to_json(),
               "cost_usd": cost_usd(self.tokens, price)}
        if self.exit_code != 0:  # a failed call keeps its diagnostics (the cache only holds calls that answered)
            out["error"] = (self.stderr or "")[-300:]
        return out


def add_model_arguments(ap, *, model: str = MODEL, effort: str = EFFORT, calls: str = "every call") -> None:
    """`--model`, `--effort`, `--no-sandbox`: the model configuration of a command line; `model` / `effort` are its
    defaults, `calls` says whose calls it configures (the verification run's: the verifier's, the base of every call;
    the task writer and the LLM judge have their own options and defaults)."""
    g = ap.add_argument_group("model", f"the model configuration of {calls}")
    g.add_argument("--model", default=model, help=f"the model of {calls} (default {model})")
    g.add_argument("--effort", default=effort, help=f"the reasoning effort of {calls} (default {effort})")
    g.add_argument("--no-sandbox", action="store_true",
                   help="call the model WITHOUT the bwrap sandbox (recorded in every output record); without this flag a machine "
                        "without bwrap refuses to call the model")


def model_config_from_args(args, *, cache_dir: str | Path | None = None) -> ModelConfig:
    return ModelConfig(model=getattr(args, "model", None) or MODEL, effort=getattr(args, "effort", None) or EFFORT,
                       cache_dir=str(cache_dir) if cache_dir else None, sandbox=not getattr(args, "no_sandbox", False))


def cache_key(prompt: str, model: str = MODEL, effort: str = EFFORT, template: str = "") -> str:
    return hashlib.sha256(f"{model}\n{effort}\n{template}\n{prompt}".encode()).hexdigest()


def key_template(template: str, config: "ModelConfig") -> str:
    """The template as it enters the cache key: itself in the sandbox, plus `|no-sandbox` without it."""
    return template if config.sandbox else f"{template}|no-sandbox"


def in_parallel(fns: list[Any], n: int) -> list[Any]:
    """Each of the callables `fns` called, up to `n` at a time (one after another when `n` <= 1), the results in their
    order; once all have returned, the first failure in their order is raised. The model calls of one query
    (`ExperimentConfig.call_parallelism`)."""
    if n <= 1 or len(fns) <= 1:
        return [f() for f in fns]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(n, len(fns))) as pool:
        futures = [pool.submit(f) for f in fns]
    return [f.result() for f in futures]


def call_key(prompt: str, template: str, config: "ModelConfig") -> str:
    """The cache key of `prompt` under `template` and `config`."""
    return cache_key(prompt, config.model, config.effort, key_template(template, config))


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write `text` to a temporary file beside `path` (mode 0600) and rename it into place: a concurrent reader sees
    the old file or the new one, never a part."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def _auth_lock():
    """The threads of this process and every other process that uses the same credentials file, one at a time (a
    lock file in the system's temporary directory, named after the credentials file's path)."""
    name = hashlib.sha256(str(master_auth().resolve()).encode()).hexdigest()[:16]
    with _lock, open(Path(tempfile.gettempdir()) / f"sqldecomp-auth-{name}.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def bwrap_command(work: Path, codex_home: Path, command: list[str], *, codex_bin: Path | None = None, net: bool = True) -> list[str]:
    """The sandbox around `command` (see the module docstring)."""
    binds = ["--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
             "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin"]
    for etc in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/ssl", "/etc/pki", "/etc/crypto-policies"):
        if os.path.exists(etc):
            binds += ["--ro-bind", etc, etc]
    return (["bwrap", "--unshare-all"] + (["--share-net"] if net else []) + ["--die-with-parent", "--new-session", "--clearenv"]
            + binds + ["--ro-bind", str(codex_bin or codex_binary()), "/opt/codex", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                       "--tmpfs", SANDBOX_HOME, "--bind", str(codex_home), f"{SANDBOX_HOME}/.codex", "--bind", str(work), "/work",
                       "--chdir", "/work", "--setenv", "HOME", SANDBOX_HOME, "--setenv", "CODEX_HOME", f"{SANDBOX_HOME}/.codex",
                       "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "TERM", "dumb", "--setenv", "NO_COLOR", "1"]
            + command)


def codex_command(answer_path: str, codex: str = "/opt/codex") -> list[str]:
    return [codex, *CODEX_COMMAND, "-o", answer_path, "-"]


def _fresh_codex_home(root: Path, model: str, effort: str) -> Path:
    home = Path(tempfile.mkdtemp(prefix="codex_home_", dir=root))
    with _auth_lock():
        shutil.copy(master_auth(), home / "auth.json")
    (home / "auth.json").chmod(0o600)
    (home / "config.toml").write_text(f'model = "{model}"\nmodel_reasoning_effort = "{effort}"\n')
    return home


def start_slot() -> float:
    """Rate-limit model-call starts across processes sharing one lock file.
    SQLDECOMP_CODEX_STARTS=N/S permits one start every S/N seconds. The lock path
    comes from SQLDECOMP_CODEX_STARTS_FILE or ~/.cache/sqldecomp/codex_starts.
    When unset, calls start immediately. Return the number of seconds waited.
    """
    spec = os.environ.get("SQLDECOMP_CODEX_STARTS", "").strip()
    if not spec:
        return 0.0
    n, _, seconds = spec.partition("/")
    gap = float(seconds or 10) / int(n)
    path = Path(os.environ.get("SQLDECOMP_CODEX_STARTS_FILE") or Path.home() / ".cache" / "sqldecomp" / "codex_starts")
    path.parent.mkdir(parents=True, exist_ok=True)
    waited = 0.0
    while True:
        with open(path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            text = fh.read().strip()
            last, now = (float(text) if text else 0.0), time.time()
            if now >= last + gap:
                fh.seek(0)
                fh.truncate()
                fh.write(f"{now:.3f}\n")
                fh.flush()
                return waited
            wait = last + gap - now
        time.sleep(wait)
        waited += wait


def call_timeout(config: ModelConfig) -> float | None:
    """Return config.timeout_s, or None when SQLDECOMP_CODEX_TIMEOUT is none.
    A call then waits for the CLI's answer or error. The timeout is excluded from
    the cache key because it does not change the requested answer.
    """
    return None if os.environ.get("SQLDECOMP_CODEX_TIMEOUT", "").strip().lower() == "none" else config.timeout_s


def _retire_codex_home(home: Path) -> None:
    """A refreshed token is kept for the next call; everything else of the per-call home is deleted."""
    try:
        fresh = home / "auth.json"
        if fresh.exists():
            new = fresh.read_text()
            with _auth_lock():
                if new != master_auth().read_text() and len(new) > 100:
                    atomic_write_text(master_auth(), new)
    finally:
        shutil.rmtree(home, ignore_errors=True)


def cached_call(prompt: str, template: str, config: ModelConfig) -> Call | None:
    """The cached record of the call, if the cache holds it."""
    if not config.cache_dir:
        return None
    key = call_key(prompt, template, config)
    hit = Path(config.cache_dir) / f"{key}.json"
    if not hit.exists():
        return None
    d = json.loads(hit.read_text())
    return Call(d["answer"], d.get("usage", {}), d.get("seconds", 0.0), d.get("exit_code", 0), True, [], d.get("stderr", ""), key,
                config.model, config.effort, template, bool(d.get("sandbox", True)), d.get("template_version", TEMPLATE_VERSION))


def call(prompt: str, template: str, config: ModelConfig, *, scratch: str | Path | None = None) -> Call:
    """One codex call for `prompt` under `config`: the cached record when the cache holds one, else a fresh call in
    the sandbox (or, with `sandbox=False`, without it), cached when it produced an answer. Raises
    `SandboxUnavailable` when the call needs the model, the sandbox is asked for and bwrap is missing."""
    hit = cached_call(prompt, template, config)
    if hit is not None:
        return hit
    require_sandbox(config)
    key = call_key(prompt, template, config)
    if scratch is not None:
        root = Path(scratch)
    elif config.cache_dir:
        root = Path(config.cache_dir) / "_scratch"
    else:
        root = Path(tempfile.gettempdir()) / "sqldecomp_llm_scratch"
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="item_", dir=root))
    home = _fresh_codex_home(root, config.model, config.effort)
    (work / "prompt.txt").write_text(prompt)
    if config.sandbox:
        command = bwrap_command(work, home, codex_command("/work/answer.txt"))
        cwd, env = None, None
    else:
        command = codex_command(str(work / "answer.txt"), str(codex_binary()))
        cwd, env = str(work), {"HOME": str(work), "CODEX_HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                               "TERM": "dumb", "NO_COLOR": "1"}
    start_slot()  # the starts spaced, when asked for: the wait is not the call's latency
    t0 = time.perf_counter()
    try:
        with (work / "prompt.txt").open("rb") as stdin:
            run = subprocess.run(command, stdin=stdin, capture_output=True, text=True, timeout=call_timeout(config), cwd=cwd, env=env)
        exit_code, stdout, stderr = run.returncode, run.stdout, run.stderr
    except subprocess.TimeoutExpired as e:
        exit_code, stdout, stderr = -1, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
    seconds = time.perf_counter() - t0
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    usage = next((e.get("usage", {}) for e in events if e.get("type") == "turn.completed"), {})
    answer = (work / "answer.txt").read_text() if (work / "answer.txt").exists() else ""
    if not answer:  # the last agent message, if the file was not written
        for e in reversed(events):
            item = e.get("item") or {}
            if e.get("type") == "item.completed" and item.get("type") == "agent_message":
                answer = item.get("text", "")
                break
    _retire_codex_home(home)
    shutil.rmtree(work, ignore_errors=True)
    result = Call(answer, usage, seconds, exit_code, False, events, stderr[-2000:], key, config.model, config.effort, template,
                  config.sandbox, config.template_version)
    if config.cache_dir and exit_code == 0 and answer.strip():
        atomic_write_text(Path(config.cache_dir) / f"{key}.json", json.dumps(
            {"key": key, "model": config.model, "effort": config.effort, "template": template, "prompt": prompt, "answer": answer,
             "usage": usage, "seconds": round(seconds, 3), "exit_code": exit_code, "stderr": stderr[-2000:],
             "sandbox": config.sandbox, "template_version": config.template_version,
             "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1) + "\n")
    return result


def usage(cache_dir: str | Path) -> dict[str, Any]:
    """Cumulative token usage over every cached call of a directory."""
    total: dict[str, int] = {}
    n = 0
    for f in Path(cache_dir).glob("*.json"):
        d = json.loads(f.read_text())
        n += 1
        for k, v in (d.get("usage") or {}).items():
            total[k] = total.get(k, 0) + int(v)
    return {"calls": n, **total}


def audit(out_dir: str | Path) -> dict[str, str]:
    """What a process inside the sandbox can see: the file system, the environment, the mounts, the user. Needs bwrap;
    the per-call codex home holds an empty auth.json (the audit never needs a token)."""
    if not sandbox_available():
        raise SandboxUnavailable(sandbox_refusal())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_scratch"
    scratch.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="audit_", dir=scratch))
    (work / "prompt.txt").write_text("audit")
    home = Path(tempfile.mkdtemp(prefix="codex_home_", dir=scratch))
    (home / "auth.json").write_text("{}")
    (home / "config.toml").write_text(f'model = "{MODEL}"\nmodel_reasoning_effort = "{EFFORT}"\n')
    results = {}
    for name, cmd in (("find", ["/usr/bin/find", "/", "-xdev", "-maxdepth", "3", "-not", "-path", "/usr/*", "-not", "-path", "/proc/*", "-not", "-path", "/dev/*"]),
                      ("env", ["/usr/bin/env"]), ("mounts", ["/usr/bin/cat", "/proc/self/mounts"]),
                      ("home", ["/usr/bin/find", SANDBOX_HOME, "/work", "/tmp"]), ("id", ["/usr/bin/id"])):
        run = subprocess.run(bwrap_command(work, home, cmd), capture_output=True, text=True, timeout=120)
        results[name] = run.stdout + (("\n[stderr]\n" + run.stderr) if run.stderr.strip() else "")
        (out / f"{name}.txt").write_text(results[name])
    (out / "command.txt").write_text(" ".join(bwrap_command(Path("<work>"), Path("<codex-home>"), codex_command("/work/answer.txt"),
                                                            codex_bin=Path("<codex binary>"))) + " < <work>/prompt.txt\n")
    shutil.rmtree(home, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    return results
