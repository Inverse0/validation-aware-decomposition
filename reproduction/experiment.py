"""Input preparation and execution of the paper configuration."""
from __future__ import annotations


import collections

import json

import os


import shutil

import subprocess

import sys

import time

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

from typing import Any

from reproduction.data import new_dir

FIELDS = ("key", "sql", "question", "evidence")
DIRECT = "direct"

def arms(name: str = "default") -> list[str]:
    common = ["--model", "gpt-5.6-sol", "--effort", "medium", "--prices", "none", "--call-parallelism", "4"]
    if name == "direct":
        return ["--direct", *common]
    if name != "default":
        raise ValueError("arm must be default or direct")
    return ["--verify", "--direct", "--semantic", "judge", "--semantic-model", "gpt-6-luna",
            "--semantic-effort", "high", "--gamma", "0.25", "--tau", "user-study", "--scoring", "human",
            "--beta", "0.6931471805599453", "--mu", "0.001", "--excess", "relative", "--link", "hinge",
            "--timeout", "3600", "--tasks", "td3-desc", "--writer-model", "gpt-6-luna",
            "--writer-effort", "max", *common]

def run_arguments(arm: str, *, alpha: float | None = None, gamma: float | None = None, verifier_model: str | None = None,
                  verifier_effort: str | None = None, decomposed_only: bool = False, replay: bool = False, decomposition: str | None = None,
                  decomposition_seed: int | None = None, k_file: str | None = None, writer_model: str | None = None,
                  writer_effort: str | None = None, semantic_model: str | None = None, semantic_effort: str | None = None,
                  call_parallelism: int | None = None) -> list[str]:
    """Build pipeline arguments from an arm and explicit run settings.
    Later arguments override arm defaults. alpha scales thresholds; gamma weights
    the semantic score. Verifier, writer and semantic-judge models are configurable.
    replay reads writer and judge answers from caches. decomposition selects the
    optimizer or a baseline; random-k uses k_file and decomposition_seed.
    call_parallelism bounds simultaneous model calls within each query.
    """
    out = arms(arm)
    if arm == DIRECT and (alpha is not None or gamma is not None or decomposed_only or replay or decomposition not in (None, "optimum")
                          or writer_model or writer_effort or semantic_model or semantic_effort):
        raise SystemExit("the direct arm decomposes nothing: --alpha, --gamma, --decomposed-only, --tasks-from, --decomposition, the writer's "
                         "and the judge's model and effort do not apply")
    if (semantic_model or semantic_effort) and out[out.index("--semantic") + 1] != "judge":
        raise SystemExit("--semantic-model / --semantic-effort are the LLM judge's: the arm must use it")
    if (decomposition == "random-k") != bool(k_file):
        raise SystemExit("random-k takes the k of every query (--k-from), and only random-k does")
    if decomposed_only:
        out = [a for a in out if a != "--direct"]
    out += (["--alpha", repr(float(alpha))] if alpha is not None else []) + (["--gamma", repr(float(gamma))] if gamma is not None else [])
    out += (["--verifier-model", verifier_model] if verifier_model else []) + (["--verifier-effort", verifier_effort] if verifier_effort else [])
    out += (["--writer-model", writer_model] if writer_model else []) + (["--writer-effort", writer_effort] if writer_effort else [])
    out += (["--semantic-model", semantic_model] if semantic_model else []) + (["--semantic-effort", semantic_effort] if semantic_effort else [])
    if decomposition and decomposition != "optimum":
        out += ["--decomposition", decomposition] + (["--decomposition-seed", str(decomposition_seed)] if decomposition_seed is not None else [])
        out += ["--k-file", str(k_file)] if k_file else []
    out += ["--call-parallelism", str(call_parallelism)] if call_parallelism is not None else []
    return out + (["--replay"] if replay else [])

def k_of_run(run: str | Path) -> dict[str, int]:
    """Read Opt block counts for matched-budget baselines; omit queries without an optimum."""
    return {r["key"]: r["optimum"]["unique_blocks"] for r in load_run(run) if (r.get("optimum") or {}).get("unique_blocks")}

CACHE_PARTS = {"tasks": ("tasks", "tasks/llm"), "semantic": ("semantic",), "verdicts": ("verdicts",)}

def parse_parts_from(text: str) -> tuple[str, tuple[str, ...]]:
    """`--parts-from PART,...=CACHE` -> (CACHE, the parts in CACHE_PARTS' order), each part named once."""
    parts, sep, cache = text.partition("=")
    names = [p.strip() for p in parts.split(",") if p.strip()]
    if not sep or not cache or not names or any(p not in CACHE_PARTS for p in names) or len(set(names)) != len(names):
        raise SystemExit(f"--parts-from PART,...=CACHE with parts among {', '.join(CACHE_PARTS)}, each once: {text!r}")
    return cache, tuple(p for p in CACHE_PARTS if p in names)

def union_caches(sources: list[tuple[str, tuple[str, ...]]], cache: Path) -> list[dict[str, Any]]:
    """A new cache directory holding the entries of the given caches' parts (`tasks`: the assignments and the writer's
    calls, `semantic`: the LLM judge's calls, `verdicts`: the verifier's calls), each a hard link to the entry (a cache
    entry is written once, by a rename, and never changed afterwards; a copy where a link is not possible). An entry
    that two sources hold is taken from the first (the same key: the same prompt, model, effort and template). Only
    the entries themselves are taken, never a `_scratch` directory (the codex homes of calls in flight). Returns, per
    source and part, the entries taken and those already there."""
    cache.mkdir(parents=True)
    out = []
    for source, parts in sources:
        for part in parts:
            taken = present = 0
            for sub in CACHE_PARTS[part]:
                target = cache / sub
                target.mkdir(parents=True, exist_ok=True)
                for f in sorted((Path(source) / sub).glob("*.json")):
                    if f.name.startswith(".") or not f.is_file():
                        continue
                    if (target / f.name).exists():
                        present += 1
                        continue
                    try:
                        os.link(f, target / f.name)
                    except OSError:
                        shutil.copy2(f, target / f.name)
                    taken += 1
            out.append({"source": str(Path(source).resolve()), "part": part, "taken": taken, "already_there": present})
    return out

def read_keys(text: str | None) -> set[str] | None:
    """--keys: comma-separated ids, or @FILE with one id per line; None when not given."""
    if not text:
        return None
    if text.startswith("@"):
        return {line.strip() for line in open(text[1:]) if line.strip()}
    return set(text.split(","))

def inputs(rows: list[dict[str, Any]], keys: set[str] | None = None) -> dict[str, list[dict[str, str]]]:
    """The queries of every schema file, in dataset order: the generated query, the question and the evidence."""
    out: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for r in rows:
        if keys is None or r["id"] in keys:
            out[r["schema_file"]].append(dict(zip(FIELDS, (r["id"], r["generated_sql"], r["question"], r["evidence"] or ""))))
    return dict(out)

def schema_dir(dataset: str, schemas: str | None) -> Path | None:
    """The directory the schemas are read from by file name: --schemas, else the `schemas/` beside the dataset when
    there is one; None: each row's `schema_file` as written."""
    if schemas:
        if not Path(schemas).is_dir():
            raise SystemExit(f"--schemas: not a directory: {schemas}")
        return Path(schemas)
    beside = Path(dataset).parent / "schemas"
    return beside if beside.is_dir() else None

POOL = "pool"  # the directory of a run's records when one pool takes every file (`run --pool`)

def pool_command(manifest: Path, queries: int, out: Path, cache: Path, jobs: int, run_tag: str, arm_args: list[str], *,
                 resume: bool = False) -> list[str]:
    """The product's command line for every file of the manifest at once (`pipeline --manifest`): one pool of `jobs`
    queries, each against its own file's schema, the records in NEWDIR/pool/."""
    return [sys.executable, "-m", "sqldecomp.pipeline", "--manifest", str(Path(manifest).resolve()), *arm_args, "--cache", str(cache),
            "--out", str(out / POOL), "--jobs", str(max(1, min(jobs, queries))), *(["--run-tag", run_tag] if run_tag else []),
            *(["--resume"] if resume else [])]

def command(entry: dict[str, Any], out: Path, cache: Path, jobs: int, run_tag: str = "", arm: str = "default", resume: bool = False,
            arm_args: list[str] | None = None) -> list[str]:
    """The product's own command line for one file of the manifest (`arm_args`: the run's arguments, `run_arguments`;
    default the arm's own; `resume`: continue the file's run that stopped)."""
    return [sys.executable, "-m", "sqldecomp.pipeline", "--queries", entry["file"], "--schema", entry["schema"],
            *(arm_args if arm_args is not None else arms(arm)),
            "--cache", str(cache), "--out", str(out / entry["name"]), "--jobs", str(max(1, min(jobs, entry["queries"]))),
            *(["--run-tag", run_tag] if run_tag else []), *(["--resume"] if resume else [])]

def cmd_inputs(args) -> int:
    rows = [json.loads(line) for line in open(args.dataset)]
    keys = read_keys(args.keys)
    unknown = sorted(keys - {r["id"] for r in rows}) if keys else []
    if unknown:
        raise SystemExit(f"keys not in the dataset: {unknown}")
    schemas = schema_dir(args.dataset, args.schemas)
    by_schema = inputs(rows, keys)
    where = {s: schemas / Path(s).name if schemas else Path(s) for s in by_schema}
    missing = sorted(str(p) for p in where.values() if schemas and not p.is_file())
    if missing:
        raise SystemExit(f"schema files not found: {missing}")
    out = new_dir(args.out)
    manifest = []
    for schema, queries in by_schema.items():
        name = Path(schema).stem
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries))
        manifest.append({"name": name, "file": str(path.resolve()), "schema": str(where[schema].resolve()), "queries": len(queries)})
    (out / "manifest.json").write_text(json.dumps({"dataset": str(Path(args.dataset).resolve()), "inputs": manifest}, indent=1) + "\n")
    print(f"{sum(m['queries'] for m in manifest)} queries in {len(manifest)} files", file=sys.stderr)
    return 0

def cmd_run(args) -> int:
    manifest = json.loads((Path(args.inputs) / "manifest.json").read_text())["inputs"]
    if args.resume:  # a run that stopped: its finished files stay, the others go on from their partial records
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = new_dir(args.out)
    earlier = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else None
    cache = Path(args.cache)
    t0 = time.time()
    # an unknown arm or a setting that does not apply is refused before any cache is made or any file runs
    k_file = out / "k.json" if args.decomposition == "random-k" else None
    if k_file is not None and not k_file.exists():
        if not args.k_from:
            raise SystemExit("--decomposition random-k needs --k-from RUN (the run whose optimum gives each query's k)")
        k_file.write_text(json.dumps(k_of_run(args.k_from), indent=1, sort_keys=True) + "\n")
    arm_args = run_arguments(args.arm, alpha=args.alpha, gamma=args.gamma, verifier_model=args.verifier_model, verifier_effort=args.verifier_effort,
                             decomposed_only=args.decomposed_only, replay=bool(args.tasks_from), decomposition=args.decomposition,
                             decomposition_seed=args.decomposition_seed if args.decomposition == "random-k" else None,
                             k_file=str(k_file.resolve()) if k_file is not None else None, writer_model=args.writer_model,
                             writer_effort=args.writer_effort, semantic_model=args.semantic_model, semantic_effort=args.semantic_effort,
                             call_parallelism=args.call_parallelism)
    arm_args += ["--prices", getattr(args, "prices", "none")]
    sources = ([(d, ("tasks", "semantic", "verdicts")) for d in args.cache_from] + [parse_parts_from(p) for p in args.parts_from]
               + [(d, ("verdicts",)) for d in args.verdicts_from] + ([(args.tasks_from, ("tasks", "semantic"))] if args.tasks_from else []))
    copied = None
    if sources and not args.resume:  # a cache that starts from others is a new directory: nothing earlier is written into
        if cache.exists():
            raise SystemExit(f"--cache {cache} exists: a cache that starts from others (--cache-from, --parts-from, --verdicts-from, --tasks-from) "
                             "is a new directory")
        missing = [d for d, _ in sources if not Path(d).is_dir()]
        if missing:
            raise SystemExit(f"not a cache directory: {missing}")
        copied = union_caches(sources, cache)
    cache.mkdir(parents=True, exist_ok=True)

    def run(entry: dict[str, Any]) -> dict[str, Any]:
        if args.resume and (out / entry["name"] / "results.jsonl").exists():
            return {"name": entry["name"], "queries": entry["queries"], "exit": 0, "seconds": 0.0, "finished_before": True}
        resume = args.resume and (out / entry["name"]).exists()
        cmd = (pool_command(Path(args.inputs) / "manifest.json", sum(e["queries"] for e in manifest), out, cache, args.jobs, args.run_tag, arm_args,
                            resume=resume) if args.pool else
               command(entry, out, cache, args.jobs, args.run_tag, args.arm, resume=resume, arm_args=arm_args))
        t = time.time()
        with open(out / f"{entry['name']}.log", "a" if args.resume else "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{entry['name']}: {entry['queries']} queries, exit {code}, {time.time() - t:.0f} s", file=sys.stderr)
        return {"name": entry["name"], "queries": entry["queries"], "exit": code, "seconds": round(time.time() - t, 1), "command": cmd}

    if args.pool:  # one pool over every file: --jobs bounds the whole run's queries at a time, whichever file they are of
        runs = [run({"name": POOL, "queries": sum(e["queries"] for e in manifest)})]
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.files)) as pool:
            runs = list(pool.map(run, sorted(manifest, key=lambda e: -e["queries"])))
    from sqldecomp.runner import code_version
    record = {"seconds": round(time.time() - t0, 1), "arm": args.arm, "arms": arm_args, "argv": list(sys.argv), "code": code_version(),
              "cwd": os.getcwd(), "cache": str(cache.resolve()), **({"cache_sources": copied} if copied is not None else {}), "runs": runs,
              **({"earlier": earlier} if earlier else {})}
    (out / "runs.json").write_text(json.dumps(record, indent=1) + "\n")
    return 0 if all(r["exit"] == 0 for r in runs) else 1

def load_run(run: str | Path) -> list[dict[str, Any]]:
    """The result records of a run directory (`run`'s NEWDIR), or of several separated by commas (a run stopped and
    continued in another directory; a key twice is refused), in layers separated by semicolons: a later layer's
    records replace the earlier ones with the same keys."""
    out: dict[str, dict[str, Any]] = {}
    for layer in str(run).split(";"):
        records = [json.loads(line) for d in layer.split(",") for f in sorted(Path(d).glob("*/results.jsonl")) for line in open(f)]
        if len({r["key"] for r in records}) != len(records):
            raise SystemExit(f"a key occurs twice in {layer}")
        out.update({r["key"]: r for r in records})
    return [out[k] for k in sorted(out)] if ";" in str(run) else list(out.values())
