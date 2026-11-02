"""Round-trip critique and synthetic execution on decomposition blocks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from reproduction import baselines as A

APPROACHES = ("rt", "se")
ARMS = ("rt_only", "se_only")
TYPES = ("BIGINT", "DOUBLE PRECISION", "DATE", "TIMESTAMP")


# --------------------------------------------------------------------------------------------------------------
# the input rows
# --------------------------------------------------------------------------------------------------------------


def _comment(text: str) -> list[str]:
    return ["-- " + line if line.strip() else "--" for line in text.splitlines()]


def never_null(c: Any) -> set[str]:
    """The columns the contract's facts establish as never NULL (`tasks.fact_sheet`'s "never NULL: ..." part:
    a base column declared NOT NULL, reached without an outer join). The contract's `nullable` flag is not this: it
    marks only a column an outer join inside the upstream block NULL-extends (the verifier's "may be NULL")."""
    m = re.search(r"(?:^|; )never NULL: (.*?)(?:; |$)", c.facts or "")
    names = set(m.group(1).split(", ")) if m else set()
    unknown = names - {col["name"] for col in c.columns}
    if unknown:
        raise ValueError(f"{c.name}: never NULL names no column of the contract: {sorted(unknown)}")
    return names


def contract_statement(c: Any) -> str:
    """An input relation as a CREATE TABLE statement: what `InputContract.text` shows the verifier (columns, types,
    keys, NULLs, the description, the facts, the parameterized mark), the lines other than the columns and keys as SQL
    comments above it. NOT NULL marks exactly the facts' never-NULL columns; the verifier's "may be NULL" line is a
    comment."""
    notes = []
    if c.requirement and c.requirement.startswith("FACTS"):
        notes.append(c.requirement)
    elif c.requirement:
        notes.append(f"{c.lead}: {c.requirement}")
    if c.facts:
        notes.append(f"established by the step that computes it (verified separately; take as given): {c.facts}")
    from sqldecomp.tasks import WEAK_KEY_LEAD, keys_text
    if any(not k for k in c.keys or ()):
        notes.append("at most one row")  # the empty key, in the verifier's words (no UNIQUE () in SQL)
    if c.weak_keys:
        notes.append(keys_text(c.weak_keys, lead=WEAK_KEY_LEAD))
    nullable = [col["name"] for col in c.columns if col.get("nullable")]
    if nullable:
        notes.append("may be NULL: " + ", ".join(nullable))
    if c.parameterized:
        notes.append("parameterized: it reads columns of the enclosing query")
    never = never_null(c)
    lines = []
    for col in c.columns:
        typ = col.get("type")
        line = f"  {A._ident(col['name'])}" + (f" {typ}" if typ and typ != "unknown" else "")
        if col["name"] in never:
            line += " NOT NULL"
        lines.append(line)
    for k in c.keys or ():
        if k:
            lines.append(f"  UNIQUE ({', '.join(map(A._ident, k))})")
    body = ",\n".join(lines)
    return "\n".join([*(x for n in notes for x in _comment(n)), f"CREATE TABLE {A._ident(c.name)} (\n{body}\n);"])


def contract_schema(c: Any) -> dict[str, Any]:
    """An input relation in their db_schema vocabulary."""
    return {"table_name": c.name, "columns": [{"name": col["name"], "type": A.THEIR_TYPE.get(str(col.get("type")).upper(), "others")}
                                              for col in c.columns], "primary_keys": [], "foreign_keys": []}


def contract_table(c: Any) -> dict[str, Any]:
    """An input relation as a schema-JSON table for DuckDB (`unknown` types are inferred from the data)."""
    return {"name": c.name, "columns": [{"name": col["name"], "type": col.get("type") or "unknown"} for col in c.columns]}


def parameters_text(task: Any) -> str:
    """The PARAMETERS section of `Task.inputs_text`, as the verifier saw it."""
    if not task.parameters:
        return ""
    return "\n".join(["PARAMETERS (columns of the enclosing query that this block reads; the block is evaluated once per row of that query)",
                      *(f"  {p}" for p in task.parameters)])


def block_rows(record: dict[str, Any], row: dict[str, Any], schema_json: dict[str, Any], prompt_of=None) -> list[dict[str, Any]]:
    """The input rows of one candidate's verdict items (their columns; ours under `q_*`). `prompt_of(cache_key)`
    returns the verifier's cached prompt: each item is checked to render to it byte for byte."""
    from sqldecomp import verify
    from sqldecomp.tasks import InputContract, Task
    tasks = [Task(**{**t, "input_contracts": [InputContract(**c) for c in t["input_contracts"]]}) for t in record["tasks"]["blocks"]]
    by_task = {t.output: t for t in tasks}
    recs = [{**b, "tables_own": b["tables"]} for b in record["optimum"]["blocks"]]
    evidence = row.get("evidence") or ""
    items = {it["item"]: it for it in verify.block_items(tasks, recs, evidence=evidence, schema_json=schema_json)}
    out = []
    for v in record["verdicts"]["items"]:
        it = items[v["item"]]
        if prompt_of is not None and verify.render(it) != prompt_of(v["key"]):
            raise SystemExit(f"{record['key']} {v['item']}: the rebuilt item does not render to the verifier's prompt")
        t = by_task[it["output"]]
        own = {x.lower() for x in next(b for b in recs if b["output"] == it["output"])["tables_own"]}
        base = [tb for tb in schema_json["tables"] if tb["name"].lower() in own]
        params = parameters_text(t)
        info = "\n\n".join(x for x in (evidence.strip(), params) if x) or None
        out.append({
            "uuid": f"{record['key']}|{v['item']}", "q_key": record["key"], "q_item": v["item"], "q_output": it["output"],
            "q_is_root": t.is_root, "q_parameterized": bool(t.parameters), "q_benchmark": row["benchmark"], "q_database": row["database"],
            "q_refused": False, "q_opt_verdict": v["verdict"], "q_verifier_key": v["key"], "q_inputs": [c.name for c in t.input_contracts],
            "q_tables_own": sorted(own), "q_unknown_columns": sum(1 for c in t.input_contracts for col in c.columns if (col.get("type") or "unknown") == "unknown"),
            "q_duck_schema": {"tables": [*base, *(contract_table(c) for c in t.input_contracts)]},
            "original_nlq": t.requirement, "additional_information": info, "generated_sql": it["sql"], "original_sql": it["sql"],
            "db_schema": {"db_id": row["database"], "tables": [*A.their_schema(schema_json, own, row["database"])["tables"],
                                                               *(contract_schema(c) for c in t.input_contracts)]},
            "db_create_statements": [*(A.create_statement(tb) for tb in base), *(contract_statement(c) for c in t.input_contracts)],
        })
    return out


def cmd_inputs(args) -> int:
    dataset = {json.loads(line)["id"]: json.loads(line) for line in open(args.dataset)}
    schemas_dir = Path(args.schemas) if args.schemas else Path(args.dataset).parent / "schemas"
    cache = Path(args.verifier_cache)

    def prompt_of(key: str) -> str:
        return json.loads((cache / "verdicts" / f"{key}.json").read_text())["prompt"]

    out = A.new_dir(args.out)
    rows, schemas, refused = [], {}, []
    for f in sorted(Path(args.opt).glob("*/results.jsonl")):
        for line in open(f):
            r = json.loads(line)
            row = dataset[r["key"]]
            if not r.get("optimum") or not (r.get("verdicts") or {}).get("items"):
                refused.append(r["key"])
                rows.append({"uuid": f"{r['key']}|", "q_key": r["key"], "q_item": None, "q_benchmark": row["benchmark"], "q_refused": True})
                continue
            name = Path(row["schema_file"]).name
            if name not in schemas:
                schemas[name] = json.loads((schemas_dir / name).read_text())
            rows += block_rows(r, row, schemas[name], prompt_of)
    (out / "inputs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    blocks = [r for r in rows if not r["q_refused"]]
    summary = {"argv": list(sys.argv), "dataset": str(Path(args.dataset).resolve()), "opt": str(Path(args.opt).resolve()),
               "verifier_cache": str(cache.resolve()), "schemas": str(schemas_dir.resolve()), "code": A.code_version(),
               "candidates": len({r["q_key"] for r in rows}), "refused": sorted(refused), "blocks": len(blocks),
               "prompts_checked": len(blocks), "parameterized": sum(r["q_parameterized"] for r in blocks), "roots": sum(r["q_is_root"] for r in blocks),
               "with_inputs": sum(1 for r in blocks if r["q_inputs"]), "unknown_columns": sum(r["q_unknown_columns"] for r in blocks),
               "with_information": sum(1 for r in blocks if r["additional_information"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("argv", "refused")}, indent=1), file=sys.stderr)
    return 0


# --------------------------------------------------------------------------------------------------------------
# SE-b's execution
# --------------------------------------------------------------------------------------------------------------

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TS = re.compile(r"\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(z|[+-]\d{2}(:?\d{2})?)?", re.I)


def infer_type(values: list[Any]) -> str:
    """The type of a column the pipeline could not type, from its generated values (None ignored): BIGINT if every
    value is an integer, DOUBLE PRECISION if every value is a number, DATE / TIMESTAMP if every value is one in ISO
    form, else TEXT (also with no value). The string "null" (any case) is a NULL: their data template asks for every
    value as a string, and TRY_CAST makes "null" NULL in a typed column."""
    vals = [v for v in values if v is not None and not (isinstance(v, str) and v.strip().lower() == "null")]
    if not vals:
        return "TEXT"

    def is_int(v):
        if isinstance(v, bool):
            return False
        if isinstance(v, int):
            return True
        return isinstance(v, str) and re.fullmatch(r"[+-]?\d+", v.strip()) is not None

    def is_num(v):
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return True
        try:
            float(v)
            return isinstance(v, str) and v.strip().lower() not in ("nan", "inf", "-inf", "infinity", "-infinity")
        except (TypeError, ValueError):
            return False

    if all(is_int(v) for v in vals):
        return "BIGINT"
    if all(is_num(v) for v in vals):
        return "DOUBLE PRECISION"
    if all(isinstance(v, str) and _DATE.fullmatch(v.strip()) for v in vals):
        return "DATE"
    if all(isinstance(v, str) and _TS.fullmatch(v.strip()) for v in vals):
        return "TIMESTAMP"
    return "TEXT"


def typed_schema(schema: dict[str, Any], data: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """`schema` with each `unknown` column typed by `infer_type` over the generated data (table and column names
    matched lowercased, as `duckdb_execute` reads the data); -> (schema, {table.column: inferred type})."""
    data = data if isinstance(data, dict) else {}
    lower = {str(k).lower(): v for k, v in data.items()}
    inferred, tables = {}, []
    for t in schema["tables"]:
        rows = lower.get(t["name"].lower()) or []
        cols = []
        for c in t["columns"]:
            if (c.get("type") or "unknown") == "unknown":
                vals = [next((v for k, v in r.items() if str(k).lower() == c["name"].lower()), None) for r in rows if isinstance(r, dict)]
                typ = infer_type(vals)
                inferred[f"{t['name']}.{c['name']}"] = typ
                c = {**c, "type": typ}
            cols.append(c)
        tables.append({**t, "columns": cols})
    return {"tables": tables}, inferred


# --------------------------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------------------------


def summarize(approach: str, row: dict[str, Any], outputs: list) -> dict[str, Any]:
    """One check's verdict on one block from their outputs."""
    if approach == "rt":
        return A.summarize("rt", row, outputs, None)
    if len(outputs) != 1:
        return {"verdict": None, "error": f"{len(outputs)} outputs (one expected)"}
    _row, gq, ga, gm, eq, ea, em, err = outputs[0]
    if err:
        return {"verdict": None, "error": err[:2000]}
    answer_sorted = ga.answer_generation.answer_dictionary_sorted
    data = ga.data_generation.generated_data
    schema, inferred = typed_schema(row["q_duck_schema"], data)
    duck = A.duckdb_execute(row["generated_sql"], schema, data)
    v = A.se_verdict(row["generated_sql"], duck, answer_sorted)
    out = {"verdict": v, "duckdb": duck, "inferred_types": inferred, "answer_sorted": answer_sorted,
           "answer_sentence": ga.answer_generation.answer_sentence, "sqlite_error": ea.generated_sql_execution_error,
           "tables_generated": sorted(data) if isinstance(data, dict) else None,
           "rows_generated": {t: len(x) for t, x in data.items() if isinstance(x, list)} if isinstance(data, dict) else None}
    if v is None:
        out["na"] = "execution_error"
    return out


async def run_block(row: dict[str, Any], approach: str, runner) -> tuple[dict[str, Any], dict[str, Any]]:
    import pandas as pd
    t0 = time.perf_counter()
    started = A._now()
    base = {"key": row["uuid"], "q_key": row["q_key"], "q_item": row.get("q_item"), "approach": approach}
    if row["q_refused"]:
        return {**base, "refused": True, approach: {"verdict": None, "not_run": "the importer refused the candidate: undecided"},
                "started": started, "ended": A._now(), "seconds": 0.0}, {}
    if approach == "se" and row["q_parameterized"]:
        return {**base, "refused": False, approach: {"verdict": None, "na": "parameterized", "calls": [],
                                                     "not_run": "the block reads columns of the enclosing query: it cannot run alone"},
                "started": started, "ended": A._now(), "seconds": 0.0}, {}
    calls: list = []
    A.CALLS.set(calls)
    obj, spec = runner
    outputs = await obj.experiment(pd.Series(row), spec)
    try:
        summary = summarize(approach, row, outputs)
    except Exception as e:  # noqa: BLE001
        summary = {"verdict": None, "error": f"summary failed: {type(e).__name__}: {e}"}
    rec = {**base, "refused": False, approach: {**summary, "calls": calls}, "started": started, "ended": A._now(),
           "seconds": round(time.perf_counter() - t0, 1)}
    return rec, {"key": row["uuid"], approach: A._flatten(outputs)}


class BlockClient(A.CodexClient):
    """Model client with an answered-call cap per block and prompt.
    RT back-translation prompts contain SQL, schema and evidence, so distinct blocks
    can share a prompt. Each block receives its own ANSWER_ATTEMPTS budget. Attempt
    labels are unique across the run; seed counts cached answers when resuming.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.per_block = self.attempts
        self._start: dict[str, int] = {}

    def start_block(self) -> None:
        self._start = {}

    def seed(self, cache: Path) -> dict[str, int]:
        for f in sorted(Path(cache).glob("*.json")):
            c = json.loads(f.read_text())
            m = re.fullmatch(r"alrashed2026-(.+)-" + re.escape(A.TEMPLATE_VERSION) + r"(\|attempt-\d+)?", c.get("template") or "")
            if m and c.get("exit_code") == 0 and (c.get("answer") or "").strip():
                pkey = hashlib.sha256(f"{m.group(1)}\n{c['prompt']}".encode()).hexdigest()
                self._answered[pkey] = self._answered.get(pkey, 0) + 1
        return dict(self._answered)

    def generate(self, model: str, contents: str, config: Any):
        role = self.role_of(model)
        pkey = hashlib.sha256(f"{role}\n{self.prompt_of(contents, config)}".encode()).hexdigest()
        with self._lock:  # one call in flight: the cap below is this block's
            start = self._start.setdefault(pkey, self._answered.get(pkey, 0))
            self.attempts = start + self.per_block
        return super().generate(model, contents, config)


class _DryCall:
    """Stands in for `sqldecomp.llm.call` in a dry run: records the prompt, answers nothing (the call fails)."""

    def __init__(self, out: Path):
        self.out, self.n = out, 0

    def __call__(self, prompt: str, template: str, config: Any):
        from sqldecomp.llm import Call
        self.n += 1
        (self.out / f"{self.n:05d}_{template.replace('|', '_')}.txt").write_text(prompt)
        return Call("", {}, 0.0, 1, key="dry", model=config.model, effort=config.effort, template=template, stderr="dry run")


def cmd_run(args) -> int:
    A.MODEL = getattr(args, "model", A.MODEL)
    A.EFFORT = getattr(args, "effort", A.EFFORT)
    from sqldecomp import llm
    rows = [json.loads(line) for line in open(Path(args.inputs) / "inputs.jsonl")]
    only = A._keys(args.keys)
    if only is not None:
        unknown = sorted(only - {r["uuid"] for r in rows})
        if unknown:
            raise SystemExit(f"--keys not in the inputs: {unknown[:5]}")
        rows = [r for r in rows if r["uuid"] in only]
    if args.limit:
        rows = rows[:args.limit]
    cache = Path(args.cache)
    if cache.exists() and not args.reuse_cache and not args.dry_run:
        raise SystemExit(f"--cache must be new (or --reuse-cache after a crash): {cache}")
    out = A.new_dir(args.out)
    (out / "raw").mkdir()
    config = llm.ModelConfig(model=A.MODEL, effort=A.EFFORT, cache_dir=None if args.dry_run else str(cache), sandbox=True)
    if not args.dry_run:
        cache.mkdir(parents=True, exist_ok=True)
        llm.require_sandbox(config)
    runners = A.approaches()
    if args.dry_run:
        (out / "prompts").mkdir()
        client = BlockClient(config, waits=(), attempts=1, call=_DryCall(out / "prompts"))
    else:
        client = BlockClient(config)
    seeded = {} if not args.seed_cache else client.seed(Path(args.seed_cache))
    A.patch_genai(client)
    runner = runners[args.approach]
    meta = {"argv": list(sys.argv), "approach": args.approach, "inputs": str(Path(args.inputs).resolve()), "cache": str(cache.resolve()),
            "out": str(out.resolve()), "model": A.MODEL, "effort": A.EFFORT, "sandbox": True, "timeout_s": config.timeout_s, "jobs": 1,
            "calls_in_flight": 1, "dry_run": args.dry_run, "answer_attempts": A.ANSWER_ATTEMPTS, "retry_waits_s": list(A.RETRY_WAITS_S),
            "rt_accept": A.RT_ACCEPT, "keys": args.keys, "reuse_cache": args.reuse_cache, "answer_cap": "per block",
            "seed_cache": args.seed_cache and str(Path(args.seed_cache).resolve()), "seeded_prompts": len(seeded), "spec": A.specs()[args.approach],
            "their_env": A.THEIR_ENV, "template_version": A.TEMPLATE_VERSION, "code": A.code_version(), "baseline_src": str(A.BASELINE_SRC),
            "codex": A._codex_version(), "python": sys.version.split()[0], "packages": A._packages(), "started": A._now(), "blocks": len(rows)}
    (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    partial = out / "records.partial.jsonl"

    async def main():
        for i, row in enumerate(rows, 1):  # one block at a time; their RoundTrip and DataGen ask their 2 calls one after another
            client.start_block()
            try:
                rec, raw = await run_block(row, args.approach, runner)
            except Exception as e:  # noqa: BLE001
                rec, raw = {"key": row["uuid"], "q_key": row["q_key"], "q_item": row.get("q_item"), "approach": args.approach,
                            "crash": f"{type(e).__name__}: {e}", args.approach: {"verdict": None, "error": "crash"}}, {}
            with open(partial, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            if raw:
                (out / "raw" / f"{row['uuid'].replace(':', '_').replace('|', '__')}.json").write_text(json.dumps(raw, default=str, indent=1) + "\n")
            r = rec.get(args.approach) or {}
            print(f"{A._now()} [{i}/{len(rows)}] {row['uuid']} {r.get('verdict')} {r.get('na') or ''} {rec.get('seconds', '')}s",
                  file=sys.stderr, flush=True)

    import concurrent.futures
    loop = asyncio.new_event_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=2))
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    recs = [json.loads(line) for line in open(partial)]
    (out / "records.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in recs))
    partial.unlink()
    calls = [c for r in recs for c in (r.get(args.approach) or {}).get("calls", [])]
    meta.update({"ended": A._now(), "calls": len(calls), "calls_cached": sum(1 for c in calls if c.get("cached")),
                 "calls_failed": sum(1 for c in calls if c.get("exit_code") != 0),
                 "verdicts": {str(v): sum(1 for r in recs if (r.get(args.approach) or {}).get("verdict") == v) for v in ("yes", "no", None)},
                 "na": {k: sum(1 for r in recs if (r.get(args.approach) or {}).get("na") == k) for k in ("parameterized", "execution_error")}})
    (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("calls", "calls_cached", "calls_failed", "verdicts", "na")}), file=sys.stderr)
    return 0


# --------------------------------------------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------------------------------------------


def block_verdict(arm: str, opt: str | None, check: dict[str, Any] | None) -> str | None:
    """A block's verdict in an arm: `opt` the verifier's, `check` RT-b's or SE-b's record part; RT-b or SE-b replaces
    the verifier."""
    check = check or {}
    if check.get("na"):
        return opt  # not applicable: the verifier's verdict stands
    v = check.get("verdict")
    return v if v in ("yes", "no") else None


def cmd_arms(args) -> int:
    rows = [json.loads(line) for line in open(Path(args.inputs) / "inputs.jsonl")]
    checks: dict[str, dict[str, dict[str, Any]]] = {"rt": {}, "se": {}}
    for approach, runs in (("rt", args.rt), ("se", args.se)):
        for run in runs:
            for r in A.run_records(Path(run)):
                if r["key"] in checks[approach]:
                    raise SystemExit(f"{r['key']} occurs twice in the {approach} runs")
                checks[approach][r["key"]] = r.get(approach) or {}
    blocks = [r for r in rows if not r["q_refused"]]
    for approach in APPROACHES:
        missing = [r["uuid"] for r in blocks if r["uuid"] not in checks[approach]]
        if missing:
            raise SystemExit(f"{len(missing)} blocks have no {approach} record, e.g. {missing[:3]}")
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault(r["q_key"], []).append(r)
    out = A.new_dir(args.out)
    counts = {}
    for arm in ARMS:
        approach = "rt" if arm == "rt_only" else "se"
        d = out / arm / "pool"
        d.mkdir(parents=True)
        lines, verdicts = [], {}
        for key, rs in by_key.items():
            if rs[0]["q_refused"]:
                v, items = None, []
            else:
                items = [{"item": r["q_item"], "opt": r["q_opt_verdict"], "check": checks[approach][r["uuid"]].get("verdict"),
                          "na": checks[approach][r["uuid"]].get("na"),
                          "verdict": block_verdict(arm, r["q_opt_verdict"], checks[approach][r["uuid"]])} for r in rs]
                v = A.conjunction([it["verdict"] for it in items])
            verdicts[key] = v
            findings = [{"stage": "import", "kind": "refused"}] if rs[0]["q_refused"] else []
            lines.append(json.dumps({"key": key, "refused": rs[0]["q_refused"], "findings": findings,
                                     "direct": {"final": v, "rule": "conjunction", "items": items}}) + "\n")
        (d / "results.jsonl").write_text("".join(lines))
        counts[arm] = {str(x): sum(1 for v in verdicts.values() if v == x) for x in ("yes", "no", None)}
    meta = {"argv": list(sys.argv), "inputs": str(Path(args.inputs).resolve()), "rt": [str(Path(r).resolve()) for r in args.rt],
            "se": [str(Path(r).resolve()) for r in args.se], "candidates": len(by_key), "blocks": len(blocks), "verdicts": counts,
            "code": A.code_version()}
    (out / "arms.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1), file=sys.stderr)
    return 0
