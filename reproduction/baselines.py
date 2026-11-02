"""Round-trip critique and synthetic execution on whole queries."""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

BASELINE_SRC = Path(__file__).resolve().parents[1]
MODEL, EFFORT = "gpt-5.6-sol", "medium"  # V1
TEMPLATE_VERSION = "v1"
ANSWER_ATTEMPTS = 3  # answered calls per prompt (a fresh call per retry of an answer that does not parse)
RETRY_WAITS_S = (30, 60, 120, 240, 480)  # a failed call (capacity, 429, no answer) is asked again after each wait
DUCKDB_TIMEOUT_S = 60.0
ROLES = ("rt-generator", "rt-evaluator", "se-answer", "se-data")
APPROACHES = ("rt", "se")
RT_ACCEPT = 4  # similarity_score >= 4 accepts (their metric: score > threshold, the paper's threshold 3)

# our schema JSON's types -> their db_schema vocabulary (DATATYPE_MAP of their DataGen); and DuckDB's types
THEIR_TYPE = {"TEXT": "text", "BIGINT": "integer", "INTEGER": "integer", "DOUBLE PRECISION": "real", "TIMESTAMP": "datetime",
              "DATE": "date"}
DUCK_TYPE = {"TEXT": "VARCHAR", "BIGINT": "BIGINT", "INTEGER": "BIGINT", "DOUBLE PRECISION": "DOUBLE", "TIMESTAMP": "TIMESTAMP",
             "DATE": "DATE"}


def new_dir(path: str | Path) -> Path:
    out = Path(path)
    if out.exists():
        raise SystemExit(f"refusing to write into an existing directory: {out}")
    out.mkdir(parents=True)
    return out


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _keys(text: str | None) -> set[str] | None:
    if not text:
        return None
    if text.startswith("@"):
        return {line.strip() for line in open(text[1:]) if line.strip()}
    return {k.strip() for k in text.split(",") if k.strip()}


def code_version() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, timeout=20).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", False
    return {"commit": commit or "unknown", "dirty": dirty}


# --------------------------------------------------------------------------------------------------------------
# the inputs
# --------------------------------------------------------------------------------------------------------------


def _ident(name: str) -> str:
    return name if re.fullmatch(r"[a-z_][a-z0-9_]*", name) else '"' + name.replace('"', '""') + '"'


def their_schema(schema_json: dict[str, Any], tables: set[str], db_id: str) -> dict[str, Any]:
    """Their `DatabaseSchemaDict` of the given tables (in the schema's order): columns with their type vocabulary,
    the primary key, one foreign-key entry per column pair."""
    out = []
    for t in schema_json["tables"]:
        if t["name"].lower() not in tables:
            continue
        fks = [{"column": c, "references_table": f["ref_table"], "references_column": rc}
               for f in t.get("foreign_keys") or () for c, rc in zip(f["columns"], f["ref_columns"])]
        out.append({"table_name": t["name"], "columns": [{"name": c["name"], "type": THEIR_TYPE.get(str(c.get("type")).upper(), "others")}
                                                         for c in t["columns"]],
                    "primary_keys": list(t.get("primary_key") or ()), "foreign_keys": fks})
    return {"db_id": db_id, "tables": out}


def create_statement(t: dict[str, Any]) -> str:
    """A PostgreSQL CREATE TABLE statement of a schema-JSON table: its columns, NOT NULL, the primary key, the UNIQUE
    keys and the foreign keys as the schema declares them."""
    pk = list(t.get("primary_key") or ())
    lines = [f"  {_ident(c['name'])} {c.get('type', 'TEXT')}" + (" NOT NULL" if c.get("nullable") is False and c["name"] not in pk else "")
             for c in t["columns"]]
    if pk:
        lines.append(f"  PRIMARY KEY ({', '.join(map(_ident, pk))})")
    for u in t.get("unique_keys") or ():
        cols = u.get("columns") if isinstance(u, dict) else u
        if cols and list(cols) != pk:
            lines.append(f"  UNIQUE ({', '.join(map(_ident, cols))})")
    for f in t.get("foreign_keys") or ():
        lines.append(f"  FOREIGN KEY ({', '.join(map(_ident, f['columns']))}) REFERENCES {_ident(f['ref_table'])} "
                     f"({', '.join(map(_ident, f['ref_columns']))})")
    return f"CREATE TABLE {_ident(t['name'])} (\n" + ",\n".join(lines) + "\n);"


def input_row(row: dict[str, Any], record: dict[str, Any], schema_json: dict[str, Any]) -> dict[str, Any]:
    """Their input row of one candidate (their column names; ours beside them under `q_*`)."""
    from sqldecomp.tasks import tables_in
    sql = record.get("sql_printed")
    base = {"uuid": row["id"], "q_key": row["id"], "q_benchmark": row["benchmark"], "q_database": row["database"],
            "original_nlq": row["question"], "additional_information": row.get("evidence") or None,
            "original_sql": row["gold_sql"]}
    if sql is None:
        return {**base, "q_refused": True, "q_refusal": (record.get("import") or {}).get("detail") or (record.get("direct") or {}).get("not_run"),
                "generated_sql": None, "db_schema": None, "db_create_statements": None, "q_tables": []}
    tables = tables_in(sql)
    known = {t["name"].lower() for t in schema_json["tables"]}
    missing = sorted(tables - known)
    return {**base, "q_refused": False, "generated_sql": sql, "q_tables": sorted(tables), "q_unknown_tables": missing,
            "db_schema": their_schema(schema_json, tables, row["database"]),
            "db_create_statements": [create_statement(t) for t in schema_json["tables"] if t["name"].lower() in tables]}


def cmd_inputs(args) -> int:
    dataset = [json.loads(line) for line in open(args.dataset)]
    records = {}
    for f in sorted(Path(args.direct).glob("*/results.jsonl")):
        for line in open(f):
            r = json.loads(line)
            records[r["key"]] = r
    schemas_dir = Path(args.schemas) if args.schemas else Path(args.dataset).parent / "schemas"
    out = new_dir(args.out)
    rows, cache = [], {}
    for row in dataset:
        if row["id"] not in records:
            continue
        name = Path(row["schema_file"]).name
        if name not in cache:
            cache[name] = json.loads((schemas_dir / name).read_text())
        if row["id"] not in records:
            raise SystemExit(f"{row['id']} has no record in {args.direct}")
        rows.append(input_row(row, records[row["id"]], cache[name]))
    (out / "inputs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    summary = {"argv": list(sys.argv), "dataset": str(Path(args.dataset).resolve()), "direct": str(Path(args.direct).resolve()),
               "schemas": str(schemas_dir.resolve()), "code": code_version(), "candidates": len(rows),
               "refused": sorted(r["uuid"] for r in rows if r["q_refused"]),
               "with_evidence": sum(1 for r in rows if r["additional_information"]),
               "unknown_tables": {r["uuid"]: r["q_unknown_tables"] for r in rows if r.get("q_unknown_tables")},
               "tables_per_candidate": {"min": min(len(r["q_tables"]) for r in rows if not r["q_refused"]),
                                        "max": max(len(r["q_tables"]) for r in rows if not r["q_refused"])}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "argv"}, indent=1), file=sys.stderr)
    return 0


# --------------------------------------------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------------------------------------------


def parse_json(text: str) -> Any:
    """The JSON of an answer: the whole text, else inside its first code fence, else from the first `{` to the last
    `}`; None when none parses."""
    t = (text or "").strip()
    candidates = [t]
    m = re.search(r"```(?:json)?\s*\n(.*?)\n?```", t, re.S)
    if m:
        candidates.append(m.group(1).strip())
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        candidates.append(t[i:j + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except ValueError:
            continue
    return None


CALLS: contextvars.ContextVar[list] = contextvars.ContextVar("alrashed_calls")
RAMP_S = 15.0  # the first wave of candidates starts one every RAMP_S seconds (12 calls at the same second fail in a burst)


def error_events(events: list[dict]) -> list[str]:
    """The messages of a failed call's error events (codex's JSON stream: `error`, `turn.failed`), the last 3."""
    out = []
    for e in events or ():
        t = str(e.get("type", ""))
        if "error" in t or "failed" in t:
            err = e.get("error")
            msg = e.get("message") or (err.get("message") if isinstance(err, dict) else err) or json.dumps(e)
            out.append(f"{t}: {str(msg)[:300]}")
    return out[-3:]


class _Usage:
    def __init__(self, total: int | None):
        self.total_token_count = total


class CodexResponse:
    """What their code reads of a Gemini response: `parsed`, `usage_metadata.total_token_count` (and its text)."""

    def __init__(self, text: str, parsed: Any, total_tokens: int | None):
        self.text, self.parsed, self.usage_metadata = text, parsed, _Usage(total_tokens)

    def __repr__(self) -> str:
        return f"CodexResponse(text={self.text[:500]!r})"


class CallFailed(RuntimeError):
    """A codex call that failed after every wait, or a prompt whose answered attempts are used up."""


class CodexModels:
    def __init__(self, client: "CodexClient"):
        self.client = client

    async def generate_content(self, *, model: str, contents: str, config: Any) -> CodexResponse:
        return await asyncio.to_thread(self.client.generate, model, contents, config)


class _Aio:
    def __init__(self, client: "CodexClient"):
        self.models = CodexModels(client)


class CodexClient:
    """Stands in for `genai.Client(...)`: the arguments of their constructor are ignored, `.aio.models
    .generate_content` asks V1 through `sqldecomp.llm.call`. One instance per run (`make_client_factory`), shared by
    every client their code builds, so that the attempt counters span their retries."""

    def __init__(self, config, *, waits: tuple[int, ...] = RETRY_WAITS_S, attempts: int = ANSWER_ATTEMPTS, call=None):
        from sqldecomp import llm
        self.config, self.waits, self.attempts = config, waits, attempts
        self._call = call or llm.call
        self._lock = threading.Lock()
        self._answered: dict[str, int] = {}
        self._dead: set[str] = set()
        self.aio = _Aio(self)

    @staticmethod
    def role_of(model: str) -> str:
        role = model[3:] if model.startswith("v1-") else model
        if role not in ROLES:
            raise ValueError(f"unknown role in the spec's llm field: {model!r}")
        return role

    @staticmethod
    def prompt_of(contents: str, config: Any) -> str:
        system = getattr(config, "system_instruction", None) or ""
        return f"{system}\n\n{contents}" if system else contents

    def generate(self, model: str, contents: str, config: Any) -> CodexResponse:
        role = self.role_of(model)
        prompt = self.prompt_of(contents, config)
        pkey = hashlib.sha256(f"{role}\n{prompt}".encode()).hexdigest()
        with self._lock:
            done = self._answered.get(pkey, 0)
            dead = pkey in self._dead
        if dead or done >= self.attempts:
            raise self._final_error(role, f"no attempt left for this prompt ({done} answered, failed for good: {dead})")
        template = f"alrashed2026-{role}-{TEMPLATE_VERSION}" + (f"|attempt-{done + 1}" if done else "")
        record = None
        for i, wait in enumerate((0, *self.waits)):
            if wait:
                time.sleep(wait)
            started = _now()
            c = self._call(prompt, template, self.config)
            record = {**c.record(), "role": role, "attempt": done + 1, "try": i + 1, "started": started, "ended": _now()}
            if c.exit_code != 0 or not c.answer.strip():
                record["error_events"] = error_events(c.events)
            calls = CALLS.get(None)
            if calls is not None:
                calls.append(record)
            if c.exit_code == 0 and c.answer.strip():
                with self._lock:
                    self._answered[pkey] = self._answered.get(pkey, 0) + 1
                tok = c.tokens
                total = (tok.input or 0) + (tok.output or 0) if tok.input is not None or tok.output is not None else None
                return CodexResponse(c.answer, parse_json(c.answer), total)
        with self._lock:
            self._dead.add(pkey)
        raise self._final_error(role, f"the call failed {1 + len(self.waits)} times: {(record or {}).get('error', '')}")

    @staticmethod
    def _final_error(role: str, message: str) -> Exception:
        if role.startswith("se-"):  # their DataGen stops retrying on its NonRetryableError
            from nl2sql_verification.approaches.datagen.main import NonRetryableError
            return NonRetryableError(f"[codex] {message}")
        return CallFailed(f"[codex] {message}")


def patch_genai(client: CodexClient) -> None:
    """Their `genai.Client(...)` returns `client` (the one change to their code's environment)."""
    from google import genai
    genai.Client = lambda *a, **k: client  # type: ignore[assignment]


# --------------------------------------------------------------------------------------------------------------
# the specs
# --------------------------------------------------------------------------------------------------------------


def specs() -> dict[str, dict[str, Any]]:
    """The paper's settings as their experiment specs (each parsed by their own parser)."""
    return {
        "rt": {"uuid": "rt-zero-shot-v1", "description": "[RT] Round-Trip, zero-shot, 1 question, additional information included, V1",
               "scaling_factor": 1, "generator_examples": [], "evaluator_examples": [], "generator_hide_additional_information": False,
               "evaluator_hide_additional_information": False, "generator_llm": "v1-rt-generator", "evaluator_llm": "v1-rt-evaluator"},
        "se": {"uuid": "se-answer-to-data-v1", "description": "[SE] DataGen ANSWER_TO_DATA, zero-shot, 1 answer x 1 dataset, additional information included, V1",
               "strategy": "ANSWER_TO_DATA", "data_gen_scaling_factor": 1, "answer_gen_scaling_factor": 1, "data_gen_examples": [],
               "answer_gen_examples": [], "data_gen_hide_additional_information": False, "answer_gen_hide_additional_information": False,
               "data_gen_llm": "v1-se-data", "answer_gen_llm": "v1-se-answer"},
    }


THEIR_ENV = {"GOOGLE_CLOUD_PROJECT": "unused",
             # their DataGen's retry loop (BASE_CONFIG, default 50 attempts with waits up to 60 s): the client bounds
             # the calls itself (ANSWER_ATTEMPTS answered, RETRY_WAITS_S), so a prompt that is used up stops there
             "NL2SQLVERIFICATION_GENAI_RETRY_MAX_ATTEMPTS": str(ANSWER_ATTEMPTS + 1)}


def approaches():
    sys.path.insert(0, str(BASELINE_SRC))
    for k, v in THEIR_ENV.items():
        os.environ[k] = v
    from nl2sql_verification.approaches.roundtrip.main import RoundTrip
    from nl2sql_verification.approaches.datagen.main import DataGen
    classes = {"rt": RoundTrip, "se": DataGen}
    return {a: (classes[a](uuid=s["uuid"]), classes[a].parse_experiment_spec(dict(s))) for a, s in specs().items()}


# --------------------------------------------------------------------------------------------------------------
# SE's execution in DuckDB
# --------------------------------------------------------------------------------------------------------------


def _text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def _cast(col: str, duck: str) -> str:
    c = f'"{col}"'
    if duck == "BIGINT":
        return f"COALESCE(TRY_CAST({c} AS BIGINT), TRY_CAST(TRY_CAST({c} AS DOUBLE) AS BIGINT))"
    if duck == "TIMESTAMP":
        return f"COALESCE(TRY_CAST({c} AS TIMESTAMP), TRY_CAST(TRY_CAST(upper({c}) AS TIMESTAMPTZ) AS TIMESTAMP))"
    if duck == "VARCHAR":
        return c
    return f"TRY_CAST({c} AS {duck})"


def result_strings(rows) -> dict[str, str]:
    """Their serialization of a result (DataGen.evaluate): each row's values as strings (`str`), sorted within the
    row, the rows joined in their order (`inner`) and sorted (`innerouter`). DuckDB's rows come as Python values
    (`fetchall`: NULL is None, as in the answer dictionary), not through a DataFrame, whose NULLs would print as
    `<NA>` or `nan` by column type."""
    rows = ["-<colsep>-".join(sorted([str(c) for c in r])) for r in rows]
    return {"inner": "-<rowsep>-".join(rows), "innerouter": "-<rowsep>-".join(sorted(rows))}


def duckdb_execute(sql: str, schema: dict[str, Any], generated_data: dict[str, Any], timeout_s: float = DUCKDB_TIMEOUT_S) -> dict[str, Any]:
    """The candidate's result on SE's generated data in DuckDB: SQL and data lowercased as their evaluation does,
    every table of `schema` (our schema JSON's tables, by lowercased name) created with its declared types and filled
    by TRY_CAST from the data's text; PostgreSQL's integer division and NULL order, UTC. -> {error} or {inner, innerouter,
    rows, cast_nulls}."""
    import duckdb
    import pandas as pd
    data = json.loads(json.dumps(generated_data, default=str).lower())
    con = duckdb.connect(":memory:")
    con.execute("SET integer_division = true")
    con.execute("SET default_null_order = 'nulls_last_on_asc_first_on_desc'")
    con.execute("SET TimeZone = 'UTC'")  # a timestamp with an offset is read in UTC, never in the server's zone
    con.execute("SET threads = 1")  # Single-threaded execution stabilizes row ordering for the baseline comparison rule.
    cast_nulls = 0
    try:
        for t in schema["tables"]:
            name = t["name"].lower()
            cols = [c["name"].lower() for c in t["columns"]]
            types = [DUCK_TYPE.get(str(c.get("type")).upper(), "VARCHAR") for c in t["columns"]]
            con.execute(f'CREATE TABLE "{name}" (' + ", ".join(f'"{c}" {ty}' for c, ty in zip(cols, types)) + ")")
            rows = data.get(name) or []
            if not isinstance(rows, list):
                rows = []
            frame = pd.DataFrame([[_text(r.get(c)) if isinstance(r, dict) else None for c in cols] for r in rows], columns=cols, dtype=object)
            if len(frame):
                con.register("_frame", frame)
                con.execute(f'INSERT INTO "{name}" SELECT ' + ", ".join(_cast(c, ty) for c, ty in zip(cols, types)) + " FROM _frame")
                con.unregister("_frame")
                for c in cols:
                    given = int(frame[c].notna().sum())
                    kept = con.execute(f'SELECT count("{c}") FROM "{name}"').fetchone()[0]
                    cast_nulls += given - kept
        timer = threading.Timer(timeout_s, con.interrupt)
        timer.start()
        try:
            rows = con.execute(sql.lower()).fetchall()
        finally:
            timer.cancel()
        return {**result_strings(rows), "rows": len(rows), "cast_nulls": cast_nulls}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:1000], "cast_nulls": cast_nulls}
    finally:
        con.close()


def se_verdict(sql: str, result: dict[str, Any] | None, answer_sorted: dict[str, Any] | None) -> str | None:
    """Their metric's rule (calculate_metrics_per_benchmark_data_gen.sql): with ORDER BY in the candidate the rows in
    their order, else sorted; lowercased; equal to the answer dictionary's -> yes, else no; None (undecided) without a
    result. An answer dictionary that could not be sorted never equals (their NULL comparison)."""
    if not result or result.get("error") is not None or "inner" not in result:
        return None
    ordered = "order by" in sql.lower()
    mine = (result["inner"] if ordered else result["innerouter"]).lower()
    theirs = (answer_sorted or {}).get("innersortedlower" if ordered else "inneroutersortedlower")
    return "yes" if theirs is not None and mine == theirs else "no"


# --------------------------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------------------------


def _dump(obj: Any) -> Any:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def _flatten(outputs: list) -> list[dict[str, Any]]:
    """Their `experiment()` outputs as their pipeline flattens them (without the input row)."""
    rows = []
    for _row, gq, ga, gm, eq, ea, em, err in outputs:
        d: dict[str, Any] = {}
        for name, obj in (("generation_question", gq), ("generation_answer", ga), ("evaluation_question", eq), ("evaluation_answer", ea)):
            for k, v in (_dump(obj) or {}).items():
                d[f"output__{name}__{k}"] = v
        for name, meta in (("generation_metadata", gm), ("evaluation_metadata", em)):
            for k, v in (meta or {}).items():
                d[f"output__{name}__{k}"] = v
        d["error_message"] = err
        rows.append(d)
    return rows


def summarize(approach: str, row: dict[str, Any], outputs: list, schema_json_tables: dict[str, Any] | None) -> dict[str, Any]:
    """One approach's verdict on one candidate from their outputs (one output row each under these specs)."""
    if len(outputs) != 1:
        return {"verdict": None, "error": f"{len(outputs)} outputs (one expected)"}
    _row, gq, ga, gm, eq, ea, em, err = outputs[0]
    if err:
        return {"verdict": None, "error": err[:2000]}
    if approach == "rt":
        s = ea.similarity_score
        return {"verdict": "yes" if s >= RT_ACCEPT else "no", "similarity_score": s, "nlq_2": ga.final_output,
                "final_judgment": ea.final_judgment, "confidence_score": ea.confidence_score}
    answer_sorted = ga.answer_generation.answer_dictionary_sorted
    sqlite = {"error": ea.generated_sql_execution_error} if ea.generated_sql_execution_error is not None else {
        "inner": ea.generated_sql_result_sorted_inner, "innerouter": ea.generated_sql_result_sorted_innerouter, "error": None}
    gold_sqlite_error = ea.original_sql_execution_error
    data = ga.data_generation.generated_data
    duck = duckdb_execute(row["generated_sql"], schema_json_tables, data)
    gold_duck = duckdb_execute(row["original_sql"], schema_json_tables, data)
    v = se_verdict(row["generated_sql"], duck, answer_sorted)
    return {"verdict": v, "verdict_sqlite": se_verdict(row["generated_sql"], sqlite, answer_sorted), "duckdb": duck,
            "sqlite_error": sqlite.get("error"), "gold_sqlite_error": gold_sqlite_error, "gold_duckdb_error": gold_duck.get("error"),
            "gold_verdict_duckdb": se_verdict(row["original_sql"], gold_duck, answer_sorted),
            "answer_sorted": answer_sorted, "answer_sentence": ga.answer_generation.answer_sentence,
            "tables_generated": sorted(data) if isinstance(data, dict) else None,
            "rows_generated": {t: len(v) for t, v in data.items() if isinstance(v, list)} if isinstance(data, dict) else None}


async def run_candidate(row: dict[str, Any], runners: dict, schema_tables: dict[str, Any] | None, *, serial: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    import pandas as pd
    t0 = time.perf_counter()
    started = _now()
    if row["q_refused"]:
        rec = {"key": row["uuid"], "benchmark": row["q_benchmark"], "refused": True, "refusal": row.get("q_refusal"),
               **{a: {"verdict": None, "not_run": "the importer refused the candidate (no sql_printed): undecided, as in Direct"} for a in APPROACHES},
               "started": started, "ended": _now(), "seconds": 0.0}
        return rec, {}
    series = pd.Series(row)

    async def one(a: str):
        calls: list = []
        CALLS.set(calls)
        approach, spec = runners[a]
        t = time.perf_counter()
        outputs = await approach.experiment(series, spec)
        return a, outputs, calls, time.perf_counter() - t

    if serial:  # Run the two approaches sequentially when requested.
        done = [await asyncio.create_task(one(a)) for a in APPROACHES]
    else:
        done = await asyncio.gather(*(asyncio.create_task(one(a)) for a in APPROACHES))
    rec: dict[str, Any] = {"key": row["uuid"], "benchmark": row["q_benchmark"], "refused": False}
    raw: dict[str, Any] = {"key": row["uuid"]}
    for a, outputs, calls, seconds in done:
        try:
            summary = summarize(a, row, outputs, schema_tables)
        except Exception as e:  # noqa: BLE001
            summary = {"verdict": None, "error": f"summary failed: {type(e).__name__}: {e}"}
        rec[a] = {**summary, "calls": calls, "seconds": round(seconds, 1)}
        raw[a] = _flatten(outputs)
    rec.update({"started": started, "ended": _now(), "seconds": round(time.perf_counter() - t0, 1)})
    return rec, raw


def _schema_tables(row: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Our schema JSON's tables the candidate reads (DuckDB creates them with their declared types)."""
    if row["q_refused"]:
        return None
    s = schemas[row["q_database"]]
    want = set(row["q_tables"])
    return {"tables": [t for t in s["tables"] if t["name"].lower() in want]}


def cmd_run(args) -> int:
    global MODEL, EFFORT
    MODEL = getattr(args, "model", MODEL)
    EFFORT = getattr(args, "effort", EFFORT)
    from sqldecomp import llm
    rows = [json.loads(line) for line in open(Path(args.inputs) / "inputs.jsonl")]
    only = _keys(args.keys)
    if only is not None:
        unknown = sorted(only - {r["uuid"] for r in rows})
        if unknown:
            raise SystemExit(f"--keys not in the inputs: {unknown[:5]}")
        rows = [r for r in rows if r["uuid"] in only]
    inputs_meta = json.loads((Path(args.inputs) / "summary.json").read_text())
    schemas_dir = Path(inputs_meta["schemas"])
    schemas = {}
    for r in rows:
        if r["q_database"] not in schemas:
            schemas[r["q_database"]] = json.loads((schemas_dir / f"{r['q_database']}.json").read_text())
    cache = Path(args.cache)
    if cache.exists() and not args.reuse_cache:
        raise SystemExit(f"--cache must be new (or --reuse-cache after a crash): {cache}")
    out = new_dir(args.out)
    (out / "raw").mkdir()
    cache.mkdir(parents=True, exist_ok=True)
    config = llm.ModelConfig(model=MODEL, effort=EFFORT, cache_dir=str(cache), sandbox=True)
    llm.require_sandbox(config)
    runners = approaches()
    client = CodexClient(config)
    patch_genai(client)
    meta = {"argv": list(sys.argv), "inputs": str(Path(args.inputs).resolve()), "cache": str(cache.resolve()), "out": str(out.resolve()),
            "model": MODEL, "effort": EFFORT, "sandbox": True, "timeout_s": config.timeout_s, "jobs": args.jobs, "serial": args.serial,
            "answer_attempts": ANSWER_ATTEMPTS, "retry_waits_s": list(RETRY_WAITS_S), "ramp_s": RAMP_S, "rt_accept": RT_ACCEPT, "keys": args.keys,
            "reuse_cache": args.reuse_cache, "specs": specs(), "their_env": THEIR_ENV, "template_version": TEMPLATE_VERSION, "code": code_version(),
            "baseline_src": str(BASELINE_SRC), "codex": _codex_version(), "python": sys.version.split()[0], "packages": _packages(),
            "started": _now(), "candidates": len(rows), "refused": sum(1 for r in rows if r["q_refused"])}
    (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    partial = out / "records.partial.jsonl"
    lock = threading.Lock()

    async def main():
        sem = asyncio.Semaphore(args.jobs)
        done = 0

        first = {r["uuid"]: i for i, r in enumerate(r for r in rows if not r["q_refused"]) if i < args.jobs}

        async def guarded(row):
            nonlocal done
            async with sem:
                if row["uuid"] in first:
                    await asyncio.sleep(RAMP_S * first[row["uuid"]])
                try:
                    rec, raw = await run_candidate(row, runners, _schema_tables(row, schemas), serial=args.serial)
                except Exception as e:  # noqa: BLE001
                    rec, raw = {"key": row["uuid"], "benchmark": row["q_benchmark"], "refused": False, "crash": f"{type(e).__name__}: {e}",
                                **{a: {"verdict": None, "error": "crash"} for a in APPROACHES}}, {}
                with lock:
                    with open(partial, "a") as fh:
                        fh.write(json.dumps(rec, default=str) + "\n")
                    if raw:
                        (out / "raw" / f"{row['uuid'].replace(':', '_')}.json").write_text(json.dumps(raw, default=str, indent=1) + "\n")
                    done += 1
                    v = " ".join(f"{a}={(rec.get(a) or {}).get('verdict')}" for a in APPROACHES)
                    print(f"{_now()} [{done}/{len(rows)}] {row['uuid']} {v} {rec.get('seconds', '')}s", file=sys.stderr, flush=True)

        await asyncio.gather(*(guarded(r) for r in rows))

    import concurrent.futures
    loop = asyncio.new_event_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=max(8, 3 * args.jobs + 2)))
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    recs = [json.loads(line) for line in open(partial)]
    order = {r["uuid"]: i for i, r in enumerate(rows)}
    recs.sort(key=lambda r: order[r["key"]])
    (out / "records.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in recs))
    partial.unlink()
    calls = [c for r in recs for a in APPROACHES for c in (r.get(a) or {}).get("calls", [])]
    meta.update({"ended": _now(), "calls": len(calls), "calls_cached": sum(1 for c in calls if c.get("cached")),
                 "calls_failed": sum(1 for c in calls if c.get("exit_code") != 0),
                 "verdicts": {a: {str(v): sum(1 for r in recs if (r.get(a) or {}).get("verdict") == v) for v in ("yes", "no", None)} for a in APPROACHES}})
    (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta["verdicts"]), file=sys.stderr)
    return 0


def _codex_version() -> str:
    from sqldecomp import llm
    try:
        return subprocess.run([str(llm.codex_binary()), "--version"], capture_output=True, text=True, timeout=60).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _packages() -> dict[str, str]:
    from importlib import metadata
    out = {}
    for p in ("pandas", "google-genai", "smart-open", "pandasql", "duckdb", "pydantic", "jinja2", "tenacity", "sqlglot", "numpy", "SQLAlchemy"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "missing"
    return out


# --------------------------------------------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------------------------------------------


def conjunction(verdicts: list[str | None]) -> str | None:
    """The pipeline's conjunction (`sqldecomp.verify.conjunction`): undecided when one is undecided, yes iff all are
    yes, else no."""
    if not verdicts or any(v not in ("yes", "no") for v in verdicts):
        return None
    return "yes" if all(v == "yes" for v in verdicts) else "no"


ARMS = {"rt": ("rt",), "se": ("se",)}


def arm_verdict(rec: dict[str, Any], arm: str) -> str | None:
    return conjunction([(rec.get(a) or {}).get("verdict") for a in ARMS[arm]])


def run_records(run: Path) -> list[dict[str, Any]]:
    """Read records.jsonl, or records.partial.jsonl for an interrupted run.
    Each completed candidate occupies one line. An incomplete final line is omitted
    so the next run can retry that candidate.
    """
    whole = run / "records.jsonl"
    if whole.exists():
        return [json.loads(line) for line in open(whole)]
    lines = open(run / "records.partial.jsonl").read().split("\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            if i < len(lines) - 2:
                raise SystemExit(f"{run}: line {i + 1} of records.partial.jsonl does not parse")
    return out


def cmd_arms(args) -> int:
    recs: dict[str, dict[str, Any]] = {}
    for run in args.run:
        for r in run_records(Path(run)):
            if r["key"] in recs:
                raise SystemExit(f"{r['key']} occurs twice in the runs")
            recs[r["key"]] = r
    out = new_dir(args.out)
    counts = {}
    for arm in ARMS:
        d = out / arm / "pool"
        d.mkdir(parents=True)
        lines = []
        for k, r in recs.items():
            v = arm_verdict(r, arm)
            findings = [{"stage": "import", "kind": "refused", "detail": r.get("refusal")}] if r.get("refused") else []
            lines.append(json.dumps({"key": k, "refused": bool(r.get("refused")), "findings": findings,
                                     "direct": {"final": v, "rule": "conjunction" if len(ARMS.get(arm, ())) > 1 else arm}}) + "\n")
        (d / "results.jsonl").write_text("".join(lines))
        counts[arm] = {str(v): sum(1 for r in recs.values() if arm_verdict(r, arm) == v) for v in ("yes", "no", None)}
    executed = sorted(k for k, r in recs.items() if (r.get("se") or {}).get("verdict") in ("yes", "no"))
    (out / "se_executed.txt").write_text("".join(k + "\n" for k in executed))
    meta = {"argv": list(sys.argv), "runs": [str(Path(r).resolve()) for r in args.run], "candidates": len(recs), "verdicts": counts,
            "se_executed": len(executed), "code": code_version()}
    (out / "arms.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1), file=sys.stderr)
    return 0
