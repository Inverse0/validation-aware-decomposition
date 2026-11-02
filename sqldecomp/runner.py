"""A VERIFICATION RUN: queries end to end, and the RESULT RECORD of each.

    SQL + question -> import -> memo -> constraints -> legality -> boundary, features, scoring -> the exact optimizer
    -> printing -> component 9's tasks -> the verifier's block verdicts -> one verdict with the rejecting blocks
    [-> the direct arm: the whole query against the question] [-> checks C1 - C3 on the optimum]

    run_one(key, sql, question, evidence, schema, config) -> the record of one query (a JSON-able dict)
    run_many(queries, schema_path, out_dir, config, jobs=) -> the run's summary; `results.jsonl` (one record per query,
        sorted by key) and `summary.json` in a NEW directory (a previous run is never overwritten)

`python -m sqldecomp.pipeline --verify [--direct]` drives it. THE RECORD (`RECORD_VERSION`): identification (the key,
the SQL and the question as given, the evidence, the schema's name and version hash, the run tag, the timestamps, the
code commit, `CONTRACT_VERSION`, every prompt and template version, the model configurations of the writer and the
verifier), the whole `ExperimentConfig`, the import result, the memo's statistics, the constraints summary, the
optimum (every block: output group, inputs, class id, features, exposed columns with types and the null_extended /
never_null flags, keys, weak keys, foreign keys, parameters, the reader-form SQL and the executable form where it
exists; the cost, the score and the search statistics), the tasks (per block: task text, input contracts,
provenance: method, prompt versions, canonical key, cache hit; the writer's calls, also when it failed), the verdicts
(per block class: verdict, reason, template, cache key, tokens, latency; the final verdict and the rejecting blocks),
the direct arm's verdict when asked for (it runs whenever the whole query printed, whatever happened to the
decomposed arm), the checks when enabled, the findings (the decomposition timeout with the stage reached, unsupported
SQL, an invalid import, print errors, a task that could not be written, a verifier call that failed) and the sandbox
flag. THE DECOMPOSITION TIMEOUT (`config.decomposition_timeout_s`, 300 s by
default) is the run's one clock: started when the query starts, it bounds import, memo, constraints, legality, scoring,
the search, printing and the task writer with its model calls; verification and the checks are not on it. RUNTIME in ms per component, each on its own (`COMPONENTS`: import, memo, constraints, blocks,
semantic (the LLM judge's two calls under `--semantic judge`, 0 otherwise), scoring, optimizer, printing, tasks,
verification, direct, checks) and the total; TOKENS per call (input, cached
input, output, reasoning) and the ESTIMATED COST in USD from the configuration's price table, summed separately for
the decomposition side (the task writer), the verification and the direct arm. A cached call keeps its original
tokens and cost with `cached: true`: `estimated_usd` is what the calls cost (would cost), `paid_usd` what this run
paid. A cost is None when a price is missing or a call's tokens are not known: unavailable, never guessed (the known
part and the number of calls without a cost stand beside it), and a token count no call reported is None, never 0. Caches
(assignments, the writer's and the verifier's calls) live where the configuration says, outside the run directory."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from sqldecomp import llm as client
from sqldecomp import model as M
from sqldecomp.config import ExperimentConfig
from sqldecomp.llm import Call, Price, cost_usd, tokens_of

RECORD_VERSION = "sqldecomp-run-v1"
REPLAY_MISS = 97  # the exit code of a replayed call (`ExperimentConfig.replay`) whose prompt is not in the cache: never asked
COMPONENTS = ("import", "memo", "constraints", "blocks", "semantic", "scoring", "optimizer", "printing", "tasks", "verification", "direct",
              "checks")
DECOMPOSITION_SIDE = ("import", "memo", "constraints", "blocks", "semantic", "scoring", "optimizer", "printing", "tasks")
Ask = Callable[[str, str], Call]


# --------------------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="milliseconds")


@contextmanager
def _timer(times: dict[str, float], component: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        times[component] = times.get(component, 0.0) + (time.perf_counter() - t0) * 1000


def code_version() -> dict[str, Any]:
    """The commit of the package's checkout and whether tracked files differ from it ('unknown' outside git)."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, timeout=20).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", False
    return {"commit": commit or "unknown", "dirty": dirty}


def schema_version(schema_json: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(schema_json, sort_keys=True).encode()).hexdigest()[:16]


def call_records_from_cache(meta: dict[str, Any], price_of: Callable[[str], Price | None]) -> list[dict[str, Any]] | None:
    """The writer's call records an assignment cache entry holds (tokens recorded when it was written), marked
    cached and re-costed under the current price table; None when the entry does not record them."""
    calls = meta.get("calls")
    if calls is None:
        return None
    out = []
    for c in calls:
        tokens = client.Tokens(**(c.get("tokens") or {}))
        out.append({**c, "cached": True, "cost_usd": cost_usd(tokens, price_of(c.get("model", "")))})
    return out


TOKEN_FIELDS = ("input", "cached_input", "output", "reasoning")


def _usd(costs: list[float | None]) -> float | None:
    """The sum of the costs when every one is known, else None (unavailable, never guessed)."""
    return None if any(x is None for x in costs) else round(sum(costs), 6) + 0.0


def _usd_known(costs: list[float | None]) -> float:
    """The sum of the costs that are known (`unpriced_calls` says how many are not)."""
    return round(sum(x for x in costs if x is not None), 6) + 0.0


def _token_sums(calls: list[dict[str, Any]]) -> dict[str, int | None]:
    """Per token field the sum over the calls that report it; None when there are calls and none of them reports it
    (a count the client did not give is never taken as 0)."""
    out: dict[str, int | None] = {}
    for k in TOKEN_FIELDS:
        known = [v for v in ((c.get("tokens") or {}).get(k) for c in calls) if v is not None]
        out[k] = sum(known) if (known or not calls) else None
    return out


def side_totals(calls: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Calls, tokens and cost of one side. `tokens` / `paid_tokens`: per field the sum over the calls that report it
    (`_token_sums`); `tokens_unreported`: per field the number of calls that do not. `estimated_usd` / `paid_usd`: the
    sum when every call's cost is known, else None; `estimated_usd_known` / `paid_usd_known`: the known part;
    `unpriced_calls`: the calls without a cost (no price for their model, or no token counts). `failed_calls`: the
    calls that exited non-zero (each record keeps its `error`)."""
    if calls is None:
        return {"calls": None, "cached_calls": None, "failed_calls": None, "tokens": None, "paid_tokens": None, "tokens_unreported": None,
                "estimated_usd": None, "estimated_usd_known": None, "unpriced_calls": None, "paid_usd": None, "paid_usd_known": None,
                "note": "the calls of a cache entry written without their records: unknown"}
    paid_calls = [c for c in calls if not c.get("cached")]
    costs = [c.get("cost_usd") for c in calls]
    paid = [c.get("cost_usd") for c in paid_calls]
    return {"calls": len(calls), "cached_calls": len(calls) - len(paid_calls),
            "failed_calls": sum(1 for c in calls if c.get("exit_code") not in (0, None)),
            "tokens": _token_sums(calls), "paid_tokens": _token_sums(paid_calls),
            "tokens_unreported": {k: sum(1 for c in calls if (c.get("tokens") or {}).get(k) is None) for k in TOKEN_FIELDS},
            "estimated_usd": _usd(costs), "estimated_usd_known": _usd_known(costs), "unpriced_calls": sum(1 for x in costs if x is None),
            "paid_usd": _usd(paid), "paid_usd_known": _usd_known(paid)}


# --------------------------------------------------------------------------------------------------------------
# one query
# --------------------------------------------------------------------------------------------------------------


def package_version() -> str:
    """The installed version of the package ('unknown' for a checkout that is not installed)."""
    try:
        from importlib.metadata import version
        return version("sqldecomp")
    except Exception:  # noqa: BLE001
        return "unknown"


def _versions() -> dict[str, Any]:
    from sqldecomp import verify
    from sqldecomp.tasks import PROMPT_VERSIONS
    return {"package": package_version(), "writer_prompts": dict(PROMPT_VERSIONS), "verifier_template": verify.TEMPLATE_FAMILY,
            "model_call_template": client.TEMPLATE_VERSION, "record": RECORD_VERSION}


def base_record(key: str, sql: str, question: str, evidence: str, schema_json: dict[str, Any], schema_name: str, config: ExperimentConfig,
                code: dict[str, Any] | None = None, started: str | None = None) -> dict[str, Any]:
    """The identification every record carries, whatever happens to its query afterwards: the key, the query as given,
    the schema's name and version, the run (tag, timestamps, code, contract version), the versions (the package's,
    the prompts' and templates'), both model configurations, the sandbox flag and the whole configuration."""
    return {
        "record": RECORD_VERSION, "key": key, "query": {"sql": sql, "question": question, "evidence": evidence or ""},
        "schema": {"name": schema_name, "version": schema_version(schema_json)},
        "run": {"tag": config.run_tag, "started_at": started or _now(), "finished_at": None, "code": code or code_version(),
                "contract_version": M.CONTRACT_VERSION},
        "versions": _versions(),
        "model": {"writer": config.writer_config().to_json(), "verifier": config.verifier_config().to_json()},
        "sandbox": bool(config.model.sandbox), "config": config.to_json(),
        "findings": [], "verdict": None,
    }


def run_one(key: str, sql: str, question: str, evidence: str, schema: M.Schema, config: ExperimentConfig, *, schema_name: str = "",
            schema_json: dict[str, Any] | None = None, writer: Ask | None = None, verifier: Ask | None = None,
            code: dict[str, Any] | None = None, semantic: Ask | None = None) -> dict[str, Any]:
    """The result record of one query under `config`. `schema_json`: the schema as the texts render it
    (`sqldecomp.schema.schema_json_of` of the schema file: its join pairs and declared case; default: the schema
    model's JSON). `writer` / `verifier` are the call functions (prompt, template) -> `Call` of the two sides
    (default: `sqldecomp.llm.call` under `config.writer_config()` / `verifier_config()`); `semantic` the LLM judge's
    (`--semantic judge`; default: `sqldecomp.llm.call` under `config.semantic_config(...)`), on the decomposition clock
    like the writer's: its labels, scores and calls are the record's `semantic`. The DIRECT arm runs whenever
    it was asked for and the whole query printed, whatever happened to the decomposed arm (its timeout, a task that
    could not be written, a block that cannot be printed are findings of that arm); the checks run whenever the
    optimum exists. The direct arm ALONE (no decomposed arm, no checks) decomposes nothing: the query is imported and
    printed (the direct item is the printed query), then judged. A query the pipeline refuses is a record with its
    findings, never an exception; a defect of the code raises."""
    from sqldecomp import verify
    from sqldecomp.clock import Deadline, DecompositionTimeout
    started, t_start = _now(), time.perf_counter()
    deadline = Deadline(config.decomposition_timeout_s)
    schema_json = schema_json if schema_json is not None else M.schema_to_json(schema)
    rec = base_record(key, sql, question, evidence, schema_json, schema_name, config, code, started)
    times: dict[str, float] = {}
    calls: dict[str, list[dict[str, Any]] | None] = {"semantic": [], "decomposition": [], "verification": [], "direct": []}
    ask_verifier = verifier or verify.model_ask(config.verifier_config())
    state: dict[str, Any] = {}
    # with call_parallelism > 1 the direct arm runs beside the decomposed arm, once the whole query is printed
    beside = ThreadPoolExecutor(max_workers=1) if config.direct_arm and config.decomposed_arm and config.call_parallelism > 1 else None
    direct = None
    try:
        try:
            if not (config.decomposed_arm or config.checks):  # the direct arm alone: imported and printed, nothing decomposed
                _import_and_print(rec, sql, schema, times, state)
            elif _decompose(rec, sql, schema, schema_json, config, times, deadline, state, question=question, evidence=evidence or "",
                            calls=calls, semantic=semantic) and config.decomposed_arm:
                if beside is not None:
                    direct = beside.submit(_direct_arm, rec, question, evidence or "", schema_json, config, times, calls, ask_verifier, state)
                _decomposed_arm(rec, question, evidence or "", schema_json, config, times, calls,
                                _on_the_clock(writer, config.writer_config(), deadline, replay=config.replay), ask_verifier, deadline, state)
        except DecompositionTimeout as e:  # never a verdict: the stage reached and the time it took
            rec["findings"].append(e.finding())
        if config.direct_arm:
            if direct is not None:
                direct.result()
            else:
                _direct_arm(rec, question, evidence or "", schema_json, config, times, calls, ask_verifier, state)
        if config.checks and "optimum" in state:
            with _timer(times, "checks"):
                rec["checks"] = _checks(sql, state["result"], state["memo"], state["blocks"], state["printer"], state["optimum"], schema,
                                        config.db)
    finally:
        if beside is not None:
            beside.shutdown(wait=True)
        times["total"] = (time.perf_counter() - t_start) * 1000
        rec["time_ms"] = {k: round(v, 1) for k, v in times.items()}
        rec["time_ms"]["decomposition_side"] = round(sum(times.get(k, 0.0) for k in DECOMPOSITION_SIDE), 1)
        rec["tokens_and_cost"] = {side: side_totals(c) for side, c in calls.items()}
        rec["run"]["finished_at"] = _now()
    return rec


def _on_the_clock(writer: Ask | None, config: client.ModelConfig, deadline, *, replay: bool = False) -> Ask:
    """The task writer's calls, and the LLM judge's: a real call runs at most what is left of the decomposition clock (the
    clock itself is looked at before and after every call, once the call is recorded: `llm_from_ask(..., deadline=)`,
    `_semantic_judge`). `replay` (`ExperimentConfig.replay`): the answer comes from the cache alone, and a prompt the
    cache does not hold is a failed call (`REPLAY_MISS`) that was never asked, so that a replay reproduces the run whose
    caches it copied, its decompositions and tasks, and writes nothing of its own."""
    from dataclasses import replace

    def ask(prompt: str, template: str) -> Call:
        if writer is not None:
            return writer(prompt, template)
        if replay:
            hit = client.cached_call(prompt, template, config)
            return hit if hit is not None else Call("", {}, 0.0, REPLAY_MISS, stderr="replay: the prompt is not in the cache (never asked)",
                                                    key=client.call_key(prompt, template, config), model=config.model, effort=config.effort,
                                                    template=template, sandbox=config.sandbox, template_version=config.template_version)
        left = deadline.remaining()
        return client.call(prompt, template, config if left == float("inf") else replace(config, timeout_s=max(0.001, min(config.timeout_s, left))))

    return ask


def _finding(rec: dict[str, Any], kind: str, stage: str, detail: str) -> None:
    rec["findings"].append({"kind": kind, "stage": stage, "detail": detail[:600]})


def _import_and_print(rec, sql, schema, times, state: dict[str, Any]) -> bool:
    """The import and the print of the whole query (a few ms; the plan alone, no memo needed): what the direct arm
    needs, and the start of the decomposition. Fills `state["result"]` and `state["whole"]`; False, with a finding,
    when the query cannot be imported or printed."""
    from sqldecomp.importer import UnsupportedSQL, import_sql
    from sqldecomp.printer import Printer, PrintError
    from sqldecomp.tasks import canonical_sql

    with _timer(times, "import"):
        try:
            result = import_sql(sql, schema)
        except UnsupportedSQL as e:
            rec["import"] = {"ok": False, "unsupported": e.reason, "detail": str(e)[:300]}
            _finding(rec, "unsupported", "import", str(e) if str(e).startswith(e.reason) else f"{e.reason}: {e}")
            return False
        problems = M.validate_import_result(result)
    rec["import"] = {"ok": not problems, "violations": problems, "scopes": len(result.scopes)}
    if problems:
        _finding(rec, "invalid_import", "import", "; ".join(problems))
        return False
    state["result"] = result
    with _timer(times, "printing"):
        try:
            state["whole"] = rec["sql_printed"] = canonical_sql(Printer(result).plan_sql(result.plan))
        except PrintError as e:
            _finding(rec, "print_error", "printing", f"the whole query: {e}")
            return False
    return True


def _decompose(rec, sql, schema, schema_json, config: ExperimentConfig, times, deadline, state: dict[str, Any], *, question: str = "",
               evidence: str = "", calls: dict[str, Any] | None = None, semantic: Ask | None = None) -> bool:
    """The decomposition side up to the printed optimum, on the clock: `_import_and_print`, then the memo, the
    constraints, the blocks, the LLM judge's labels when the scoring asks for them (`--semantic judge`), the scoring,
    the exact search and the block records. Fills `state` (the import result, the memo, ..., the optimum and its block
    records); False, with a finding, when the decomposed arm cannot go on."""
    from sqldecomp.blocks import Blocks
    from sqldecomp.boundary import Boundary
    from sqldecomp.clock import memo_budget, memo_stopped
    from sqldecomp.constraints import derive_constraints
    from sqldecomp.memo import build_memo_with_stats, signature
    from sqldecomp.optimizer import Optimizer
    from sqldecomp.printer import Printer
    from sqldecomp.scoring_args import is_judge, make_scoring, scoring_label
    from sqldecomp.tasks import MERGED_METHODS, block_records

    if not _import_and_print(rec, sql, schema, times, state):
        return False
    result = state["result"]
    with _timer(times, "memo"):
        deadline.check("memo")
        memo, stats = build_memo_with_stats(result, budget_s=memo_budget(deadline))
        rec["memo"] = {"groups": len(memo.groups), "expressions": stats.expressions, "complete": memo.complete}
        memo_stopped(memo.complete, deadline)
        signatures = {g: signature(memo, result, g) for g in memo.groups}
    if not memo.complete:
        _finding(rec, "incomplete_memo", "memo", "the saturation stopped before the end: an optimum would range over fewer decompositions")
        return False
    with _timer(times, "constraints"):
        deadline.check("constraints")
        constraints = derive_constraints(memo, result, schema)
    with _timer(times, "blocks"):
        deadline.check("blocks")
        blocks = Blocks(memo, result)
        blocks.forced_cuts()
    printer = Printer(result, memo)
    rec["constraints"] = _constraint_summary(constraints, memo, blocks)
    hook = None
    if is_judge(config.scoring):
        hook = _semantic_judge(rec, config, times, calls if calls is not None else {}, deadline, semantic, memo, result, blocks, printer,
                               signatures, question, evidence, schema_json)
    else:
        times["semantic"] = 0.0
    with _timer(times, "scoring"):
        deadline.check("scoring")
        boundary = Boundary(memo, result, blocks, printer, constraints, schema)
        scoring = make_scoring(config.scoring, memo, result, blocks, boundary, semantic_hook=hook)
    rec["scoring"] = scoring_label(config.scoring)
    t0 = time.perf_counter()
    opt = Optimizer(memo, result, blocks, scoring, schema=schema, constraints=constraints, signatures=signatures, printer=printer,
                    boundary=boundary, deadline=deadline)
    try:
        optimum = opt.optimize() if config.decomposition == "optimum" else _baseline(rec, config, opt, memo, result, blocks, schema, deadline)
        if optimum is None:  # a baseline that could not be made: its finding says why
            return False
    finally:  # a timeout too: the optimizer's time goes to the stage it was spent in
        total = (time.perf_counter() - t0) * 1000
        enum, scored = opt.stats.get("enumeration_ms"), opt.stats.get("scoring_ms")
        if enum is None:  # stopped in the enumeration
            times["blocks"] += total
        elif scored is None:  # stopped while scoring the classes
            times["blocks"] += enum
            times["scoring"] += total - enum
        else:
            times["blocks"] += enum
            times["scoring"] += scored
            times["optimizer"] = total - enum - scored
        rec["optimizer_stats"] = {k: opt.stats.get(k) for k in ("candidate_work", "candidates_held", "classes", "states_expanded",
                                                                "enumeration_ms", "scoring_ms", "search_ms")}
    with _timer(times, "printing"):
        deadline.check("printing")
        records = block_records(optimum, memo, result, blocks, printer, boundary, merged=config.tasks_method in MERGED_METHODS)
    rec["optimum"] = _optimum_record(optimum, opt, scoring, records, schema_json)
    state.update(result=result, memo=memo, blocks=blocks, printer=printer, optimum=optimum, records=records)
    broken = [r for r in records if "print_error" in r]
    for r in broken:
        _finding(rec, "print_error", "printing", f"block {r['output']}: {r['print_error']}")
    return not broken


def _baseline(rec, config: ExperimentConfig, opt, memo, result, blocks, schema, deadline):
    """A baseline decomposition in place of the optimizer's choice (`config.decomposition`, `sqldecomp.baselines`):
    syntactic, or random-k with the k of this query (`config.k_by_key`) and the query's seed; evaluated by the
    optimizer's bookkeeping (`Optimizer.optimum_of`), checked by C2 first. The record's `decomposition` says how it was
    made; a query without a k, or a decomposition C2 refuses, is a finding and no optimum (None)."""
    from sqldecomp import baselines
    from sqldecomp.printer import check_recomposition
    opt.prepare()
    deadline.check("baseline")
    if config.decomposition == "syntactic":
        d, info = baselines.syntactic(blocks, memo, result)
    else:
        k = (config.k_by_key or {}).get(rec["key"])
        if k is None:
            _finding(rec, "no_k", "baseline", "random-k: the run that gives k has no optimum for this query")
            return None
        d, info = baselines.random_k(blocks, lambda x: opt.optimum_of(x).unique_blocks, int(k),
                                     baselines.query_seed(config.decomposition_seed, rec["key"]), tick=lambda: deadline.check("baseline"))
    c2 = check_recomposition(d, result, memo, schema)
    rec["decomposition"] = {**info, "c2": {"ok": c2.ok, "detail": "" if c2.ok else c2.detail[:300]}}
    if not c2.ok:
        _finding(rec, "c2_failed", "baseline", f"the {config.decomposition} decomposition does not recompose the query: {c2.detail[:300]}")
        return None
    return opt.optimum_of(d)


def _semantic_judge(rec, config: ExperimentConfig, times, calls, deadline, ask: Ask | None, memo, result, blocks, printer, signatures,
                    question: str, evidence: str, schema_json: dict[str, Any]):
    """The LLM judge's labels of the query's cut points (`sqldecomp.semantic_llm`), on the clock, and the hook the
    scoring reads them through. The two calls are reported whatever happens (`calls["semantic"]`, the record's
    `semantic`); a failed call or a label that could not be read is a finding, and the label is missing (scored 1)."""
    from sqldecomp import semantic_llm
    judge = semantic_llm.JudgeSpec.from_json(config.scoring["semantic"])
    with _timer(times, "semantic"):
        deadline.check("semantic")
        points = semantic_llm.cut_points(memo, result, blocks, printer, signatures)
        rating = semantic_llm.rate(points, question, evidence, schema_json, rec["key"],
                                   _on_the_clock(ask, config.semantic_config(judge), deadline, replay=config.replay), judge,
                                   parallelism=config.call_parallelism)
        calls["semantic"] = [c.record(config.price(c.model)) for c in rating.calls]
        rec["semantic"] = {**rating.to_json(judge), "calls": calls["semantic"]}
        for c in calls["semantic"]:
            if c.get("exit_code") not in (0, None):
                _finding(rec, "model_call_failed", "semantic", f"{c['template']}: exit code {c['exit_code']}: {c.get('error') or 'no message'}")
        if rating.problems:
            _finding(rec, "semantic_labels", "semantic", f"{len(rating.problems)} labels missing or unreadable: " + "; ".join(rating.problems[:5]))
        deadline.check("semantic")  # after the calls are recorded: a call that ran past the clock is still reported
    return semantic_llm.judge_hook(rating, judge, signatures[memo.root])


def _decomposed_arm(rec, question, evidence, schema_json, config: ExperimentConfig, times, calls, writer: Ask, verifier: Ask, deadline,
                    state: dict[str, Any]) -> None:
    """Component 9's tasks (on the clock) and the verifier's block verdicts with their aggregation. The writer's calls
    are reported whatever happens: a task that could not be written is a finding, the clock a timeout."""
    from sqldecomp import verify
    from sqldecomp.tasks import AssignmentError, TaskCache, assign_from_records, canonical_key, llm_from_ask, prompt_versions, writer_identity

    price_of = config.price
    records = state["records"]
    with _timer(times, "tasks"):
        cache = TaskCache(config.task_cache) if config.task_cache else None
        identity = writer_identity(config.writer_config())  # tasks written by another model, effort or sandbox are never reused
        key = canonical_key(records, question, evidence, schema_json, config.tasks_method, writer=identity)
        entry = cache.get_entry(key) if cache is not None else None
        made: list[Call] = []
        head = {"method": config.tasks_method, "canonical_key": key, "prompt_versions": prompt_versions(config.tasks_method)}
        try:
            tasks = assign_from_records(records, question=question, evidence=evidence, schema_json=schema_json, method=config.tasks_method,
                                        llm=llm_from_ask(writer, made, deadline), cache=cache, writer=identity, parallelism=config.call_parallelism)
        except Exception as e:  # a task that could not be written, or the clock: the calls made are still reported
            calls["decomposition"] = [c.record(price_of(c.model)) for c in made]
            rec["tasks"] = {**head, "cache_hit": False, "blocks": [], "calls": calls["decomposition"]}
            if isinstance(e, AssignmentError):
                _finding(rec, "assignment_error", "tasks", str(e))
                return
            raise
        if entry is not None:
            calls["decomposition"] = call_records_from_cache(entry[1], price_of)
        else:
            calls["decomposition"] = [c.record(price_of(c.model)) for c in made]
            if cache is not None:  # the entry keeps the records of the calls that wrote it (a rerun reports their cost)
                cache.put(key, tasks, {"method": config.tasks_method, "question": question,
                                       "calls": [{k: v for k, v in c.items() if k != "cost_usd"} for c in calls["decomposition"]]})
    rec["tasks"] = {**head, "cache_hit": entry is not None, "blocks": [t.to_json() for t in tasks],
                    "calls": calls["decomposition"]}  # every writer call: keys, tokens, latency, cost, cache flag (None: unknown)
    with _timer(times, "verification"):
        v = verify.verify_tasks(tasks, records, question=question, evidence=evidence, schema_json=schema_json,
                                ask=lambda p, t: verifier(p, t), rule=config.aggregation, price=None, parallelism=config.call_parallelism)
    _price_calls(v, price_of)
    calls["verification"] = v.calls
    rec["verdicts"] = v.to_json()
    rec["verdict"] = v.final
    _call_findings(rec, v, "verification")


def _direct_arm(rec, question, evidence, schema_json, config: ExperimentConfig, times, calls, verifier: Ask, state: dict[str, Any]) -> None:
    """The whole query judged against the question with the same prompt family and rule; not under the clock."""
    from sqldecomp import verify
    if state.get("whole") is None:
        rec["direct"] = {"final": None, "not_run": "the whole query could not be imported or printed (see the findings)"}
        return
    with _timer(times, "direct"):
        d = verify.verify_direct(state["whole"], question=question, evidence=evidence, schema_json=schema_json, ask=lambda p, t: verifier(p, t),
                                 rule=config.aggregation, price=None)
    _price_calls(d, config.price)
    calls["direct"] = d.calls
    rec["direct"] = d.to_json()
    _call_findings(rec, d, "direct")


def _call_findings(rec: dict[str, Any], v, stage: str) -> None:
    """A verifier call that exited non-zero (its verdict is then undecided unless it answered anyway) is a finding."""
    for it in v.items:
        for c in it.calls:
            if c.get("exit_code") not in (0, None):
                _finding(rec, "model_call_failed", stage, f"{it.item}: exit code {c['exit_code']}: {c.get('error') or 'no message'}")


def _price_calls(v, price_of: Callable[[str], Price | None]) -> None:
    """The estimated cost of every call of a verification, under the price of the model that answered it."""
    for item in v.items:
        for c in item.calls:
            c["cost_usd"] = cost_usd(client.Tokens(**c["tokens"]), price_of(c["model"]))


def _constraint_summary(constraints: dict[str, M.GroupConstraints], memo: M.Memo, blocks) -> dict[str, Any]:
    keyed = sorted(g for g, c in constraints.items() if c.keys)
    return {"groups": len(constraints), "groups_with_key": len(keyed), "groups_with_fk": sum(bool(c.fks) for c in constraints.values()),
            "groups_with_weak_key": sum(bool(c.weak) for c in constraints.values()),
            "cut_points_with_key": sum(blocks.can_cut(g) for g in keyed), "root_has_key": memo.root in keyed}


def _exposed(record: dict[str, Any], schema_json: dict[str, Any]) -> list[dict[str, Any]]:
    """The exposed columns of a block in the record: name, type, `null_extended` (the column comes from the
    NULL-extended side of an outer join inside the block, the boundary's flag), `never_null` (a base column declared
    NOT NULL reached without an outer join: the `never NULL` fact of the contracts), and where it comes from."""
    from sqldecomp.tasks import never_null_flags
    return [{"name": c["name"], "type": c.get("type"), "null_extended": bool(c.get("nullable")), "never_null": never,
             "alias": c.get("alias"), "column": c.get("column")}
            for c, never in zip(record.get("output_columns", []), never_null_flags(record, schema_json))]


def _optimum_record(optimum, opt, scoring, records: list[dict[str, Any]], schema_json: dict[str, Any]) -> dict[str, Any]:
    features = getattr(scoring, "features", None)
    out_blocks = []
    for r in records:
        cid = r.get("class_id")
        ctx = optimum.contexts.get(cid) if cid else None
        out_blocks.append({
            "output": r["output"], "inputs": r["inputs"], "class_id": cid, "shared_with": r.get("shared_with", []), "is_root": r["is_root"],
            "kind": r.get("kind"), "readers": r.get("readers", []), "parameters": r.get("parameters", []),
            "exposed": _exposed(r, schema_json), "star": r.get("output_star"), "keys": r.get("output_keys", []),
            "weak_keys": r.get("output_weak_keys", []), "fks": r.get("output_fks", []),
            "features": dict(features.of(ctx).values) if (features is not None and ctx is not None) else None,
            "cost": round(optimum.cost_by_class[cid], 9) if cid in optimum.cost_by_class else None,
            "meaningfulness": r.get("meaningfulness"), "sql": r.get("sql"), "sql_exec": r.get("sql_exec") or None, "tables": r.get("tables_own", [])})
    s = optimum.stats
    return {"blocks": out_blocks, "n_blocks": len(out_blocks), "unique_blocks": optimum.unique_blocks, "cost": round(optimum.cost, 9),
            "score": optimum.score, "omega": optimum.omega, "classes_shared": optimum.classes_shared,
            "search": {k: s.get(k) for k in ("method", "classes", "f_evaluations", "blocks_evaluated", "repeatable_classes", "forced_cuts",
                                             "states_expanded", "states_generated", "enumeration_ms", "scoring_ms", "search_ms", "time_ms")}}


def _checks(sql: str, result, memo, blocks, printer, optimum, schema, db: str | None) -> dict[str, Any]:
    """C1 (round trip) on every memo group, C2 (recomposition) on the optimum, and with a database C3 (read-only
    EXPLAIN) on the original, the recomposed optimum and every block of it."""
    from sqldecomp.printer import PrintError, check_recomposition, check_round_trip, check_valid
    out: dict[str, Any] = {"c1": {"groups": 0, "passed": 0, "failed": []}}
    for g in sorted(memo.groups):
        check = check_round_trip(printer.group_plan(g), result, memo, schema)
        out["c1"]["groups"] += 1
        out["c1"]["passed"] += check.ok
        if not check.ok:
            out["c1"]["failed"].append({"group": g, "detail": check.detail[:300]})
    c2 = check_recomposition(optimum.decomposition, result, memo, schema)
    out["c2"] = {"ok": c2.ok, "detail": "" if c2.ok else c2.detail[:300]}
    if db:
        jobs = [("original", sql)]
        try:
            jobs.append(("recomposed", printer.decomposition_sql(optimum.decomposition)))
            jobs += [(f"block {b.output}", printer.block_sql(b, inputs="inline", bind_outer=True)) for b in optimum.decomposition.blocks]
        except PrintError as e:
            out["c3"] = {"database": db, "error": str(e)[:300]}
            return out
        verdicts = {what: check_valid(text, dbname=db) for what, text in jobs}
        original_ok = verdicts["original"].ok
        out["c3"] = {"database": db, "original_ok": original_ok,
                     "statements": {what: {"ok": v.ok, "detail": "" if v.ok else v.detail[:200]} for what, v in verdicts.items()},
                     "consistent": all(v.ok for v in verdicts.values()) if original_ok else not verdicts["recomposed"].ok}
    return out


# --------------------------------------------------------------------------------------------------------------
# a run over many queries
# --------------------------------------------------------------------------------------------------------------


def load_manifest_queries(manifest: str | Path) -> list[dict[str, Any]]:
    """The queries of every file of an inputs manifest (`experiments.data_prep.experiment inputs`: {"inputs": [{name,
    file, schema, queries}]}), each with its file's schema (`schema`), sorted by key: one pool over every file."""
    out = []
    for entry in json.loads(Path(manifest).read_text())["inputs"]:
        for q in load_run_queries(queries=entry["file"]):
            out.append({**q, "schema": entry["schema"]})
    keys = [q["key"] for q in out]
    if len(set(keys)) != len(keys):
        raise ValueError("a key occurs in two files of the manifest")
    return sorted(out, key=lambda q: q["key"])


def load_run_queries(corpus: str | None = None, queries: str | None = None) -> list[dict[str, Any]]:
    """The queries of a run: a directory of JSON files (`sql`, `question`, optional `evidence`; the key is the file
    name) or a JSONL file (`key`, `sql`, `question`, optional `evidence`), sorted by key."""
    out = []
    if corpus:
        for path in sorted(Path(corpus).glob("*.json")):
            try:
                d = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and isinstance(d.get("sql"), str) and d["sql"].strip():
                out.append({"key": path.name, "sql": d["sql"], "question": d.get("question") or "", "evidence": d.get("evidence") or ""})
    if queries:
        for line in Path(queries).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out.append({"key": str(d["key"]), "sql": d["sql"], "question": d.get("question") or "", "evidence": d.get("evidence") or ""})
    return sorted(out, key=lambda q: q["key"])


def _worker(args: tuple) -> dict[str, Any]:
    q, schema_path, config_json, code = args
    from sqldecomp.pipeline import load_any_schema
    from sqldecomp.schema import schema_json_of
    config = ExperimentConfig.from_json(config_json)
    schema = load_any_schema(schema_path)
    schema_json = schema_json_of(schema_path)
    name = Path(schema_path).stem
    if not q["question"].strip():
        rec = base_record(q["key"], q["sql"], "", q["evidence"], schema_json, name, config, code)
        rec["findings"].append({"kind": "no_question", "stage": "input", "detail": "the query has no question"})
        rec.update(time_ms={"total": 0.0}, tokens_and_cost={})
        rec["run"]["finished_at"] = rec["run"]["started_at"]
        return rec
    try:
        return run_one(q["key"], q["sql"], q["question"], q["evidence"], schema, config, schema_name=name, schema_json=schema_json, code=code)
    except client.SandboxUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 - a defect is reported in the record of its query; the run goes on
        rec = base_record(q["key"], q["sql"], q["question"], q["evidence"], schema_json, name, config, code)
        rec["findings"].append({"kind": f"error:{type(e).__name__}", "stage": "run", "detail": str(e)[:300], "trace": traceback.format_exc()[-1500:]})
        rec.update(time_ms={}, tokens_and_cost={})
        rec["run"]["finished_at"] = _now()
        return rec


def _identity(q: dict[str, Any]) -> tuple[str, str, str]:
    return (q["sql"], q.get("question") or "", q.get("evidence") or "")


PARTIAL = "results.partial.jsonl"  # the records of a run in progress, one appended as each query finishes


def run_many(queries: list[dict[str, Any]], schema_path: str | Path | None, out_dir: str | Path, config: ExperimentConfig, *, jobs: int = 1,
             progress: Callable[[int, int], None] | None = None, resume: bool = False) -> dict[str, Any]:
    """Every query, `results.jsonl` (sorted by key) and `summary.json` into `out_dir`, which must not exist yet. A
    query that carries its own `schema` (a path: `load_manifest_queries`) is run against it, so that one pool of
    `jobs` takes the queries of several schemas; `schema_path` is the others' (None when every query carries one). Each
    record is appended to `results.partial.jsonl` as soon as its query finishes, so a run that stops keeps every query
    it finished; with `resume` the directory may exist, the queries whose records that file holds are not run again,
    and the run goes on with the others. Once every query is done the two files are written and the partial file is
    removed (a directory holding `results.jsonl` is a finished run: never resumed). A query that repeats an earlier
    one (same SQL, question and evidence, another key) runs after all first occurrences, so that its calls come from
    the caches the first one filled instead of being paid twice in parallel."""
    from sqldecomp.schema import schema_json_of
    out = Path(out_dir)
    if out.exists() and not resume:
        raise FileExistsError(f"{out} exists; a run never writes into an existing directory (resume continues a run that stopped)")
    if (out / "results.jsonl").exists():
        raise FileExistsError(f"{out} holds a finished run")
    if config.decomposed_arm or config.direct_arm:
        client.require_sandbox(config.model)  # before any work
    started = _now()
    code = code_version()
    out.mkdir(parents=True, exist_ok=resume)
    partial = out / PARTIAL
    done: list[dict[str, Any]] = [json.loads(line) for line in open(partial)] if partial.exists() else []
    finished = {r["key"] for r in done}
    seen: set[tuple[str, str, str]] = set()
    first, repeats = [], []
    for q in queries:
        if q["key"] in finished:
            continue
        (repeats if _identity(q) in seen else first).append(q)
        seen.add(_identity(q))
    if schema_path is None and any(not q.get("schema") for q in queries):
        raise ValueError("a query without a schema, and no schema_path")
    work = [(q, str(q.get("schema") or schema_path), config.to_json(), code) for q in first + repeats]
    total = len(done) + len(work)
    with open(partial, "a") as sink:
        def keep(rec: dict[str, Any]) -> None:
            done.append(rec)
            sink.write(json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            sink.flush()
            if progress:
                progress(len(done), total)

        if jobs > 1:
            with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(_worker, w) for w in work]):
                    keep(future.result())
        else:
            for w in work:
                keep(_worker(w))
    done.sort(key=lambda r: r["key"])
    schemas = sorted({str(q.get("schema") or schema_path) for q in queries})
    summary = summarize(done, config, started, _now(), code,
                        schema=({"name": Path(schemas[0]).stem, "version": schema_version(schema_json_of(schemas[0]))} if len(schemas) == 1 else
                                {"pool": [{"name": Path(sp).stem, "version": schema_version(schema_json_of(sp))} for sp in schemas]}))
    (out / "results.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False, default=str) + "\n" for r in done))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    partial.unlink()
    return summary


def _dist(values: list[float]) -> dict[str, float]:
    """n, median, p90 (the nearest-rank percentile: the smallest value with at least 90% of the values at or below
    it), max (ms) and the sum (s)."""
    values = sorted(values)
    if not values:
        return {}
    return {"n": len(values), "median_ms": round(statistics.median(values), 1),
            "p90_ms": round(values[max(0, math.ceil(0.9 * len(values)) - 1)], 1),
            "max_ms": round(values[-1], 1), "sum_s": round(sum(values) / 1000, 2)}


FINDINGS_LISTED = 200  # findings listed one by one in the summary (all are counted by kind)


def summarize(records: list[dict[str, Any]], config: ExperimentConfig, started: str, finished: str, code: dict[str, Any],
              schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """The run's aggregates: per component the time (median / p90 / max, over the queries where it ran), the tokens and
    the estimated and paid cost per side (with the known part, the calls without a cost and the unreported token
    counts), the verdicts of both arms (`not requested` for an arm the configuration did not ask for, `not reached`
    for a query that did not get that far), the findings by kind, and the identification of the run."""
    def counts(path: str, requested: bool) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in records:
            node = r.get(path) or {}
            value = node.get("final") if isinstance(node, dict) else None
            label = value if value in ("yes", "no", "undecided") else ("not reached" if requested else "not requested")
            out[label] = out.get(label, 0) + 1
        return dict(sorted(out.items()))

    sides: dict[str, Any] = {}
    for side in ("decomposition", "verification", "direct"):
        per = [r.get("tokens_and_cost", {}).get(side) for r in records if r.get("tokens_and_cost", {}).get(side) is not None]
        known = [p for p in per if p.get("calls") is not None]
        tokens: dict[str, int | None] = {}
        for k in TOKEN_FIELDS:
            vals = [(p.get("tokens") or {}).get(k) for p in known if p.get("calls")]
            reported = [v for v in vals if v is not None]
            tokens[k] = sum(reported) if (reported or not vals) else None
        sides[side] = {"queries": len(per), "calls": sum(p.get("calls") or 0 for p in known),
                       "cached_calls": sum(p.get("cached_calls") or 0 for p in known), "failed_calls": sum(p.get("failed_calls") or 0 for p in known),
                       "queries_with_unknown_calls": len(per) - len(known), "queries_with_unpriced_calls": sum(1 for p in known if p.get("unpriced_calls")),
                       "tokens": tokens,
                       "tokens_unreported": {k: sum((p.get("tokens_unreported") or {}).get(k, 0) for p in known) for k in TOKEN_FIELDS},
                       "unpriced_calls": sum(p.get("unpriced_calls") or 0 for p in known),
                       "estimated_usd": _usd([p.get("estimated_usd") for p in per]),
                       "estimated_usd_known": _usd_known([p.get("estimated_usd_known") for p in known]),
                       "paid_usd": _usd([p.get("paid_usd") for p in per]),
                       "paid_usd_known": _usd_known([p.get("paid_usd_known") for p in known])}
    findings: dict[str, int] = {}
    for r in records:
        for f in r.get("findings", []):
            findings[f["kind"]] = findings.get(f["kind"], 0) + 1
    listed = [{"key": r["key"], **f} for r in records for f in r.get("findings", [])]
    blocks = [r["optimum"]["n_blocks"] for r in records if "optimum" in r]
    return {
        "record": RECORD_VERSION, "config": config.to_json(), "schema": schema, "versions": _versions(),
        "run": {"tag": config.run_tag, "started_at": started, "finished_at": finished, "code": code, "contract_version": M.CONTRACT_VERSION},
        "queries": len(records), "distinct_queries": len({_identity(r["query"]) for r in records}), "sandbox": bool(config.model.sandbox),
        "verdicts": {"decomposed": counts("verdicts", config.decomposed_arm), "direct": counts("direct", config.direct_arm)},
        "findings": dict(sorted(findings.items())),
        "findings_list": listed[:FINDINGS_LISTED], "findings_list_truncated": max(0, len(listed) - FINDINGS_LISTED),
        "blocks_per_optimum": {"n": len(blocks), "median": statistics.median(blocks) if blocks else None, "max": max(blocks, default=None)},
        "time_ms": {c: _dist([r["time_ms"][c] for r in records if c in r.get("time_ms", {})]) for c in COMPONENTS + ("decomposition_side", "total")},
        "tokens_and_cost": sides,
    }
