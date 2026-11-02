"""Command line of the optimizer (component 6).

    python -m sqldecomp.optimize --sql "SELECT ..." --schema benchmarks/toy/schema.json [--db sqldecomp_test_x] [--json]
    python -m sqldecomp.optimize --sql ... --schema ... [--scoring human] [--tau inner_join=2,pred=3,...] [--beta B]
                                 [--excess relative|absolute] [--link hinge|logistic] [--mu M] [--theta-json FILE]
    python -m sqldecomp.optimize --corpus DIR --schema beaver_table_dw_schema.json --out NEW_DIR [--jobs N] [--seed S]

One query: the optimal decomposition under the chosen scoring (`sqldecomp.scoring_args`: by default the human-facing
model of component 3 with its default theta; every block then also shows its nineteen feature values, the excess over
each threshold and -log F), every block in the reader form
(`print_block(inputs="name", within=d)`) with its class, its cost -log F, its boundary columns (with types and the
keys that hold on them), its parameters and its inputs' columns; then the search statistics. `--db` runs check C3
(a read-only EXPLAIN) on the recomposed optimum and on every block of it. `--compare` also prints the cost of the
coarsest and the finest decomposition and of seeded random ones, which the optimum must not exceed.

Corpus: one JSON record per query into `optima.jsonl`, a `summary.json`, and the timings apart (`timings.jsonl`,
`timings_summary.json`). The two result files are a function of (corpus, schema, scoring, seed): the same seed gives
byte-identical files. A query whose decomposition does not finish within the decomposition timeout (`--timeout`,
300 s by default, over import, memo, constraints, legality, scoring and the exact search) is a FINDING in the record
(`finding`, `timeout` with the stage reached), never an approximation. The output directory must not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from sqldecomp.blocks import Blocks, decomposition_key
from sqldecomp.clock import DECOMPOSITION_TIMEOUT_S, Deadline, DecompositionTimeout, memo_budget, memo_stopped, timeout_seconds
from sqldecomp.constraints import derive_constraints
from sqldecomp.importer import UnsupportedSQL, import_sql
from sqldecomp.memo import build_memo, signature
from sqldecomp.model import Block, Decomposition, Schema, validate_import_result
from sqldecomp.optimizer import Optimizer, Optimum, decomposition_cost
from sqldecomp.pipeline import load_any_schema
from sqldecomp.printer import Printer, PrintError, check_valid
from sqldecomp.boundary import Boundary
from sqldecomp.scoring import Scoring
from sqldecomp.scoring_args import add_scoring_arguments, is_judge, make_scoring, scoring_spec_from_args


def scoring_from_args(args) -> dict[str, Any]:
    """The scoring SPEC of the command line (a JSON-able dict; `make_scoring` builds the Scoring per query), or the spec
    a caller hands over directly as `args.spec`."""
    spec = getattr(args, "spec", None)
    spec = spec if spec is not None else scoring_spec_from_args(args)
    if is_judge(spec):
        raise SystemExit("error: --semantic judge rates the cut points with the question and two model calls: it runs in the verification "
                         "run (python -m sqldecomp.pipeline --verify)")
    return spec


# --------------------------------------------------------------------------------------------------------------
# one query
# --------------------------------------------------------------------------------------------------------------


class IncompleteMemo(Exception):
    """The memo's saturation hit its time budget (sound, not complete): an optimum over it would be an optimum over
    fewer decompositions than the query has, so it is reported as a finding, never computed silently."""


def prepare(sql: str, schema: Schema, deadline: Deadline | None = None):
    """(result, memo, blocks, printer) of a query; the memo's saturation runs on what `deadline` has left (a timeout
    at stage `memo` when it runs out; without a deadline the library's budget applies, an incomplete memo then being
    `IncompleteMemo`)."""
    if deadline is not None:
        deadline.check("import")
    result = import_sql(sql, schema)
    problems = validate_import_result(result)
    if problems:
        raise ValueError("invalid import result: " + "; ".join(problems))
    budget = memo_budget(deadline)
    memo = build_memo(result, budget_s=budget)
    memo_stopped(memo.complete, deadline)
    if not memo.complete:
        raise IncompleteMemo(f"the memo's saturation hit its budget of {budget:g} s")
    return result, memo, Blocks(memo, result), Printer(result, memo)


def optimize_one(sql: str, schema: Schema, scoring: Scoring | dict[str, Any], *, timeout_s: float | None = DECOMPOSITION_TIMEOUT_S):
    """(result, memo, blocks, printer, optimizer, optimum) under the decomposition timeout `timeout_s` (None: no
    limit). Raises DecompositionTimeout (TooManyBlocks at the memory valve) when the clock runs out first. `scoring`
    is a Scoring, or a spec (`sqldecomp.scoring_args`) from which the Scoring of this query is built."""
    deadline = Deadline(timeout_s)
    result, memo, blocks, printer = prepare(sql, schema, deadline)
    deadline.check("constraints")
    signatures = {g: signature(memo, result, g) for g in memo.groups}
    constraints = derive_constraints(memo, result, schema)
    boundary = Boundary(memo, result, blocks, printer, constraints, schema)  # the scoring and the contexts share it
    if isinstance(scoring, dict):
        scoring = make_scoring(scoring, memo, result, blocks, boundary)
    opt = Optimizer(memo, result, blocks, scoring, schema=schema, constraints=constraints, signatures=signatures,
                    printer=printer, boundary=boundary, deadline=deadline)
    return result, memo, blocks, printer, opt, opt.optimize()


def alternatives(blocks: Blocks, opt: Optimizer, scoring: Scoring, seed: int, wanted: int = 8,
                 opt_decomposition: Decomposition | None = None) -> dict[str, Any]:
    """The cost of the coarsest, the finest and seeded random decompositions under the same scoring: what the
    optimum is compared with (a non-vacuity check: it must not exceed any of them)."""
    key_of = lambda b: (opt._key_of(b), opt.w[opt._key_of(b)])  # noqa: E731
    out: dict[str, Any] = {}
    for label, d in (("coarsest", blocks.coarsest()), ("finest", blocks.finest())):
        cost, k = decomposition_cost(blocks, d, key_of, scoring.omega)
        out[label] = {"blocks": len(d.blocks), "unique_blocks": k, "cost": round(cost, 9)}
    costs = []
    optimum_key = decomposition_key(opt_decomposition) if opt_decomposition is not None else None
    others = [c["cost"] for label, c in out.items()
              if optimum_key is None or decomposition_key(blocks.coarsest() if label == "coarsest" else blocks.finest()) != optimum_key]
    for d in blocks.samples(seed, wanted=wanted, tries=10 * wanted):
        cost = round(decomposition_cost(blocks, d, key_of, scoring.omega)[0], 9)
        costs.append(cost)
        if optimum_key is None or decomposition_key(d) != optimum_key:
            others.append(cost)
    out["random"] = {"decompositions": len(costs), "min_cost": min(costs) if costs else None}
    out["other_costs"] = others  # the costs of the alternatives that are not the optimum itself
    return out


def opt_boundary_columns(optimum: Optimum, blocks: Blocks, g: str):
    """What THIS position exposes (its own `Boundary.columns`), as its printed SQL shows it. The class context holds
    the union over the class's positions in the REPRESENTATIVE's names, which need not be this position."""
    boundary = getattr(optimum, "boundary", None)
    if boundary is None:
        ctx = optimum.contexts[optimum.class_of_block[g]]
        return ctx.exposed, ctx.star
    return boundary.columns(g)


def block_lines(blocks: Blocks, printer: Printer, optimum: Optimum, scoring: Scoring | None = None) -> list[str]:
    """The blocks of the optimum in the reader form, with class, cost, boundary columns and inputs; under the human
    scoring also the nineteen feature values, the excess over each threshold and -log F."""
    from sqldecomp.complexity import HumanScoring
    d = optimum.decomposition
    lines = []
    for b in d.blocks:
        cid = optimum.class_of_block[b.output]
        ctx = optimum.contexts[cid]
        info = blocks.describe(b)
        try:
            text = printer.block_sql(b, inputs="name", within=d)
            params = printer.block_parameters(b, within=d)
        except (PrintError, Exception) as e:  # noqa: BLE001
            text, params = f"-- {type(e).__name__}: {e}", []
        shared = optimum.classes_shared.get(cid)
        head = (f"### block {b.output}{'(' + ', '.join(params) + ')' if params else ''} <- {sorted(b.inputs)}  "
                f"class {cid}  -log F = {optimum.cost_by_class[cid]:.6f}  [{', '.join(info.kinds)}]")
        if shared:
            head += f"  SHARED with {', '.join(g for g in shared if g != b.output)} (paid once)"
        lines.append(head)
        own, own_star = opt_boundary_columns(optimum, blocks, b.output)
        columns = ", ".join(f"{c.name}:{c.type}" for c in own) + (" (SELECT *)" if own_star else "")
        lines.append(f"exposes: {columns or '(nothing by name)'}")
        if ctx.block.output != b.output or len(ctx.occurrences) > 1:  # the class union, in the representative's names
            union = ", ".join(f"{c.name}:{c.type}" for c in ctx.exposed) + (" (SELECT *)" if ctx.star else "")
            lines.append(f"the class exposes (union over its positions, named as at {ctx.block.output}): {union or '(nothing by name)'}")
        if ctx.output.keys or ctx.output.fks:
            lines.append("keys on the output: " + "; ".join(" + ".join(k) or "(at most one row)" for k in ctx.output.keys)
                         + ("  fks: " + "; ".join(f"({', '.join(c)}) -> {t}({', '.join(r)})" for c, t, r in ctx.output.fks)
                            if ctx.output.fks else ""))
        for i in sorted(b.inputs):  # THIS block's inputs, with what each exposes at its position
            cols, star = opt_boundary_columns(optimum, blocks, i)
            lines.append(f"input {i} exposes: " + (", ".join(f"{c.name}:{c.type}" for c in cols) or "(SELECT *)")
                         + (" [parameterized]" if blocks.parameters(Block(i)) else ""))
        if ctx.parameters:
            lines.append("parameters: " + "; ".join(f"{', '.join(q.columns) or q.alias} (of {q.alias!r}, {'own' if q.own else 'through an input'})"
                                                    for q in ctx.parameters))
        lines.append(f"operators: {ctx.operators}  depth {ctx.depth}  meaningfulness {ctx.meaningfulness}")
        if isinstance(scoring, HumanScoring):
            lines += feature_lines(scoring, ctx)
        lines.append(text)
        extra = optimum.expose_at(b)
        if extra:  # the text verified ONCE for the class: this position's SQL exposing the class's union (OPT-O1); a
            # PrintError here is an error of the run, never a line of the report
            union_text = printer.block_sql(b, inputs="name", within=d, expose=extra)
            lines += [f"verified once for class {cid}, exposing the union ({', '.join(f'{a}.{c}' for a, c in extra)}):", union_text]
        lines.append("")
    return lines


def feature_lines(scoring, ctx) -> list[str]:
    """The human model's view of one class: the feature values, the excess per feature over its threshold with the
    contribution to -log F, and -log F itself."""
    r = scoring.risk(ctx)
    v = scoring.features.of(ctx)
    over = " ".join(f"{row.feature}:{row.value}>{row.threshold:g}(excess {row.excess:.3f}, +{row.contribution:.4f})"
                    for row in r.over_threshold())
    return [f"features: {v}",
            f"excess: {over or 'none: every feature within its threshold'}  -log F = {r.total:.6f}"
            + (f"  (semantic {r.semantic_contribution:+.4f})" if r.semantic_contribution else "")]


def features_of_optimum(scoring, optimum: Optimum) -> dict[str, Any] | None:
    """Per class of the decomposition: the nineteen values, the breakdown extras, the excess rows and -log F (human
    scoring only; None otherwise)."""
    from sqldecomp.complexity import HumanScoring
    if not isinstance(scoring, HumanScoring):
        return None
    out: dict[str, Any] = {}
    for cid, ctx in optimum.contexts.items():
        r = scoring.risk(ctx)
        v = scoring.features.of(ctx)
        out[cid] = {"values": dict(v.values), "risk": round(r.total, 9), "F": r.F,
                    "over_threshold": {row.feature: {"value": row.value, "threshold": row.threshold,
                                                     "excess": round(row.excess, 9), "contribution": round(row.contribution, 9)}
                                       for row in r.over_threshold()},
                    "bool_depth": v.breakdown["bool_depth"], "order_keys": v.breakdown["order_keys"],
                    "scope_nesting": v.breakdown["scope_nesting"], "optypes": v.breakdown["optypes"],
                    "notes": list(v.notes)}
    return out


def record_of(name: str, sql: str, blocks: Blocks, opt: Optimizer, optimum: Optimum, scoring: Scoring, seed: int) -> dict[str, Any]:
    d = optimum.decomposition
    s = optimum.stats
    rec: dict[str, Any] = {
        "query": name, "sql_sha1": hashlib.sha1(sql.encode()).hexdigest()[:12], "scoring": scoring.name,
        "blocks": len(d.blocks), "unique_blocks": optimum.unique_blocks, "cost": round(optimum.cost, 9),
        "score": optimum.score, "omega": optimum.omega,
        "decomposition": [[b.output, sorted(b.inputs)] for b in d.blocks],
        "classes_of_blocks": {g: cid for g, cid in optimum.class_of_block.items()},
        "cost_by_class": {cid: round(w, 9) for cid, w in optimum.cost_by_class.items()},
        "classes_shared": optimum.classes_shared,
        "boundary": {b.output: {"exposes": [[c.name, c.type] for c in opt_boundary_columns(optimum, blocks, b.output)[0]],
                                "star": opt_boundary_columns(optimum, blocks, b.output)[1],
                                "class_exposes": [[c.name, c.type] for c in optimum.contexts[optimum.class_of_block[b.output]].exposed],
                                "keys": [list(k) for k in optimum.contexts[optimum.class_of_block[b.output]].output.keys],
                                "inputs": len(b.inputs), "parameters": len(blocks.parameters(b))}
                     for b in d.blocks},
        "stats": {k: v for k, v in s.items() if not k.endswith("_ms")},
        "alternatives": alternatives(blocks, opt, scoring, seed, opt_decomposition=d),
        "problems": blocks.problems(d),
        "union_notes": list(opt.builder.union_notes),
    }
    rec["differs_from_coarsest"] = decomposition_key(d) != decomposition_key(blocks.coarsest())
    rec["differs_from_finest"] = decomposition_key(d) != decomposition_key(blocks.finest())
    features = features_of_optimum(scoring, optimum)
    if features is not None:
        rec["features"] = features
        rec["classes_over_threshold"] = sum(1 for f in features.values() if f["over_threshold"])
        rec["blocks_over_threshold"] = sum(1 for b in d.blocks if features[optimum.class_of_block[b.output]]["over_threshold"])
        rec["max_excess"] = max((row["excess"] for f in features.values() for row in f["over_threshold"].values()), default=0.0)
        rec["feature_notes"] = sorted(set(scoring.features.notes))
    alt = rec["alternatives"]
    worst = min(alt["coarsest"]["cost"], alt["finest"]["cost"], alt["random"]["min_cost"] if alt["random"]["min_cost"] is not None else alt["finest"]["cost"])
    rec["optimum_beats_alternatives"] = optimum.cost <= worst + 1e-9
    # strictly cheaper than every alternative that is another decomposition (ties are common: under the human model
    # every block within its thresholds costs 0)
    rec["other_alternatives"] = len(alt["other_costs"])
    rec["optimum_strictly_cheaper_than_alternatives"] = bool(alt["other_costs"]) and all(
        optimum.cost < c - 1e-9 for c in alt["other_costs"])  # False, not vacuously True, without another alternative
    return rec


def c3_of(sql: str, blocks: Blocks, printer: Printer, optimum: Optimum, db: str) -> dict[str, Any]:
    """Check C3 on the recomposed optimum and on every block of it (inline form, closed form for parameterized ones)."""
    d = optimum.decomposition
    jobs = [("original", check_valid(sql, dbname=db))]
    try:
        jobs.append(("recomposed", check_valid(printer.decomposition_sql(d), dbname=db)))
    except PrintError as e:
        jobs.append(("recomposed", type("R", (), {"ok": False, "detail": str(e)})()))
    for b in d.blocks:
        try:
            jobs.append((f"block {b.output}", check_valid(printer.block_sql(b, inputs="inline", bind_outer=True), dbname=db)))
        except PrintError as e:
            jobs.append((f"block {b.output}", type("R", (), {"ok": False, "detail": str(e)})()))
        extra = optimum.expose_at(b)
        if extra:  # the union-exposing text of a shared class must be valid too
            try:
                jobs.append((f"block {b.output} with the class union",
                             check_valid(printer.block_sql(b, inputs="inline", bind_outer=True, expose=extra), dbname=db)))
            except PrintError as e:
                jobs.append((f"block {b.output} with the class union", type("R", (), {"ok": False, "detail": str(e)})()))
    return {what: {"ok": v.ok, "detail": "" if v.ok else str(v.detail)[:300]} for what, v in jobs}


# --------------------------------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------------------------------


def _worker(args: tuple) -> tuple[dict, dict]:
    name, sql, schema_path, spec, seed, timeout = args
    schema = load_any_schema(schema_path)
    timings: dict[str, float] = {"query": name}
    t0 = time.perf_counter()
    try:
        result, memo, blocks, printer, opt, optimum = optimize_one(sql, schema, spec, timeout_s=timeout)
    except UnsupportedSQL as e:
        return {"query": name, "finding": f"unsupported: {e.reason}"}, timings
    except DecompositionTimeout as e:  # the clock, or the memory valve of the enumeration
        return ({"query": name, "finding": f"timeout at stage {e.stage}: {e}", "timeout": e.finding()},
                {**timings, "total_ms": round((time.perf_counter() - t0) * 1000, 2)})
    except IncompleteMemo as e:
        return {"query": name, "finding": f"incomplete memo: {e}"}, {**timings, "total_ms": round((time.perf_counter() - t0) * 1000, 2)}
    try:
        rec = record_of(name, sql, blocks, opt, optimum, opt.scoring, seed)
    except PrintError as e:  # a block of the optimum (or its text verified once) that does not print: a finding
        return {"query": name, "finding": f"print: {e}"}, {**timings, "total_ms": round((time.perf_counter() - t0) * 1000, 2)}
    # what it would cost if F read the printed SQL: measured on up to 20 class representatives
    t1 = time.perf_counter()
    sample = list(opt.classes.values())[:20]
    for info in sample:
        try:
            printer.block_sql(info.representative, inputs="name")
        except Exception:  # noqa: BLE001
            pass
    per_print = (time.perf_counter() - t1) * 1000 / max(1, len(sample))
    timings.update({k: v for k, v in optimum.stats.items() if k.endswith("_ms")},
                   total_ms=round((time.perf_counter() - t0) * 1000, 2), per_print_ms=round(per_print, 3),
                   prints_for_all_classes_ms=round(per_print * len(opt.classes), 1))
    return rec, timings


def run_corpus(queries: list[tuple[str, str]], schema_path: str, out_dir: str | Path, args, *, jobs: int = 1) -> dict[str, Any]:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; a corpus run never writes into an existing directory")
    spec = scoring_from_args(args)
    work = [(name, sql, str(schema_path), spec, args.seed, args.timeout) for name, sql in queries]
    if jobs > 1:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            done = list(pool.map(_worker, work, chunksize=2))
    else:
        done = list(map(_worker, work))
    done.sort(key=lambda item: item[0]["query"])
    records, timings = [r for r, _ in done], [t for _, t in done]
    summary = summarize(records, args)
    out.mkdir(parents=True)
    (out / "optima.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (out / "timings.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in timings))
    (out / "timings_summary.json").write_text(json.dumps(summarize_timings(timings), indent=2, sort_keys=True) + "\n")
    return summary


def _dist(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    if not values:
        return {}
    return {"median": statistics.median(values), "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1],
            "sum": round(sum(values), 3)}


def summarize(records: list[dict], args) -> dict[str, Any]:
    ok = [r for r in records if "finding" not in r]
    findings = [{"query": r["query"], "finding": r["finding"]} for r in records if "finding" in r]
    stat = lambda key: _dist([r["stats"][key] for r in ok if key in r["stats"]])  # noqa: E731
    spec = scoring_from_args(args)
    out = {
        "scoring": {**spec, "seed": args.seed},
        "files": len(records), "solved": len(ok), "findings": findings,
        "blocks": _dist([r["blocks"] for r in ok]), "unique_blocks": _dist([r["unique_blocks"] for r in ok]),
        "cost": _dist([r["cost"] for r in ok]),
        "queries_with_a_shared_class_in_the_optimum": sum(bool(r["classes_shared"]) for r in ok),
        "shared_classes_in_optima": sum(len(r["classes_shared"]) for r in ok),
        "optimum_differs_from_coarsest": sum(r["differs_from_coarsest"] for r in ok),
        "optimum_differs_from_finest": sum(r["differs_from_finest"] for r in ok),
        "optimum_differs_from_both": sum(r["differs_from_coarsest"] and r["differs_from_finest"] for r in ok),
        "optimum_beats_every_alternative": sum(r["optimum_beats_alternatives"] for r in ok),
        "optimum_strictly_cheaper_than_every_alternative": sum(r["optimum_strictly_cheaper_than_alternatives"] for r in ok),
        "queries_without_another_alternative": sum(r["other_alternatives"] == 0 for r in ok),
        "one_block_optima": sum(r["blocks"] == 1 for r in ok),
        "timeout_s": args.timeout,
        "optima_with_problems": [r["query"] for r in ok if r["problems"]],
        "union_notes": sum(len(r["union_notes"]) for r in ok),
        "methods": {m: sum(r["stats"]["method"] == m for r in ok) for m in sorted({r["stats"]["method"] for r in ok})},
        "f_evaluations": stat("f_evaluations"), "blocks_evaluated": stat("blocks_evaluated"), "classes": stat("classes"),
        "repeatable_classes": stat("repeatable_classes"), "states_expanded": stat("states_expanded"),
        "states_generated": stat("states_generated"), "forced_cuts": stat("forced_cuts"),
        "queries_with_repeatable_classes": sum(r["stats"]["repeatable_classes"] > 0 for r in ok),
    }
    if spec.get("kind") == "human":  # what the tau sweep reads
        over = [r for r in ok if "features" in r]
        out["human"] = {
            "one_block_optima": sum(r["blocks"] == 1 for r in ok),
            "queries_with_a_block_over_threshold": sum(r["blocks_over_threshold"] > 0 for r in over),
            "blocks_over_threshold": sum(r["blocks_over_threshold"] for r in over),
            "classes_over_threshold": sum(r["classes_over_threshold"] for r in over),
            "blocks_in_optima": sum(r["blocks"] for r in ok),
            "max_excess": _dist([r["max_excess"] for r in over]),
            "risk_of_optimum": _dist([sum(f["risk"] for f in r["features"].values()) for r in over]),
            "score_of_optimum": _dist([r["score"] for r in ok if r["score"] != float("inf")]),
            "feature_notes": sorted({n for r in over for n in r.get("feature_notes", ())}),
        }
    return out


def summarize_timings(timings: list[dict]) -> dict[str, Any]:
    with_total = [t for t in timings if "total_ms" in t]
    slowest = sorted(with_total, key=lambda t: -t["total_ms"])[:5]
    return {"per_query_total_ms": _dist([t["total_ms"] for t in with_total]),
            "per_stage_ms": {k: _dist([t[k] for t in with_total if k in t])
                             for k in ("enumeration_ms", "scoring_ms", "search_ms", "time_ms")},
            "per_print_ms": _dist([t["per_print_ms"] for t in with_total if "per_print_ms" in t]),
            "prints_for_all_classes_ms": _dist([t["prints_for_all_classes_ms"] for t in with_total if "prints_for_all_classes_ms" in t]),
            "slowest": [{k: v for k, v in t.items()} for t in slowest]}


# --------------------------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sqldecomp.optimize", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sql", help="one query (text, or @file)")
    ap.add_argument("--corpus", help="directory of *.json files with a `sql` field")
    ap.add_argument("--schema", required=True, help="schema file: contract JSON or the BEAVER format")
    ap.add_argument("--out", help="corpus mode: NEW directory for optima.jsonl, summary.json and the timings")
    add_scoring_arguments(ap)
    ap.add_argument("--db", help="run check C3 (read-only EXPLAIN) on the optimum's recomposed query and blocks")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random decompositions the optimum is compared with")
    ap.add_argument("--jobs", type=int, default=1, help="corpus mode: worker processes")
    ap.add_argument("--every", type=int, default=1, help="corpus mode: every k-th file")
    ap.add_argument("--timeout", type=timeout_seconds, default=DECOMPOSITION_TIMEOUT_S,
                    help="the decomposition timeout per query in seconds (import to the optimum; a timeout finding with the stage reached)")
    ap.add_argument("--json", action="store_true", help="--sql only: print the record as JSON instead of the report")
    ap.add_argument("--compare", action="store_true", help="--sql only: print the costs of the alternatives too")
    args = ap.parse_args(argv)
    if bool(args.sql) == bool(args.corpus):
        ap.error("give exactly one of --sql and --corpus")
    if args.sql:
        sql = Path(args.sql[1:]).read_text() if args.sql.startswith("@") else args.sql
        schema = load_any_schema(args.schema)
        try:
            spec = scoring_from_args(args)
            result, memo, blocks, printer, opt, optimum = optimize_one(sql, schema, spec, timeout_s=args.timeout)
        except (UnsupportedSQL, DecompositionTimeout, IncompleteMemo, PrintError, ValueError) as e:
            out = {"finding": f"{type(e).__name__}: {e}"}
            if isinstance(e, DecompositionTimeout):  # the structured finding of the other outputs: kind, stage, budget, time
                out["timeout"] = e.finding()
            print(json.dumps(out, indent=1))
            return 1
        scoring = opt.scoring
        rec = record_of("query", sql, blocks, opt, optimum, scoring, args.seed)
        if args.db:
            rec["c3"] = c3_of(sql, blocks, printer, optimum, args.db)
        if args.json:
            print(json.dumps(rec, indent=1, sort_keys=True))
        else:
            lines = [f"# optimum under {scoring.name}: {len(optimum.decomposition.blocks)} blocks, "
                     f"{optimum.unique_blocks} unique, cost {optimum.cost:.6f} (score {optimum.score:.6g}, "
                     f"Omega {optimum.omega:.4f})", ""]
            lines += block_lines(blocks, printer, optimum, scoring)
            lines += ["# recomposed", printer.decomposition_sql(optimum.decomposition), ""]
            lines += [f"# stats: {json.dumps(rec['stats'])}"]
            if args.compare:
                lines += [f"# alternatives: {json.dumps(rec['alternatives'])}",
                          f"# differs from coarsest: {rec['differs_from_coarsest']}, from finest: {rec['differs_from_finest']}, "
                          f"beats every alternative: {rec['optimum_beats_alternatives']}"]
            if "c3" in rec:
                lines += ["# C3", json.dumps(rec["c3"], indent=1)]
            if rec["union_notes"]:
                lines += ["# union notes", *rec["union_notes"]]
            print("\n".join(lines))
        bad = rec["problems"] or not rec["optimum_beats_alternatives"] or ("c3" in rec and rec["c3"]["original"]["ok"]
                                                                            and not all(v["ok"] for v in rec["c3"].values()))
        return 1 if bad else 0
    if not args.out:
        ap.error("--corpus needs --out")
    from sqldecomp.importer.corpus import load_queries
    queries = load_queries(args.corpus)[::max(1, args.every)]
    summary = run_corpus(queries, args.schema, args.out, args, jobs=args.jobs)
    print(json.dumps({k: summary[k] for k in ("files", "solved", "findings", "methods", "optimum_differs_from_coarsest",
                                              "optimum_differs_from_finest", "queries_with_a_shared_class_in_the_optimum",
                                              "optimum_beats_every_alternative")}, indent=1))
    return 0 if summary["solved"] == summary["files"] and summary["optimum_beats_every_alternative"] == summary["solved"] else 1


if __name__ == "__main__":
    sys.exit(main())
