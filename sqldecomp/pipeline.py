"""The pipeline end to end. Two modes.

THE VERIFICATION RUN (`--verify`, `--direct`; `sqldecomp.runner`): SQL + question -> decomposition -> tasks -> the
verifier's verdicts -> one verdict with the rejecting blocks named (and the direct arm on request), one result record
per query with the whole experiment configuration, the time per component, the tokens and the estimated cost:

    python -m sqldecomp.pipeline --sql "SELECT ..." --question TEXT [--evidence TEXT] --schema S --verify [--direct]
                                 --cache DIR [--json]
    python -m sqldecomp.pipeline (--corpus DIR | --queries FILE.jsonl) --schema S --verify [--direct] --cache DIR --out NEW_DIR

THE ACCEPTANCE HARNESS (no `--verify`): the components and the checks, no LLM unless `--tasks`:

    import -> validate -> memo -> signatures -> constraints -> blocks [-> optimizer] -> print -> C1 -> C2 -> C3

(the decomposition timeout covers the memo, the blocks of the decompositions and the optimizer; the census of every
legal block that the record and the sampled checks use runs after the optimizer on a clock of its own, so that a
statistic of the harness never takes the budget of the decomposition)

    python -m sqldecomp.pipeline --sql "SELECT ..." --schema benchmarks/toy/schema.json [--db NAME] [--execute]
    python -m sqldecomp.pipeline --corpus DIR --schema beaver_table_dw_schema.json --out NEW_DIR [--db beaver_dw]

What is checked for one query:

- C1 (round trip) on EVERY group of the memo;
- C2 (recomposition) on the coarsest decomposition (one block, plus a block for every group that is read from two or
  more places: design decision D2), the finest one, `samples` random ones and, when the query has predicate or
  select-item subqueries, one targeted decomposition per cut point inside a subquery body, so that every way a block
  can hold part of a subquery is exercised where it exists;
- with `--optimize` (Options.optimize), the OPTIMUM under the chosen scoring (`sqldecomp.scoring_args`: the human-facing
  model with its default theta unless another is chosen) is one more decomposition (label `optimum`): it goes through C2 and C3 like the others, and `record["optimizer"]` holds its
  cost, its unique blocks, its shared classes and the search statistics. An exact search that does not finish within
  its budget is a FINDING in the record, never an approximation;
- C3 (Postgres EXPLAIN, read-only) when a database is given: the original, the printed query, every group, every
  block of those decompositions, a seeded sample of further legal blocks, every recomposed query and every TEXTUAL
  composition. A print has to be judged like the original; an original that fails EXPLAIN itself is counted
  separately, with the reason of every failing print beside it;
- names: the columns that a block printed by itself leaves unbound are exactly the parameters `Blocks.parameters`
  reports (the printer and the legality rules agree on which blocks are parameterized, and by what);
- textual composition: the blocks AS PRINTED with named inputs (what a reader of one block sees), pasted into one
  another. C2 certifies the printer's own recomposition, which is built from plans; this composes the texts;
- negative controls, so that no run consists of checks that never say no: C1 and C2 are given the print of ANOTHER
  group and must reject it, C3 is given the print with a column that does not exist and must reject it.

A corpus run writes `results.jsonl` (one record per query) and `summary.json` into a NEW directory. Both are a
function of (corpus, schema, database, seed): the same seed gives byte-identical files. Timings change from run to
run and go to `timings.jsonl` / `timings_summary.json`. `--execute` (single query only) runs the original, the print
and every recomposed query and compares their rows; it is refused for any database not named `sqldecomp_test_*`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import sqlglot
from sqlglot import exp

from sqldecomp import model as M
from sqldecomp.blocks import BlockInfo, Blocks, decomposition_key
from sqldecomp.constraints import derive_constraints
from sqldecomp.importer import UnsupportedSQL, import_sql
from sqldecomp.memo import build_memo_with_stats, signature
from sqldecomp.model import Block, Decomposition, GroupId, ImportResult, Memo, Schema
from sqldecomp.printer import CheckResult, Printer, PrintError, check_recomposition, check_round_trip, check_valid, sqltext
from sqldecomp.printer.build import boundary_ident
from sqldecomp.schema import load_beaver_schema, load_schema, schema_json_of
from sqldecomp.clock import timeout_seconds

STAGES = ("import", "memo", "signatures", "constraints", "blocks", "optimize", "print", "c1", "c2", "names", "controls", "c3")


@dataclass
class Options:
    seed: int = 0
    samples: int = 4  # random decompositions per query, beside the coarsest and the finest one
    db: str | None = None  # database for C3; None skips C3
    block_sample: int = 0  # further legal blocks per query (seeded) that are printed, name-checked and EXPLAINed
    workers: int = 8  # parallel EXPLAINs
    keep_sql: bool = False  # put every printed SQL into the record (single-query mode)
    optimize: bool = False  # run the optimizer (component 6) under the scoring below; its optimum joins the checks
    timeout_s: float | None = 300.0  # the decomposition timeout per query (import to the optimum and its tasks; the checks are outside)
    scoring: dict | None = None  # a scoring spec (`sqldecomp.scoring_args`); None: the default (`default_spec()`, the human model)
    tasks: str | None = None  # component 9: the method whose tasks are written for the optimum's blocks (None: none)
    tasks_cache: str | None = None  # component 9: the persistent cache of assignments (model calls under <dir>/llm)
    question: str = ""  # component 9: the question N the query is supposed to answer
    evidence: str = ""  # component 9: the evidence given with the question
    model: Any = None  # the task writer's `sqldecomp.llm.ModelConfig` (None: GPT-6 Luna at max effort, the default writer, in the sandbox)
    schema_json: dict | None = None  # component 9: the schema as the texts render it (`schema.schema_json_of`; None: the model's JSON)


@dataclass
class Run:
    """Everything the pipeline built for one query, for callers that want more than the record (the CLI, the tests)."""

    result: ImportResult
    memo: Memo
    printer: Printer
    blocks: Blocks
    constraints: dict[GroupId, M.GroupConstraints]
    signatures: dict[GroupId, str]
    decompositions: list[tuple[str, Decomposition]]
    record: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------------------------
# one query
# --------------------------------------------------------------------------------------------------------------


def run_query(name: str, sql: str, schema: Schema, options: Options | None = None) -> Run | dict[str, Any]:
    """The whole pipeline for one query. Returns a `Run`; when the query cannot be imported or its memo cannot be
    built, returns just the record (with `verdict` = "fail" or "unsupported")."""
    from sqldecomp.clock import Deadline, memo_budget
    options = options or Options()
    deadline = Deadline(options.timeout_s)  # the decomposition side's clock: the checks are not on it
    record: dict[str, Any] = {"query": name, "sql_sha1": hashlib.sha1(sql.encode()).hexdigest()[:12], "failures": []}
    timings: dict[str, float] = {}
    clock = _Clock(timings)

    with clock("import"):
        try:
            result = import_sql(sql, schema)
        except UnsupportedSQL as e:
            record.update(verdict="unsupported", **{"import": {"ok": False, "reason": e.reason, "detail": str(e)}})
            record["timings_ms"] = _ms(timings)
            return record
        problems = M.validate_import_result(result)
    record["import"] = {"ok": not problems, "scopes": len(result.scopes), "side_entries": len(result.side),
                        "conjuncts": sum(e.kind == "conjunct" for e in result.side.values()),
                        "region_inputs": len(result.facts.inputs), "violations": problems}
    if problems:
        return _fail(record, timings, "import", "; ".join(problems))

    with clock("memo"):
        try:
            memo, stats = build_memo_with_stats(result, budget_s=memo_budget(deadline))
        except Exception as e:  # InvalidImportResult, or a tripwire of the read-out
            return _fail(record, timings, "memo", f"{type(e).__name__}: {e}")
    record["memo"] = {"groups": len(memo.groups), "expressions": stats.expressions, "complete": memo.complete,
                      "iterations": stats.iterations}
    if not memo.complete:
        record["failures"].append({"stage": "memo", "detail": f"the decomposition timeout ({options.timeout_s} s) stopped the saturation"})

    blocks = Blocks(memo, result)
    printer = Printer(result, memo)

    with clock("signatures"):
        signatures = {g: signature(memo, result, g) for g in memo.groups}
    record["signatures"] = _sharing(signatures, blocks)

    with clock("constraints"):
        try:
            constraints = derive_constraints(memo, result, schema)
        except Exception as e:
            return _fail(record, timings, "constraints", f"{type(e).__name__}: {e}")
    record["constraints"] = _constraint_summary(constraints, memo, blocks)

    with clock("blocks"):
        decompositions = _decompositions(blocks, options)
    if options.optimize:
        with clock("optimize"):
            record["optimizer"] = _optimize(memo, result, blocks, printer, signatures, constraints, schema, options, decompositions, deadline)
    with clock("blocks"):
        # the census of every legal block (the record, the sampled checks) is the harness's, not the decomposition's:
        # after the optimizer and on a clock of its own, so that it never takes the budget the product has (the
        # verification run's decomposition side takes no census; on BEAVER neutron_220 it took all 300 s)
        legal, complete = _legal_blocks(blocks, Deadline(options.timeout_s))
    cut_points, pipeline_groups = blocks.cut_points(), blocks.pipeline_groups()
    # every group is exactly one of: the root, a cut point, inside one SELECT's pipeline, or no block at all (a bare
    # scan, a `ref`, a fragment that reads no table)
    record["blocks"] = {"cut_points": len(cut_points), "pipeline_groups": len(pipeline_groups),
                        "scan_and_ref_groups": len(memo.groups) - 1 - len(cut_points) - len(pipeline_groups),
                        "legal_blocks": len(legal), "legal_blocks_complete": complete,
                        "finest_blocks": len(decompositions[1][1].blocks)}
    tested = _tested_blocks(decompositions, legal, options)
    record["subqueries"] = _subquery_summary(blocks, decompositions)

    with clock("print"):
        try:
            whole = printer.plan_sql(result.plan)
            group_sql = {g: printer.group_sql(g, bind_outer=True) for g in memo.groups}
        except PrintError as e:
            return _fail(record, timings, "print", f"{type(e).__name__}: {e}")
    record["print"] = {"whole_sha1": hashlib.sha1(whole.encode()).hexdigest()[:12], "groups": len(group_sql),
                       "distinct_group_sql": len(set(group_sql.values()))}
    if options.keep_sql:
        record["print"].update(whole=whole, group_sql=group_sql)

    with clock("c1"):
        record["c1"] = _c1(memo, result, printer, schema, record["failures"])
    with clock("c2"):
        record["c2"] = _c2(decompositions, blocks, memo, result, printer, schema, record["failures"], options)
    with clock("names"):
        record["names"] = _names(tested, blocks, printer, record["failures"])
        textual = _textual(decompositions, blocks, printer, record["failures"])
    with clock("controls"):
        record["controls"] = _controls(memo, result, printer, blocks, schema, group_sql, record["failures"])
    if options.db:
        with clock("c3"):
            record["c3"] = _c3(sql, whole, group_sql, decompositions, tested, textual, blocks, printer, options,
                               record["controls"], record["failures"])

    record["verdict"] = "pass" if not record["failures"] else "fail"
    record["timings_ms"] = _ms(timings)
    return Run(result, memo, printer, blocks, constraints, signatures, decompositions, record, timings)


def _fail(record: dict, timings: dict, stage: str, detail: str) -> dict:
    record["failures"].append({"stage": stage, "detail": detail})
    record["verdict"] = "fail"
    record["timings_ms"] = _ms(timings)
    return record


def _ms(timings: dict[str, float]) -> dict[str, float]:
    return {k: round(v * 1000, 3) for k, v in timings.items()}


class _Clock:
    def __init__(self, timings: dict[str, float]):
        self.timings = timings

    def __call__(self, stage: str) -> "_Clock":
        self.stage = stage
        return self

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, *exc) -> None:
        self.timings[self.stage] = self.timings.get(self.stage, 0.0) + time.perf_counter() - self.start


# --------------------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------------------


def _sharing(signatures: dict[GroupId, str], blocks: Blocks) -> dict[str, Any]:
    """Same task = same signature. Classes that hold two or more occurrences, split by whether the occurrences can
    be blocks (a bare scan of one table trivially repeats)."""
    classes: dict[str, list[GroupId]] = {}
    for g, s in signatures.items():
        classes.setdefault(s, []).append(g)
    shared = [sorted(gs) for gs in classes.values() if len(gs) > 1]
    cuttable = [gs for gs in shared if all(blocks.can_cut(g) or g == blocks.memo.root for g in gs)]
    unique = sum('"unique"' in s for s in signatures.values())
    return {"classes": len(classes), "shared_classes": len(shared), "shared_block_classes": len(cuttable),
            "shared_block_groups": sorted(cuttable), "occurrences_in_shared_block_classes": sum(map(len, cuttable)),
            "groups_made_unique": unique}


def _constraint_summary(constraints: dict[GroupId, M.GroupConstraints], memo: Memo, blocks: Blocks) -> dict[str, Any]:
    keyed = sorted(g for g, c in constraints.items() if c.keys)
    return {"groups": len(constraints), "groups_with_key": len(keyed), "groups_with_fk": sum(bool(c.fks) for c in constraints.values()),
            "cut_points_with_key": sum(blocks.can_cut(g) for g in keyed), "root_has_key": memo.root in keyed,
            "keys": {g: sorted(sorted([c.origin, c.column] for c in k.columns) for k in constraints[g].keys) for g in keyed}}


def _decompositions(blocks: Blocks, options: Options) -> list[tuple[str, Decomposition]]:
    """(label, decomposition): the coarsest one (one block plus the forced cuts), the finest one, random ones, and one
    per cut point inside a subquery body. Distinct, in a fixed order."""
    out: list[tuple[str, Decomposition]] = [("coarsest", blocks.coarsest()), ("finest", blocks.finest())]
    seen = {decomposition_key(d) for _, d in out[:1]}  # the finest one is kept even when it IS the coarsest one
    seen.add(decomposition_key(out[1][1]))
    for d in blocks.samples(options.seed, wanted=options.samples + 2, tries=20 * (options.samples + 2)):
        if decomposition_key(d) not in seen and sum(label == "sample" for label, _ in out) < options.samples:
            seen.add(decomposition_key(d))
            out.append(("sample", d))
    for g in blocks.cut_points():
        if blocks.body_of(g) is not None:
            d = blocks.with_cuts({g})
            if decomposition_key(d) not in seen:
                seen.add(decomposition_key(d))
                out.append((f"cut-in-body:{g}", d))
    return out


def _optimize(memo: Memo, result: ImportResult, blocks: Blocks, printer: Printer, signatures: dict[GroupId, str],
              constraints: dict[GroupId, M.GroupConstraints], schema: Schema, options: Options,
              decompositions: list[tuple[str, Decomposition]], deadline: Any = None) -> dict[str, Any]:
    """The optimum under the chosen scoring (`options.scoring`, a spec of `sqldecomp.scoring_args`; `default_spec()`
    when None), appended to the decompositions under the label `optimum` (so that C2, C3, the names check and the
    textual composition cover it), and what the record says about it. The scoring is built with the query's
    `Boundary`, which the optimizer then shares: the human and llm models read it. The optimizer and the task writer
    run on the query's decomposition timeout (`deadline`): its expiry is a finding with the stage reached."""
    from sqldecomp.boundary import Boundary
    from sqldecomp.clock import DecompositionTimeout
    from sqldecomp.optimizer import Optimizer, decomposition_cost
    from sqldecomp.scoring_args import default_spec, make_scoring
    spec = options.scoring or default_spec()
    boundary = Boundary(memo, result, blocks, printer, constraints, schema)
    scoring = make_scoring(spec, memo, result, blocks, boundary)
    if not memo.complete:  # an optimum over an incomplete memo would range over fewer decompositions
        return {"scoring": scoring.name, "finding": f"incomplete memo: the decomposition timeout ({options.timeout_s} s) stopped the saturation",
                "stats": {}}
    opt = Optimizer(memo, result, blocks, scoring, schema=schema, constraints=constraints, signatures=signatures,
                    printer=printer, boundary=boundary, deadline=deadline)
    try:
        optimum = opt.optimize()
    except DecompositionTimeout as e:  # the clock or the memory valve: reported, never silently approximated
        return {"scoring": scoring.name, "finding": f"timeout at stage {e.stage}: {e}", "timeout": e.finding(), "stats": dict(opt.stats)}
    d = optimum.decomposition
    key_of = lambda b: (opt._key_of(b), opt.w[opt._key_of(b)])  # noqa: E731
    # one entry per tested decomposition (several carry the label `sample`)
    alternatives = {f"{label}#{i}": round(decomposition_cost(blocks, x, key_of, scoring.omega)[0], 9)
                    for i, (label, x) in enumerate(decompositions)}
    others = [c for (label, x), c in zip(decompositions, alternatives.values()) if decomposition_key(x) != decomposition_key(d)]
    decompositions.append(("optimum", d))
    from sqldecomp.optimize import features_of_optimum
    features = features_of_optimum(scoring, optimum)
    human = {} if features is None else {
        "features": features,
        "blocks_over_threshold": sum(1 for b in d.blocks if features[optimum.class_of_block[b.output]]["over_threshold"]),
        "max_excess": max((row["excess"] for f in features.values() for row in f["over_threshold"].values()), default=0.0)}
    tasks: dict[str, Any] = {}
    if options.tasks:  # component 9 on the optimum (`sqldecomp.tasks`)
        from sqldecomp.tasks import MERGED_METHODS, AssignmentError, TaskCache, assign_from_records, block_records, codex_llm, writer_identity
        try:
            records = block_records(optimum, memo, result, blocks, printer, opt.boundary, merged=options.tasks in MERGED_METHODS)
            from sqldecomp.config import WRITER_EFFORT, WRITER_MODEL
            from sqldecomp.llm import ModelConfig
            writer = options.model or ModelConfig(model=WRITER_MODEL, effort=WRITER_EFFORT)
            config = writer.with_(cache_dir=str(Path(options.tasks_cache) / "llm")) if options.tasks_cache else None
            llm = codex_llm(config=config, deadline=deadline) if config is not None else None
            written = assign_from_records(records, question=options.question, evidence=options.evidence,
                                          schema_json=options.schema_json if options.schema_json is not None else M.schema_to_json(schema),
                                          method=options.tasks, llm=llm, cache=TaskCache(options.tasks_cache) if options.tasks_cache else None,
                                          writer=writer_identity(config) if config is not None else None)
            tasks = {"tasks": {"method": options.tasks, "sandbox": config.sandbox if config else None,
                               "blocks": [t.to_json() for t in written]}}
        except AssignmentError as e:
            tasks = {"tasks": {"method": options.tasks, "finding": str(e)[:300]}}
        except DecompositionTimeout as e:
            tasks = {"tasks": {"method": options.tasks, "finding": f"timeout at stage {e.stage}: {e}", "timeout": e.finding()}}
    return {"scoring": scoring.name, "blocks": len(d.blocks), "unique_blocks": optimum.unique_blocks, **human, **tasks,
            "cost": round(optimum.cost, 9), "omega": optimum.omega,
            "decomposition": [[b.output, sorted(b.inputs)] for b in d.blocks],
            "classes_shared": optimum.classes_shared, "cost_by_class": {k: round(v, 9) for k, v in optimum.cost_by_class.items()},
            "differs_from_coarsest": decomposition_key(d) != decomposition_key(blocks.coarsest()),
            "differs_from_finest": decomposition_key(d) != decomposition_key(blocks.finest()),
            "beats_every_other_tested_decomposition": all(optimum.cost <= c + 1e-9 for c in alternatives.values()),
            "strictly_cheaper_than_every_other_tested_decomposition": bool(others) and all(optimum.cost < c - 1e-9 for c in others),
            "other_tested_decompositions": len(others),
            "tested_decompositions": len(alternatives), "cost_of_tested_decompositions": alternatives,
            "problems": blocks.problems(d), "union_notes": list(opt.builder.union_notes),
            "stats": {k: v for k, v in optimum.stats.items() if not k.endswith("_ms") and k != "repeatable_outputs"}}


def _subquery_summary(blocks: Blocks, decompositions: list[tuple[str, Decomposition]]) -> dict[str, Any]:
    bodies = [s for s in blocks.result.scopes if s.kind == "subquery"]
    correlated = []
    for s in bodies:
        g = blocks.memo.group_of(s.plan)
        if g is not None and blocks.parameters(Block(g)):
            correlated.append(s.id)
    inside = [g for g in blocks.cut_points() if blocks.body_of(g) is not None]
    kinds: dict[str, int] = {}
    for _, d in decompositions:
        for b in d.blocks:
            for kind in blocks.describe(b).kinds:
                kinds[kind] = kinds.get(kind, 0) + 1
    return {"bodies": len(bodies), "correlated_bodies": len(correlated), "cut_points_inside_bodies": len(inside),
            "blocks_by_kind": dict(sorted(kinds.items()))}


def _legal_blocks(blocks: Blocks, deadline: Any = None) -> tuple[list[Block], bool]:
    """Every legal block, for the record and the sampled checks; incomplete (a lower bound) when the decomposition
    timeout or the memory valve stops the enumeration."""
    from sqldecomp.clock import DecompositionTimeout
    found: list[Block] = []
    try:
        for b in blocks.legal_blocks(deadline=deadline):
            found.append(b)
    except DecompositionTimeout:
        return found, False
    return found, True


def _tested_blocks(decompositions: list[tuple[str, Decomposition]], legal: list[Block], options: Options) -> list[tuple[str, Block]]:
    """(origin, block): every distinct block of the tested decompositions, then a seeded sample of the other legal ones."""
    seen: dict[tuple, tuple[str, Block]] = {}
    for label, d in decompositions:
        for b in d.blocks:
            seen.setdefault((b.output, tuple(sorted(b.inputs))), ("decomposition", b))
    rest = [b for b in legal if (b.output, tuple(sorted(b.inputs))) not in seen]
    rng = random.Random(options.seed)
    for b in rng.sample(rest, min(options.block_sample, len(rest))):
        seen[(b.output, tuple(sorted(b.inputs)))] = ("sample", b)
    return list(seen.values())


def _names(tested: list[tuple[str, Block]], blocks: Blocks, printer: Printer, failures: list) -> dict[str, Any]:
    """Printer and legality rules must agree on what a block leaves unbound. The printer writes a block by itself with
    its inputs by name; the columns whose alias no FROM clause of that text defines are its free names. They have to
    be exactly the columns `Blocks.parameters` reports, and none at all for a block that is a statement."""
    out = {"blocks": 0, "agree": 0, "parameterized": 0, "parameters": 0}
    for _, b in tested:
        tag = f"{b.output} <- {sorted(b.inputs)}"
        try:
            text = printer.block_sql(b, inputs="name")
            tree = sqlglot.parse_one(text, dialect=sqltext.DIALECT)
        except Exception as e:  # PrintError, or text that does not parse
            failures.append({"stage": "names", "block": tag, "detail": f"{type(e).__name__}: {e}"[:600]})
            continue
        free = sorted({".".join(sqltext.column_key(c)) for c in sqltext.outer_columns(tree)})
        wanted = sorted({c.lower() for p in blocks.parameters(b) for c in p.columns})
        out["blocks"] += 1
        out["parameterized"] += bool(wanted)
        out["parameters"] += len(wanted)
        if free == wanted:
            out["agree"] += 1
        else:
            failures.append({"stage": "names", "block": tag, "detail": f"the printed block leaves {free} unbound, "
                             f"its parameters are {wanted}", "sql": text[:1500]})
    return out


class NameCapture(Exception):
    """Pasting is not possible: a block's text names a base table, and the block that reads it defines a CTE of that
    name, which would capture the reference (each text is right by itself; the printer's own recomposition keeps the
    CTE order that avoids this)."""


def compose_textually(printer: Printer, d: Decomposition) -> str:
    """The query obtained by PASTING the printed blocks into one another: every reference `b_<id>` (or
    `b_<id>(<parameters>)`) in a block's text is replaced by the text of the block that computes it, as a derived
    table of that name. Nothing of the printer's own recomposition is used. Raises NameCapture where pasting would
    change what a table name refers to."""
    prefix = printer.ctx.boundary_prefix
    by_name = {sqltext.ident_key(boundary_ident(b.output, prefix)): b for b in d.blocks}
    trees: dict[str, exp.Expression] = {}
    free_tables: dict[str, set[str]] = {}

    def tree_of(name: str) -> exp.Expression:
        if name not in trees:
            tree = sqlglot.parse_one(printer.block_sql(by_name[name], inputs="name", within=d), dialect=sqltext.DIALECT)
            own_ctes = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
            free_tables[name] = {t.name.lower() for t in tree.find_all(exp.Table)
                                 if t.name and t.name.lower() not in own_ctes and t.name.lower() not in by_name}
            for table in list(tree.find_all(exp.Table)):
                inner = table.this
                key = (inner.this if isinstance(inner, exp.Anonymous) else table.name or "").lower()
                if key in by_name and key != name:
                    pasted = tree_of(key)
                    if own_ctes & free_tables[key]:
                        raise NameCapture(f"{sorted(own_ctes & free_tables[key])} in block {key}")
                    free_tables[name] |= free_tables[key]
                    # the reference's own alias node, quotes and case kept (a keyword alias such as "or" stays quoted)
                    alias = table.args["alias"].this.copy() if table.args.get("alias") is not None and table.args["alias"].this else sqltext.make_ident(key)
                    pasted_item = exp.Subquery(this=pasted.copy(), alias=exp.TableAlias(this=alias))
                    if table.args.get("joins"):  # the first item of a parenthesized join carries the chain: keep it
                        pasted_item.set("joins", [j for j in table.args["joins"]])
                    table.replace(pasted_item)
            trees[name] = tree
        return trees[name]

    root = next(b for b in d.blocks if b.output == printer.memo.root)
    return tree_of(sqltext.ident_key(boundary_ident(root.output, prefix))).sql(dialect=sqltext.DIALECT, pretty=True)


def _textual(decompositions: list[tuple[str, Decomposition]], blocks: Blocks, printer: Printer, failures: list) -> list[tuple[str, str]]:
    out = []
    for label, d in decompositions:
        if len(d.blocks) < 2:
            continue
        try:
            out.append((label, compose_textually(printer, d)))
        except NameCapture:
            continue  # pasting does not apply here; the blocks themselves and the printer's recomposition are checked
        except Exception as e:
            failures.append({"stage": "textual", "label": label, "detail": f"{type(e).__name__}: {e}"[:600]})
    return out


def _controls(memo: Memo, result: ImportResult, printer: Printer, blocks: Blocks, schema: Schema,
              group_sql: dict[GroupId, str], failures: list) -> dict[str, Any]:
    """Negative controls. C1 for the root and C2 for the coarsest decomposition are given the re-import of ANOTHER
    group's print (the largest cut point, else any other group): both must say no."""
    others = sorted((g for g in memo.groups if g != memo.root), key=lambda g: (-len(blocks.occurrences_under(g)), g))
    cut = [g for g in others if blocks.can_cut(g)]
    out: dict[str, Any] = {"c1_run": 0, "c1_rejected": 0, "c2_run": 0, "c2_rejected": 0}
    if not others:
        return out
    wrong = printer.group_sql((cut or others)[0], bind_outer=True)
    substitute = lambda sql, schema_: import_sql(wrong, schema_)  # noqa: E731
    for what, check in (("c1", lambda: check_round_trip(result.plan, result, memo, schema, substitute)),
                        ("c2", lambda: check_recomposition(blocks.coarsest(), result, memo, schema, substitute))):
        verdict = check()
        out[f"{what}_run"] += 1
        out[f"{what}_rejected"] += not verdict.ok
        if verdict.ok:
            failures.append({"stage": "controls", "detail": f"{what.upper()} accepted the print of group "
                             f"{(cut or others)[0]} as the whole query"})
    return out


def _c1(memo: Memo, result: ImportResult, printer: Printer, schema: Schema, failures: list) -> dict[str, Any]:
    out = {"groups": 0, "passed": 0, "ids": 0, "scopes": 0, "subquery_bodies": 0, "modulo_projection": 0,
           "not_first_candidate": 0, "mapping_covers_every_id": 0}
    for g in sorted(memo.groups):
        check = check_round_trip(printer.group_plan(g), result, memo, schema)
        out["groups"] += 1
        if not check.ok:
            failures.append({"stage": "c1", "group": g, "detail": check.detail[:600]})
            continue
        out["passed"] += 1
        out["modulo_projection"] += "projection of plain columns" in check.detail
        out["not_first_candidate"] += check.stats.get("candidates", 1) > 1
        out["mapping_covers_every_id"] += check.stats.get("ids", 0) > 0  # `_verify` refuses a mapping with orphans
        for k in ("ids", "scopes", "subquery_bodies"):
            out[k] += check.stats.get(k, 0)
    return out


def _c2(decompositions: list[tuple[str, Decomposition]], blocks: Blocks, memo: Memo, result: ImportResult,
        printer: Printer, schema: Schema, failures: list, options: Options) -> dict[str, Any]:
    out: dict[str, Any] = {"decompositions": 0, "passed": 0, "with_two_or_more_blocks": 0, "blocks": 0, "ids": 0,
                           "scopes": 0, "subquery_bodies": 0, "by_label": {}, "by_kind": {}, "list": []}
    for label, d in decompositions:
        shape = [[b.output, sorted(b.inputs)] for b in d.blocks]
        illegal = blocks.problems(d)
        check = CheckResult(False, "not a legal decomposition: " + "; ".join(illegal)) if illegal else \
            check_recomposition(d, result, memo, schema)
        family = label.split(":")[0]
        out["decompositions"] += 1
        out["with_two_or_more_blocks"] += len(d.blocks) > 1
        out["blocks"] += len(d.blocks)
        tally = out["by_label"].setdefault(family, {"checked": 0, "passed": 0})
        tally["checked"] += 1
        tally["passed"] += check.ok
        for kind in sorted({k for b in d.blocks for k in blocks.describe(b).kinds}):
            per_kind = out["by_kind"].setdefault(kind, {"checked": 0, "passed": 0})
            per_kind["checked"] += 1
            per_kind["passed"] += check.ok
        entry: dict[str, Any] = {"label": label, "blocks": shape, "ok": check.ok}
        if check.ok:
            out["passed"] += 1
            for k in ("ids", "scopes", "subquery_bodies"):
                out[k] += check.stats.get(k, 0)
        else:
            failures.append({"stage": "c2", "decomposition": shape, "label": label, "detail": check.detail[:600]})
        if options.keep_sql:
            try:
                entry["sql"] = printer.decomposition_sql(d)
            except PrintError as e:
                entry["sql"] = f"-- {type(e).__name__}: {e}"
        out["list"].append(entry)
    return out


# --------------------------------------------------------------------------------------------------------------
# C3
# --------------------------------------------------------------------------------------------------------------

_TYPE_MISMATCH = re.compile(r"^(operator does not exist: .*|function [^ ]+\(.*\) does not exist)$")


def explain_reason(detail: str) -> str:
    """The one-line reason of a failed EXPLAIN (`operator does not exist: text = integer`)."""
    m = re.search(r"ERROR:\s*(.*)", detail)
    return (m.group(1) if m else (detail.strip().splitlines() or ["?"])[0]).strip()


def reason_class(reason: str) -> str:
    """Two prints of one invalid expression can fail with differently spelled messages (the printer writes VARIANCE
    as VAR_SAMP; another clause is analysed first). What must agree is the KIND of failure: an operator or function
    that does not exist for the column types the database has."""
    return "no such operator or function for these types" if _TYPE_MISMATCH.match(reason) else reason


def _broken(sql: str) -> str | None:
    """The same statement with its first column reference renamed to a column that does not exist."""
    try:
        tree = sqlglot.parse_one(sql, dialect=sqltext.DIALECT)
    except Exception:
        return None
    column = next((c for c in tree.find_all(exp.Column) if c.table), None)
    if column is None:
        return None
    column.set("this", exp.to_identifier("no_such_column_sqldecomp"))
    return tree.sql(dialect=sqltext.DIALECT)


def _c3(sql: str, whole: str, group_sql: dict[GroupId, str], decompositions: list[tuple[str, Decomposition]],
        tested: list[tuple[str, Block]], textual: list[tuple[str, str]], blocks: Blocks, printer: Printer,
        options: Options, controls: dict[str, Any], failures: list) -> dict[str, Any]:
    jobs: list[tuple[str, str, str]] = [("original", "", sql), ("whole", "", whole)]
    jobs += [("group", g, s) for g, s in sorted(group_sql.items())]
    block_kinds: dict[str, list[str]] = {}
    for label, d in decompositions:
        try:
            jobs.append(("recomposed", label + " " + json.dumps([[b.output, sorted(b.inputs)] for b in d.blocks]),
                         printer.decomposition_sql(d)))
        except PrintError as e:
            failures.append({"stage": "c3", "what": "recomposed", "label": label, "detail": f"{type(e).__name__}: {e}"})
    for origin, b in tested:
        tag = f"{b.output} <- {sorted(b.inputs)}"
        block_kinds[tag] = blocks.describe(b).kinds
        try:
            jobs.append(("block" if origin == "decomposition" else "sampled_block", tag,
                         printer.block_sql(b, inputs="inline", bind_outer=True)))
        except PrintError as e:
            failures.append({"stage": "c3", "what": "block", "block": tag, "detail": f"{type(e).__name__}: {e}"})
    jobs += [("textual", label, text) for label, text in textual]
    broken = _broken(whole)
    if broken:
        jobs.append(("control", "", broken))
    with ThreadPoolExecutor(max(1, options.workers)) as pool:
        verdicts = list(pool.map(lambda job: check_valid(job[2], dbname=options.db), jobs))

    original = verdicts[0]
    o_reason = "" if original.ok else explain_reason(original.detail)
    out: dict[str, Any] = {"database": options.db, "original_ok": original.ok, "original_reason": o_reason,
                           "statements": len(jobs), "whole_reason": "", "print_reasons": {}, "textual_untyped_null": 0}
    kinds_of_statement = ("whole", "group", "recomposed", "block", "sampled_block", "textual")
    for what in kinds_of_statement:
        out[what] = {"checked": 0, "ok": 0, "failed_same_reason": 0, "failed_same_class": 0, "failed_other": 0}
    out["block_by_kind"] = {}
    out["recomposed_by_label"] = {}
    for (what, tag, text), verdict in zip(jobs[1:], verdicts[1:]):
        reason = "" if verdict.ok else explain_reason(verdict.detail)
        if what == "control":  # the print with a column that does not exist: EXPLAIN has to say no, and say why
            if original.ok and verdicts[1].ok:  # only a VALID print can show that the broken column is what is refused
                controls["c3_run"] = 1
                controls["c3_rejected"] = int(not verdict.ok and "no_such_column_sqldecomp" in verdict.detail)
                if not controls["c3_rejected"]:
                    failures.append({"stage": "controls", "detail": "C3 accepted a query with a column that does not exist"})
            continue
        tally = out[what]
        tally["checked"] += 1
        if reason:
            out["print_reasons"][reason] = out["print_reasons"].get(reason, 0) + 1
        if what == "whole":
            out["whole_reason"] = reason
        if verdict.ok:
            tally["ok"] += 1
            outcome = "ok"
        elif not original.ok and reason == o_reason:
            tally["failed_same_reason"] += 1
            outcome = "same"
        elif not original.ok and reason_class(reason) == reason_class(o_reason):
            tally["failed_same_class"] += 1
            outcome = "class"
        else:
            tally["failed_other"] += 1
            outcome = "other"
        # A print of the WHOLE query has to be judged exactly like the original. A part of a valid query has to be
        # valid. A part of an invalid query may be valid (it may not contain the offending expression) or fail for the
        # kind of reason the original fails for; anything else is the printer's fault.
        wrong = (what in ("whole", "recomposed") and verdict.ok != original.ok) or outcome == "other"
        if what == "textual" and outcome == "other" and reason.startswith("UNION types"):
            # KNOWN LIMIT of SQL, not of the print: a set-operation branch with an untyped `NULL AS x` that is cut out
            # as a block of its own is of type text once it stands alone, so the union of the block outputs does not
            # type-check although the original (where the NULL takes its type from the sibling branch) does.
            out["textual_untyped_null"] += 1
            wrong = False
        if wrong:
            failures.append({"stage": "c3", "what": what, "which": tag, "original_ok": original.ok,
                             "original_reason": o_reason, "reason": reason, "sql": text[:2000]})
        if what in ("block", "sampled_block"):
            for kind in block_kinds.get(tag, ["plain"]):
                per_kind = out["block_by_kind"].setdefault(kind, {"checked": 0, "ok": 0, "failed_like_original": 0, "wrong": 0})
                per_kind["checked"] += 1
                per_kind["ok"] += verdict.ok
                per_kind["failed_like_original"] += outcome in ("same", "class")
                per_kind["wrong"] += wrong
        if what == "recomposed":
            family = tag.split(" ")[0].split(":")[0]
            per_label = out["recomposed_by_label"].setdefault(family, {"checked": 0, "ok": 0, "failed_like_original": 0, "wrong": 0})
            per_label["checked"] += 1
            per_label["ok"] += verdict.ok
            per_label["failed_like_original"] += outcome in ("same", "class")
            per_label["wrong"] += wrong
    out["block_by_kind"] = dict(sorted(out["block_by_kind"].items()))
    out["print_reasons"] = dict(sorted(out["print_reasons"].items()))
    return out


# --------------------------------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------------------------------


def _corpus_worker(args: tuple[str, str, str, dict]) -> tuple[dict, dict]:
    name, sql, schema_path, opts = args
    done = run_query(name, sql, load_any_schema(schema_path), Options(**opts))
    record = done.record if isinstance(done, Run) else done
    record = dict(record)
    timings = record.pop("timings_ms", {})
    return record, {"query": name, **timings}


def run_corpus(queries: list[tuple[str, str]], schema_path: str, out_dir: str | Path, options: Options | None = None,
               *, jobs: int = 1, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Runs every query, writes results.jsonl / summary.json (deterministic) and timings.jsonl /
    timings_summary.json into `out_dir`, which must not exist yet (previous results are never overwritten)."""
    options = options or Options()
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; a corpus run never writes into an existing directory")
    opts = {k: getattr(options, k) for k in ("seed", "samples", "db", "block_sample", "workers", "optimize", "timeout_s", "scoring")}
    work = [(name, sql, str(schema_path), opts) for name, sql in queries]
    done: list[tuple[dict, dict]] = []
    if jobs > 1:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, item in enumerate(pool.map(_corpus_worker, work, chunksize=4)):
                done.append(item)
                if progress:
                    progress(i + 1, len(work))
    else:
        for i, item in enumerate(map(_corpus_worker, work)):
            done.append(item)
            if progress:
                progress(i + 1, len(work))
    done.sort(key=lambda item: item[0]["query"])
    records, timings = [r for r, _ in done], [t for _, t in done]
    texts = {name: hashlib.sha1(sql.encode()).hexdigest() for name, sql in queries}
    summary = summarize(records, texts, options)
    out.mkdir(parents=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (out / "timings.jsonl").write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in timings))
    (out / "timings_summary.json").write_text(json.dumps(summarize_timings(timings), indent=2, sort_keys=True) + "\n")
    return summary


def summarize(records: list[dict], texts: dict[str, str], options: Options) -> dict[str, Any]:
    """Counts over all files and over distinct SQL texts, the non-vacuity counters, and every failure by stage."""
    def total(path: str) -> int:
        n = 0
        for r in records:
            value: Any = r
            for key in path.split("."):
                value = value.get(key, 0) if isinstance(value, dict) else 0
            n += int(value or 0)
        return n

    passed = [r for r in records if r.get("verdict") == "pass"]
    distinct = {texts[r["query"]] for r in records}
    distinct_passed = {texts[r["query"]] for r in passed}
    distinct_failed = {texts[r["query"]] for r in records if r.get("verdict") != "pass"}
    with_cut = [r for r in records if r.get("blocks", {}).get("cut_points", 0) > 0]
    failures: dict[str, list[dict]] = {}
    for r in records:
        for f in r.get("failures", []):
            failures.setdefault(f.get("stage", "?"), []).append({"query": r["query"], **f})
    summary: dict[str, Any] = {
        "options": {"seed": options.seed, "samples": options.samples, "timeout_s": options.timeout_s, "database": options.db},
        "files": len(records), "distinct_sql_texts": len(distinct),
        "passed_files": len(passed), "passed_distinct_sql_texts": len(distinct_passed - distinct_failed),
        "pass_rate_files": round(len(passed) / len(records), 4) if records else 0.0,
        "unsupported": sum(r.get("verdict") == "unsupported" for r in records),
        "import_valid": sum(bool(r.get("import", {}).get("ok")) for r in records),
        "memo_complete": sum(bool(r.get("memo", {}).get("complete")) for r in records),
        "groups": total("memo.groups"), "expressions": total("memo.expressions"),
        "c1": {k: total(f"c1.{k}") for k in ("groups", "passed", "ids", "scopes", "subquery_bodies", "modulo_projection",
                                             "not_first_candidate", "mapping_covers_every_id")},
        "c2": {k: total(f"c2.{k}") for k in ("decompositions", "passed", "with_two_or_more_blocks", "blocks", "ids", "scopes",
                                             "subquery_bodies")},
        "non_vacuity": {
            "queries_with_zero_groups_checked": sum(r.get("c1", {}).get("groups", 0) == 0 for r in records),
            "queries_with_a_legal_cut": len(with_cut),
            "queries_with_a_legal_cut_but_no_multi_block_decomposition": sorted(
                r["query"] for r in with_cut if r.get("c2", {}).get("with_two_or_more_blocks", 0) == 0),
            "c1_passes_whose_mapping_covers_every_id": total("c1.mapping_covers_every_id"),
            "subquery_bodies_compared": total("c1.subquery_bodies") + total("c2.subquery_bodies"),
        },
        "blocks": {"cut_points": total("blocks.cut_points"), "pipeline_groups": total("blocks.pipeline_groups"),
                   "scan_and_ref_groups": total("blocks.scan_and_ref_groups"), "legal_blocks": total("blocks.legal_blocks"),
                   "queries_whose_block_count_is_a_lower_bound": sum(
                       not r.get("blocks", {}).get("legal_blocks_complete", True) for r in records)},
        "print": {"groups": total("print.groups"), "distinct_group_sql": total("print.distinct_group_sql")},
        "names": {k: total(f"names.{k}") for k in ("blocks", "agree", "parameterized", "parameters")},
        "controls": {k: total(f"controls.{k}") for k in ("c1_run", "c1_rejected", "c2_run", "c2_rejected", "c3_run", "c3_rejected")},
        "constraints": {"groups_with_key": total("constraints.groups_with_key"), "groups_with_fk": total("constraints.groups_with_fk"),
                        "cut_points_with_key": total("constraints.cut_points_with_key"),
                        "queries_whose_root_has_a_key": sum(bool(r.get("constraints", {}).get("root_has_key")) for r in records)},
        "sharing": {"queries_with_a_shared_block_class": sum(r.get("signatures", {}).get("shared_block_classes", 0) > 0 for r in records),
                    "shared_block_classes": total("signatures.shared_block_classes"),
                    "groups_made_unique": total("signatures.groups_made_unique")},
        "subqueries": {"queries_with_a_subquery": sum(r.get("subqueries", {}).get("bodies", 0) > 0 for r in records),
                       "queries_with_a_correlated_subquery": sum(r.get("subqueries", {}).get("correlated_bodies", 0) > 0 for r in records),
                       "cut_points_inside_bodies": total("subqueries.cut_points_inside_bodies")},
        "failures_by_stage": {stage: len(items) for stage, items in sorted(failures.items())},
        "failures": {stage: items[:50] for stage, items in sorted(failures.items())},
    }
    if any("optimizer" in r for r in records):
        opt = [r["optimizer"] for r in records if "optimizer" in r]
        solved = [o for o in opt if "finding" not in o]
        summary["optimizer"] = {
            "queries": len(opt), "solved": len(solved),
            "findings": [{"query": r["query"], "finding": r["optimizer"]["finding"]} for r in records
                         if "optimizer" in r and "finding" in r["optimizer"]],
            "scoring": sorted({o["scoring"] for o in opt}),
            "optimum_differs_from_coarsest": sum(o["differs_from_coarsest"] for o in solved),
            "optimum_differs_from_finest": sum(o["differs_from_finest"] for o in solved),
            "optimum_differs_from_both": sum(o["differs_from_coarsest"] and o["differs_from_finest"] for o in solved),
            "optimum_beats_every_other_tested_decomposition": sum(o["beats_every_other_tested_decomposition"] for o in solved),
            "optimum_strictly_cheaper_than_every_other_tested_decomposition": sum(
                o["strictly_cheaper_than_every_other_tested_decomposition"] for o in solved),
            "queries_without_another_tested_decomposition": sum(o["other_tested_decompositions"] == 0 for o in solved),
            "tested_decompositions": sum(o["tested_decompositions"] for o in solved),
            "optima_with_problems": [r["query"] for r in records if "optimizer" in r and r["optimizer"].get("problems")],
            "timeout_s": options.timeout_s,
            "queries_with_a_shared_class_in_the_optimum": sum(bool(o["classes_shared"]) for o in solved),
            "methods": {m: sum(o["stats"]["method"] == m for o in solved) for m in sorted({o["stats"]["method"] for o in solved})},
            "f_evaluations": sum(o["stats"]["f_evaluations"] for o in solved),
            "blocks_evaluated": sum(o["stats"]["blocks_evaluated"] for o in solved),
            "states_expanded": sum(o["stats"]["states_expanded"] for o in solved),
            "union_notes": sum(len(o["union_notes"]) for o in solved),
        }
        if any("features" in o for o in solved):  # the human scoring: what the tau sweep reads
            human = [o for o in solved if "features" in o]
            summary["optimizer"]["human"] = {
                "one_block_optima": sum(o["blocks"] == 1 for o in solved),
                "queries_with_a_block_over_threshold": sum(o["blocks_over_threshold"] > 0 for o in human),
                "blocks_over_threshold": sum(o["blocks_over_threshold"] for o in human),
                "blocks_in_optima": sum(o["blocks"] for o in solved),
                "max_excess_max": max((o["max_excess"] for o in human), default=0.0),
            }
    for table, path in (("c2_by_label", "c2.by_label"), ("c2_by_kind", "c2.by_kind"), ("c3_block_by_kind", "c3.block_by_kind"),
                        ("c3_recomposed_by_label", "c3.recomposed_by_label")):
        merged: dict[str, dict[str, int]] = {}
        for r in records:
            value: Any = r
            for key in path.split("."):
                value = value.get(key, {}) if isinstance(value, dict) else {}
            for kind, counts in value.items():
                for k, v in counts.items():
                    merged.setdefault(kind, {}).setdefault(k, 0)
                    merged[kind][k] += v
        summary[table] = merged
    if options.db:
        c3 = [r["c3"] for r in records if "c3" in r]
        invalid = [r for r in records if "c3" in r and not r["c3"]["original_ok"]]
        reasons: dict[str, int] = {}
        for r in invalid:
            reasons[r["c3"]["original_reason"]] = reasons.get(r["c3"]["original_reason"], 0) + 1
        summary["c3"] = {
            "queries_checked": len(c3), "statements": sum(x["statements"] for x in c3),
            "originals_valid": len(c3) - len(invalid), "originals_invalid": len(invalid),
            "originals_invalid_by_reason": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
            **{what: {k: sum(x[what][k] for x in c3) for k in ("checked", "ok", "failed_same_reason", "failed_same_class", "failed_other")}
               for what in ("whole", "group", "recomposed", "block", "sampled_block", "textual")},
            "valid_originals": {what: {k: sum(r["c3"][what][k] for r in records if "c3" in r and r["c3"]["original_ok"])
                                       for k in ("checked", "ok")}
                                for what in ("whole", "group", "recomposed", "block", "sampled_block", "textual")},
            "textual_compositions_that_fail_only_for_an_untyped_null_in_a_union_branch": sum(x["textual_untyped_null"] for x in c3),
            "whole_prints_whose_message_differs_from_the_originals": sorted(
                [r["query"], r["c3"]["original_reason"], r["c3"]["whole_reason"]] for r in records
                if "c3" in r and not r["c3"]["original_ok"] and r["c3"]["whole_reason"] != r["c3"]["original_reason"]),
            "distinct_blocks_explained": sum(x["block"]["checked"] + x["sampled_block"]["checked"] for x in c3),
        }
    return summary


def summarize_timings(timings: list[dict]) -> dict[str, Any]:
    def dist(values: list[float]) -> dict[str, float]:
        values = sorted(values)
        if not values:
            return {}
        return {"median_ms": round(statistics.median(values), 2), "p90_ms": round(values[int(0.9 * (len(values) - 1))], 2),
                "max_ms": round(values[-1], 2), "sum_s": round(sum(values) / 1000, 2)}

    totals = [(sum(v for k, v in t.items() if k != "query"), t["query"]) for t in timings]
    no_c3 = [(sum(v for k, v in t.items() if k not in ("query", "c3")), t["query"]) for t in timings]
    slowest = sorted(no_c3, reverse=True)[:5]
    by_query = {t["query"]: t for t in timings}
    return {"per_stage": {s: dist([t[s] for t in timings if s in t]) for s in STAGES},
            "per_query_total": dist([x for x, _ in totals]), "per_query_without_c3": dist([x for x, _ in no_c3]),
            "slowest_without_c3": [{"query": q, "total_ms": round(ms, 1),
                                    "stages_ms": {k: v for k, v in by_query[q].items() if k != "query"}} for ms, q in slowest]}


# --------------------------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------------------------


def load_any_schema(path: str | Path) -> Schema:
    """The contract's JSON format (an object with `tables`) or the BEAVER format (a list of tables)."""
    data = json.loads(Path(path).read_text())
    return load_beaver_schema(path) if isinstance(data, list) else load_schema(path)


def report(run: Run, options: Options) -> str:
    """A readable account of one query: what the `--sql` mode prints."""
    from sqldecomp.importer import render
    r, memo, printer, blocks = run.result, run.memo, run.printer, run.blocks
    rec = run.record
    lines = ["# import", render(r).rstrip(), "", f"# memo: {len(memo.groups)} groups, root {memo.root}, complete={memo.complete}"]
    classes: dict[str, list[str]] = {}
    for g, s in run.signatures.items():
        classes.setdefault(s, []).append(g)
    for g in sorted(memo.groups):
        flags = [blocks.kind(g)]
        flags.append("cut point" if blocks.can_cut(g) else f"no boundary ({blocks.why_no_cut(g)})" if g != memo.root else "root")
        if blocks.body_of(g):
            flags.append(f"inside subquery scope {blocks.body_of(g)}")
        same = [x for x in classes[run.signatures[g]] if x != g]
        if same:
            flags.append("same task as " + ", ".join(same))
        c = run.constraints.get(g)
        keys = [sorted(f"{k.origin}.{k.column}" for k in key.columns) for key in (c.keys if c else ())]
        fks = [f"({', '.join(f'{x.origin}.{x.column}' for x in f.columns)}) -> {f.ref_table}({', '.join(f.ref_columns)})"
               for f in (c.fks if c else ())]
        lines += ["", f"## group {g}: " + "; ".join(flags),
                  "expressions: " + " | ".join(f"{e.kind}({', '.join(e.children)}{'; ' + (e.arg or ','.join(e.conjuncts)) if (e.arg or e.conjuncts) else ''})"
                                               for e in memo.groups[g]),
                  f"keys: {sorted(keys)}   fks: {sorted(fks)}", rec["print"]["group_sql"][g] if "group_sql" in rec["print"] else ""]
    lines += ["", "# whole query as printed", rec["print"].get("whole", ""), "", "# C1", json.dumps(rec["c1"]),
              "", f"# names: {json.dumps(rec['names'])}", f"# negative controls: {json.dumps(rec['controls'])}",
              "", f"# blocks: {json.dumps(rec['blocks'])}", f"# subqueries: {json.dumps(rec['subqueries'])}", "", "# decompositions and C2"]
    for (label, d), entry in zip(run.decompositions, rec["c2"]["list"]):
        lines += ["", f"## {label}: C2 {'ok' if entry['ok'] else 'FAILED'}"]
        for b in d.blocks:
            info: BlockInfo = blocks.describe(b)
            by_column: dict[tuple[str, str], list[str]] = {}  # one entry per PARAMETER, however many conjuncts read it
            for q in info.parameters:
                for column in q.columns or (q.alias,):
                    by_column.setdefault((column, q.alias), []).append(q.entry + ("" if q.own else " inside an input block"))
            params = "; ".join(f"{column} (of FROM item {alias!r}, read by {', '.join(readers)})"
                               for (column, alias), readers in sorted(by_column.items(), key=lambda kv: (kv[0][1].lower(), kv[0][0].lower())))
            try:
                text = printer.block_sql(b, inputs="name", within=d)
                # a parameterized block is a function: its header lists the outer columns AS ITS TEXT NAMES THEM here
                # (through a block boundary when another block of this decomposition hides their alias), in the order
                # in which a reader passes them
                free = printer.block_parameters(b, within=d)
            except Exception as e:  # PrintError, or text that does not parse
                text, free = f"-- {type(e).__name__}: {e}", []
            lines += [f"### block {b.output}{'(' + ', '.join(free) + ')' if free else ''} <- {sorted(b.inputs)}  "
                      f"[{', '.join(info.kinds)}]" + (f"  PARAMETERS: {params}" if params else ""), text]
        lines += ["### recomposed", entry.get("sql", "")]
    if "optimizer" in rec:
        lines += ["", "# optimizer", json.dumps({k: v for k, v in rec["optimizer"].items() if k != "cost_by_class"}, indent=1)]
    if "c3" in rec:
        lines += ["", "# C3", json.dumps(rec["c3"], indent=1)]
        if not rec["c3"]["original_ok"]:
            lines += [f"NOTE: the ORIGINAL query fails EXPLAIN on {rec['c3']['database']}: {rec['c3']['original_reason']}"]
    if "execution" in rec:
        lines += ["", "# execution", json.dumps(rec["execution"], indent=1)]
    lines += ["", f"# verdict: {rec['verdict']}"] + [json.dumps(f) for f in rec["failures"]]
    return "\n".join(lines)


def execute_and_compare(run: Run, options: Options) -> dict[str, Any]:
    """Rows of the original against the printed query and every recomposed query, as multisets, on a SCRATCH database.
    A query with LIMIT / DISTINCT ON over ties may legitimately differ; that is reported, not judged."""
    if not (options.db or "").startswith("sqldecomp_test_"):
        raise SystemExit("--execute only runs on a scratch database named sqldecomp_test_*; beaver_dw is never executed on")
    original = _rows(run.result.sql, options.db)
    out: dict[str, Any] = {"original_rows": None if isinstance(original, str) else len(original)}
    if isinstance(original, str):
        out["original_error"] = original
        return out
    texts = [("printed", run.printer.plan_sql(run.result.plan))]
    for label, d in run.decompositions:
        try:
            texts.append((f"recomposed {label}", run.printer.decomposition_sql(d)))
        except PrintError as e:
            out[f"recomposed {label}"] = f"{type(e).__name__}: {e}"
    for what, text in texts:
        got = _rows(text, options.db)
        out[what] = "same rows" if got == original else ("ERROR " + got if isinstance(got, str) else f"DIFFERENT ROWS: {len(got)} rows")
    return out


def _rows(sql: str, dbname: str) -> list[str] | str:
    psql = os.environ.get("SQLDECOMP_PSQL") or shutil.which("psql") or "/Library/PostgreSQL/16/bin/psql"  # as check C3 finds it
    script = f"BEGIN TRANSACTION READ ONLY;\n{sql.strip().rstrip(';')};\nROLLBACK;\n"
    command = [psql, "-X", "-q", "-A", "-t", "-F", "|", "-v", "ON_ERROR_STOP=1", "-h", os.environ.get("PGHOST", "localhost"),
               "-p", os.environ.get("PGPORT", "5432"), "-d", dbname, "-f", "-"]
    done = subprocess.run(command, input=script, capture_output=True, text=True, timeout=120)
    return sorted(done.stdout.splitlines()) if done.returncode == 0 else explain_reason(done.stderr or done.stdout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sqldecomp.pipeline", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sql", help="one query (text, or @file)")
    ap.add_argument("--corpus", help="directory of *.json files with a `sql` field (and, for --verify, `question`, optional `evidence`)")
    ap.add_argument("--queries", help="--verify: a JSONL file of queries (`key`, `sql`, `question`, optional `evidence`)")
    ap.add_argument("--manifest", help="--verify / --direct: an inputs manifest (experiments.data_prep.experiment inputs: every file with its "
                                       "schema); one pool of --jobs over the queries of every file, each against its own schema (no --schema)")
    ap.add_argument("--verify", action="store_true", help="the verification run: tasks for the optimum's blocks, the verifier's verdicts, "
                                                          "their conjunction (or --aggregation), one result record per query")
    ap.add_argument("--direct", action="store_true", help="the direct arm: the whole query judged against the question (with or without --verify)")
    ap.add_argument("--schema", help="schema file: contract JSON or the BEAVER format (required but with --manifest)")
    ap.add_argument("--out", help="corpus mode: NEW directory for results.jsonl, summary.json and the timings")
    ap.add_argument("--resume", action="store_true", help="--verify / --direct with --out: continue a run that stopped; the queries whose "
                                                          "records --out/results.partial.jsonl holds are not run again")
    ap.add_argument("--db", help="database for C3 (EXPLAIN only, read-only), e.g. beaver_dw")
    ap.add_argument("--execute", action="store_true", help="--sql only: also run the SQL and compare rows (scratch databases only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=None, help="the acceptance harness: random decompositions per query (default 4)")
    ap.add_argument("--block-sample", type=int, default=None,
                    help="the acceptance harness: further legal blocks per query to print, name-check and EXPLAIN (default 0)")
    ap.add_argument("--jobs", type=int, default=1, help="corpus mode: worker processes")
    ap.add_argument("--every", type=int, default=1, help="corpus mode: every k-th file")
    ap.add_argument("--json", action="store_true", help="--sql only: print the record as JSON instead of the report")
    ap.add_argument("--optimize", action="store_true", help="run the optimizer; its optimum goes through C2 and C3")
    from sqldecomp.scoring_args import add_scoring_arguments, is_judge, scoring_spec_from_args
    add_scoring_arguments(ap)
    ap.add_argument("--timeout", type=timeout_seconds, default=None,
                    help="the decomposition timeout per query in seconds (import to the optimum and its tasks; the checks and the "
                         "verification are not on it; default 3,600 when tasks are written, --verify or --tasks, else 300); a query "
                         "that hits it is a timeout finding with the stage reached")
    from sqldecomp.tasks import METHODS
    ap.add_argument("--tasks", default=None, choices=METHODS,
                    help="component 9's method: the tasks of the optimum's blocks (--optimize --sql), or the verification run's (default td3-desc)")
    ap.add_argument("--tasks-cache", default=None, help="--tasks: the persistent cache of assignments; the model calls are cached under <dir>/llm")
    ap.add_argument("--question", default="", help="--tasks: the question the query is supposed to answer")
    ap.add_argument("--evidence", default="", help="--tasks: the evidence given with the question")
    from sqldecomp import llm as client
    from sqldecomp.config import add_experiment_arguments
    client.add_model_arguments(ap, calls="the verifier's calls (and the base of every call: the sandbox, the call timeout; the task writer "
                                         "has --writer-model / --writer-effort, the LLM judge --semantic-model / --semantic-effort)")
    add_experiment_arguments(ap)
    args = ap.parse_args(argv)
    if args.manifest and not (args.verify or args.direct):
        ap.error("--manifest belongs to the verification run (--verify / --direct)")
    if not args.schema and not args.manifest:
        ap.error("the following arguments are required: --schema")
    if args.verify or args.direct:
        return verification_main(ap, args)
    if args.tasks and not (args.optimize and args.sql and args.question and args.tasks_cache):
        ap.error("--tasks needs --optimize, --sql, --question and --tasks-cache")
    from sqldecomp.clock import DECOMPOSITION_TIMEOUT_S, TASKS_TIMEOUT_S
    from sqldecomp.config import WRITER_EFFORT, WRITER_MODEL
    if args.replay:
        ap.error("--replay belongs to the verification run (--verify)")
    model = client.model_config_from_args(args)
    model = model.with_(model=args.writer_model or WRITER_MODEL, effort=args.writer_effort or WRITER_EFFORT)  # --tasks: the writer's calls
    if args.tasks:
        try:
            client.require_sandbox(model)  # before any work: a run never stops halfway for want of bwrap
        except client.SandboxUnavailable as e:
            ap.error(f"refused: {e}")
    spec = scoring_spec_from_args(args)
    if is_judge(spec):
        ap.error("--semantic judge rates the cut points with the question and two model calls: it runs in the verification run (--verify)")
    options = Options(seed=args.seed, samples=4 if args.samples is None else args.samples, db=args.db,
                      block_sample=0 if args.block_sample is None else args.block_sample,
                      optimize=args.optimize, scoring=spec, tasks=args.tasks, tasks_cache=args.tasks_cache,
                      timeout_s=args.timeout if args.timeout is not None else (TASKS_TIMEOUT_S if args.tasks else DECOMPOSITION_TIMEOUT_S),
                      question=args.question, evidence=args.evidence, model=model,
                      schema_json=schema_json_of(args.schema) if args.tasks else None)
    if bool(args.sql) == bool(args.corpus):
        ap.error("give exactly one of --sql and --corpus")
    if args.execute and not (args.db or "").startswith("sqldecomp_test_"):
        ap.error("--execute only runs on a scratch database named sqldecomp_test_*; beaver_dw is never executed on")
    if args.sql:
        sql = Path(args.sql[1:]).read_text() if args.sql.startswith("@") else args.sql
        options.keep_sql = True
        done = run_query("query", sql, load_any_schema(args.schema), options)
        if not isinstance(done, Run):
            print(json.dumps(done, indent=1))
            return 1
        if args.execute:  # rows decide too: an importer that misreads a query is invisible to C1 / C2 (the print and
            # its re-import go through the same misreading), and only a comparison of results shows it
            done.record["execution"] = execute_and_compare(done, options)
            for what, outcome in done.record["execution"].items():
                if what != "original_rows" and outcome != "same rows":
                    done.record["failures"].append({"stage": "execute", "what": what, "detail": str(outcome)})
            done.record["verdict"] = "pass" if not done.record["failures"] else "fail"
        print(json.dumps(done.record, indent=1) if args.json else report(done, options))
        return 0 if done.record["verdict"] == "pass" else 1
    if not args.out:
        ap.error("--corpus needs --out")
    if Path(args.out).exists():
        ap.error(f"{args.out} exists; a corpus run never writes into an existing directory")
    from sqldecomp.importer.corpus import load_queries
    queries = load_queries(args.corpus)[::max(1, args.every)]
    summary = run_corpus(queries, args.schema, args.out, options, jobs=args.jobs,
                         progress=lambda i, n: print(f"{i}/{n}", file=sys.stderr) if i % 50 == 0 or i == n else None)
    print(json.dumps({k: summary[k] for k in ("files", "distinct_sql_texts", "passed_files", "passed_distinct_sql_texts",
                                              "pass_rate_files", "failures_by_stage", "non_vacuity")}, indent=1))
    return 0 if summary["passed_files"] == summary["files"] else 1


def verification_main(ap: argparse.ArgumentParser, args) -> int:
    """`--verify` / `--direct`: the verification run of `sqldecomp.runner`."""
    from sqldecomp import llm as client
    from sqldecomp.config import config_from_args
    from sqldecomp.runner import load_run_queries, run_many, run_one
    from sqldecomp.scoring_args import scoring_spec_from_args
    if sum(bool(x) for x in (args.sql, args.corpus, args.queries, args.manifest)) != 1:
        ap.error("give exactly one of --sql, --corpus, --queries and --manifest")
    if args.manifest and args.schema:
        ap.error("--manifest names every file's schema: no --schema")
    if args.execute or args.samples is not None or args.block_sample is not None:
        ap.error("--execute, --samples and --block-sample belong to the acceptance harness, not to the verification run")
    if args.sql and not args.question:
        ap.error("--verify / --direct need --question with --sql")
    model = client.model_config_from_args(args)
    config = config_from_args(args, scoring_spec_from_args(args), model)
    if config.decomposed_arm and not config.task_cache:
        ap.error("--verify needs --cache (or --tasks-cache): the assignments and the writer's calls are cached there")
    if not config.verdict_cache:
        ap.error("--verify / --direct need --cache (or --verdict-cache): the verifier's calls are cached there")
    try:
        client.require_sandbox(model)  # before any work: a run never stops halfway for want of bwrap
    except client.SandboxUnavailable as e:
        ap.error(f"refused: {e}")
    if args.manifest:
        if not args.out:
            ap.error("--manifest needs --out")
        if Path(args.out).exists() and not args.resume:
            ap.error(f"{args.out} exists; a run never writes into an existing directory (--resume continues a run that stopped)")
        if (Path(args.out) / "results.jsonl").exists():
            ap.error(f"{args.out} holds a finished run")
        from sqldecomp.runner import load_manifest_queries
        summary = run_many(load_manifest_queries(args.manifest)[::max(1, args.every)], None, args.out, config, jobs=args.jobs, resume=args.resume,
                           progress=lambda i, n: print(f"{i}/{n}", file=sys.stderr) if i % 25 == 0 or i == n else None)
        print(json.dumps({k: summary[k] for k in ("queries", "verdicts", "findings", "tokens_and_cost")}, indent=1, default=str))
        return 0
    schema = load_any_schema(args.schema)
    if args.sql:
        sql = Path(args.sql[1:]).read_text() if args.sql.startswith("@") else args.sql
        rec = run_one("query", sql, args.question, args.evidence, schema, config, schema_name=Path(args.schema).stem,
                      schema_json=schema_json_of(args.schema))
        print(json.dumps(rec, indent=1, ensure_ascii=False, default=str) if args.json else verification_report(rec))
        return 1 if rec["findings"] else 0
    if not args.out:
        ap.error("--corpus / --queries need --out")
    if Path(args.out).exists() and not args.resume:
        ap.error(f"{args.out} exists; a run never writes into an existing directory (--resume continues a run that stopped)")
    if (Path(args.out) / "results.jsonl").exists():
        ap.error(f"{args.out} holds a finished run")
    queries = load_run_queries(args.corpus, args.queries)[::max(1, args.every)]
    summary = run_many(queries, args.schema, args.out, config, jobs=args.jobs, resume=args.resume,
                       progress=lambda i, n: print(f"{i}/{n}", file=sys.stderr) if i % 25 == 0 or i == n else None)
    print(json.dumps({k: summary[k] for k in ("queries", "verdicts", "findings", "tokens_and_cost")}, indent=1, default=str))
    return 0


def verification_report(rec: dict[str, Any]) -> str:
    """A readable account of one verification record."""
    lines = [f"# {rec['key']}: {rec.get('scoring', '')}",
             f"# model {rec['model']['writer']['model']} ({rec['model']['writer']['effort']}) for the writer, "
             f"{rec['model']['verifier']['model']} ({rec['model']['verifier']['effort']}) for the verifier"
             + ("" if rec.get("sandbox", True) else "; WITHOUT the sandbox (--no-sandbox)")]
    tasks = {t["output"]: t for t in (rec.get("tasks") or {}).get("blocks", [])}
    for item in (rec.get("verdicts") or {}).get("items", []):
        t = tasks.get(item["output"]) or {}
        shared = f" (class shared with {', '.join(item['shared_with'])})" if item["shared_with"] else ""
        lines += ["", f"## block {item['output']}{' (root)' if item['is_root'] else ''}{shared}: {item['verdict'].upper()}",
                  f"reason: {item['reason']}", f"task: {t.get('text', '')}"]
    if rec.get("verdicts"):
        v = rec["verdicts"]
        lines += ["", f"# verdict ({v['rule']}): {v['final']}" + (f"; rejecting: {', '.join(v['rejecting_blocks'])}" if v["rejecting_blocks"] else "")
                  + (f"; undecided: {', '.join(v['undecided_blocks'])}" if v["undecided_blocks"] else "")]
    if rec.get("direct"):
        d = rec["direct"]
        lines.append(f"# direct arm: {d['final']}: {d['items'][0]['reason']}" if d.get("items") else f"# direct arm: not run: {d.get('not_run')}")
    lines.append("# time (ms): " + ", ".join(f"{k} {v:,.1f}" for k, v in rec.get("time_ms", {}).items()))
    def n(v: int | None) -> str:
        return "unknown" if v is None else f"{v:,}"

    for side, t in rec.get("tokens_and_cost", {}).items():
        if t.get("calls"):
            tok = t["tokens"]
            paid = "unavailable" if t["paid_usd"] is None else f"{t['paid_usd']:.4f} USD"
            cost = ("unavailable" if t["estimated_usd"] is None else f"{t['estimated_usd']:.4f} USD") + f" (paid in this run: {paid})"
            if t["estimated_usd"] is None and t.get("unpriced_calls"):
                cost += f"; {t['unpriced_calls']} call(s) without a cost, the others {t['estimated_usd_known']:.4f} USD"
            lines.append(f"# {side}: {t['calls']} calls ({t['cached_calls']} cached, {t.get('failed_calls', 0)} failed), tokens in "
                         f"{n(tok['input'])} (cached {n(tok['cached_input'])}) out {n(tok['output'])}, estimated cost {cost}")
    for f in rec.get("findings", []):
        lines.append(f"# FINDING {f['kind']} at {f['stage']}: {f['detail']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
