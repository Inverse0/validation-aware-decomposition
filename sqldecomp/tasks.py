"""Component 9: TASK ASSIGNMENT. The blocks of the optimal decomposition become verification tasks: for every block
(once per shared class) the TASK TEXT a verifier is shown, the INPUT CONTRACTS of its inputs, the parameter
descriptions and the provenance (`Task`).

THE RULE: no circularity. The STANDARD a block is judged against (its REQUIREMENT) is derived from the question N, the
schema, the constraints and the ROLE of the block in the decomposition (what its readers need from it: the reader's
own task, the reader's SQL where the block appears as a named relation, the block's interface: exposed columns, keys,
parameters), NEVER from the block's own SQL, which the requirement writer is not shown. A description of what the
block's SQL computes (the CLAIM) may be shown beside the requirement as something to confirm or refute; it is never
the standard. The root block's task is N itself.

The method (`td3-desc`): the writer's model writes every non-root block's requirement top-down from N, the evidence,
the schema, the block's interface and every reader's task and SQL (`REQUIREMENT_PROMPT`); a FIXED judging stance is
appended to every block's text (`STANCE_ROOT`, `STANCE_SUPPLIER`). What a reader is told about an input is a
description of what the input computes, written by the writer's model from the input merged with everything
upstream of it into ONE block, as the printer renders it (it runs as it is), without the question
(`DESCRIBE_PROMPT`), shown as "computes (described from its SQL)" beside the input's established facts (its fact
sheet, `fact_sheet`).

STABILITY: the assignment of a decomposition is a deterministic function of its canonical key (`canonical_key`: method
and prompt versions, N, the evidence, the schema, the blocks in decomposition order with their reader-form SQL,
interfaces, keys, parameters, class ids, readers), kept in a PERSISTENT cache (`TaskCache`): the same decomposition
gets the same tasks across runs and days. Every LLM call is itself cached by prompt (the sandbox cache). Shared
classes get ONE task, written for the representative (the first occurrence in the decomposition's block order) with
the readers of every occurrence.

    python -m sqldecomp.tasks --sql FILE --schema FILE.json --question TEXT [--evidence TEXT]
                              (--cache DIR | --llm-cache DIR) [--model gpt-6-luna] [--effort max] [--no-sandbox]
                              [--timeout S] [--json] [--scoring human|llm and the options of sqldecomp.scoring_args]

The model calls go through `sqldecomp.llm` under one `ModelConfig` (the writer's: GPT-6 Luna at max effort by default,
in the bwrap sandbox; a machine without bwrap refuses unless `--no-sandbox` is given, which the JSON output records).

The scoring the optimizer runs under is chosen per run (`--scoring`, default `human`: the human-facing model of
component 3 with its default theta, `sqldecomp.scoring_args.default_spec()`); the tasks are written for the optimum
found under it.
"""
from __future__ import annotations

from prompts import load_prompt

import argparse
import collections
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Any, Callable

import sqlglot
from sqlglot import exp

from sqldecomp.llm import atomic_write_text, in_parallel
from sqldecomp.model import Block, GroupId

METHODS = ("td3-desc",)
MERGED_METHODS = METHODS  # the methods that read the merged blocks of `block_records(merged=True)`
# The labels of the writer's calls (part of their cache keys) and of the fixed texts, as the paper's runs recorded them,
# so that their cached answers apply. The schema text's label: v2 added UNIQUE keys, v3 declared NOT NULL columns.
PROMPT_VERSIONS = {"requirement": "c9-requirement-v3", "describe": "c9-describe-v1", "stance": "c9-stance-v1",
                   "writer-inputs": "c9-writer-interface-v1", "schema-text": "c9-schema-text-v3"}


def prompt_versions(method: str) -> dict[str, str]:
    """The prompt labels a method's assignment depends on."""
    return dict(PROMPT_VERSIONS)


SCHEMA_LEGEND = load_prompt('tasks/schema_legend.txt')
UNIQUE_SENTENCE = load_prompt('tasks/unique_sentence.txt')
NOT_NULL_SENTENCE = load_prompt('tasks/not_null_sentence.txt')
NOT_NULL_MARK = " NOT NULL"


def schema_legend(text: str) -> str:
    """The legend line above a schema text: the plain legend, plus the NOT NULL sentence when the text marks a NOT NULL
    column and the UNIQUE sentence when it renders a UNIQUE key; only what is rendered is explained, so a prompt over
    a schema that declares neither (DW's) is byte-identical to the prompt the published experiments used."""
    parts = ([NOT_NULL_SENTENCE] if NOT_NULL_MARK in text else []) + ([UNIQUE_SENTENCE] if " UNIQUE(" in text else [])
    return SCHEMA_LEGEND if not parts else SCHEMA_LEGEND[:-1] + "".join("; " + x for x in parts) + ")"


WEAK_KEY_LEAD = "unique where non-null: "  # a weak key of component 8: no two rows that are non-null in all of it agree on it
REQUIRED_LEAD = "required to provide"  # an input contract's line for the input's requirement summary
DESCRIBED_LEAD = "computes (described from its SQL)"  # td3-desc: for the description of what the input computes


def keys_text(keys: list[list[str]], *, lead: str = "unique: ") -> str:
    """The keys of a relation for a reader: `unique: (a, b); (c)`, and the empty key (at most one row) in words."""
    named = [k for k in keys if k]
    parts = ([lead + "; ".join("(" + ", ".join(k) + ")" for k in named)] if named else []) + (["at most one row"] if len(named) < len(keys) else [])
    return "; ".join(parts)

LLM = Callable[[str, str], str]  # (prompt, template) -> the model's answer


class AssignmentError(Exception):
    """A task could not be written: the model's answer did not parse, or the record is inconsistent."""


# --------------------------------------------------------------------------------------------------------------
# what component 9 reads: the decomposition record
# --------------------------------------------------------------------------------------------------------------


def tables_in(sql: str) -> set[str]:
    """Base-table names a statement reads (CTE names and block inputs `b_<id>` / `b1_<id>` excluded)."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001
        return set()
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    out = set()
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if name and name not in ctes and not re.fullmatch(r"b\d*_g\d+", name):
            out.add(name)
    return out


def canonical_sql(sql: str) -> str:
    """One layout for every text a verifier sees (SQLGlot's pretty Postgres print, the printer's own layout)."""
    try:
        return sqlglot.parse_one(sql, read="postgres").sql(dialect="postgres", pretty=True)
    except Exception:  # noqa: BLE001
        return sql


def with_inputs(reader_sql: str, bodies: dict[str, str]) -> str:
    """The executable form of a block's reader text: every input defined as a CTE over its own executable statement."""
    tree = sqlglot.parse_one(reader_sql, read="postgres")
    ctes = [exp.CTE(this=sqlglot.parse_one(body, read="postgres"), alias=exp.TableAlias(this=exp.to_identifier(name)))
            for name, body in bodies.items()]
    existing = tree.args.get("with_")
    if existing is not None:
        existing.set("expressions", ctes + list(existing.expressions))
    else:
        tree.set("with_", exp.With(expressions=ctes))
    return tree.sql(dialect="postgres", pretty=True)


def schema_text(schema_json: dict[str, Any], tables: set[str]) -> str:
    """One line per table: `name(col TYPE[ NOT NULL][ PK][ FK->table.col][ JOINS table.col|...], ...)[ UNIQUE(col,
    ...)]...`: ` NOT NULL` marks a column the schema declares not nullable (`"nullable": false`) unless it is part of
    the primary key (which implies it), and the declared UNIQUE keys of the table (`unique_keys` of the schema JSON,
    as `schema_from_json` reads them) follow its columns (version `c9-schema-text-v3`; `schema_legend` says what each
    mark means, only where one is rendered)."""
    from sqldecomp.model import _unique_keys, Column
    by_name = {t["name"].lower(): t for t in schema_json["tables"]}
    lines = []
    for name in sorted(tables):
        t = by_name.get(name)
        if t is None:
            lines.append(f"{name}(?)")
            continue
        pk = set(t.get("primary_key") or ())
        uniques = _unique_keys(t, tuple(Column(str(c["name"]).lower()) for c in t["columns"]), tuple(str(x).lower() for x in pk))
        fk = {}
        for f in t.get("foreign_keys") or ():
            for c, rc in zip(f["columns"], f["ref_columns"]):
                fk[c] = f"{f['ref_table']}.{rc}"
        jp: dict[str, list[str]] = {}
        for j in t.get("join_pairs") or ():
            jp.setdefault(j["column"], []).append(f"{j['table']}.{j['ref_column']}")
        cols = []
        for c in t["columns"]:
            s = f"{c['name']} {c.get('type', 'unknown')}"
            if c.get("nullable") is False and c["name"] not in pk:
                s += NOT_NULL_MARK
            if c["name"] in pk:
                s += " PK"
            if c["name"] in fk:
                s += f" FK->{fk[c['name']]}"
            if c["name"] in jp:
                s += " JOINS " + "|".join(sorted(set(jp[c["name"]])))
            cols.append(s)
        lines.append(f"{name}({', '.join(cols)})" + "".join(f" UNIQUE({', '.join(u)})" for u in uniques))
    return "\n".join(lines)


WRITER_SCHEMA_WHOLE_MAX = 40  # a database with at most this many tables is shown whole to the requirement writer


def interface_tables(record: dict[str, Any], schema_json: dict[str, Any]) -> set[str]:
    """The base tables the exposed column NAMES name (`<table>[_<k>]__<column>`, the printer's naming of base columns):
    part of the block's interface, so a mutation that keeps the interface keeps them; the block's own `tables_own`
    (read from its SQL) must not reach its requirement writer."""
    names = {t["name"].lower() for t in schema_json.get("tables", [])}
    out: set[str] = set()
    for c in record.get("output_columns", []) or []:
        m = re.match(r"^(.*?)(?:_\d+)?__", c.get("name") or "")
        if m and m.group(1).lower() in names:
            out.add(m.group(1).lower())
    return out


def writer_tables(record: dict[str, Any], readers: list[dict[str, Any]], schema_json: dict[str, Any], question_tables: Iterable[str] | None) -> set[str]:
    """The schema excerpt of a block's requirement writer: the question's own database when it is small or
    given, the readers' base tables and the tables the interface names; never the block's own base tables."""
    tables = {t.lower() for t in (question_tables or ())}
    all_tables = [t["name"].lower() for t in schema_json.get("tables", [])]
    if len(all_tables) <= WRITER_SCHEMA_WHOLE_MAX:
        tables |= set(all_tables)
    for rd in readers:
        tables |= {t.lower() for t in rd.get("tables_own", [])}
    tables |= interface_tables(record, schema_json)
    return tables


def _input_names(reader_sql: str) -> dict[str, str]:
    """group id -> the relation name under which the reader text reads it (`b_<id>`, or `b1_<id>` ...)."""
    out: dict[str, str] = {}
    try:
        tree = sqlglot.parse_one(reader_sql, read="postgres")
    except Exception:  # noqa: BLE001
        return out
    for t in tree.find_all(exp.Table):
        m = re.fullmatch(r"b\d*_(g\d+)", t.name, re.I)
        if m:
            out[m.group(1)] = t.name
    return out


def upstream_of(by_output: dict[str, dict[str, Any]], group: str) -> set[str]:
    """Every block that `group` reads, directly or through other blocks."""
    out: set[str] = set()
    stack = list(by_output[group].get("inputs", []))
    while stack:
        x = stack.pop()
        if x not in out:
            out.add(x)
            stack += by_output[x].get("inputs", [])
    return out


def _uses(reader_sql: str, relation: str) -> list[str]:
    """The columns of `relation` that the reader text reads, in first-use order (`*` for `relation.*`)."""
    try:
        tree = sqlglot.parse_one(reader_sql, read="postgres")
    except Exception:  # noqa: BLE001
        return []
    seen: list[str] = []
    for c in tree.find_all(exp.Column):
        if c.table and c.table.lower() == relation.lower() and c.name not in seen:
            seen.append(c.name)
    for s in tree.find_all(exp.Star):
        parent = s.parent
        if isinstance(parent, exp.Column) and parent.table and parent.table.lower() == relation.lower():
            seen.append("*")
    return seen


def _columns_json(columns) -> list[dict[str, Any]]:
    return [{"name": c.name, "type": c.type, "nullable": bool(c.nullable), "alias": c.alias, "column": c.column} for c in columns]


def block_records(optimum, memo, result, blocks, printer, boundary, *, merged: bool = False) -> list[dict[str, Any]]:
    """Everything component 9 may read about the blocks of `optimum.decomposition`, root first:
    the DAG (inputs, readers), the reader-form SQL with the class union (`expose_at`), the executable form where one
    exists, the parameters, the exposed columns with types and `nullable` flags, the keys and foreign keys at every
    boundary (component 8), the base tables each block reads, the columns each reader uses of each input, the class
    id and the other occurrences of the class. JSON-able dicts; a block the printer refuses carries `print_error`.

    The exposed columns of a block, and of every input in its readers' contracts, are the columns its PRINTED text
    exposes (`Boundary.columns_with(group, expose_at(block))`): for a member of a shared class the union of the
    class's columns at that position, which is more than the position alone reads. Listing only a position's own columns would describe a narrower interface than
    the printed text. `contract_gaps` checks that every column used by a reader
    appears in its input contract.

    `merged` (MERGED_METHODS): every non-root block also carries the two MERGED blocks the printer
    renders for them, each ONE block of the memo printed by `Printer.block_sql`: `sql_downstream`, the root merged with
    every block downstream of the block's occurrences (inputs by name: the block itself and the inputs those blocks
    read from elsewhere; `sql_downstream_inline`, the same block with its inputs inlined, is the whole query and runs
    as it is), and `sql_merged`, the block with everything upstream of it as one block (no input; it runs as it is
    unless it has `merged_parameters`), with the base tables it reads (`tables_merged`). A merged block the printer
    refuses leaves `merged_error`.
    """
    from sqldecomp.printer import PrintError

    d = optimum.decomposition
    by_output = {b.output: b for b in d.blocks}
    readers: dict[str, list[str]] = collections.defaultdict(list)
    for b in d.blocks:
        for i in b.inputs:
            readers[i].append(b.output)
    out: list[dict[str, Any]] = []
    for b in d.blocks:
        cid = optimum.class_of_block.get(b.output)
        ctx = optimum.contexts.get(cid) if cid else None
        expose = optimum.expose_at(b)
        rb: dict[str, Any] = {"output": b.output, "inputs": sorted(b.inputs), "class_id": cid,
                              "shared_with": sorted(g for g in optimum.classes_shared.get(cid, []) if g != b.output) if cid else [],
                              "is_root": b.output == memo.root, "scope_top": blocks.is_scope_top(b.output), "kind": blocks.kind(b.output),
                              "readers": sorted(readers.get(b.output, [])), "expose": expose}
        try:
            reader = printer.block_sql(b, inputs="name", within=d, expose=expose or None)
            inline = printer.block_sql(b, inputs="inline", within=d, expose=expose or None)
            params = printer.block_parameters(b, within=d)
        except PrintError as e:
            rb["print_error"] = str(e)[:300]
            out.append(rb)
            continue
        rb["sql"] = canonical_sql(reader)
        rb["sql_inline"] = canonical_sql(inline)
        rb["parameters"] = list(params)
        rb["tables_own"] = sorted(tables_in(reader))
        rb["tables_inline"] = sorted(tables_in(inline))
        cols, star = boundary.columns_with(b.output, expose)
        keys, fks = boundary.exposed_constraints(b.output, cols, star)
        rb["output_columns"] = _columns_json(cols)
        rb["output_star"] = star
        rb["output_keys"] = [list(k) for k in keys]
        rb["output_weak_keys"] = [list(k) for k in boundary.exposed_weak_keys(b.output, cols, star)]
        rb["output_fks"] = [{"columns": list(c), "ref_table": t, "ref_columns": list(rc)} for c, t, rc in fks]
        rb["class_exposed"] = [c.name for c in ctx.exposed] if ctx is not None else None
        rb["meaningfulness"] = ctx.meaningfulness if ctx is not None else None
        names = _input_names(reader)
        contracts, bodies, executable = [], {}, not params
        for g in sorted(b.inputs):
            name = names.get(g, f"b_{g}")
            icols, istar = boundary.columns_with(g, optimum.expose_at(by_output[g]) if g in by_output else None)
            ikeys, ifks = boundary.exposed_constraints(g, icols, istar)
            iweak = boundary.exposed_weak_keys(g, icols, istar)
            parameterized = bool(blocks.parameters(Block(g)))
            try:
                itables = sorted(tables_in(printer.block_sql(by_output[g], inputs="inline", within=d))) if g in by_output else []
            except PrintError:
                itables = []
            contracts.append({"name": name, "group": g, "columns": _columns_json(icols), "star": istar, "keys": [list(k) for k in ikeys],
                              "weak_keys": [list(k) for k in iweak],
                              "fks": [{"columns": list(c), "ref_table": t, "ref_columns": list(rc)} for c, t, rc in ifks],
                              "parameterized": parameterized, "uses": _uses(reader, name), "tables": itables})
            if parameterized:
                executable = False
            if executable and g in by_output:
                try:
                    bodies[name] = printer.block_sql(by_output[g], inputs="inline", within=d, expose=optimum.expose_at(by_output[g]) or None)
                except PrintError:
                    executable = False
        rb["input_contracts"] = contracts
        rb["sql_exec"] = None
        if executable:
            try:
                rb["sql_exec"] = with_inputs(reader, bodies) if bodies else canonical_sql(reader)
            except Exception:  # noqa: BLE001 - an executable form is a convenience, not a requirement
                rb["sql_exec"] = None
        if merged and b.output != memo.root:
            try:
                rb.update(_merged_blocks(b, d, by_output, readers, rb["shared_with"], memo.root, printer, expose))
            except Exception as e:  # noqa: BLE001 - recorded; the variant that needs it refuses the query
                rb["merged_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out.append(rb)
    return out


def _merged_blocks(b: Block, d: Any, by_output: dict[str, Block], readers: dict[str, list[str]], shared_with: list[str], root: str,
                   printer: Any, expose: Any) -> dict[str, Any]:
    """The two merged blocks of `block_records(merged=True)` for the non-root block `b`."""
    from sqldecomp.model import Decomposition
    down: set[str] = set()
    stack = [x for g in [b.output, *shared_with] for x in readers.get(g, [])]
    while stack:
        x = stack.pop()
        if x not in down:
            down.add(x)
            stack += readers.get(x, [])
    # the blocks downstream as one: no block outside them reads one of them, so the rest of the decomposition stands
    above = Block(root, frozenset(g for x in down for g in by_output[x].inputs if g not in down))
    within = Decomposition([above] + [x for x in d.blocks if x.output not in down])
    whole = Block(b.output, frozenset())
    sql_merged = canonical_sql(printer.block_sql(whole, inputs="name", expose=expose or None))
    return {"sql_downstream": canonical_sql(printer.block_sql(above, inputs="name", within=within)),
            "sql_downstream_inline": canonical_sql(printer.block_sql(above, inputs="inline", within=within)),
            "sql_merged": sql_merged, "merged_parameters": list(printer.block_parameters(whole)),
            "tables_merged": sorted(tables_in(sql_merged))}


def contract_gaps(records: list[dict[str, Any]]) -> list[str]:
    """Every column a reader's text uses of an input that the input's contract does not list (`*` apart): empty for
    a consistent set of records."""
    out = []
    for r in records:
        for c in r.get("input_contracts", []):
            listed = {col["name"].lower() for col in c["columns"]}
            missing = [u for u in c.get("uses", []) if u != "*" and u.lower() not in listed]
            if missing and not c.get("star"):
                out.append(f"block {r['output']} reads {', '.join(missing)} of {c['name']}, which its contract does not list")
    return out


# --------------------------------------------------------------------------------------------------------------
# the output
# --------------------------------------------------------------------------------------------------------------


@dataclass
class InputContract:
    name: str  # the relation name in the reader's SQL (`b_<id>`)
    group: GroupId
    columns: list[dict[str, Any]]  # name, type, nullable
    keys: list[list[str]]
    fks: list[dict[str, Any]]
    parameterized: bool
    requirement: str | None = None  # what the input is required to provide (consistent with its own task)
    facts: str | None = None  # what the block computing it establishes (its fact sheet), shown as given
    weak_keys: list[list[str]] = field(default_factory=list)  # keys of the rows that are non-null in them (component 8's weak keys)
    lead: str = REQUIRED_LEAD  # the line's words for `requirement` (td3-desc: DESCRIBED_LEAD)

    def text(self) -> str:
        cols = ", ".join(f"{c['name']} {c['type']}" if c.get("type") and c["type"] != "unknown" else c["name"] for c in self.columns)
        line = f"{self.name}({cols})"
        if self.keys:
            line += " -- " + keys_text(self.keys)
        if self.weak_keys:
            line += " -- " + keys_text(self.weak_keys, lead=WEAK_KEY_LEAD)
        nullable = [c["name"] for c in self.columns if c.get("nullable")]
        if nullable:
            line += " -- may be NULL: " + ", ".join(nullable)
        if self.parameterized:
            line += " -- parameterized: it reads columns of the enclosing query"
        if self.requirement and self.requirement.startswith("FACTS"):
            line += f"\n  {self.requirement}"
        elif self.requirement:
            line += f"\n  {self.lead}: {self.requirement}"
        if self.facts:
            line += f"\n  established by the step that computes it (verified separately; take as given): {self.facts}"
        return line


@dataclass
class Task:
    output: GroupId
    inputs: list[GroupId]
    class_id: str | None
    shared_with: list[GroupId]  # the other occurrences of the class (one task for all of them)
    is_root: bool
    text: str  # the TASK TEXT the verifier is shown (the root: the question itself)
    requirement: str  # the standard (derived from N and the role; the root: N)
    claim: str | None  # the bottom-up part, if the method shows one
    summary: str  # one clause: what the block provides, for the readers' input contracts
    input_contracts: list[InputContract]
    parameters: list[str]  # the parameter descriptions (the printer's parameter texts)
    provenance: dict[str, Any] = field(default_factory=dict)

    def inputs_text(self) -> str:
        """The INPUT RELATIONS section of the verifier's prompt, and the PARAMETERS section for a parameterized block
        (a correlated block reads enclosing-query columns, which must be declared as parameters)."""
        lines = []
        if self.input_contracts:
            lines.append("INPUT RELATIONS (results of earlier steps of the same query; use each as a table with exactly these columns)")
            lines += [c.text() for c in self.input_contracts]
        if self.parameters:
            lines.append("PARAMETERS (columns of the enclosing query that this block reads; the block is evaluated once per row of that query)")
            lines += [f"  {p}" for p in self.parameters]
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------------------------------
# the canonical key and the persistent cache
# --------------------------------------------------------------------------------------------------------------


def writer_identity(config: Any) -> dict[str, Any]:
    """What of a `ModelConfig` the written tasks depend on: the model, the reasoning effort and whether the calls ran
    in the sandbox (part of the canonical key: tasks written by one writer are never served for another)."""
    return {"model": config.model, "effort": config.effort, "sandbox": bool(config.sandbox)}


def canonical_key(records: list[dict[str, Any]], question: str, evidence: str, schema_json: dict[str, Any], method: str,
                  question_tables: Iterable[str] | None = None, writer: dict[str, Any] | None = None) -> str:
    """The SHA-256 of everything the assignment depends on, serialised deterministically: the method and prompt
    versions, N, the evidence, the schema, the blocks in decomposition order (output, inputs, reader-form SQL,
    exposed columns with types and nullable flags, keys, parameters, class id, the other occurrences, readers) and,
    when given, the writer (`writer_identity`: model, effort, sandbox; the pipeline's runs always give it, an injected
    writer function has none)."""
    blocks = [{"output": r["output"], "inputs": r["inputs"], "sql": r.get("sql"), "columns": r.get("output_columns"),
               "keys": r.get("output_keys"), "weak_keys": r.get("output_weak_keys"), "parameters": r.get("parameters"), "class_id": r.get("class_id"),
               "shared_with": r.get("shared_with"), "readers": r.get("readers"),
               "contracts": [{k: c.get(k) for k in ("name", "group", "columns", "keys", "weak_keys", "parameterized")} for c in r.get("input_contracts", [])]}
              for r in records]
    payload = {"component": "9", "method": method, "prompts": prompt_versions(method), "question": question.strip(), "evidence": (evidence or "").strip(),
               "schema": schema_json, "blocks": blocks, "question_tables": sorted(question_tables) if question_tables else None}
    if writer is not None:
        payload["writer"] = writer
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class TaskCache:
    """A persistent cache of assignments: canonical key -> the tasks (JSON, one file per key)."""

    def __init__(self, directory: str | Path | None):
        self.dir = Path(directory) if directory else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> list[Task] | None:
        entry = self.get_entry(key)
        return entry[0] if entry is not None else None

    def get_entry(self, key: str) -> tuple[list[Task], dict[str, Any]] | None:
        """(tasks, meta) of a key, or None; `meta` holds what the writer recorded (method, question, and the records
        of the model calls that wrote the tasks, when a verification run stored them)."""
        if self.dir is None or not (self.dir / f"{key}.json").exists():
            return None
        d = json.loads((self.dir / f"{key}.json").read_text())
        return [Task(**{**t, "input_contracts": [InputContract(**c) for c in t["input_contracts"]]}) for t in d["tasks"]], d.get("meta") or {}

    def put(self, key: str, tasks: list[Task], meta: dict[str, Any] | None = None) -> None:
        if self.dir is None:
            return
        atomic_write_text(self.dir / f"{key}.json", json.dumps({"key": key, "meta": meta or {}, "tasks": [t.to_json() for t in tasks]},
                                                         indent=1, ensure_ascii=False, default=str) + "\n")


# --------------------------------------------------------------------------------------------------------------
# the prompts
# --------------------------------------------------------------------------------------------------------------

REQUIREMENT_PROMPT = load_prompt('tasks/requirement_prompt.txt')

STANCE_SUPPLIER = load_prompt('tasks/stance_supplier.txt')

STANCE_ROOT = load_prompt('tasks/stance_root.txt')


READER_TEMPLATE = load_prompt('tasks/reader_template.txt')

# the description of what an input computes, for its readers, from the input merged with everything upstream;
# the describer is not shown the question, so it describes what the SQL does rather than what it should do
DESCRIBE_PROMPT = load_prompt('tasks/describe_prompt.txt')


def _conjuncts(cond: exp.Expression) -> list[exp.Expression]:
    if isinstance(cond, exp.And):
        return _conjuncts(cond.left) + _conjuncts(cond.right)
    if isinstance(cond, exp.Paren) and isinstance(cond.this, exp.And):
        return _conjuncts(cond.this)
    return [cond]


def _json_answer(answer: str, keys: tuple[str, ...]) -> dict[str, str]:
    m = re.search(r"\{.*\}", answer or "", re.S)
    if not m:
        raise AssignmentError(f"no JSON object in the answer: {answer[:200]!r}")
    try:
        d = json.loads(m.group(0))
    except ValueError as e:
        raise AssignmentError(f"the answer is not valid JSON: {e}: {answer[:200]!r}") from e
    out = {}
    for k in keys:
        v = d.get(k)
        if not isinstance(v, str) or not v.strip():
            raise AssignmentError(f"the answer lacks {k!r}: {answer[:200]!r}")
        out[k] = " ".join(v.split())
    return out


def _ask(llm: LLM, prompt: str, template: str, keys: tuple[str, ...]) -> dict[str, str]:
    """One cached call; an unparsable answer is asked once more under a retry template, then it is an error."""
    try:
        return _json_answer(llm(prompt, template), keys)
    except AssignmentError:
        return _json_answer(llm(prompt, template + "|retry"), keys)


def _base_columns_of(sql: str) -> dict[str, str]:
    """FROM alias -> base table name for the top-level FROM items of a statement (relations `b_<id>` excluded)."""
    out: dict[str, str] = {}
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001
        return out
    for t in tree.find_all(exp.Table):
        if re.fullmatch(r"b\d*_g\d+", t.name, re.I):
            continue
        out[(t.alias or t.name).lower()] = t.name.lower()
    return out


def _top_select(sql: str):
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001
        return None
    top = tree
    while isinstance(top, (exp.Subquery, exp.Paren)):
        top = top.this
    return top if isinstance(top, exp.Select) else None


def never_null_flags(record: dict[str, Any], schema_json: dict[str, Any]) -> list[bool]:
    """For every exposed column of a block (`output_columns`, in order): whether it is never NULL, i.e. a base column
    the schema declares NOT NULL, reached without an outer join (the `never NULL` fact of the contracts)."""
    aliases = _base_columns_of(record.get("sql", ""))
    by_table = {t["name"].lower(): t for t in schema_json.get("tables", [])}
    out = []
    for c in record.get("output_columns", []):
        table = aliases.get((c.get("alias") or "").lower())
        col = by_table.get(table, {}).get("columns", []) if table else []
        declared = next((x for x in col if x["name"] == (c.get("column") or "").lower()), None)
        out.append(declared is not None and not declared.get("nullable", True) and not c.get("nullable"))
    return out


def fact_sheet(record: dict[str, Any], schema_json: dict[str, Any]) -> str:
    """The established facts of a block's output, for the contracts of its readers: what holds on every row of the
    relation, read from the pipeline's facts and the block's own text by the parser: the keys (component 8), the
    foreign keys, which exposed columns cannot be NULL (a base column declared NOT NULL, reached without an outer
    join), the base column behind each exposed column, the conditions that hold on every row (the top-level WHERE /
    HAVING conjuncts and the inner-join conditions, in the block's own words), the join type of every base table, the
    grain (GROUP BY or DISTINCT) and LIMIT."""
    parts = []
    cols = record.get("output_columns", [])
    if record.get("output_keys"):
        parts.append(keys_text(record["output_keys"], lead="no two rows agree on ").replace("; (", " or on ("))
    if record.get("output_fks"):
        parts.append("foreign keys: " + "; ".join(f"({', '.join(f['columns'])}) references {f['ref_table']}({', '.join(f['ref_columns'])})" for f in record["output_fks"]))
    aliases = _base_columns_of(record.get("sql", ""))
    not_null = [c["name"] for c, never in zip(cols, never_null_flags(record, schema_json)) if never]
    if not_null:
        parts.append("never NULL: " + ", ".join(not_null))
    prov = [f"{c['name']} is {aliases[(c.get('alias') or '').lower()]}.{c['column']}" for c in cols
            if c.get("column") and (c.get("alias") or "").lower() in aliases]
    if prov:
        parts.append("columns: " + ", ".join(prov))
    top = _top_select(record.get("sql", ""))
    if top is not None:
        conds = []
        for w in list(top.find_all(exp.Where)) + list(top.find_all(exp.Having)):
            if w.find_ancestor(exp.Select) is top:
                conds += [c.sql(dialect="postgres") for c in _conjuncts(w.this)]
        joins = []
        for j in top.args.get("joins") or []:
            side = (j.args.get("side") or "").upper() or "INNER"
            name = j.this.alias_or_name if hasattr(j.this, "alias_or_name") else j.this.sql(dialect="postgres")
            joins.append(f"{name}: {side} join")
            on = j.args.get("on")
            if on is not None and side == "INNER":
                conds += [c.sql(dialect="postgres") for c in _conjuncts(on)]
        if conds:
            parts.append("every row satisfies: " + "; ".join(conds))
        if joins:
            parts.append("joins: " + ", ".join(joins))
        group = top.args.get("group")
        if group is not None:
            parts.append("one row per distinct (" + ", ".join(g.sql(dialect="postgres") for g in group.expressions) + ")")
        elif top.args.get("distinct") is not None:
            parts.append("rows are distinct over the listed columns")
        limit = top.args.get("limit")
        if limit is not None:
            parts.append(f"at most {limit.expression.sql(dialect='postgres')} rows" + (", chosen by " + ", ".join(o.sql(dialect="postgres") for o in top.args["order"].expressions) if top.args.get("order") else ""))
    return "; ".join(parts) + "." if parts else ""


# --------------------------------------------------------------------------------------------------------------
# the assignment
# --------------------------------------------------------------------------------------------------------------


def _order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Blocks in an order where every block comes after all its readers (the root first). A shared class is written
    once, for the readers of EVERY occurrence, so it is ready only when all those readers are done, and its other
    occurrences count as done with it (they are skipped here and filled from the representative)."""
    by_output = {r["output"]: r for r in records}
    done: list[dict[str, Any]] = []
    seen: set[str] = set()
    pending = list(records)
    while pending:
        progress = False
        for r in list(pending):
            if r not in pending:  # removed as another occurrence of a class written in this pass
                continue
            occurrences = [r["output"]] + [o for o in r.get("shared_with", []) if o in by_output]
            readers_all = {x for o in occurrences for x in by_output[o].get("readers", [])}
            if all(x in seen for x in readers_all):
                done.append(r)
                for o in occurrences:
                    seen.add(o)
                    for other in list(pending):
                        if other is not r and other["output"] == o:
                            pending.remove(other)  # filled from the representative
                pending.remove(r)
                progress = True
        if not progress:  # a cycle cannot happen in a decomposition; refuse rather than loop
            raise AssignmentError("the readers of " + ", ".join(r["output"] for r in pending) + " never complete")
    return done


def assign_from_records(records: list[dict[str, Any]], *, question: str, evidence: str, schema_json: dict[str, Any],
                        method: str = "td3-desc", llm: LLM | None = None, cache: TaskCache | None = None,
                        llm_tag: str = "", question_tables: Iterable[str] | None = None,
                        writer: dict[str, Any] | None = None, parallelism: int = 1) -> list[Task]:
    """The tasks of a decomposition given its block records (`block_records(merged=True)`), in decomposition order
    (root first). `llm_tag`: a suffix on every model call's template to request fresh answers; setting it bypasses the
    assignment cache. `writer`: the identity of the writer behind `llm` (`writer_identity`), part of the canonical key.
    `parallelism`: the model calls made at a time (the writer's of the blocks whose readers' tasks are written, and the
    descriptions, which then run beside the writer's waves under the same limit); task dependencies remain fixed."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if any("print_error" in r for r in records):
        raise AssignmentError("a block of the decomposition cannot be printed: " + "; ".join(r["print_error"] for r in records if "print_error" in r))
    key = canonical_key(records, question, evidence, schema_json, method, question_tables=question_tables, writer=writer)
    if llm_tag:
        cache = None
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    if llm is None and any(not r["is_root"] for r in records):
        raise AssignmentError("writing the requirements of the non-root blocks needs an LLM")
    by_output = {r["output"]: r for r in records}
    evidence_block = f"\nADDITIONAL INFORMATION GIVEN WITH THE QUESTION\n{evidence.strip()}\n" if (evidence or "").strip() else ""
    requirement: dict[str, str] = {}
    summary: dict[str, str] = {}
    provenance: dict[str, dict[str, Any]] = {}
    # 1. the requirements, top-down: the root's is N; every other block's is written from its readers' tasks and SQL.
    # With `parallelism` > 1 the blocks whose readers' tasks are all written are written together, up to
    # `parallelism` calls at a time: the same prompts in the same waves
    tag = f"|{llm_tag}" if llm_tag else ""

    def prepare(r: dict[str, Any]) -> dict[str, Any]:
        """The writer's prompt for the non-root block `r` (its readers' tasks written)."""
        g = r["output"]
        # a shared class: the readers of every occurrence, each under the relation name it uses
        occurrences = [g] + [o for o in r.get("shared_with", []) if o in by_output]
        reader_blocks: list[tuple[str, str, str]] = []  # (reader output, relation name, occurrence)
        for occ in occurrences:
            for reader in by_output[occ].get("readers", []):
                name = next((c["name"] for c in by_output[reader]["input_contracts"] if c["group"] == occ), f"b_{occ}")
                reader_blocks.append((reader, name, occ))
        if not reader_blocks:
            raise AssignmentError(f"block {g} has no reader and is not the root")
        relation = reader_blocks[0][1]
        readers_text = "\n".join(READER_TEMPLATE.format(k=i + 1, reader=reader, task=requirement[reader], relation=name, sql=by_output[reader]["sql"])
                                 for i, (reader, name, _) in enumerate(reader_blocks))
        cols = ", ".join(f"{c['name']} ({c['type']})" if c.get("type") and c["type"] != "unknown" else c["name"] for c in r.get("output_columns", [])) or "(every column of its sources)"
        params = ("; it depends on the enclosing query's " + ", ".join(r["parameters"])) if r.get("parameters") else ""
        # The writer receives neither base tables nor derived keys from this block's SQL.
        tables = writer_tables(r, [by_output[reader] for reader, _, _ in reader_blocks], schema_json, question_tables)
        text = schema_text(schema_json, tables)
        prompt = REQUIREMENT_PROMPT.format(question=question.strip(), evidence=evidence_block, schema_text=text, schema_legend=schema_legend(text),
                                           relation=relation, columns=cols, keys="", parameters=params, readers=readers_text)
        return {"g": g, "occurrences": occurrences, "readers": [x[0] for x in reader_blocks], "prompt": prompt,
                "version": PROMPT_VERSIONS["requirement"]}

    def store(w: dict[str, Any], answer: dict[str, str]) -> None:
        g = w["g"]
        requirement[g], summary[g] = answer["requirement"], answer["summary"]
        provenance[g] = {"requirement": w["version"], "derived_from": {"question": True, "readers": w["readers"], "reader_sql": True,
                                                                        "interface": True, "own_sql": False, "own_tables": False, "own_keys": False}}
        for occ in w["occurrences"][1:]:
            requirement[occ], summary[occ], provenance[occ] = requirement[g], summary[g], {**provenance[g], "representative": g}

    def write(w: dict[str, Any]) -> dict[str, str]:
        return _ask(llm, w["prompt"], w["version"] + tag, ("requirement", "summary"))

    # Describe each non-root block from its upstream-merged SQL, without the question.
    # Schedule descriptions alongside requirement waves under one shared parallelism
    # limit; each reader sees its input description instead of a requirement summary.
    description: dict[str, str] = {}
    described: list[tuple[dict[str, Any], str]] = []  # (the class's first occurrence, its prompt), in decomposition order
    version = PROMPT_VERSIONS["describe"]
    seen: set[str] = set()
    for r in records:
        g = r["output"]
        if r["is_root"] or g in seen:
            continue
        if "sql_merged" not in r:
            raise AssignmentError(f"block {g}: no merged block ({r.get('merged_error', 'block_records(merged=True) was not asked for')})")
        seen.update([g] + [o for o in r.get("shared_with", []) if o in by_output])
        text = schema_text(schema_json, set(r.get("tables_merged", [])))
        cols = ", ".join(f"{c['name']} ({c['type']})" if c.get("type") and c["type"] != "unknown" else c["name"] for c in r.get("output_columns", [])) or "(every column of its sources)"
        params = ("; it depends on the enclosing query's " + ", ".join(r["merged_parameters"])) if r.get("merged_parameters") else ""
        described.append((r, DESCRIBE_PROMPT.format(schema_legend=schema_legend(text), schema_text=text, columns=cols, parameters=params,
                                                    sql=r["sql_merged"])))
    limit = threading.BoundedSemaphore(max(1, parallelism))

    def limited(fn: Callable[[], Any]) -> Any:
        with limit:
            return fn()

    describe = [lambda prompt=prompt: limited(lambda: _ask(llm, prompt, version + tag, ("description",))) for _, prompt in described]
    beside = ThreadPoolExecutor(max_workers=1) if parallelism > 1 and describe else None
    descriptions = beside.submit(in_parallel, describe, parallelism) if beside is not None else None
    try:
        ordered = _order(records)  # the representatives, each after every reader of every occurrence (the root first)
        for r in ordered:
            if r["is_root"]:
                requirement[r["output"]] = question.strip()
                summary[r["output"]] = "the answer to the question"
                provenance[r["output"]] = {"requirement": "the question (the root block's task is the question itself)"}
        pending = [r for r in ordered if not r["is_root"]]
        if parallelism <= 1:
            for r in pending:
                w = prepare(r)
                store(w, write(w))
            pending = []
        while pending:  # parallelism > 1: in waves
            wave = [r for r in pending if all(x in requirement for occ in [r["output"]] + list(r.get("shared_with", []))
                                              if occ in by_output for x in by_output[occ].get("readers", []))]
            if not wave:
                raise AssignmentError("the readers of " + ", ".join(r["output"] for r in pending) + " never complete")
            prepared = [prepare(r) for r in wave]
            for w, answer in zip(prepared, in_parallel([lambda w=w: limited(lambda: write(w)) for w in prepared], parallelism)):
                store(w, answer)
            written = {id(r) for r in wave}
            pending = [r for r in pending if id(r) not in written]
        # 1b. the descriptions' answers, applied in decomposition order once every requirement is written
        answers = descriptions.result() if descriptions is not None else in_parallel(describe, parallelism)
    finally:
        if beside is not None:
            beside.shutdown(wait=True)  # a failed wave still waits for the descriptions' calls, which the record reports
    for (r, _), answer in zip(described, answers):
        g = r["output"]
        upstream = upstream_of(by_output, g)
        for occ in [g] + [o for o in r.get("shared_with", []) if o in by_output]:
            description[occ] = answer["description"]
            provenance[occ] = {**provenance[occ], "description": {"version": version, "question": False, "own_sql": True,
                                                                  "upstream": sorted(upstream)}}
    # 2. the task texts and the contracts: the requirement plus a FIXED judging stance (no SQL enters it); the root keeps
    # N and is told to take its inputs as given
    tasks: list[Task] = []
    for r in records:
        g = r["output"]
        if r["is_root"]:
            text = requirement[g] + (("\n\n" + STANCE_ROOT) if r.get("inputs") else "")
        else:
            text = requirement[g] + "\n\n" + STANCE_SUPPLIER
        contracts = [InputContract(c["name"], c["group"], c["columns"], c["keys"], c["fks"], c["parameterized"], description.get(c["group"]),
                                   fact_sheet(by_output[c["group"]], schema_json) if c["group"] in by_output else None,
                                   weak_keys=list(c.get("weak_keys") or []), lead=DESCRIBED_LEAD)
                     for c in r.get("input_contracts", [])]
        tasks.append(Task(g, list(r["inputs"]), r.get("class_id"), list(r.get("shared_with", [])), r["is_root"], text, requirement[g], None,
                          summary[g], contracts, list(r.get("parameters", [])),
                          {"method": method, "prompts": prompt_versions(method), "key": key, **provenance.get(g, {}),
                           "stance": PROMPT_VERSIONS["stance"], "contracts": "description+facts"}))
    if cache is not None:
        cache.put(key, tasks, {"method": method, "question": question})
    return tasks


def assign_tasks(optimum, context: "AssignmentContext", *, method: str = "td3-desc") -> dict[Block, Task]:
    """The entry point: the tasks of the optimum, per block of its decomposition (shared classes get one
    task, repeated under each occurrence's block). `context` carries the question, the evidence, the schema and the
    pipeline objects (`AssignmentContext`)."""
    records = block_records(optimum, context.memo, context.result, context.blocks, context.printer, context.boundary, merged=method in MERGED_METHODS)
    tasks = assign_from_records(records, question=context.question, evidence=context.evidence, schema_json=context.schema_json, question_tables=context.question_tables,
                                method=method, llm=context.llm, cache=context.cache)
    by_output = {t.output: t for t in tasks}
    return {b: by_output[b.output] for b in optimum.decomposition.blocks}


@dataclass
class AssignmentContext:
    question: str
    evidence: str
    schema_json: dict[str, Any]
    memo: Any
    result: Any
    blocks: Any
    printer: Any
    boundary: Any
    llm: LLM | None = None
    cache: TaskCache | None = None
    question_tables: list[str] | None = None  # the question's own database's tables, when the database holds several (BIRD)


# --------------------------------------------------------------------------------------------------------------
# the default LLM client (`sqldecomp.llm`: GPT-5.5 medium through the codex CLI in the bwrap sandbox, cached by prompt)
# --------------------------------------------------------------------------------------------------------------


def llm_from_ask(ask: Callable[[str, str], Any], calls: list | None = None, deadline: Any = None) -> LLM:
    """An `LLM` over a call function (prompt, template) -> `sqldecomp.llm.Call`: the answer, every call appended to
    `calls` when given; a failed call without an answer is an `AssignmentError`. With a `deadline` (the decomposition
    timeout) the clock is looked at before and after every call: the writer's calls are on it."""

    def answer(prompt: str, template: str) -> str:
        if deadline is not None:
            deadline.check("tasks")
        call = ask(prompt, template)
        if calls is not None:
            calls.append(call)
        if deadline is not None:  # after the call is recorded: a call that ran past the clock is still reported
            deadline.check("tasks")
        if call.exit_code != 0 and not call.answer.strip():
            raise AssignmentError(f"the model call failed (exit code {call.exit_code}): {call.stderr[-300:].strip() or 'no message'}")
        return call.answer

    return answer


def codex_llm(cache_dir: str | Path | None = None, *, model: str | None = None, effort: str | None = None,
              config: Any = None, calls: list | None = None, deadline: Any = None) -> LLM:
    """An `LLM` over `sqldecomp.llm.call` under `config` (a `ModelConfig`, the one the verifier takes too; default
    `ModelConfig()`: GPT-5.5, medium, in the sandbox); `cache_dir`, `model` and `effort` override its fields. Every
    call is cached by (model, effort, template, prompt) and appended to `calls` when given (the run record's tokens,
    latency and cost). With a `deadline` (the decomposition timeout) no call outlives the clock: each runs at most
    what is left, and the run stops with a timeout at stage `tasks`. A call without bwrap and without `--no-sandbox`
    raises `SandboxUnavailable`."""
    from dataclasses import replace

    from sqldecomp import llm as client
    cfg = config if config is not None else client.ModelConfig()
    changes = {k: v for k, v in (("cache_dir", str(cache_dir) if cache_dir else None), ("model", model), ("effort", effort)) if v}
    cfg = replace(cfg, **changes) if changes else cfg

    def ask(prompt: str, template: str):
        if deadline is None or deadline.end is None:
            return client.call(prompt, template, cfg)
        return client.call(prompt, template, replace(cfg, timeout_s=max(0.001, min(cfg.timeout_s, deadline.remaining()))))

    return llm_from_ask(ask, calls, deadline)


# --------------------------------------------------------------------------------------------------------------
# the command line: the whole pipeline under the chosen scoring (the human model by default), then the tasks
# --------------------------------------------------------------------------------------------------------------


def run(sql: str, schema_json: dict[str, Any], question: str, evidence: str = "", *, method: str = "td3-desc", llm: LLM | None = None,
        cache: TaskCache | None = None, timeout_s: float | None = None, deadline: Any = None,
        spec: dict[str, Any] | None = None, writer: dict[str, Any] | None = None, parallelism: int = 1) -> tuple[Any, list[Task], list[dict[str, Any]]]:
    """Import, memo, constraints, legality, boundary, the scoring of `spec` (`sqldecomp.scoring_args`; None or an
    empty spec = the default, `default_spec()`), the exact optimizer, then the tasks, all under the decomposition
    timeout: `deadline`, or one of `timeout_s` seconds started here (default 300 s); its expiry is a
    `DecompositionTimeout` with the stage, the writer's calls included. `writer`: the identity of the writer behind
    `llm` (`writer_identity`), part of the canonical key. Returns (optimum, tasks, records). An import result the
    validator rejects is a `RuntimeError`."""
    from sqldecomp.clock import DECOMPOSITION_TIMEOUT_S, Deadline, memo_budget, memo_stopped
    from sqldecomp.blocks import Blocks
    from sqldecomp.boundary import Boundary
    from sqldecomp.constraints import derive_constraints
    from sqldecomp.importer import import_sql
    from sqldecomp.memo import build_memo, signature
    from sqldecomp.model import schema_from_json, validate_import_result
    from sqldecomp.optimizer import Optimizer
    from sqldecomp.printer import Printer
    from sqldecomp.scoring_args import default_spec, make_scoring

    deadline = deadline if deadline is not None else Deadline(DECOMPOSITION_TIMEOUT_S if timeout_s is None else timeout_s)
    schema = schema_from_json(schema_json)
    result = import_sql(sql, schema)
    problems = validate_import_result(result)
    if problems:
        raise RuntimeError("invalid import result: " + "; ".join(problems))
    memo = build_memo(result, budget_s=memo_budget(deadline))
    memo_stopped(memo.complete, deadline)
    deadline.check("constraints")
    blocks = Blocks(memo, result)
    printer = Printer(result, memo)
    signatures = {g: signature(memo, result, g) for g in memo.groups}
    constraints = derive_constraints(memo, result, schema)
    boundary = Boundary(memo, result, blocks, printer, constraints, schema)
    scoring = make_scoring(spec if spec else default_spec(), memo, result, blocks, boundary)
    optimum = Optimizer(memo, result, blocks, scoring, schema=schema, constraints=constraints, signatures=signatures, printer=printer,
                        boundary=boundary, deadline=deadline).optimize()
    deadline.check("printing")
    records = block_records(optimum, memo, result, blocks, printer, boundary, merged=method in MERGED_METHODS)

    def timed(prompt: str, template: str) -> str:  # the writer's calls are on the clock
        deadline.check("tasks")
        answer = llm(prompt, template)
        deadline.check("tasks")
        return answer
    tasks = assign_from_records(records, question=question, evidence=evidence, schema_json=schema_json, method=method,
                                llm=timed if llm is not None else None, cache=cache, writer=writer, parallelism=parallelism)
    return optimum, tasks, records


def main(argv: list[str] | None = None) -> int:
    from sqldecomp.clock import timeout_seconds
    ap = argparse.ArgumentParser(prog="python -m sqldecomp.tasks", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sql", required=True, help="a file holding the query")
    ap.add_argument("--schema", required=True, help="the schema file: contract JSON or the BEAVER format")
    ap.add_argument("--question", required=True)
    ap.add_argument("--evidence", default="")
    from sqldecomp.config import TASKS_METHOD, WRITER_EFFORT, WRITER_MODEL
    ap.add_argument("--method", choices=METHODS, default=TASKS_METHOD, help=f"component 9's method (default {TASKS_METHOD})")
    ap.add_argument("--cache", default=None, help="the persistent cache of assignments (a directory)")
    ap.add_argument("--llm-cache", default=None, help="the cache of model calls (default: <cache>/llm, or none)")
    ap.add_argument("--json", action="store_true")
    from sqldecomp.clock import TASKS_TIMEOUT_S
    ap.add_argument("--timeout", type=timeout_seconds, default=TASKS_TIMEOUT_S,
                    help=f"the decomposition timeout in seconds: import to the written tasks, the writer's calls included (default {TASKS_TIMEOUT_S:g})")
    from sqldecomp import llm as client
    from sqldecomp.schema import schema_json_of
    from sqldecomp.scoring_args import add_scoring_arguments, is_judge, scoring_label, scoring_spec_from_args
    add_scoring_arguments(ap)
    client.add_model_arguments(ap, model=WRITER_MODEL, effort=WRITER_EFFORT, calls="the task writer's calls (td3-desc's describer's too)")
    args = ap.parse_args(argv)
    llm_cache = args.llm_cache or (str(Path(args.cache) / "llm") if args.cache else None)
    model_config = client.model_config_from_args(args, cache_dir=llm_cache)
    from sqldecomp.clock import Deadline
    deadline = Deadline(args.timeout)
    llm = codex_llm(config=model_config, deadline=deadline) if llm_cache else None
    if llm is None:
        print("an LLM cache directory is needed (--cache or --llm-cache)", file=sys.stderr)
        return 2
    try:
        client.require_sandbox(model_config)  # before any work: a run never stops halfway for want of bwrap
    except client.SandboxUnavailable as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    spec = scoring_spec_from_args(args)
    if is_judge(spec):
        raise SystemExit("error: --semantic judge rates the cut points with two model calls of its own: it runs in the verification run "
                         "(python -m sqldecomp.pipeline --verify)")
    try:
        optimum, tasks, records = run(Path(args.sql).read_text(), schema_json_of(args.schema), args.question, args.evidence,
                                      method=args.method, llm=llm, cache=TaskCache(args.cache) if args.cache else None,
                                      deadline=deadline, spec=spec, writer=writer_identity(model_config))
    except Exception as e:  # noqa: BLE001 - UnsupportedSQL, an invalid import, the decomposition timeout, AssignmentError: reported, not a traceback
        if type(e).__name__ not in ("UnsupportedSQL", "RuntimeError", "AssignmentError", "DecompositionTimeout", "TooManyBlocks", "ValueError"):
            raise
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"method": args.method, "scoring": scoring_label(spec), "model_config": model_config.to_json(),
                          "sandbox": model_config.sandbox, "blocks": len(tasks), "tasks": [t.to_json() for t in tasks]},
                         indent=1, ensure_ascii=False, default=str))
        return 0
    by_output = {r["output"]: r for r in records}
    print(f"scoring: {scoring_label(spec)}")
    print(f"model: {model_config.model} ({model_config.effort}){'' if model_config.sandbox else ', WITHOUT the sandbox (--no-sandbox)'}")
    for t in tasks:
        print(f"=== block {t.output} <- {t.inputs or 'base tables'}{' (root)' if t.is_root else ''}{' (class shared with ' + ', '.join(t.shared_with) + ')' if t.shared_with else ''}")
        print("TASK:", t.text)
        if t.inputs_text():
            print(t.inputs_text())
        print("SQL:")
        print(by_output[t.output]["sql"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
