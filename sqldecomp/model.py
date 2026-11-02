"""Shared data types and validation contracts for the SQL decomposition pipeline.
This module depends only on the Python standard library.

SQL -> importer -> ImportResult -> memo -> Memo
Memo feeds the printer/checks, constraints, block enumeration and optimizer.
sqldecomp.blocks defines legality; sqldecomp.pipeline assembles the stages.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal, Union

CONTRACT_VERSION = "0.5"

# --------------------------------------------------------------------------------------------------------------
# Database schema and declared constraints (PK, FK, UNIQUE)
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    name: str
    type: str = "unknown"
    nullable: bool = True


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    """`unique_keys`: the declared UNIQUE constraints beside the primary key, with SQL's NULLS DISTINCT meaning
    (Postgres by default, MySQL, SQLite): two rows conflict only when they are non-null in every column of the
    constraint and agree on all of them, so a column that may be NULL holds any number of NULLs. A UNIQUE whose columns
    are all declared NOT NULL is a candidate key like the primary key; one with a nullable column is a key of the rows
    that are non-null in those columns (component 8 carries it as a WEAK key and promotes it where the rows are known
    to be non-null). Lower-case column names; never the primary key's own column set, never a column the table does
    not declare (`schema_from_json` drops both)."""

    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class Schema:
    """All names are lower case. A schema that declares no constraints is legal (the current beaver_dw load)."""

    tables: tuple[Table, ...]

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name.lower():
                return t
        raise KeyError(name)

    def as_sqlglot(self) -> dict[str, dict[str, str]]:
        return {t.name: {c.name: c.type for c in t.columns} for t in self.tables}


# --------------------------------------------------------------------------------------------------------------
# The skeleton plan: the ONLY plan language the memo sees. Five constructors.
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scan:
    """One FROM item that is a base table. Two pseudo tables stand in where SQL has no table: `$dual` for a SELECT
    without FROM and `$values` for a VALUES list; their side entry has `data.table = None`, so never look them up in
    the schema (sqldecomp.importer.DUAL_TABLE / VALUES_TABLE)."""

    table: str
    occ: str  # occurrence id of this FROM item, unique in the query, e.g. "c@0" (alias @ scope)


@dataclass(frozen=True)
class Join:
    """Inner join. Its condition is a parameter of the join, exactly as a Filter carries its conjunct: `conjuncts`
    holds the ids of the join conjuncts that span the two sides, as a sorted tuple. Normalization N places every join
    conjunct at the lowest join that covers its inputs, so this set is exactly the edges of Facts.edges whose inputs
    are all covered by the join but not by one side alone (validate_import_result checks that). An empty tuple is a
    cross join that the query itself contains. Semantically the two inputs are unordered; left/right only record the
    order written in the query, which the printer may use for readability."""

    left: "Plan"
    right: "Plan"
    conjuncts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Filter:
    """One conjunct applied to one plan. WHERE conjuncts that are no join conjuncts, and HAVING conjuncts: those sit
    directly above the aggregate, which is then a region input of a one-input region of its own."""

    input: "Plan"
    conjunct: str  # id of ONE conjunct in the side table; a WHERE with k local conjuncts is k stacked Filters


@dataclass(frozen=True)
class Op1:
    """Any unary operator the rewrite rules never touch: aggregate, distinct, order, limit, projection, and `ref`:
    one USE of a nested scope (CTE or derived table) as a FROM item, which carries that use's alias. A barrier.

    A select list is a `project` node when it computes something (window functions included) and ALSO where the
    listed columns decide the ROWS: under SELECT DISTINCT, and in the branches of a set operation other than
    UNION ALL. Elsewhere a select list of plain columns is no node (ScopeInfo.outputs has it). Order of the nodes of
    one SELECT: aggregate, HAVING filters, project, distinct, order, limit; for DISTINCT ON: order, distinct, limit."""

    input: "Plan"
    op: str  # id in the side table


@dataclass(frozen=True)
class Op2:
    """Any binary operator the rewrite rules never touch: set operation, outer / semi / anti join. A barrier."""

    op: str
    left: "Plan"
    right: "Plan"


Plan = Union[Scan, Join, Filter, Op1, Op2]


def plan_children(p: Plan) -> tuple[Plan, ...]:
    if isinstance(p, Scan):
        return ()
    if isinstance(p, (Filter, Op1)):
        return (p.input,)
    return (p.left, p.right)


def walk(p: Plan) -> Iterator[Plan]:
    yield p
    for c in plan_children(p):
        yield from walk(c)


# --------------------------------------------------------------------------------------------------------------
# Side table and scopes: where ALL real SQL lives. The memo never reads these.
# --------------------------------------------------------------------------------------------------------------

SideKind = Literal["scan", "conjunct", "aggregate", "project", "distinct", "order", "limit", "setop", "outer_join", "ref"]


@dataclass(frozen=True)
class SideEntry:
    """sql is the PostgreSQL fragment with each column qualified by its FROM alias.
    data holds kind-specific structured fields used by the importer and consumers.
    """

    id: str
    kind: SideKind
    sql: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputColumn:
    name: str  # the name the enclosing query sees
    sql: str  # the expression that produces it (qualified)


@dataclass(frozen=True)
class ScopeInfo:
    """One SELECT (or set operation) of the query."""

    id: str  # "0", "1", ...
    kind: Literal["root", "cte", "derived", "subquery", "setop_branch"]
    name: str | None  # CTE name or derived-table alias; None for root and for predicate subqueries
    # id of the scope that references this one; for a CTE the FIRST scope that uses it, which need not be the scope
    # whose WITH clause declares it (that one is `data["declared_in"]` of every `ref` entry of the CTE)
    parent: str | None
    outputs: tuple[OutputColumn, ...]
    plan: Plan  # the skeleton of this scope, including its Op1 nodes


# --------------------------------------------------------------------------------------------------------------
# Facts: everything the memo, the constraint rules and the optimizer may know about ids. Emit a fact only when
# sure. A missing fact costs completeness, never soundness.
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnRef:
    """A column, named by what EXPOSES it where the fact lives: a region input (a scan occurrence, a `ref`, or the
    outer join through which the column is visible: behind an outer join it may be NULL-extended and is not the base
    column any more), or the operator that produces it (aggregate, project, set operation), with its output name."""

    origin: str
    column: str


@dataclass(frozen=True)
class RegionInput:
    """A leaf of an inner-join region: a Scan, or a plan rooted at Op1 / Op2. A nested scope is never a leaf directly:
    each use of it is wrapped in its own `ref` Op1, so every region input is one specific OCCURRENCE and no two region
    inputs ever share a plan value."""

    id: str  # for a Scan this equals Scan.occ; for an operator-rooted input it equals the operator id
    plan: Plan


@dataclass(frozen=True)
class Edge:
    """A join conjunct: references two or more region inputs (a hyperedge when more than two)."""

    conjunct: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class Equate:
    """A join conjunct that is a plain `column = column` between two different inputs."""

    conjunct: str
    left: ColumnRef
    right: ColumnRef


@dataclass(frozen=True)
class Keep:
    """Output column `name` of operator `op` is an unmodified copy of `source`."""

    op: str
    name: str
    source: ColumnRef


@dataclass
class Facts:
    inputs: list[RegionInput] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    # local conjunct id -> the single input it depends on: its own columns AND the inputs its subqueries are correlated
    # with (`NOT EXISTS (... c.c_id ...)` is owned by c). A conjunct that depends on no input has no owner.
    owners: dict[str, str] = field(default_factory=dict)
    equates: list[Equate] = field(default_factory=list)
    # aggregate op id -> ALL its grouping columns, present only when every grouping expression is a plain column and
    # there are no grouping sets; [] = no GROUP BY (at most one row); a missing entry means "unknown", never "empty"
    group_cols: dict[str, list[ColumnRef]] = field(default_factory=dict)
    keeps: list[Keep] = field(default_factory=list)  # about VALUES, not rows: also across a DISTINCT or an aggregate


@dataclass
class ImportResult:
    sql: str
    plan: Plan  # P0: the normalized skeleton of the whole query (equals the root scope's plan)
    scopes: list[ScopeInfo]
    side: dict[str, SideEntry]
    facts: Facts
    dialect: str = "postgres"
    contract: str = CONTRACT_VERSION


# --------------------------------------------------------------------------------------------------------------
# Memo read-out: plain Python, no e-graph types. Everything downstream of the memo works on this.
# --------------------------------------------------------------------------------------------------------------

GroupId = str


@dataclass(frozen=True)
class MExpr:
    """One way to compute a group. kind/arg: scan -> occ, filter -> conjunct id, op1/op2 -> op id, join -> None.
    A join expression carries its condition in `conjuncts` (sorted ids of the join conjuncts spanning its two sides)."""

    kind: Literal["scan", "join", "filter", "op1", "op2"]
    children: tuple[GroupId, ...]
    arg: str | None = None
    conjuncts: tuple[str, ...] = ()


@dataclass
class Memo:
    groups: dict[GroupId, list[MExpr]]
    root: GroupId
    complete: bool = True  # False when saturation hit its time budget (sound, not complete)

    def _index(self) -> dict[tuple, GroupId]:
        idx = getattr(self, "_idx", None)
        if idx is None:
            idx = {}
            for g, exprs in self.groups.items():
                for e in exprs:
                    kids = tuple(sorted(e.children)) if e.kind == "join" else e.children
                    idx[(e.kind, kids, e.arg, tuple(sorted(e.conjuncts)))] = g
            self._idx = idx
        return idx

    def group_of(self, plan: Plan) -> GroupId | None:
        """The group a plan belongs to, or None if the memo does not represent it. Component 5 for two sub-plans
        of the same query is `memo.group_of(p) == memo.group_of(q)` (and not None)."""
        if isinstance(plan, Scan):
            key = ("scan", (), plan.occ, ())
        else:
            kids = tuple(self.group_of(c) for c in plan_children(plan))
            if any(k is None for k in kids):
                return None
            if isinstance(plan, Join):
                key = ("join", tuple(sorted(kids)), None, tuple(sorted(plan.conjuncts)))
            elif isinstance(plan, Filter):
                key = ("filter", kids, plan.conjunct, ())
            elif isinstance(plan, Op1):
                key = ("op1", kids, plan.op, ())
            else:
                key = ("op2", kids, plan.op, ())
        return self._index().get(key)

    def scans(self, g: GroupId, _seen: frozenset = frozenset()) -> frozenset[str]:
        """Occurrence ids of every base table under a group (the same for all of its expressions, by M1)."""
        if g in _seen:
            return frozenset()
        e = self.groups[g][0]
        if e.kind == "scan":
            return frozenset({e.arg})
        out: set[str] = set()
        for c in e.children:
            out |= self.scans(c, _seen | {g})
        return frozenset(out)

    def filters(self, g: GroupId, _seen: frozenset = frozenset()) -> frozenset[str]:
        """Conjunct ids already applied under a group (the same for all of its expressions, by M1)."""
        if g in _seen:
            return frozenset()
        e = self.groups[g][0]
        out: set[str] = {e.arg} if e.kind == "filter" else set()
        for c in e.children:
            out |= self.filters(c, _seen | {g})
        return frozenset(out)


@dataclass(frozen=True)
class Block:
    """(output group, input groups): some fragment computes `output` from `inputs` and base tables. An input may lie
    inside the body of a predicate / select-item subquery of the block. Which pairs are blocks, and which blocks are
    PARAMETERIZED (they read a column of an enclosing query), is decided by sqldecomp.blocks.Blocks."""

    output: GroupId
    inputs: frozenset[GroupId] = frozenset()


@dataclass
class Decomposition:
    """Blocks such that one has the root group as its output, every input is the output of exactly one block, every
    block is reachable from the root block, every forced cut is made (a group read from two or more places is a
    block of its own) and each group is cut one way (sqldecomp.blocks.Blocks.problems lists what is wrong otherwise)."""

    blocks: list[Block]


# --------------------------------------------------------------------------------------------------------------
# Constraint propagation results (component 8)
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    """No two rows of the group agree on all of `columns`, where NULL agrees with NULL. The empty key: at most one
    row. Keys made by operators (GROUP BY, global aggregate, DISTINCT, UNION) are derived even when the schema
    declares no constraint (a design decision)."""

    columns: frozenset[ColumnRef]


@dataclass(frozen=True)
class FKey:
    """Every row whose `columns` are all non-null has a match in `ref_table` (MATCH SIMPLE)."""

    columns: tuple[ColumnRef, ...]
    ref_table: str
    ref_columns: tuple[str, ...]


@dataclass(frozen=True)
class WeakKey:
    """A key of the rows that are non-null in every column of `nullable`: no two rows of the group that are non-null in
    all of `nullable` agree on all of `columns` (NULL agreeing with NULL elsewhere). `nullable` is a non-empty subset
    of `columns`; a weak key whose `nullable` set has become empty is a `Key`. Made by a declared UNIQUE constraint
    with a nullable column: it is not a key of the group (two rows may both be NULL there), but it may serve as the
    equated key of the other side of a join (the join condition being TRUE, the equated values are non-null) and it
    becomes a key once every column of `nullable` is null-rejected for the rows of the group."""

    columns: frozenset[ColumnRef]
    nullable: frozenset[ColumnRef]


@dataclass
class GroupConstraints:
    keys: set[Key] = field(default_factory=set)
    fks: set[FKey] = field(default_factory=set)
    weak: set[WeakKey] = field(default_factory=set)  # UNIQUE constraints with a nullable column, carried by component 8


# --------------------------------------------------------------------------------------------------------------
# JSON (fixtures, artifacts, cross-component hand-off)
# --------------------------------------------------------------------------------------------------------------


def plan_to_json(p: Plan) -> dict:
    if isinstance(p, Scan):
        return {"k": "scan", "table": p.table, "occ": p.occ}
    if isinstance(p, Join):
        return {"k": "join", "conjuncts": list(p.conjuncts), "left": plan_to_json(p.left), "right": plan_to_json(p.right)}
    if isinstance(p, Filter):
        return {"k": "filter", "conjunct": p.conjunct, "input": plan_to_json(p.input)}
    if isinstance(p, Op1):
        return {"k": "op1", "op": p.op, "input": plan_to_json(p.input)}
    return {"k": "op2", "op": p.op, "left": plan_to_json(p.left), "right": plan_to_json(p.right)}


def plan_from_json(d: dict) -> Plan:
    k = d["k"]
    if k == "scan":
        return Scan(d["table"], d["occ"])
    if k == "join":
        return Join(plan_from_json(d["left"]), plan_from_json(d["right"]), tuple(d.get("conjuncts", ())))
    if k == "filter":
        return Filter(plan_from_json(d["input"]), d["conjunct"])
    if k == "op1":
        return Op1(plan_from_json(d["input"]), d["op"])
    if k == "op2":
        return Op2(d["op"], plan_from_json(d["left"]), plan_from_json(d["right"]))
    raise ValueError(f"unknown plan kind {k!r}")


def _cref(c: ColumnRef) -> list[str]:
    return [c.origin, c.column]


def schema_to_json(s: Schema) -> dict:
    """`unique_keys` is written only for a table that has some, so the field leaves the JSON of a schema without
    UNIQUE constraints unchanged, and with it the canonical keys of component 9 that this JSON enters."""
    return {"tables": [{"name": t.name, "columns": [{"name": c.name, "type": c.type, "nullable": c.nullable} for c in t.columns],
                        "primary_key": list(t.primary_key),
                        "foreign_keys": [{"columns": list(f.columns), "ref_table": f.ref_table, "ref_columns": list(f.ref_columns)}
                                         for f in t.foreign_keys],
                        **({"unique_keys": [list(u) for u in t.unique_keys]} if t.unique_keys else {})} for t in s.tables]}


def _unique_keys(t: dict, columns: tuple[Column, ...], primary_key: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """The table's `unique_keys` as declared, lower case, in the order given: without a key that repeats a column or
    names a column the table does not declare (when it declares its columns), without a key that contains the primary
    key's column set (the primary key implies it), and each column set once."""
    known = {c.name for c in columns}
    seen: set[frozenset[str]] = set()
    out: list[tuple[str, ...]] = []
    for u in t.get("unique_keys") or ():
        key = tuple(str(x).lower() for x in u)
        if not key or len(set(key)) != len(key) or (known and not set(key) <= known) or frozenset(key) in seen:
            continue
        if primary_key and set(primary_key) <= set(key):
            continue
        seen.add(frozenset(key))
        out.append(key)
    return tuple(out)


def schema_from_json(d: dict) -> Schema:
    tables = []
    for t in d["tables"]:
        columns = tuple(Column(c["name"].lower(), c.get("type", "unknown"), c.get("nullable", True)) for c in t["columns"])
        primary_key = tuple(x.lower() for x in t.get("primary_key", ()))
        tables.append(Table(t["name"].lower(), columns, primary_key,
                            tuple(ForeignKey(tuple(x.lower() for x in f["columns"]), f["ref_table"].lower(),
                                             tuple(x.lower() for x in f["ref_columns"])) for f in t.get("foreign_keys", ())),
                            _unique_keys(t, columns, primary_key)))
    return Schema(tuple(tables))


def import_result_to_json(r: ImportResult) -> dict:
    f = r.facts
    return {
        "contract": r.contract, "dialect": r.dialect, "sql": r.sql, "plan": plan_to_json(r.plan),
        "scopes": [{"id": s.id, "kind": s.kind, "name": s.name, "parent": s.parent,
                    "outputs": [{"name": o.name, "sql": o.sql} for o in s.outputs], "plan": plan_to_json(s.plan)} for s in r.scopes],
        "side": [{"id": e.id, "kind": e.kind, "sql": e.sql, "data": e.data} for e in r.side.values()],
        "facts": {
            "inputs": [{"id": i.id, "plan": plan_to_json(i.plan)} for i in f.inputs],
            "edges": [{"conjunct": e.conjunct, "inputs": list(e.inputs)} for e in f.edges],
            "owners": dict(f.owners),
            "equates": [{"conjunct": q.conjunct, "left": _cref(q.left), "right": _cref(q.right)} for q in f.equates],
            "group_cols": {op: [_cref(c) for c in cols] for op, cols in f.group_cols.items()},
            "keeps": [{"op": k.op, "name": k.name, "source": _cref(k.source)} for k in f.keeps],
        },
    }


def import_result_from_json(d: dict) -> ImportResult:
    fd = d["facts"]
    facts = Facts(
        inputs=[RegionInput(i["id"], plan_from_json(i["plan"])) for i in fd["inputs"]],
        edges=[Edge(e["conjunct"], tuple(e["inputs"])) for e in fd["edges"]],
        owners=dict(fd["owners"]),
        equates=[Equate(q["conjunct"], ColumnRef(*q["left"]), ColumnRef(*q["right"])) for q in fd["equates"]],
        group_cols={op: [ColumnRef(*c) for c in cols] for op, cols in fd["group_cols"].items()},
        keeps=[Keep(k["op"], k["name"], ColumnRef(*k["source"])) for k in fd["keeps"]],
    )
    scopes = [ScopeInfo(s["id"], s["kind"], s["name"], s["parent"],
                        tuple(OutputColumn(o["name"], o["sql"]) for o in s["outputs"]), plan_from_json(s["plan"])) for s in d["scopes"]]
    side = {e["id"]: SideEntry(e["id"], e["kind"], e["sql"], e.get("data", {})) for e in d["side"]}
    return ImportResult(d["sql"], plan_from_json(d["plan"]), scopes, side, facts, d.get("dialect", "postgres"),
                        d.get("contract", CONTRACT_VERSION))


def memo_to_json(m: Memo) -> dict:
    return {"root": m.root, "complete": m.complete,
            "groups": {g: [{"kind": e.kind, "children": list(e.children), "arg": e.arg, "conjuncts": list(e.conjuncts)} for e in exprs]
                       for g, exprs in m.groups.items()}}


def memo_from_json(d: dict) -> Memo:
    return Memo({g: [MExpr(e["kind"], tuple(e["children"]), e.get("arg"), tuple(e.get("conjuncts", ()))) for e in exprs]
                 for g, exprs in d["groups"].items()},
                d["root"], d.get("complete", True))


def dumps(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False)


# --------------------------------------------------------------------------------------------------------------
# Invariants every ImportResult must satisfy. The importer's tests and the pipeline both call this.
# --------------------------------------------------------------------------------------------------------------


def validate_import_result(r: ImportResult) -> list[str]:
    """Returns a list of violations; empty means valid."""
    bad: list[str] = []
    side = r.side
    occs = [n.occ for n in walk(r.plan) if isinstance(n, Scan)]
    for s in r.scopes:
        occs += [n.occ for n in walk(s.plan) if isinstance(n, Scan)]
    for n in _all_nodes(r):
        if isinstance(n, Scan) and (n.occ not in side or side[n.occ].kind != "scan"):
            bad.append(f"scan {n.occ} has no side entry of kind 'scan'")
        if isinstance(n, Filter) and (n.conjunct not in side or side[n.conjunct].kind != "conjunct"):
            bad.append(f"filter conjunct {n.conjunct} has no side entry of kind 'conjunct'")
        if isinstance(n, (Op1, Op2)) and n.op not in side:
            bad.append(f"operator {n.op} has no side entry")
    input_ids = {i.id for i in r.facts.inputs}
    if len(input_ids) != len(r.facts.inputs):
        bad.append("duplicate region input ids")
    by_plan: dict[Plan, str] = {}
    for i in r.facts.inputs:
        if isinstance(i.plan, (Join, Filter)):
            bad.append(f"region input {i.id} must be a Scan or be rooted at Op1 / Op2 (wrap a nested scope in a 'ref' Op1)")
        if i.plan in by_plan:
            bad.append(f"region inputs {by_plan[i.plan]} and {i.id} share one plan value; every use needs its own 'ref' Op1")
        by_plan[i.plan] = i.id
    for n in _all_nodes(r):
        if isinstance(n, Join):
            left, right = region_inputs_under(n.left, by_plan), region_inputs_under(n.right, by_plan)
            if left is None or right is None:
                bad.append("a join input contains an operator that is not registered as a region input")
                continue
            expected = sorted(e.conjunct for e in r.facts.edges
                              if set(e.inputs) <= left | right and not set(e.inputs) <= left and not set(e.inputs) <= right)
            if sorted(n.conjuncts) != expected:
                bad.append(f"join over {sorted(left | right)} carries {sorted(n.conjuncts)} but its spanning edges are {expected}")
            if list(n.conjuncts) != sorted(n.conjuncts):
                bad.append("Join.conjuncts must be a sorted tuple")
    for occ in set(occs):
        if occ not in input_ids:
            bad.append(f"scan {occ} is not registered as a region input")
    applied = {n.conjunct for n in _all_nodes(r) if isinstance(n, Filter)}
    for e in r.facts.edges:
        if len(set(e.inputs)) < 2:
            bad.append(f"edge {e.conjunct} must reference at least two different inputs")
        if not set(e.inputs) <= input_ids:
            bad.append(f"edge {e.conjunct} references unknown inputs")
        if e.conjunct not in side or side[e.conjunct].kind != "conjunct":
            bad.append(f"edge conjunct {e.conjunct} has no side entry of kind 'conjunct'")
        if e.conjunct in applied:
            bad.append(f"conjunct {e.conjunct} is both a join edge and a Filter")
    for c, owner in r.facts.owners.items():
        if owner not in input_ids:
            bad.append(f"owner of {c} is not a region input")
        if c not in applied:
            bad.append(f"owned conjunct {c} never appears as a Filter")
    for q in r.facts.equates:
        if q.conjunct not in {e.conjunct for e in r.facts.edges}:
            bad.append(f"equate {q.conjunct} is not a join edge")
    root_scopes = [s for s in r.scopes if s.kind == "root"]
    if len(root_scopes) != 1:
        bad.append("exactly one root scope is required")
    elif root_scopes[0].plan != r.plan:
        bad.append("ImportResult.plan must equal the root scope's plan")
    return bad


def region_inputs_under(plan: Plan, by_plan: dict[Plan, str]) -> frozenset[str] | None:
    """Ids of the region inputs covered by a plan inside ONE inner-join region (Joins and Filters over inputs).
    None when the plan contains an operator that is not a registered region input."""
    if plan in by_plan:
        return frozenset({by_plan[plan]})
    if isinstance(plan, (Join, Filter)):
        out: set[str] = set()
        for c in plan_children(plan):
            sub = region_inputs_under(c, by_plan)
            if sub is None:
                return None
            out |= sub
        return frozenset(out)
    return None


def _all_nodes(r: ImportResult) -> Iterable[Plan]:
    seen: list[Plan] = list(walk(r.plan))
    for s in r.scopes:
        seen += list(walk(s.plan))
    for i in r.facts.inputs:
        seen += list(walk(i.plan))
    return seen
