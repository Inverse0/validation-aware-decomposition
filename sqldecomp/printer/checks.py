"""Checks C1 (round trip), C2 (recomposition) and C3 (validity).

C1 and C2 decide equivalence the way component 5 does: re-import the printed SQL, map its ids onto the original
import, and look the mapped plan up in the ORIGINAL memo. By default the importer is `sqldecomp.importer.import_sql`
and the mappings come from `sqldecomp.memo.alignments`, the memo's own search; both can be injected (the printer's
unit tests inject test doubles).

An aligner only PROPOSES mappings. Whatever it proposes, the check itself verifies (`_verify`): the mapped plan is in
the wanted group of the original memo; the body of every predicate / select-item subquery is, mapped, in the group of
its partner's body (the outer plan holds such a body only as opaque text); every nested scope outputs what its
partner outputs (a rename-or-drop projection is no plan node); and every id of the original has a partner. So a
wrong proposal can only make a check fail, never pass.

C2 has one step of its own. In this build the importer keeps every derived table as a `ref` barrier, so the derived
tables `b_<group id>` that the printer puts at block boundaries come back as operators the original query never had.
Such a boundary only renames and drops columns. `dissolve_boundaries` therefore removes exactly these refs again and
undoes their renaming before the aligned plan is looked up; every other derived table or CTE stays a barrier."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from sqldecomp.importer import import_sql
from sqldecomp.importer._names import UNKNOWN as NO_DEFAULT_NAME, pg_default_name
from sqldecomp.memo import SearchLimit, alignments, explain_alignment
from sqldecomp.model import (Decomposition, Edge, Facts, Filter, GroupId, ImportResult, Join, Memo, Op1, Op2,
                             OutputColumn, Plan, RegionInput, Scan, Schema, ScopeInfo, SideEntry,
                             validate_import_result, walk)
from sqldecomp.printer import sqltext
from sqldecomp.printer.build import Builder
from sqldecomp.printer.context import PrintContext
from sqldecomp.printer.errors import PrintError
from sqldecomp.printer.printer import Printer

ImportFn = Callable[[str, Schema], ImportResult]
AlignFn = Callable[[ImportResult, ImportResult], "dict[str, str] | None"]  # (other, onto) -> one mapping, or None


@dataclass
class CheckResult:
    """`stats` of a passed C1 / C2 says what was really compared, so that a run can show its checks were not vacuous:
    `ids` mapped, nested `scopes` whose output lists were compared, `subquery_bodies` looked up in the memo, and the
    number of `candidates` (mappings) tried."""

    ok: bool
    detail: str = ""
    stats: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------------------------
# C1 and C2
# --------------------------------------------------------------------------------------------------------------


def check_round_trip(plan: Plan, result: ImportResult, memo: Memo, schema: Schema,
                     import_fn: ImportFn | None = None, align_fn: AlignFn | None = None) -> CheckResult:
    """C1: re-importing the printed SQL lands in the same memo group.

    A plan from inside a correlated subquery is printed in closed form (`Printer.plan_sql(bind_outer=True)`); the
    check then follows the re-import into the EXISTS subquery that holds the plan. The alignment is made onto the part
    of the original import that the plan covers, so that a query which contains the same fragment twice (one CTE
    used in two places, two equal UNION branches) does not make the alignment of ONE of them ambiguous."""
    want = memo.group_of(plan)
    if want is None:
        return CheckResult(False, "the plan is not represented in the memo, so there is no group to come back to")
    try:
        builder = Builder(PrintContext(result))
        sql = builder.statement(plan, bind_outer=True).sql(dialect=sqltext.DIALECT, pretty=True)
    except PrintError as e:
        return CheckResult(False, f"cannot print the plan: {e}")
    try:
        other = (import_fn or import_sql)(sql, schema)
    except Exception as e:  # the importer can be injected; whatever it raises is a failed check, not a crash
        return CheckResult(False, f"re-import failed: {type(e).__name__}: {e}\n{sql}")
    try:
        if builder.closed:
            other = restrict(other, _exists_body(other))
        ctx = PrintContext(result)
        # an inner fragment selects the columns the rest of the query reads; only the top of a scope has a select list
        check = _lands_in(other, want, restrict(result, plan), memo, align_fn, sql, root_outputs=ctx.is_scope_top(plan))
        return check if check.ok else _modulo_projection(plan, other, check, result, memo, align_fn, sql, ctx)
    except PrintError as e:
        return CheckResult(False, f"{e}\n{sql}")


def _modulo_projection(plan: Plan, other: ImportResult, failed: CheckResult, result: ImportResult, memo: Memo,
                       align_fn: AlignFn, sql: str, ctx: PrintContext) -> CheckResult:
    """A select list of plain columns is no plan node (normalization iii). An importer may still make it one where
    the listed columns decide the rows: `sqldecomp.importer` does so below DISTINCT. Such a projection printed on its
    own, without the DISTINCT, comes back without the node. The round trip then holds when the re-import lands in the
    group below the projection AND selects exactly the projection's columns."""
    if not (isinstance(plan, Op1) and ctx.entry(plan.op).kind == "project"):
        return failed
    items = ctx.entry(plan.op).data.get("select") or ()
    parsed = [sqltext.parse_item(i["sql"]) for i in items]
    if not all(isinstance(t, exp.Column) for t in parsed):
        return failed
    below = memo.group_of(plan.input)
    if below is None:
        return failed
    check = _lands_in(other, below, restrict(result, plan.input), memo, align_fn, sql, root_outputs=False)
    if not check.ok:
        return failed
    want = [((i.get("alias") or t.name).lower(), sqltext.to_sql(sqltext.unquote_safe_identifiers(t)))
            for i, t in zip(items, parsed)]
    got = _output_list(next(sc for sc in other.scopes if sc.kind == "root"))
    if got != want:
        return CheckResult(False, f"the re-import selects {got}, the projection {plan.op} selects {want}\n{sql}")
    return CheckResult(True, f"group {memo.group_of(plan)} (a projection of plain columns over group {below})", check.stats)


def check_recomposition(decomposition: Decomposition, result: ImportResult, memo: Memo, schema: Schema,
                        import_fn: ImportFn | None = None, align_fn: AlignFn | None = None) -> CheckResult:
    """C2: inlining every block into its consumers gives a query equivalent to the original."""
    try:
        tree, boundaries = Printer(result, memo).recompose(decomposition)
    except PrintError as e:
        return CheckResult(False, f"cannot recompose the decomposition: {e}")
    sql = tree.sql(dialect=sqltext.DIALECT, pretty=True)
    try:
        other = (import_fn or import_sql)(sql, schema)
    except Exception as e:
        return CheckResult(False, f"re-import failed: {type(e).__name__}: {e}\n{sql}")
    try:
        if boundaries:
            other = dissolve_boundaries(other, boundaries)
            problems = validate_import_result(other)
            if problems:
                return CheckResult(False, "without its block boundaries the re-imported query breaks the contract: "
                                   + "; ".join(problems) + f"\n{sql}")
        # aligned onto what the root plan reaches: a CTE that the query defines but never uses is not part of it
        return _lands_in(other, memo.root, restrict(result, result.plan), memo, align_fn, sql, root_outputs=True)
    except PrintError as e:
        return CheckResult(False, f"{e}\n{sql}")


def _lands_in(other: ImportResult, want: GroupId, onto: ImportResult, memo: Memo, align_fn: AlignFn | None,
              sql: str, *, root_outputs: bool) -> CheckResult:
    """Does the re-import `other` lie in group `want` of the original memo? `onto` is the part of the original import
    it is compared with. The first proposed mapping that survives `_verify` decides; when none does, the first
    failure is reported."""
    theirs, ours = canonical(other), canonical(onto)
    failure, tried = None, 0
    try:
        for mapping in _proposals(theirs, ours, memo, align_fn):
            tried += 1
            problem, stats = _verify(other, want, onto, memo, mapping, root_outputs)
            if problem is None:
                return CheckResult(True, f"group {want}", {**stats, "candidates": tried})
            failure = failure or problem
    except SearchLimit as e:
        failure = failure or f"cannot tell: {e}"
    if failure is None:
        failure = "the re-imported query cannot be aligned with the original (ambiguous or incomplete)"
        if align_fn is None:  # the search found no mapping at all; say what a strict alignment stumbles over
            why = explain_alignment(theirs, ours)
            failure = ("no mapping of ids places the re-imported plan in the memo of the original"
                       + (": " + "; ".join(why[:3]) if why else ""))
    return CheckResult(False, f"{failure}\n{sql}")


def _proposals(other: ImportResult, onto: ImportResult, memo: Memo, align_fn: AlignFn | None):
    """Candidate mappings of the ids of `other` onto the ids of `onto`. Product path: every mapping under which the
    plan is in the memo (`sqldecomp.memo.alignments`). An injected aligner proposes one mapping or none."""
    if align_fn is None:
        return (mapping for _, mapping in alignments(memo, onto, other))
    mapping = align_fn(other, onto)
    return [] if mapping is None else [mapping]


def _verify(other: ImportResult, want: GroupId, onto: ImportResult, memo: Memo, mapping: dict[str, str],
            root_outputs: bool) -> tuple[str | None, dict[str, int]]:
    """None when `mapping` proves that `other` lies in group `want`, else what is wrong. Nothing here trusts the
    aligner: a wrong mapping can only produce a failure."""
    try:
        mapped = map_plan(other.plan, mapping)
    except KeyError as e:
        return f"the alignment has no image for id {e.args[0]!r}", {}
    got = memo.group_of(mapped)
    if got is None:
        return f"the re-imported plan is not in the memo (expected group {want})", {}
    if got != want:
        return f"the re-imported plan is in group {got}, expected {want}", {}
    bodies = _subquery_bodies(other, onto, memo, mapping)
    if isinstance(bodies, str):
        return bodies, {}
    scopes = _different_outputs(other, onto, mapping, root_outputs)
    if isinstance(scopes, str):
        return scopes, {}
    orphans = sorted(set(onto.side) - set(mapping.values()))
    if orphans:  # by M1 every plan of a group has the same ids, so this can only be a bug (memo, restrict or aligner)
        return f"ids of the original without a partner in the re-import: {', '.join(orphans[:6])}", {}
    return None, {"ids": len(mapping), "scopes": scopes, "subquery_bodies": bodies}


def _subquery_bodies(other: ImportResult, onto: ImportResult, memo: Memo, mapping: dict[str, str]) -> int | str:
    """The plan of `other` holds a predicate or select-item subquery only as opaque text inside a conjunct or
    operator, and `canonical` reduces that text to a digest of labels, which does not see the STRUCTURE of the body
    (which side of a LEFT JOIN, which branch of an EXCEPT). So every subquery scope that a mapped entry names has to
    be, mapped, in the memo group of the scope its partner names at the same position. Returns how many bodies were
    compared, or the first difference."""
    theirs, ours = {s.id: s for s in other.scopes}, {s.id: s for s in onto.scopes}
    compared = 0
    for src, dst in mapping.items():
        if src not in other.side or dst not in onto.side:
            return f"the alignment maps {src!r} to {dst!r}, which is not an id of the compared imports"
        a_subs = list(other.side[src].data.get("subqueries") or ())
        b_subs = list(onto.side[dst].data.get("subqueries") or ())
        if len(a_subs) != len(b_subs):
            return f"{src} holds {len(a_subs)} subqueries, its partner {dst} holds {len(b_subs)}"
        for sa, sb in zip(a_subs, b_subs):
            if sa not in theirs or sb not in ours:
                return f"{src} / {dst} name a subquery scope ({sa} / {sb}) that the import result does not contain"
            try:
                got = memo.group_of(map_plan(theirs[sa].plan, mapping))
            except KeyError as e:
                return f"the alignment has no image for id {e.args[0]!r} inside the subquery of {src}"
            expected = memo.group_of(ours[sb].plan)
            if got is None or got != expected:
                return (f"the subquery of {src} is in group {got}, the subquery of its partner {dst} is in group "
                        f"{expected}: same text, another body")
            compared += 1
    return compared


def _different_outputs(other: ImportResult, onto: ImportResult, mapping: dict[str, str], root: bool) -> int | str:
    """A select list that only drops or renames columns is no plan node, so the group does not see it: a query that
    returns the wrong columns of the right rows would pass. Every scope of the re-import therefore has to output what
    its partner outputs: the same names for the same expressions, in the same order. The root is left out for an inner
    fragment, whose select list is the printer's choice. A predicate or select-item subquery is compared by its
    expressions only: its reader takes its values, never its names (a select item that IS such a subquery is named in
    the enclosing scope, whose own output list is compared), and a cut inside its body makes Postgres name an
    unaliased item after the boundary column (`CAST(b_<id>.belts__id AS VARCHAR)` is `belts__id`, the original
    `CAST(id AS VARCHAR)` is `id`). Returns how many scopes were compared, or the difference."""
    theirs, ours = PrintContext(other), PrintContext(onto)
    compared = 0
    for scope in other.scopes:
        if scope.kind == "root" and not root:
            continue
        partners = {ours.owner.get(mapping[i]) for i in theirs.scope_ids(scope.id) if i in mapping} - {None}
        if not partners:
            continue  # a scope that the compared plan does not reach (the mapping covers what the plan reaches)
        if len(partners) > 1:
            return f"scope {scope.id} of the re-import is spread over the scopes {sorted(partners)} of the original"
        partner = ours.scopes[partners.pop()]
        mine, theirs_ = _output_list(scope), _output_list(partner)
        if scope.kind == "subquery" and partner.kind == "subquery":
            mine, theirs_ = [sql for _, sql in mine], [sql for _, sql in theirs_]
        if mine != theirs_:
            return (f"scope {scope.id} of the re-import outputs {_output_list(scope)}, its partner, scope {partner.id} "
                    f"of the original, outputs {_output_list(partner)}")
        compared += 1
    return compared


def _output_list(scope: ScopeInfo) -> list[tuple[str, str]]:
    return [(o.name.lower() if not o.name.startswith('"') else o.name, _without_bodies(o.sql)) for o in scope.outputs]


def _without_bodies(sql: str) -> str:
    """An output expression with every query nested in it masked: subquery bodies are compared as scopes of their own
    (`_subquery_bodies`), and their TEXT changes when a block boundary lies inside them."""
    try:
        tree = sqltext.parse_item(sql)
        holder = exp.Paren(this=tree)
        for query in sqltext.top_level_queries(tree):
            query.replace(exp.select(exp.Literal.string("subquery")))
        return sqltext.to_sql(sqltext.unquote_safe_identifiers(holder.this))
    except PrintError:
        return sql


def _exists_body(other: ImportResult) -> Plan:
    """The plan of the EXISTS subquery of a closed form `SELECT 1 FROM ... WHERE EXISTS (...)`."""
    ctx = PrintContext(other)
    root = next((s for s in other.scopes if s.kind == "root"), None)
    if root is not None:
        for id_, e in other.side.items():
            nested = e.data.get("subqueries") or ()
            if e.kind == "conjunct" and ctx.owner.get(id_) == root.id and len(nested) == 1 and nested[0] in ctx.scopes:
                return ctx.scopes[nested[0]].plan
    raise PrintError("the re-import of a closed form does not show its EXISTS subquery as a scope of its own")


def restrict(result: ImportResult, plan: Plan) -> ImportResult:
    """The part of an import result that a plan of it covers, as an import result whose root is that plan: the plan,
    every nested scope it reaches (through refs, set operations and subqueries) and their side entries and facts.
    What the plan reaches is told by IDS, not by plan values: a plan of a group can reach a nested scope as another
    plan of that scope's group than the import's (the memo's read-out may stack a CTE body's filters in another
    order), and that scope and its region inputs still belong to the part."""
    ctx = PrintContext(result)
    home = ctx.scope_of(plan)
    ids: set[str] = set()
    plans: set[Plan] = set()
    todo = [plan]
    while todo:
        for node in walk(todo.pop()):
            if node in plans:
                continue
            plans.add(node)
            here = [node.occ] if isinstance(node, Scan) else list(node.conjuncts) if isinstance(node, Join) else \
                [node.conjunct] if isinstance(node, Filter) else [node.op]
            ids.update(here)
            for id_ in here:
                for sid in ctx.entry(id_).data.get("subqueries") or ():
                    if sid in ctx.scopes:
                        todo.append(ctx.scopes[sid].plan)
    reached = {ctx.owner.get(i) for i in ids} - {None, home.id}
    kept = [s for s in result.scopes if s.id in reached]
    known = {s.id for s in kept} | {home.id}
    scopes = [ScopeInfo(home.id, "root", None, None, home.outputs, plan)]
    scopes += [s if s.parent in known else replace(s, parent=home.id) for s in kept]
    f = result.facts
    facts = Facts([i for i in f.inputs if i.id in ids],
                  [e for e in f.edges if e.conjunct in ids],
                  {c: o for c, o in f.owners.items() if c in ids},
                  [q for q in f.equates if q.conjunct in ids],
                  {op: cols for op, cols in f.group_cols.items() if op in ids},
                  [k for k in f.keeps if k.op in ids])
    side = {id_: e for id_, e in result.side.items() if id_ in ids}
    return ImportResult(result.sql, plan, scopes, side, facts, result.dialect, result.contract)


def map_plan(p: Plan, mapping: dict[str, str]) -> Plan:
    """The plan of one import expressed in the ids of another. KeyError names the first id without an image."""
    if isinstance(p, Scan):
        return Scan(p.table, mapping[p.occ])
    if isinstance(p, Join):
        return Join(map_plan(p.left, mapping), map_plan(p.right, mapping), tuple(sorted(mapping[c] for c in p.conjuncts)))
    if isinstance(p, Filter):
        return Filter(map_plan(p.input, mapping), mapping[p.conjunct])
    if isinstance(p, Op1):
        return Op1(map_plan(p.input, mapping), mapping[p.op])
    return Op2(mapping[p.op], map_plan(p.left, mapping), map_plan(p.right, mapping))


# --------------------------------------------------------------------------------------------------------------
# Block boundaries of a re-imported recomposition
# --------------------------------------------------------------------------------------------------------------


def dissolve_boundaries(other: ImportResult, names: set[str]) -> ImportResult:
    """Remove the block boundaries from a re-imported recomposition. `names` are the (folded) names `b_<group id>` that
    the printer gave its boundaries; everything else comes from the re-import itself, never from the printer's
    bookkeeping, so that a boundary which exposes the wrong column is caught:

    - a boundary is a `ref` without a CTE name whose alias is one of `names`; its body must live in a FROM clause
      (scans, joins, filters, refs, outer joins), otherwise it is more than a projection and the check fails;
    - the body takes the place of the ref in every plan;
    - a column `b_<id>.<x>` is replaced by the expression that the boundary's own select list gives for `<x>`, the way
      a view is expanded (recursively, when that expression reads another boundary).

    Ids are kept. A derived table or CTE of the query itself is not touched and stays a barrier."""
    if not names:
        return other
    ctx = PrintContext(other)
    refs = {id_: e for id_, e in other.side.items()
            if e.kind == "ref" and not e.data.get("name") and sqltext.ident_key(e.data.get("alias")) in names}
    missing = names - {sqltext.ident_key(e.data.get("alias")) for e in refs.values()}
    if missing:
        raise PrintError(f"the re-import does not show the block boundaries {sorted(missing)} as derived tables "
                         "(kind 'ref', alias b_<group id>, no name)")
    at = {(ctx.owner.get(id_), sqltext.ident_key(e.data.get("alias"))): e for id_, e in refs.items()}
    for e in refs.values():
        if not ctx.is_region_level(ctx.nested_scope(e).plan):
            raise PrintError(f"block boundary {e.data.get('alias')} is more than a projection: its body ends in an "
                             "aggregate, DISTINCT, ORDER BY, LIMIT, computed column or set operation")

    def definition(column: exp.Column, scope_id: str | None) -> exp.Expression | None:
        """What `b_<id>.<x>` stands for, read from the select list of the boundary visible in scope `scope_id`."""
        table, name = sqltext.column_key(column)
        if table not in names:
            return None
        home = scope_id
        while home is not None and (home, table) not in at:
            home = ctx.scopes[home].parent if home in ctx.scopes else None
        if home is None:
            raise PrintError(f"{sqltext.to_sql(column)} refers to a block boundary that no enclosing scope defines")
        body = ctx.nested_scope(at[(home, table)])
        for out in body.outputs:
            if _names(column.this, out.name):
                return restore(sqltext.parse_item(out.sql), body.id)
        raise PrintError(f"the re-import reads {sqltext.to_sql(column)}, which the block boundary does not define")

    def restore(tree: exp.Expression, scope_id: str | None) -> exp.Expression:
        if isinstance(tree, exp.Alias):
            tree.set("this", restore(tree.this, scope_id))
            return tree
        return sqltext.rename_columns(tree, lambda column: definition(column, scope_id))

    def text(sql: str, scope_id: str | None) -> str:
        return sqltext.to_sql(restore(sqltext.parse_item(sql), scope_id))


    def splice(p: Plan) -> Plan:
        if isinstance(p, Scan):
            return p
        if isinstance(p, Join):
            return Join(splice(p.left), splice(p.right), p.conjuncts)
        if isinstance(p, Filter):
            return Filter(splice(p.input), p.conjunct)
        if isinstance(p, Op1):
            return splice(p.input) if p.op in refs else Op1(splice(p.input), p.op)
        return Op2(p.op, splice(p.left), splice(p.right))

    side: dict[str, SideEntry] = {}
    for id_, e in other.side.items():
        if id_ in refs:
            continue
        home = ctx.owner.get(id_)
        data = dict(e.data)
        if "select" in data:
            data["select"] = [{**i, "sql": text(i["sql"], home)} for i in data["select"] or ()]
        if "on" in data:
            data["on"] = [text(sql, home) for sql in data["on"] or ()]
        if "group_by" in data:  # an item may be ROLLUP (...) / CUBE (...) / GROUPING SETS (...): no expression by itself
            data["group_by"] = [sqltext.group_item_sql(restore(sqltext.parse_group_item(sql), home))
                                for sql in data["group_by"] or ()]
        if "keys" in data:
            data["keys"] = [{**k, "sql": text(k["sql"], home)} for k in data["keys"] or ()]
        side[id_] = SideEntry(id_, e.kind, text(e.sql, home) if e.kind == "conjunct" else e.sql, data)

    gone = {e.data.get("scope") for e in refs.values()}
    scopes = []
    for sc in other.scopes:
        if sc.id in gone:
            continue
        parent = sc.parent
        while parent in gone:  # a scope inside a dissolved boundary now hangs below the reader of that boundary
            parent = ctx.scopes[parent].parent
        outputs = tuple(OutputColumn(o.name, text(o.sql, sc.id)) for o in sc.outputs)
        scopes.append(ScopeInfo(sc.id, sc.kind, sc.name, parent, outputs, splice(sc.plan)))
    dissolved = ImportResult(other.sql, splice(other.plan), scopes, side, Facts(), other.dialect, other.contract)
    dissolved.facts = _facts_without(other.facts, set(refs), splice, dissolved)
    return dissolved


def _names(identifier: exp.Expression, output_name: str) -> bool:
    """Does a column reference name this output column? Output names are stored bare, so Postgres' rule is applied
    from the side of the reference: a quoted identifier matches exactly, an unquoted one whatever its case."""
    if not isinstance(identifier, exp.Identifier):
        return False
    bare = output_name[1:-1].replace('""', '"') if output_name.startswith('"') and output_name.endswith('"') else output_name
    return identifier.name == bare if identifier.args.get("quoted") else identifier.name.lower() == bare.lower()


def _facts_without(facts: Facts, refs: set[str], splice: Callable[[Plan], Plan], dissolved: ImportResult) -> Facts:
    """Facts of a dissolved import: the boundary refs are no region inputs any more, and a conjunct that mentioned one
    now mentions the inputs behind it. Which input an alias belongs to depends on the region: above an outer join its
    aliases belong to the outer join. Facts for component 8 that mention a boundary are dropped (always allowed)."""
    ctx = PrintContext(dissolved)
    inputs = [RegionInput(i.id, splice(i.plan)) for i in facts.inputs if i.id not in refs]
    at: dict[str, Plan] = {}  # conjunct -> the Join or Filter node that carries it
    for scope in dissolved.scopes:
        for node in walk(scope.plan):
            for c in (node.conjuncts if isinstance(node, Join) else [node.conjunct] if isinstance(node, Filter) else ()):
                at.setdefault(c, node)

    def region_leaves(p: Plan) -> list[Plan]:
        if isinstance(p, (Join, Filter)):
            return [leaf for child in ((p.left, p.right) if isinstance(p, Join) else (p.input,))
                    for leaf in region_leaves(child)]
        return [p]

    def inputs_of(conjunct: str) -> tuple[str, ...]:
        if conjunct not in at:
            raise PrintError(f"conjunct {conjunct!r} is not part of any scope of the dissolved import")
        by_alias: dict[str, str] = {}
        for leaf in region_leaves(at[conjunct]):
            for alias in ctx.visible_aliases(leaf):
                by_alias.setdefault(alias, leaf.occ if isinstance(leaf, Scan) else leaf.op)
        found: list[str] = []
        for column in sqltext.outer_columns(sqltext.parse_predicate(dissolved.side[conjunct].sql)):
            input_id = by_alias.get(sqltext.column_key(column)[0])
            if input_id is not None and input_id not in found:
                found.append(input_id)
        return tuple(found)

    edges, owners = [], {}
    for e in facts.edges:
        edges.append(Edge(e.conjunct, inputs_of(e.conjunct)) if refs & set(e.inputs) else e)
    for conjunct, owner in facts.owners.items():
        if owner in refs:
            behind = inputs_of(conjunct)
            if len(behind) != 1:
                raise PrintError(f"conjunct {conjunct!r} sits above a block boundary as a filter but reads "
                                 f"{len(behind)} inputs behind it")
            owner = behind[0]
        owners[conjunct] = owner
    clean = lambda c: c.origin not in refs  # noqa: E731
    return Facts(inputs, edges, owners,
                 [q for q in facts.equates if clean(q.left) and clean(q.right)],
                 {op: cols for op, cols in facts.group_cols.items() if all(clean(c) for c in cols)},
                 [k for k in facts.keeps if k.op not in refs and clean(k.source)])


def canonical(result: ImportResult) -> ImportResult:
    """A copy whose side-table SQL is in one normal form, for alignment. Ids are untouched. Both sides of an alignment
    go through this, so a difference that is only one of rendering cannot make two equal operators look different:

    - every fragment is parsed and printed again by the same SQLGlot, without needless quotes;
    - an alias that repeats the column's own name is dropped (`c.c_id AS c_id`);
    - the descriptive `sql` of an operator is rebuilt from its `data`;
    - a query nested in a conjunct or select item is replaced by a digest of what its scope contains. The text of a
      subquery changes when a block boundary lies inside it, its scope (after `dissolve_boundaries`) does not."""
    ctx = PrintContext(result)
    digests: dict[str, str] = {}
    labels: dict[str, str] = {}

    def digest(scope_id: str) -> str:
        if scope_id not in digests:
            digests[scope_id] = "..."  # a scope cannot contain itself; this only stops a broken import from looping
            content = sorted(f"{result.side[i].kind} {label(i)}" for i in ctx.scope_ids(scope_id))
            digests[scope_id] = hashlib.sha1("\n".join(content).encode()).hexdigest()[:12]
        return digests[scope_id]

    def text(sql: str, nested: list[str]) -> str:
        """`nested`: scope ids of the queries inside `sql`, consumed from the front in the order they are written."""
        try:
            tree = sqltext.parse_item(sql)
        except PrintError:
            return sql
        holder = exp.Paren(this=tree)
        for node in sqltext.top_level_queries(tree):
            if nested:
                node.replace(exp.select(exp.Literal.string(f"scope {digest(nested.pop(0))}")))
        tree = holder.this
        if isinstance(tree, exp.Alias) and sqltext.natural_name(tree.this) == sqltext.ident_key(tree.args["alias"]):
            tree = tree.this
        return sqltext.to_sql(sqltext.unquote_safe_identifiers(tree))

    def item(i: dict[str, Any], nested: list[str]) -> str:
        """A select item without an alias that only repeats the name the column has anyway: its own name for a plain
        column, and for an expression the name Postgres gives it (`COUNT(*) AS count`). The importer spells that
        default name out as soon as the query READS the column by name, which a printed `SELECT * FROM (...) AS z`
        does and the original `SELECT COUNT(*) FROM (...) z` does not."""
        sql, alias = text(i["sql"], nested), i.get("alias")
        try:
            tree = sqltext.parse_item(i["sql"])
            natural = sqltext.natural_name(tree)
            if natural is None:
                default = pg_default_name(tree)
                natural = None if default == NO_DEFAULT_NAME else sqltext.ident_key(default)
        except PrintError:
            natural = None
        return sql if not alias or natural == sqltext.ident_key(alias) else f"{sql} AS {alias}"

    def label(id_: str) -> str:
        if id_ not in labels:
            e = result.side[id_]
            d = e.data
            nested = [s for s in d.get("subqueries") or () if s in ctx.scopes]
            if e.kind == "scan":
                sql = f"{str(d.get('table')).lower()} AS {str(d.get('alias')).lower()}"
            elif e.kind == "conjunct":
                sql = text(e.sql, nested)
            elif e.kind == "aggregate":
                sql = "SELECT " + ", ".join(item(i, nested) for i in d.get("select") or ()) + " ... GROUP BY " + ", ".join(
                    text(g, nested) for g in d.get("group_by") or ())
            elif e.kind == "project":
                sql = "SELECT " + ", ".join(item(i, nested) for i in d.get("select") or ())
            elif e.kind == "order":
                sql = "ORDER BY " + ", ".join(
                    sqltext.to_sql(sqltext.parse_order_key(text(k["sql"], nested), k.get("desc"), k.get("nulls")))
                    for k in d.get("keys") or ())
            elif e.kind == "outer_join":  # `ON TRUE` is what the printer writes for an outer join without a condition
                sql = f"{str(d.get('join_type') or '').upper()} JOIN ON " + " AND ".join(
                    c for c in (text(c, nested) for c in d.get("on") or ()) if c.upper() != "TRUE")
            elif e.kind == "limit":
                sql = f"LIMIT {d.get('limit')} OFFSET {d.get('offset')}"
            elif e.kind == "distinct":  # DISTINCT ON keys name columns, which a dissolved block boundary renames back
                keys = [text(k, nested) for k in d.get("on") or ()]
                sql = "DISTINCT" + (f" ON ({', '.join(keys)})" if keys else "")
            elif e.kind == "setop":
                sql = str(d.get("op") or e.sql).upper()
            elif e.kind == "ref":  # the body of a derived table is compared through its own scope, never as text
                sql = f"{d.get('name') or '(derived table)'} AS {d.get('alias')}"
            else:
                sql = e.sql
            labels[id_] = sql
        return labels[id_]

    side = {id_: (e if e.kind == "scan" else SideEntry(id_, e.kind, label(id_), e.data)) for id_, e in result.side.items()}
    # The aligner proposes partners for a ref or a set operation only when the output lists below them agree, and it
    # compares their TEXT. A select item of a derived table or CTE that holds a subquery reads differently once a block
    # boundary lies inside that subquery, so the text of the body is masked here too; `_verify` compares the bodies.
    scopes = [replace(s, outputs=tuple(replace(o, sql=_without_bodies(o.sql)) for o in s.outputs)) for s in result.scopes]
    return replace(result, side=side, scopes=scopes)


# --------------------------------------------------------------------------------------------------------------
# C3
# --------------------------------------------------------------------------------------------------------------


def check_valid(sql: str, *, dbname: str = "beaver_dw") -> CheckResult:
    """C3: Postgres EXPLAIN accepts the SQL. Read-only; never modifies the database."""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=sqltext.DIALECT) if s is not None]
    except SqlglotError as e:
        return CheckResult(False, f"not parseable: {e}")
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation, exp.Subquery)):
        return CheckResult(False, "C3 takes exactly one query (SELECT, set operation or WITH ... SELECT)")
    if statements[0].find(exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Command,
                          exp.Into):
        return CheckResult(False, "C3 takes a query that only reads (no data-modifying CTE, no SELECT INTO)")
    psql = os.environ.get("SQLDECOMP_PSQL") or shutil.which("psql") or "/Library/PostgreSQL/16/bin/psql"
    if not (shutil.which(psql) or os.path.exists(psql)):
        return CheckResult(False, f"psql not found ({psql}); set SQLDECOMP_PSQL")
    # the ';' on a line of its own: a query that ends with a line comment (`... = 'x' -- note`) must not swallow it
    script = f"BEGIN TRANSACTION READ ONLY;\nEXPLAIN {sql.strip().rstrip(';')}\n;\nROLLBACK;\n"
    command = [psql, "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-h", os.environ.get("PGHOST", "localhost"),
               "-p", os.environ.get("PGPORT", "5432"), "-d", dbname, "-f", "-"]
    try:
        run = subprocess.run(command, input=script, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return CheckResult(False, f"psql did not run: {e}")
    if run.returncode != 0:
        return CheckResult(False, (run.stderr or run.stdout).strip())
    return CheckResult(True, "EXPLAIN accepted the query")
