"""Read-only indexes over one ImportResult that the printer needs again and again: which scope owns an id, the
written order of the FROM items, which aliases a plan makes visible, and which columns the rest of the query reads
from a plan (the columns a block boundary has to expose)."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterator, Union

from sqlglot import exp

from sqldecomp.model import Filter, ImportResult, Join, Op1, Op2, Plan, Scan, ScopeInfo, SideEntry
from sqldecomp.printer import sqltext
from sqldecomp.printer.errors import PrintError


@dataclass(frozen=True)
class BlockInput:
    """A leaf of a block fragment: the output of another block. `plan` is any plan of that group (all of them read the
    same tables and apply the same conjuncts), used to learn its aliases and the columns it has to expose."""

    group: str
    plan: Plan


PPlan = Union[Plan, BlockInput]  # a plan whose leaves may be block inputs

OUTER_JOIN_HIDES_RIGHT = ("SEMI", "ANTI")


def children(p: PPlan) -> tuple[PPlan, ...]:
    if isinstance(p, (Scan, BlockInput)):
        return ()
    if isinstance(p, (Filter, Op1)):
        return (p.input,)
    return (p.left, p.right)


def walk(p: PPlan) -> Iterator[PPlan]:
    yield p
    for c in children(p):
        yield from walk(c)


@dataclass(frozen=True)
class NeededColumn:
    """One column that the query reads from a plan: `column` is the reference as the side table writes it."""

    key: tuple[str, str]  # (alias key, column key), folded the way Postgres folds identifiers
    column: exp.Column


def boundary_prefix(sql: str) -> str:
    """What the name of a block starts with: `b_` (block g4 is `b_g4`), or the first of `b1_`, `b2_`, ... when the
    query has a name of its own that looks like a block's. A FROM alias `b_g4` next to a block boundary `b_g4` would
    be one name for two things."""
    text = sql.lower()
    for n in itertools.count():
        prefix = f"b{n or ''}_"
        if not re.search(rf"(?<![a-z0-9_$]){prefix}g\d", text):
            return prefix
    raise AssertionError("unreachable")


class PrintContext:
    def __init__(self, result: ImportResult):
        self.result = result
        self.boundary_prefix = boundary_prefix(result.sql)
        self.side = result.side
        self.side_pos = {id_: i for i, id_ in enumerate(result.side)}
        self.scopes: dict[str, ScopeInfo] = {s.id: s for s in result.scopes}
        self.scope_pos = {s.id: i for i, s in enumerate(result.scopes)}
        self._branch_plans = {s.plan: s.id for s in result.scopes if s.kind == "setop_branch"}
        self._edge_inputs = {e.conjunct: tuple(e.inputs) for e in result.facts.edges}
        self.owner: dict[str, str] = {}  # id of a scan, conjunct or operator -> id of the scope it belongs to
        self.scans: dict[str, Scan] = {}  # occurrence id -> the importer's own Scan node (its table name is not in the memo)
        self.written: dict[str, int] = {}  # id of a region input -> position in the written order of the query
        self._scope_ids: dict[str, frozenset[str]] = {}
        self._items: dict[str, list[tuple[str, exp.Expression]]] = {}
        self._alias_inputs: dict[tuple[str | None, str], Plan] | None = None
        self._quoted_names: set[str] | None = None
        self._keeps_written_case = False
        for s in result.scopes:
            claimed: list[str] = []
            self._claim(s.plan, s.id, claimed)
            self._scope_ids[s.id] = frozenset(claimed)

    # ---------------------------------------------------------------------------------------------------------
    # side table access
    # ---------------------------------------------------------------------------------------------------------

    def entry(self, id_: str, *kinds: str) -> SideEntry:
        e = self.side.get(id_)
        if e is None:
            raise PrintError(f"id {id_!r} has no side-table entry")
        if kinds and e.kind not in kinds:
            raise PrintError(f"side-table entry {id_!r} is a {e.kind!r}, expected {' or '.join(kinds)}")
        return e

    def kind(self, p: PPlan) -> str:
        """scan, join, filter, input, or the side-table kind of an operator (aggregate, ref, setop, outer_join, ...)."""
        if isinstance(p, Scan):
            return "scan"
        if isinstance(p, Join):
            return "join"
        if isinstance(p, Filter):
            return "filter"
        if isinstance(p, BlockInput):
            return "input"
        return self.entry(p.op).kind

    def by_written_order(self, conjuncts) -> list[str]:
        return sorted(conjuncts, key=lambda c: (self.side_pos.get(c, len(self.side_pos)), c))

    # ---------------------------------------------------------------------------------------------------------
    # scopes
    # ---------------------------------------------------------------------------------------------------------

    def _claim(self, p: Plan, sid: str, claimed: list[str]) -> None:
        """Ids of scope `sid`: everything reachable from its plan without crossing into a nested scope."""
        if isinstance(p, Scan):
            self.scans.setdefault(p.occ, p)
            self._own(p.occ, sid, claimed, region_input=True)
        elif isinstance(p, Join):
            for c in p.conjuncts:
                self._own(c, sid, claimed)
            self._claim(p.left, sid, claimed)
            self._claim(p.right, sid, claimed)
        elif isinstance(p, Filter):
            self._own(p.conjunct, sid, claimed)
            self._claim(p.input, sid, claimed)
        elif isinstance(p, Op1):
            kind = self.entry(p.op).kind
            self._own(p.op, sid, claimed, region_input=kind == "ref")
            if kind != "ref":  # below a ref starts the nested scope, which claims its own ids
                self._claim(p.input, sid, claimed)
        elif isinstance(p, Op2):
            kind = self.entry(p.op).kind
            self._own(p.op, sid, claimed, region_input=kind == "outer_join")
            for child in (p.left, p.right):
                if kind == "setop" and child in self._branch_plans:
                    continue  # a branch is a scope of its own
                self._claim(child, sid, claimed)
        else:  # pragma: no cover
            raise PrintError(f"not a plan node: {p!r}")

    def _own(self, id_: str, sid: str, claimed: list[str], region_input: bool = False) -> None:
        self.owner.setdefault(id_, sid)
        claimed.append(id_)
        if region_input:
            self.written.setdefault(id_, len(self.written))

    def ids(self, p: PPlan, *, through_inputs: bool = False) -> frozenset[str]:
        """Ids of a plan down to the boundaries of its scope: not below a ref or a set-operation branch, and not
        inside a block input unless `through_inputs` is set."""
        out: list[str] = []
        home = self.scope_of(p).id

        def visit(n: PPlan) -> None:
            if isinstance(n, BlockInput):
                if through_inputs:
                    visit(n.plan)
                return
            if isinstance(n, Scan):
                out.append(n.occ)
            elif isinstance(n, Join):
                out.extend(n.conjuncts)
                visit(n.left), visit(n.right)
            elif isinstance(n, Filter):
                out.append(n.conjunct)
                visit(n.input)
            elif isinstance(n, Op1):
                out.append(n.op)
                if self.entry(n.op).kind != "ref":
                    visit(n.input)
            else:
                out.append(n.op)
                for child in (n.left, n.right):
                    if self.scope_of(child).id == home:  # a set-operation branch is a scope of its own
                        visit(child)

        visit(p)
        return frozenset(out)

    def scope_ids(self, scope_id: str) -> frozenset[str]:
        """Ids of the scans, conjuncts and operators of one scope (nested scopes not included)."""
        return self._scope_ids.get(scope_id, frozenset())

    def scope_of(self, p: PPlan) -> ScopeInfo:
        """The scope whose SELECT a plan node is part of."""
        n = p
        while True:
            if isinstance(n, BlockInput):
                n = n.plan
            elif isinstance(n, Scan):
                id_ = n.occ
                break
            elif isinstance(n, Filter):
                id_ = n.conjunct
                break
            elif isinstance(n, (Op1, Op2)):
                id_ = n.op
                break
            elif n.conjuncts:
                id_ = n.conjuncts[0]
                break
            else:
                n = n.left
        sid = self.owner.get(id_)
        if sid is None:
            raise PrintError(f"id {id_!r} does not belong to any scope of the import result")
        return self.scopes[sid]

    def is_scope_top(self, p: PPlan) -> bool:
        """True when `p` computes everything its scope computes: it applies every operator and conjunct of the scope,
        in whatever order. The select list of such a plan is the select list of the scope."""
        return self.ids(p, through_inputs=True) == self._scope_ids[self.scope_of(p).id]

    def outer_input(self, p: PPlan, alias: str) -> Plan:
        """The FROM item of an enclosing scope that a correlated reference `alias.column` inside `p` refers to."""
        if self._alias_inputs is None:
            self._alias_inputs = {}
            for i in self.result.facts.inputs:
                if isinstance(i.plan, Scan):
                    key = sqltext.ident_key(self.scan_alias(i.plan))
                elif isinstance(i.plan, Op1) and self.entry(i.plan.op).kind == "ref":
                    key = sqltext.ident_key(self.ref_alias(self.entry(i.plan.op)))
                else:
                    continue
                self._alias_inputs.setdefault((self.owner.get(i.id), key), i.plan)
        sid = self.scope_of(p).parent
        while sid is not None and sid in self.scopes:
            found = self._alias_inputs.get((sid, alias))
            if found is not None:
                return found
            sid = self.scopes[sid].parent
        raise PrintError(f"alias {alias!r} is read inside scope {self.scope_of(p).id!r} but no enclosing scope defines it")

    def explicit_output_names(self, scope: ScopeInfo) -> list[str] | None:
        """The output column names of a scope when the query itself spells every one of them (a plain column or an
        alias); None when Postgres would have to invent one (`SELECT AVG(x)`), or for a set operation."""
        plan = scope.plan
        while isinstance(plan, Op1) and self.entry(plan.op).kind in ("limit", "order", "distinct"):
            plan = plan.input
        while isinstance(plan, Filter):
            plan = plan.input
        if isinstance(plan, Op2) and self.entry(plan.op).kind == "setop":
            return None
        if isinstance(plan, Op1) and self.entry(plan.op).kind in ("project", "aggregate"):
            names = []
            for i in self.entry(plan.op).data.get("select") or ():
                item = sqltext.parse_item(i["sql"])
                name = i.get("alias") or (item.name if isinstance(item, (exp.Column, exp.Alias)) else None)
                if not name or name == "*":
                    return None
                names.append(name)
            return names
        names = [o.name for o in scope.outputs]
        return names if all(n and n != "?column?" for n in names) and len(set(names)) == len(names) else None

    def names_every_output(self, scope: ScopeInfo) -> bool:
        """Whether the query itself names every output column of a scope: a plain column or an alias, in the first
        branch for a set operation."""
        plan = scope.plan
        while isinstance(plan, (Op1, Filter)) and (isinstance(plan, Filter) or self.entry(plan.op).kind in (
                "limit", "order", "distinct")):
            plan = plan.input
        if isinstance(plan, Op2) and self.entry(plan.op).kind == "setop":
            first = self._branch_plans.get(plan.left)
            return first is not None and self.names_every_output(self.scopes[first])
        return self.explicit_output_names(scope) is not None

    def star_reads_an_unnamed_column(self, p: PPlan) -> bool:
        """`SELECT *` over the FROM items visible above `p` would read a column of a derived table or CTE that has no
        name of its own. Postgres derives one from the expression and the importer invents another: once a statement
        READS such a column the invented name has to be written into the body, which the original never did."""
        def visit(n: PPlan) -> bool:
            if isinstance(n, BlockInput):
                return visit(n.plan)
            if isinstance(n, Scan):
                return False
            if isinstance(n, (Join, Filter)):
                return any(visit(c) for c in children(n))
            e = self.entry(n.op)
            if isinstance(n, Op1):
                return e.kind == "ref" and not self.names_every_output(self.nested_scope(e))
            if e.kind != "outer_join":
                return False
            hides_right = str(e.data.get("join_type", "")).upper() in OUTER_JOIN_HIDES_RIGHT
            return visit(n.left) or (not hides_right and visit(n.right))
        return visit(p)

    def nested_scope(self, ref: SideEntry) -> ScopeInfo:
        sid = ref.data.get("scope")
        if sid not in self.scopes:
            raise PrintError(f"ref {ref.id!r} names scope {sid!r}, which is not in ImportResult.scopes")
        return self.scopes[sid]

    # ---------------------------------------------------------------------------------------------------------
    # regions
    # ---------------------------------------------------------------------------------------------------------

    def is_region_level(self, p: PPlan) -> bool:
        """True for a plan that lives in a FROM clause: region inputs (scan, ref, outer join) and the joins and WHERE
        filters over them. False for the operators of a select list pipeline (aggregate, HAVING filter, projection,
        distinct, order, limit) and for a set operation."""
        if isinstance(p, BlockInput):
            return self.is_region_level(p.plan)
        while isinstance(p, Filter):
            p = p.input
        if isinstance(p, (Scan, Join)):
            return True
        return self.entry(p.op).kind in ("ref", "outer_join")

    def region_input_ids(self, p: PPlan) -> frozenset[str]:
        """Ids of the region inputs that a region-level plan covers (the ids that Facts.edges talk about)."""
        if isinstance(p, BlockInput):
            return self.region_input_ids(p.plan)
        if isinstance(p, Scan):
            return frozenset({p.occ})
        if isinstance(p, (Op1, Op2)):
            return frozenset({p.op})
        out: frozenset[str] = frozenset()
        for c in children(p):
            out |= self.region_input_ids(c)
        return out

    def written_pos(self, p: PPlan) -> int:
        ids = self.region_input_ids(p)
        return min((self.written.get(i, len(self.written)) for i in ids), default=len(self.written))

    def conjunct_inputs(self, conjunct: str) -> tuple[str, ...]:
        """Region inputs that a join conjunct references: Facts.edges, else the `refs` of its side entry."""
        if conjunct in self._edge_inputs:
            return self._edge_inputs[conjunct]
        return tuple(self.entry(conjunct).data.get("refs") or ())

    def visible_aliases(self, p: PPlan) -> dict[str, exp.Identifier]:
        """Alias key -> alias identifier, for every FROM alias whose columns can be read above `p`, in the written order
        of the FROM items (never in the order of the plan, which differs between equivalent plans)."""
        found: list[tuple[int, exp.Identifier]] = []

        def visit(n: PPlan) -> None:
            if isinstance(n, BlockInput):
                visit(n.plan)
            elif isinstance(n, Scan):
                if not self.is_dual(n):
                    found.append((self.written.get(n.occ, len(self.written)), self.scan_alias(n)))
            elif isinstance(n, (Join, Filter)):
                for c in children(n):
                    visit(c)
            elif isinstance(n, Op1):
                e = self.entry(n.op)
                if e.kind == "ref":
                    found.append((self.written.get(n.op, len(self.written)), self.ref_alias(e)))
                # any other unary operator closes its SELECT: nothing below it is visible by alias
            else:
                e = self.entry(n.op)
                if e.kind == "outer_join":
                    visit(n.left)
                    if str(e.data.get("join_type", "")).upper() not in OUTER_JOIN_HIDES_RIGHT:
                        visit(n.right)

        visit(p)
        out: dict[str, exp.Identifier] = {}
        for _, alias in sorted(found, key=lambda f: f[0]):
            out.setdefault(sqltext.ident_key(alias), alias)
        return out

    def is_dual(self, p: PPlan) -> bool:
        """A SELECT without FROM. A plan leaf has to be a Scan, so an importer stands in a pseudo table for the missing
        FROM clause: a scan whose side entry has neither SQL nor a table (the importer's `$dual`)."""
        if not isinstance(p, Scan):
            return False
        e = self.entry(p.occ, "scan")
        return not e.sql and not e.data.get("table")

    def scan_alias(self, scan: Scan) -> exp.Identifier:
        item = self.scan_item(scan)
        alias = item.args.get("alias")
        if alias is not None and alias.this is not None:
            return alias.this
        return item.this if isinstance(item.this, exp.Identifier) else sqltext.make_ident(item.name)

    def scan_item(self, scan: Scan) -> exp.Expression:
        """The FROM item of a scan, as the side table writes it, minus an alias that only repeats the table name
        (`FROM orders AS orders`, which is what qualification makes of `FROM orders`)."""
        e = self.entry(scan.occ, "scan")
        if e.sql:
            item = sqltext.parse_from_item(e.sql)
        else:
            item = exp.to_table(e.data.get("table") or scan.table, dialect=sqltext.DIALECT)
            if e.data.get("alias"):
                item = exp.alias_(item, self.identifier(e.data["alias"]), table=True, copy=False)
        alias = item.args.get("alias")
        if isinstance(item, exp.Table) and alias is not None and alias.this is not None and not alias.args.get("columns") \
                and isinstance(item.this, exp.Identifier) and sqltext.ident_key(alias.this) == sqltext.ident_key(item.this):
            item.set("alias", None)
        return item

    def ref_alias(self, e: SideEntry) -> exp.Identifier:
        alias = e.data.get("alias") or e.data.get("name")
        if not alias:
            raise PrintError(f"ref {e.id!r} has neither an alias nor a name")
        return self.identifier(alias)

    def identifier(self, name: str) -> exp.Identifier:
        """An alias, CTE name or output name taken from a `data` field of the side table. Such names are stored bare,
        so whether they were quoted is lost: `Ord` may be the identifier "Ord" or a spelling of ord. The SQL of the
        side table knows. A name that it writes in quotes somewhere (and that needs them) is quoted here as well. And
        when that SQL has no unquoted identifier with an upper-case letter, the importer folds unquoted identifiers the
        way Postgres does, so a bare name with an upper-case letter can only come from a quoted identifier. When the
        SQL does keep the written case (BEAVER is unquoted upper case), upper case alone never adds quotes."""
        if name.startswith('"') and name.endswith('"') and len(name) > 1:
            return exp.Identifier(this=name[1:-1].replace('""', '"'), quoted=True)
        if self._quoted_names is None:
            self._quoted_names = set()
            self._keeps_written_case = False
            for e in self.side.values():
                for sql in ([e.sql] if e.kind in ("scan", "conjunct") else []) + self.entry_sqls(e):
                    try:
                        tree = sqltext.parse_from_item(sql) if e.kind == "scan" else sqltext.parse_item(sql)
                    except PrintError:
                        continue
                    for i in tree.find_all(exp.Identifier):
                        if not i.args.get("quoted"):
                            self._keeps_written_case |= i.name != i.name.lower()
                        elif not sqltext.is_safe_unquoted(i.name):
                            self._quoted_names.add(i.name)
        if name in self._quoted_names or (name != name.lower() and not self._keeps_written_case):
            return exp.Identifier(this=name, quoted=True)
        return sqltext.make_ident(name)

    # ---------------------------------------------------------------------------------------------------------
    # SQL-bearing items of a scope, and the columns that the rest of the query reads from a plan
    # ---------------------------------------------------------------------------------------------------------

    def entry_sqls(self, e: SideEntry) -> list[str]:
        """Every SQL text of a side entry that can mention columns."""
        d = e.data
        if e.kind == "conjunct":
            return [e.sql]
        if e.kind == "aggregate":
            return [i["sql"] for i in d.get("select") or ()] + list(d.get("group_by") or ())
        if e.kind == "project":
            return [i["sql"] for i in d.get("select") or ()]
        if e.kind == "order":
            return [k["sql"] for k in d.get("keys") or ()]
        if e.kind == "outer_join":
            return list(d.get("on") or ())
        if e.kind == "limit":
            return [x for x in (d.get("limit"), d.get("offset")) if isinstance(x, str)]
        if e.kind == "distinct":
            return list(d.get("on") or ())
        return []

    def entry_trees(self, e: SideEntry) -> list[exp.Expression]:
        """The same texts parsed. A GROUP BY item is parsed as one (`ROLLUP (...)`, `CUBE (...)` and
        `GROUPING SETS (...)` are no expressions on their own)."""
        if e.kind == "aggregate":
            d = e.data
            return [sqltext.parse_item(i["sql"]) for i in d.get("select") or ()] + \
                   [sqltext.parse_group_item(g) for g in d.get("group_by") or ()]
        return [sqltext.parse_item(sql) for sql in self.entry_sqls(e)]

    def scope_items(self, sid: str) -> list[tuple[str, exp.Expression]]:
        """(id, parsed SQL) for everything scope `sid` evaluates. What defines the select list comes first (aggregate,
        projection, and the output columns under the pseudo id `outputs@<sid>`), then conjuncts, join conditions and
        sort keys in side-table order. `needed_columns` lists columns in order of first use, so a fragment selects
        the columns the query outputs before the columns that only a later condition reads."""
        if sid not in self._items:
            first: list[tuple[str, exp.Expression]] = []
            then: list[tuple[str, exp.Expression]] = []
            for id_ in sorted(self._scope_ids[sid], key=lambda i: self.side_pos.get(i, 0)):
                e = self.entry(id_)
                for tree in self.entry_trees(e):
                    (first if e.kind in ("aggregate", "project") else then).append((id_, tree))
            for o in self.scopes[sid].outputs:
                first.append((f"outputs@{sid}", sqltext.parse_item(o.sql)))
            self._items[sid] = first + then
        return self._items[sid]

    def needed_columns(self, p: PPlan) -> list[NeededColumn]:
        """Columns of the aliases visible above `p` that the query reads OUTSIDE `p`: what a block that computes `p`
        has to expose. Ordered by the FROM position of the alias, then by first use. The answer depends only on the
        group of `p`, never on the decomposition."""
        visible = self.visible_aliases(p)
        if not visible:
            return []
        inside = self.ids(p, through_inputs=True)
        order = {key: i for i, key in enumerate(visible)}
        found: dict[tuple[str, str], NeededColumn] = {}
        for id_, tree in self.scope_items(self.scope_of(p).id):
            if id_ in inside:
                continue
            for column in sqltext.outer_columns(tree):
                key = sqltext.column_key(column)
                if key[0] not in visible:
                    continue
                if key[1] == "*":
                    raise PrintError(f"{sqltext.to_sql(column)} is read across a block boundary; the importer has to "
                                     "expand stars")
                found.setdefault(key, NeededColumn(key, column))
        return sorted(found.values(), key=lambda n: order[n.key[0]])  # stable: first use breaks ties
