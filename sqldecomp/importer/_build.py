"""Step 2 of the import: one skeleton plan per scope, the side table and the facts.

Vocabulary used below.

region      A tree of inner joins over its inputs, plus the conjuncts collected for it (ON clauses of its inner joins;
            for the top region of a SELECT also the WHERE clause). A region is *closed* when its conjuncts are
            classified: join conjuncts go onto the lowest join that covers their inputs, the others become Filters
            stacked on top. An outer / semi / anti join closes the regions on both of its sides and the resulting Op2
            is ONE input of the region that continues above it.
input       A leaf of a region: a Scan, a `ref` Op1 (one use of a CTE or derived table), an outer-join Op2, or the
            aggregate Op1 under its HAVING filters. `alias_map` of a region says which input exposes a FROM alias.
occurrence  One FROM item: a Scan (id = occ) or a `ref` (id = operator id).
column      Facts name a column by the region input that EXPOSES it where the fact lives: ColumnRef(occurrence id, name)
            for a FROM item that is an input itself, ColumnRef(outer-join id, name) for a column that is only visible
            through an outer join (it may be NULL-extended there, so it is not the base column any more; when both
            sides expose the name there is no fact). Above an aggregate / project / ref: ColumnRef(operator id, name).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlglot import exp
from sqlglot.optimizer.scope import Scope, build_scope

from sqldecomp.importer._errors import UnsupportedSQL
from sqldecomp.importer._names import pg_default_name
from sqldecomp.importer._prepare import COMMA_JOIN, OUTPUT_NAME, is_query, unparen
from sqldecomp.model import (ColumnRef, Edge, Equate, Facts, Filter, ImportResult, Join, Keep, Op1, Op2, OutputColumn,
                             Plan, RegionInput, Scan, Schema, ScopeInfo, SideEntry)

DUAL_TABLE = "$dual"  # Scan.table of the one-row input of a SELECT without FROM
VALUES_TABLE = "$values"  # Scan.table of a VALUES list in FROM

# the names that other ids of a scope are made of: conjuncts `p<n>@<scope>`, operators `<kind>@<scope>`, the pseudo
# entry `outputs@<scope>` of a scope's output list
_ID_NAMES = re.compile(r"p\d+|aggregate|project|distinct|order|limit|setop|outer_join|ref|outputs|scan|conjunct")


def occurrence_id(alias: str, sid: str) -> str:
    """`<alias>@<scope>`, the id of a FROM item; an alias that would read as another id of the scope (a table aliased
    `p2` beside the conjunct `p2@0`) gets a trailing `'`, and so does an alias that already ends with one, so that no
    two FROM items and no FROM item and another entry ever share an id. Every other id is unchanged."""
    return f"{alias}'@{sid}" if _ID_NAMES.fullmatch(alias) or alias.endswith("'") else f"{alias}@{sid}"


# ----------------------------------------------------------------------------------------------------------------------
# working data
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class _Source:
    """One FROM item of a scope."""

    alias: str
    id: str  # occurrence id: Scan.occ, or the id of the `ref` Op1
    columns: frozenset[str] | None = None  # the names it exposes; None when they are not known (VALUES without names)
    table: str | None = None  # the schema table of a base-table FROM item; None for a CTE, a derived table, VALUES


@dataclass
class _Ctx:
    """One scope while it is being built."""

    sid: str
    sg: Scope
    parent: "_Ctx | None"  # the enclosing scope whose FROM items this one may reference (correlation), if any
    sources: dict[str, _Source] = field(default_factory=dict)
    free: list[tuple["_Ctx", _Source]] = field(default_factory=list)  # references that leave this scope
    counts: dict[str, int] = field(default_factory=dict)
    nconj: int = 0
    from_entries: list[SideEntry] = field(default_factory=list)
    conj_entries: list[SideEntry] = field(default_factory=list)
    op_entries: list[SideEntry] = field(default_factory=list)
    subscopes: dict[int, Scope] | None = None

    def op_id(self, kind: str) -> str:
        k = self.counts[kind] = self.counts.get(kind, 0) + 1
        return f"{kind}@{self.sid}" if k == 1 else f"{kind}@{self.sid}#{k}"

    def conj_id(self) -> str:
        self.nconj += 1
        return f"p{self.nconj - 1}@{self.sid}"

    def add_free(self, octx: "_Ctx", src: _Source) -> None:
        if not any(o is octx and s is src for o, s in self.free):
            self.free.append((octx, src))


@dataclass
class _Tree:
    """The join tree of a region as written. A leaf has `input_id` and `plan`; a join has `left` and `right`."""

    input_id: str | None = None
    plan: Plan | None = None
    left: "_Tree | None" = None
    right: "_Tree | None" = None
    conjuncts: list[str] = field(default_factory=list)

    def ids(self) -> frozenset[str]:
        if self.input_id is not None:
            return frozenset({self.input_id})
        return self.left.ids() | self.right.ids()


@dataclass
class _Region:
    tree: _Tree
    alias_map: dict[str, str]  # FROM alias -> id of the region input that exposes it
    pool: list[tuple[exp.Expr, str]] = field(default_factory=list)  # (conjunct, source) in the order written


@dataclass
class _Item:
    """One select item: its expression without the alias, the alias the side table shows, and the output name."""

    expr: exp.Expr
    sql: str
    alias: str | None
    name: str


@dataclass
class _Out:
    name: str
    sql: str
    identity: ColumnRef | None  # the column at the top of the scope's plan that this output copies, when there is one


@dataclass
class _Built:
    sid: str
    plan: Plan
    outputs: list[_Out]
    free: list[tuple[_Ctx, _Source]]
    sg: Scope  # keeps the SQLGlot scope alive: the cache below is keyed by id(scope)


def _conjuncts(e: exp.Expr):
    e = unparen(e)
    if isinstance(e, exp.And):
        yield from _conjuncts(e.this)
        yield from _conjuncts(e.expression)
    else:
        yield e


def _unique(names: list[str]) -> set[str]:
    return {n for n in names if names.count(n) == 1}


# ----------------------------------------------------------------------------------------------------------------------
# the builder
# ----------------------------------------------------------------------------------------------------------------------


class Builder:
    def __init__(self, sql: str, tree: exp.Expr, schema: Schema, dialect: str):
        self.sql_text = sql
        self.tree = tree
        self.tables = {t.name: frozenset(c.name for c in t.columns) for t in schema.tables}
        self.primary_keys = {t.name: t.primary_key for t in schema.tables if t.primary_key}
        self.dialect = dialect
        self.facts = Facts()
        self.ctxs: list[_Ctx] = []
        self.infos: dict[str, ScopeInfo] = {}
        self.built: dict[int, _Built] = {}
        self.ctx_of: dict[int, _Ctx] = {}

    def run(self) -> ImportResult:
        root = build_scope(self.tree)
        if root is None:
            raise UnsupportedSQL("not a query", "no SELECT found")
        top = self._scope(root, "root", None, None, None)
        side: dict[str, SideEntry] = {}
        for ctx in self.ctxs:
            ctx.conj_entries.sort(key=lambda e: int(e.id[1:e.id.index("@")]))
            for entry in ctx.from_entries + ctx.conj_entries + ctx.op_entries:
                side[entry.id] = entry
        scopes = [self.infos[c.sid] for c in self.ctxs]
        return ImportResult(self.sql_text, top.plan, scopes, side, self.facts, self.dialect)

    def _sql(self, e: exp.Expr) -> str:
        return e.sql(dialect=self.dialect, comments=False)

    def _conj_sql(self, e: exp.Expr) -> str:
        # A conjunct is an operand of AND; OR is the only connective that binds weaker, so it keeps its parentheses.
        return f"({self._sql(e)})" if isinstance(e, exp.Or) else self._sql(e)

    # ------------------------------------------------------------------------------------------------------------------
    # scopes
    # ------------------------------------------------------------------------------------------------------------------

    def _scope(self, sg: Scope, kind: str, name: str | None, parent_sid: str | None, lexical: _Ctx | None,
               rows_depend_on_columns: bool = False) -> _Built:
        """Build the scope once. A CTE that is referenced again gets the SAME _Built: one plan value, many refs.
        `rows_depend_on_columns`: the scope is a branch of a set operation that compares whole rows."""
        done = self.built.get(id(sg))
        if done is not None:
            return done
        ctx = _Ctx(str(len(self.ctxs)), sg, lexical)
        self.ctxs.append(ctx)
        self.ctx_of[id(sg)] = ctx
        expr = sg.expression
        if isinstance(expr, exp.Select):
            plan, outputs = self._select(expr, ctx, rows_depend_on_columns)
        elif isinstance(expr, exp.SetOperation):
            plan, outputs = self._setop(expr, ctx, rows_depend_on_columns)
        else:
            raise UnsupportedSQL("unsupported query form", f"{expr.key.upper()} where a SELECT is expected")
        for sub in sg.subquery_scopes:
            if id(sub) not in self.built:
                raise UnsupportedSQL("subquery in an unsupported position", self._sql(sub.expression)[:120])
        self.infos[ctx.sid] = ScopeInfo(ctx.sid, kind, name, parent_sid,
                                        tuple(OutputColumn(o.name, o.sql) for o in outputs), plan)
        done = self.built[id(sg)] = _Built(ctx.sid, plan, outputs, ctx.free, sg)
        return done

    def _setop(self, expr: exp.SetOperation, ctx: _Ctx, rows_depend_on_columns: bool) -> tuple[Plan, list[_Out]]:
        name = {exp.Union: "UNION", exp.Intersect: "INTERSECT", exp.Except: "EXCEPT"}.get(type(expr))
        if name is None or expr.args.get("by_name") or expr.args.get("side") or expr.args.get("kind"):
            raise UnsupportedSQL("unsupported set operation", expr.key.upper())
        op = name if expr.args.get("distinct") else f"{name} ALL"
        # Everything but UNION ALL compares whole rows, so WHICH columns a branch returns decides which rows survive.
        compares_rows = rows_depend_on_columns or op != "UNION ALL"
        branches = []
        for node in (expr.this, expr.expression):
            while isinstance(node, (exp.Subquery, exp.Paren)) and not node.alias:
                if any(node.args.get(k) for k in ("order", "limit", "offset")):
                    raise UnsupportedSQL("modifiers on a parenthesized query", "ORDER BY / LIMIT outside the parentheses")
                node = node.this
            sg = next((s for s in ctx.sg.union_scopes if s.expression is node), None)
            if sg is None:
                raise UnsupportedSQL("unsupported set operation branch", f"{node.key.upper()}")
            branch = self._scope(sg, "setop_branch", None, ctx.sid, ctx, compares_rows)
            for octx, src in branch.free:
                ctx.add_free(octx, src)
            branches.append(branch)
        op_id = ctx.op_id("setop")
        ctx.op_entries.append(SideEntry(op_id, "setop", op, {"op": op}))
        plan: Plan = Op2(op_id, branches[0].plan, branches[1].plan)
        # The output names of a set operation are those of its first branch; nothing is a copy of one column.
        outputs = [_Out(o.name, self._sql(exp.to_identifier(o.name)), None) for o in branches[0].outputs]
        return self._limit(expr, self._order(expr, plan, ctx, {}, outputs), ctx, {}), outputs

    def _subquery(self, node: exp.Expr, ctx: _Ctx) -> _Built:
        if ctx.subscopes is None:
            ctx.subscopes = {id(s.expression): s for s in ctx.sg.subquery_scopes}
        sg = ctx.subscopes.get(id(node))
        if sg is None:
            raise UnsupportedSQL("subquery in an unsupported position", self._sql(node)[:120])
        return self._scope(sg, "subquery", None, ctx.sid, ctx)

    # ------------------------------------------------------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------------------------------------------------------

    def _select(self, select: exp.Select, ctx: _Ctx, rows_depend_on_columns: bool) -> tuple[Plan, list[_Out]]:
        region = self._from(select, ctx)
        where = select.args.get("where")
        if where is not None:
            region.pool += [(c, "where") for c in _conjuncts(where.this)]
        visible = dict(region.alias_map)  # every alias the clauses after FROM can see, and the input that exposes it
        plan = self._close(region, ctx)

        items = self._items(select)
        group = select.args.get("group")
        having = select.args.get("having")
        order = select.args.get("order")
        distinct = select.args.get("distinct")
        distinct_on = distinct is not None and distinct.args.get("on") is not None
        beyond = [i.expr for i in items] + ([having.this] if having is not None else []) \
            + ([o.this for o in order.expressions] if order else [])
        producer: str | None = None  # the operator whose output columns the scope returns, if there is one
        if group is not None or having is not None or any(_has_aggregate(e) for e in beyond):
            agg_items = [i for i in items if not _has_window(i.expr)]
            agg_id = producer = ctx.op_id("aggregate")
            keys = ([o.this for o in order.expressions] if order else []) + _distinct_on_exprs(distinct)
            implied = self._implied_grouping(group, [i.expr for i in items], having, keys, ctx, visible)
            plan = self._aggregate(agg_id, plan, agg_items, group, ctx, visible, implied)
            if having is not None:
                plan = self._having(agg_id, plan, having, ctx, visible)
            if len(agg_items) < len(items):  # window functions are evaluated after HAVING: a projection of its own
                producer, plan = self._project(plan, items, ctx, visible, (agg_id, agg_items))
        elif any(not isinstance(i.expr, exp.Column) for i in items) or (distinct is not None and not distinct_on) \
                or rows_depend_on_columns:
            # A select list that only drops or renames columns is normally not a node. Under DISTINCT, or in a branch
            # of a row-comparing set operation, WHICH columns are listed decides which rows come out: a node.
            producer, plan = self._project(plan, items, ctx, visible)

        unique = _unique([i.name for i in items])
        outputs = []
        for i in items:
            identity = None
            if i.name in unique:
                identity = ColumnRef(producer, i.name) if producer is not None else self._column(i.expr, ctx, visible)
            outputs.append(_Out(i.name, i.sql, identity))

        if distinct_on:  # Postgres keeps the FIRST row of each group in ORDER BY order: sort, then DISTINCT ON
            plan = self._distinct(distinct, self._order(select, plan, ctx, visible), items, outputs, ctx, visible)
        else:
            if distinct is not None:
                plan = self._distinct(distinct, plan, items, outputs, ctx, visible)
            plan = self._order(select, plan, ctx, visible)
        return self._limit(select, plan, ctx, visible), outputs

    def _distinct(self, distinct: exp.Distinct, plan: Plan, items: list[_Item], outputs: list[_Out], ctx: _Ctx,
                  visible: dict[str, str]) -> Plan:
        """`key` says which columns end up unique together: every output, or the DISTINCT ON expressions (when each
        of them is one of the select items)."""
        op_id = ctx.op_id("distinct")
        on_exprs = _distinct_on_exprs(distinct)
        for e in on_exprs:
            self._input_names(e, ctx, visible)
        data: dict = {"on": [self._sql(e) for e in on_exprs]} if on_exprs else {}
        if on_exprs:
            by_sql = {i.sql: o.identity for i, o in zip(items, outputs)}
            key = [by_sql.get(self._sql(e)) for e in on_exprs]
        else:
            key = [o.identity for o in outputs]
        if key and all(k is not None for k in key):
            data["key"] = [[k.origin, k.column] for k in key]
        self._beyond(data, on_exprs, ctx, visible)
        sql = "DISTINCT ON (" + ", ".join(data["on"]) + ")" if on_exprs else "DISTINCT"
        ctx.op_entries.append(SideEntry(op_id, "distinct", sql, data))
        return Op1(plan, op_id)

    def _items(self, select: exp.Select) -> list[_Item]:
        items = []
        for item in select.selects:
            if isinstance(item, exp.Alias):
                expr, alias, name = item.this, item.alias, item.alias
            else:  # `_prepare` dropped the alias that qualify had added and left the output name on the item
                expr, alias, name = item, None, item.meta.get(OUTPUT_NAME) or pg_default_name(item, self.dialect)
            items.append(_Item(expr, self._sql(expr), alias, name))
        return items

    def _select_data(self, items: list[_Item]) -> list[dict]:
        return [{"sql": i.sql, "alias": i.alias} for i in items]

    def _select_sql(self, items: list[_Item]) -> str:
        return "SELECT " + ", ".join(i.sql if i.alias is None else f"{i.sql} AS {self._sql(exp.to_identifier(i.alias))}"
                                     for i in items)

    def _column(self, e: exp.Expr, ctx: _Ctx, alias_map: dict[str, str]) -> ColumnRef | None:
        """The ColumnRef of `e` if it is a plain column of one of this scope's own FROM items, named by the region
        input that exposes it (see the module docstring); None when it is anything else, or when we are not sure."""
        e = unparen(e)
        if not (isinstance(e, exp.Column) and e.table in ctx.sources and e.table in alias_map):
            return None
        source, input_id = ctx.sources[e.table], alias_map[e.table]
        if input_id == source.id:
            return ColumnRef(source.id, e.name)
        # Visible only through an outer join. Name the join, unless another FROM item behind it may expose the name too.
        behind = [ctx.sources[a] for a, i in alias_map.items() if i == input_id]
        if sum(1 for s in behind if s.columns is None or e.name in s.columns) == 1:
            return ColumnRef(input_id, e.name)
        return None

    def _keeps(self, op_id: str, items: list[_Item], source_of) -> None:
        unique = _unique([i.name for i in items])
        for i in items:
            source = source_of(i) if i.name in unique else None
            if source is not None:
                self.facts.keeps.append(Keep(op_id, i.name, source))

    def _beyond(self, data: dict, exprs: list[exp.Expr], ctx: _Ctx, alias_map: dict[str, str]) -> None:
        """Record what the expressions of an operator reach outside of it: outer occurrences and subquery scopes."""
        outer: list[str] = []
        subs: list[str] = []
        for e in exprs:
            self._analyze(e, ctx, alias_map, [], outer, subs)
        if outer:
            data["outer_refs"] = outer
        if subs:
            data["subqueries"] = subs

    def _aggregate(self, op_id: str, plan: Plan, items: list[_Item], group: exp.Group | None, ctx: _Ctx,
                   visible: dict[str, str], implied: list[exp.Column]) -> Plan:
        """`implied`: the columns the scope reads by functional dependency (`_implied_grouping`); they are grouped by
        after the GROUP BY items as written, before any ROLLUP / CUBE / GROUPING SETS."""
        group_exprs: list[exp.Expr] = []
        plain = True  # plain GROUP BY: no ROLLUP / CUBE / GROUPING SETS, which put NULLs into grouping columns
        if group is not None:
            group_exprs += group.expressions
            group_exprs += implied
            for key in ("rollup", "cube", "grouping_sets"):
                extra = group.args.get(key)
                if extra:
                    group_exprs += extra if isinstance(extra, list) else [extra]
                    plain = False
            if group.args.get("all") or group.args.get("totals") or any(isinstance(g, exp.Tuple) for g in group_exprs):
                plain = False  # GROUP BY () is a grouping set too
        group_sql = [self._sql(g) for g in group_exprs]
        data = {"group_by": group_sql, "select": self._select_data(items)}
        self._beyond(data, [i.expr for i in items] + group_exprs, ctx, visible)
        sql = self._select_sql(items) + " ..." + (" GROUP BY " + ", ".join(group_sql) if group_sql else "")
        ctx.op_entries.append(SideEntry(op_id, "aggregate", sql, data))
        if plain:
            cols = [self._column(g, ctx, visible) for g in group_exprs]
            if all(c is not None for c in cols):  # no GROUP BY at all: one group, the empty list
                self.facts.group_cols[op_id] = cols
            self._keeps(op_id, items, lambda i: self._column(i.expr, ctx, visible))
        return Op1(plan, op_id)

    def _implied_grouping(self, group: exp.Group | None, exprs: list[exp.Expr], having: exp.Having | None,
                          keys: list[exp.Expr], ctx: _Ctx, visible: dict[str, str]) -> list[exp.Column]:
        """The columns a grouped SELECT reads without grouping or aggregating them, legal only because its GROUP BY
        holds the PRIMARY KEY of their table: a functional dependency. Postgres, like the SQL standard, accepts
        `SELECT p.id, p.status, COUNT(*) FROM ports p ... GROUP BY p.id` when `id` is the primary key of `ports`; it
        honours a primary key only (no UNIQUE constraint) of a base table only (no CTE, no derived table), with every
        key column a plain grouping item that all grouping sets share. A block boundary below the aggregate hides the
        key: the block is a derived relation without one, so Postgres rejects the block and the recomposition. The
        aggregate therefore groups by these columns as well, after the GROUP BY items as written (`GROUP BY p.id,
        p.status`): the groups are the same, since the key determines the columns.

        What is looked at is what Postgres evaluates after grouping: the select list (`exprs`, window functions
        included), HAVING and the ORDER BY and DISTINCT ON keys (`keys`, where a bare name standing alone names an
        output and one inside an expression an input column, as `_input_names` reads it), with the outer references
        of a subquery there; never the inside of an aggregate call or of a part that is a GROUP BY item. The columns
        come in the order the select list, HAVING and the keys read them, each once. [] for every SELECT that does
        not rely on the dependency: its import is unchanged."""
        if group is None or not self.primary_keys:
            return []

        def flat(nodes) -> list[exp.Expr]:
            out: list[exp.Expr] = []
            for n in nodes:
                n = unparen(n)
                out += flat(n.expressions) if isinstance(n, (exp.Tuple, exp.Rollup, exp.Cube, exp.GroupingSets)) else [n]
            return out

        def columns(nodes) -> set[tuple[str, str]]:
            return {(n.table, n.name) for n in nodes if isinstance(n, exp.Column) and n.table in ctx.sources}

        # the items every grouping set holds: those outside ROLLUP / CUBE / GROUPING SETS (Postgres reads the
        # implicit row `GROUP BY (a, b)` as `GROUP BY a, b`; an explicit ROW (a, b) is one value, no column of it)
        common = columns(flat(group.expressions))
        determined = {alias for alias, src in ctx.sources.items() if src.table in self.primary_keys
                      and all((alias, c) in common for c in self.primary_keys[src.table])}
        if not determined:
            return []
        every = flat(group.expressions) + flat(x for k in ("rollup", "cube", "grouping_sets")
                                               for x in (group.args.get(k) or []))
        grouped = columns(every)
        written = {self._sql(g) for g in every if not isinstance(g, exp.Column)}  # GROUP BY UPPER(p.name) covers p.name
        found: list[exp.Column] = []

        def note(column: exp.Column, alias: str) -> None:
            if alias in determined and (alias, column.name) not in grouped and column.name in ctx.sources[alias].columns \
                    and not column.args.get("db"):
                grouped.add((alias, column.name))
                table = column.args.get("table")
                found.append(exp.Column(this=column.this.copy(), table=table.copy() if table is not None else
                                        exp.to_identifier(alias, quoted=alias != alias.lower() or None)))

        def read(e: exp.Expr, key: bool) -> None:
            closed = _aggregate_calls(e)  # parts not to look into: aggregate calls and parts that are GROUP BY items
            for node in e.walk(bfs=False, prune=lambda n: is_query(n) or id(n) in closed):
                if id(node) in closed:
                    continue
                if written and not isinstance(node, (exp.Column, exp.Identifier, exp.Literal)) \
                        and self._sql(node) in written:
                    closed.add(id(node))  # pruned right after this
                    continue
                if is_query(node):  # a subquery: what its body reads of this SELECT's FROM items
                    if ctx.subscopes is None:
                        ctx.subscopes = {id(s.expression): s for s in ctx.sg.subquery_scopes}
                    sub = ctx.subscopes.get(id(node))
                    for column in sub.external_columns if sub is not None else ():
                        if column.table in ctx.sources:  # no FROM item of the body or below shadows the alias
                            note(column, column.table)
                elif isinstance(node, exp.Column):
                    if node.table in ctx.sources:
                        note(node, node.table)
                    elif key and not node.table and node is not e:  # an input column inside an ORDER BY expression
                        owners = [a for a in visible if node.name in (ctx.sources[a].columns or ())]
                        if len(owners) == 1:
                            note(node, owners[0])

        for e in exprs:
            read(e, False)
        if having is not None:
            read(having.this, False)
        for e in keys:
            read(e, True)
        return found

    def _having(self, agg_id: str, plan: Plan, having: exp.Having, ctx: _Ctx, visible: dict[str, str]) -> Plan:
        """HAVING conjuncts are Filters over the aggregate: a region of their own whose single input is the aggregate."""
        self.facts.inputs.append(RegionInput(agg_id, plan))
        above = {a: agg_id for a in visible}
        seen: set[str] = set()
        for c in _conjuncts(having.this):
            sql = self._conj_sql(c)
            if sql in seen and _repeatable(c):
                continue
            seen.add(sql)
            refs: list[str] = []
            outer: list[str] = []
            subs: list[str] = []
            self._analyze(c, ctx, above, refs, outer, subs)
            if _has_aggregate(c):  # COUNT(*) > 1 mentions no column, but it is about the aggregate's groups
                refs = [agg_id]
            cid = ctx.conj_id()
            ctx.conj_entries.append(SideEntry(cid, "conjunct", sql, {"source": "having", "refs": refs,
                                                                   "outer_refs": outer, "subqueries": subs}))
            if refs:  # a constant, or a reference to an enclosing query only, has no owner, as everywhere else
                self.facts.owners[cid] = agg_id
            plan = Filter(plan, cid)
        return plan

    def _project(self, plan: Plan, items: list[_Item], ctx: _Ctx, visible: dict[str, str],
                 aggregate: tuple[str, list[_Item]] | None = None) -> tuple[str, Plan]:
        op_id = ctx.op_id("project")
        data = {"select": self._select_data(items), "has_window": any(_has_window(i.expr) for i in items)}
        self._beyond(data, [i.expr for i in items], ctx, visible)
        ctx.op_entries.append(SideEntry(op_id, "project", self._select_sql(items), data))
        if aggregate is None:
            self._keeps(op_id, items, lambda i: self._column(i.expr, ctx, visible))
        else:  # over an aggregate: every item without a window function IS an output of the aggregate, copied
            agg_id, agg_items = aggregate
            agg_names = _unique([i.name for i in agg_items])
            self._keeps(op_id, items, lambda i: ColumnRef(agg_id, i.name) if i in agg_items and i.name in agg_names else None)
        return op_id, Op1(plan, op_id)

    def _input_names(self, e: exp.Expr, ctx: _Ctx, visible: dict[str, str]) -> None:
        """In ORDER BY and DISTINCT ON an unqualified name that stands alone is an output name. INSIDE an expression
        Postgres reads it as an input column, also when a select alias has the same name; qualify leaves exactly
        those unqualified. Give them their FROM alias."""
        for node in list(e.walk(bfs=False, prune=is_query)):
            if not isinstance(node, exp.Column) or node.table or node is e:
                continue
            owners = [a for a in visible if node.name in (ctx.sources[a].columns or ())]
            if len(owners) != 1:
                raise UnsupportedSQL("name resolution", f"{node.name} inside {self._sql(e)} names {len(owners)} input columns")
            node.set("table", exp.to_identifier(owners[0], quoted=owners[0] != owners[0].lower() or None))

    def _order(self, query: exp.Expr, plan: Plan, ctx: _Ctx, alias_map: dict[str, str],
               outputs: list[_Out] | None = None) -> Plan:
        order = query.args.get("order")
        if order is not None and order.expressions:
            for o in order.expressions:
                if outputs is None:
                    self._input_names(o.this, ctx, alias_map)
                elif not (isinstance(o.this, exp.Column) and not o.this.table and o.this.name in {x.name for x in outputs}):
                    # Postgres: "only result column names can be used, not expressions or functions"
                    raise UnsupportedSQL("name resolution", f"ORDER BY {self._sql(o.this)} of a set operation is not one of its outputs")
            op_id = ctx.op_id("order")
            keys = []
            for o in order.expressions:
                text = self._sql(o)
                nulls = "first" if text.endswith("NULLS FIRST") else "last" if text.endswith("NULLS LAST") else None
                keys.append({"sql": self._sql(o.this), "desc": bool(o.args.get("desc")), "nulls": nulls})
            data: dict = {"keys": keys}
            self._beyond(data, [o.this for o in order.expressions], ctx, alias_map)
            ctx.op_entries.append(SideEntry(op_id, "order", "ORDER BY " + ", ".join(self._sql(o) for o in order.expressions), data))
            plan = Op1(plan, op_id)
        return plan

    def _limit(self, query: exp.Expr, plan: Plan, ctx: _Ctx, alias_map: dict[str, str]) -> Plan:
        limit, offset = query.args.get("limit"), query.args.get("offset")
        if limit is not None or offset is not None:
            count = None
            if isinstance(limit, exp.Fetch):
                options = limit.args.get("limit_options")
                if options is not None and (options.args.get("percent") or options.args.get("with_ties")):
                    raise UnsupportedSQL("FETCH ... PERCENT / WITH TIES", self._sql(limit))
                count = limit.args.get("count")
            elif limit is not None:
                count = limit.expression
            start = offset.expression if offset is not None else None
            op_id = ctx.op_id("limit")
            data = {"limit": self._sql(count) if count is not None else None,
                    "offset": self._sql(start) if start is not None else None}
            self._beyond(data, [e for e in (count, start) if e is not None], ctx, alias_map)
            sql = " ".join(p for p in (f"LIMIT {data['limit']}" if count is not None else "",
                                       f"OFFSET {data['offset']}" if start is not None else "") if p)
            ctx.op_entries.append(SideEntry(op_id, "limit", sql, data))
            plan = Op1(plan, op_id)
        return plan

    # ------------------------------------------------------------------------------------------------------------------
    # FROM: regions
    # ------------------------------------------------------------------------------------------------------------------

    def _from(self, select: exp.Select, ctx: _Ctx) -> _Region:
        from_ = select.args.get("from_")
        if from_ is None:
            occ = f"{DUAL_TABLE}@{ctx.sid}"
            ctx.from_entries.append(SideEntry(occ, "scan", "", {"table": None, "alias": None}))
            return self._leaf(occ, Scan(DUAL_TABLE, occ), [])
        # A comma binds weaker than JOIN: `a, b LEFT JOIN c` is a x (b LEFT JOIN c). Each comma starts a new chain.
        chains: list[tuple[exp.Expr, list[exp.Join]]] = [(from_.this, [])]
        for join in select.args.get("joins") or []:
            if join.meta.get(COMMA_JOIN):
                chains.append((join.this, []))
            else:
                chains[-1][1].append(join)
        region: _Region | None = None
        for head, joins in chains:
            chain = self._chain(self._item(head, ctx), joins, ctx)
            region = chain if region is None else self._merge(region, chain)
        return region

    def _chain(self, region: _Region, joins: list[exp.Join], ctx: _Ctx) -> _Region:
        """Joins are left-associative: each one takes everything written before it in its chain as its left side."""
        for join in joins:
            if join.args.get("using"):
                raise UnsupportedSQL("name resolution", "a USING clause could not be expanded")
            right = self._item(join.this, ctx)
            side, kind, method = ((join.args.get(k) or "").upper() for k in ("side", "kind", "method"))
            on = join.args.get("on")
            if method == "NATURAL" and on is None:
                # qualify turns NATURAL into ON equalities; what is left found no common column: a cross join
                common = self._columns(region, ctx) & self._columns(right, ctx)
                if common:
                    raise UnsupportedSQL("name resolution", f"NATURAL JOIN over {sorted(common)} was not expanded")
            elif method:
                raise UnsupportedSQL("unsupported join type", f"{method} JOIN")
            if kind in ("SEMI", "ANTI") or (side in ("LEFT", "RIGHT", "FULL") and kind in ("", "OUTER")):
                region = self._barrier(region, right, join, side, kind, ctx)
            elif not side and kind in ("", "INNER", "CROSS"):
                region = self._merge(region, right)
                if on is not None:
                    region.pool += [(c, "on") for c in _conjuncts(on)]
            else:
                raise UnsupportedSQL("unsupported join type", " ".join(p for p in (side, kind, "JOIN") if p))
        return region

    @staticmethod
    def _columns(region: _Region, ctx: _Ctx) -> set[str]:
        return {c for alias in region.alias_map for c in ctx.sources[alias].columns or ()}

    def _leaf(self, input_id: str, plan: Plan, aliases: list[str]) -> _Region:
        self.facts.inputs.append(RegionInput(input_id, plan))
        return _Region(_Tree(input_id=input_id, plan=plan), {a: input_id for a in aliases})

    @staticmethod
    def _merge(a: _Region, b: _Region) -> _Region:
        return _Region(_Tree(left=a.tree, right=b.tree), {**a.alias_map, **b.alias_map}, a.pool + b.pool)

    def _item(self, node: exp.Expr, ctx: _Ctx) -> _Region:
        if isinstance(node, exp.Table):
            nested = node.args.get("joins")
            if nested:  # a parenthesized join: (a JOIN b ON ...)
                head = node.copy()
                head.set("joins", None)
                return self._chain(self._table(node, head, ctx), nested, ctx)
            return self._table(node, node, ctx)
        if isinstance(node, exp.Subquery):
            nested = node.args.get("joins")
            if nested and node.alias:
                # a parenthesized join whose FIRST item is a derived table or a VALUES list: `((SELECT ...) AS d JOIN t
                # ON ...)`. SQLGlot hangs the chain off the derived table itself; without this branch the chain was
                # dropped silently (wrong rows, invisible to C1 / C2)
                head = node.copy()
                head.set("joins", None)
                return self._chain(self._item(head, ctx), nested, ctx)
            inner = node.this
            if isinstance(inner, exp.Table) or (isinstance(inner, exp.Subquery) and not node.alias):
                return self._item(inner, ctx)  # parentheses around a join, possibly several pairs
            if is_query(inner) or isinstance(inner, exp.Subquery):
                return self._ref(node, node.alias, None, ctx)
            if isinstance(inner, exp.Values):
                return self._values(inner if inner.alias else node, ctx)
        if isinstance(node, exp.Values):
            return self._values(node, ctx)
        raise UnsupportedSQL("unsupported FROM item", f"{node.key.upper()}: {self._sql(node)[:80]}")

    def _table(self, node: exp.Table, printable: exp.Table, ctx: _Ctx) -> _Region:
        alias = node.alias_or_name
        source = ctx.sg.sources.get(alias)
        if isinstance(source, Scope):
            return self._ref(printable, alias, node.name, ctx)
        if node.name not in self.tables:
            raise UnsupportedSQL("unknown table", node.name)
        occ = occurrence_id(alias, ctx.sid)
        self._add_source(ctx, _Source(alias, occ, self.tables[node.name], node.name))
        ctx.from_entries.append(SideEntry(occ, "scan", self._sql(printable), {"table": node.name, "alias": alias}))
        return self._leaf(occ, Scan(node.name, occ), [alias])

    def _values(self, node: exp.Expr, ctx: _Ctx) -> _Region:
        alias = node.alias
        if not alias:
            raise UnsupportedSQL("unsupported FROM item", "VALUES without an alias")
        occ = occurrence_id(alias, ctx.sid)
        table_alias = node.args.get("alias")
        columns = [c.name for c in table_alias.columns] if table_alias is not None else []
        self._add_source(ctx, _Source(alias, occ, frozenset(columns) if columns else None))
        data: dict = {"table": None, "alias": alias, "columns": columns}
        # the rows may read an enclosing query (`(VALUES (e.event_date)) AS v(x)` inside a correlated subquery): the
        # scan then depends on those occurrences like any correlated entry, and a block holding it is parameterized
        values = node.this if isinstance(node, exp.Subquery) else node
        rows = values.expressions if isinstance(values, exp.Values) else []
        self._beyond(data, [e for row in rows for e in (row.expressions if isinstance(row, exp.Tuple) else [row])], ctx, {})
        ctx.from_entries.append(SideEntry(occ, "scan", self._sql(node), data))
        return self._leaf(occ, Scan(VALUES_TABLE, occ), [alias])

    def _ref(self, node: exp.Expr, alias: str, cte_name: str | None, ctx: _Ctx) -> _Region:
        """One USE of a CTE or derived table: its own `ref` Op1 over the nested scope's (shared) plan."""
        sg = ctx.sg.sources.get(alias)
        if not isinstance(sg, Scope) or not is_query(sg.expression):
            raise UnsupportedSQL("unsupported FROM item", self._sql(node)[:80])
        # A nested scope never sees the FROM items next to it (that would be LATERAL). It does see the query levels
        # around the query that DEFINES it: a derived table is defined here, a CTE wherever its WITH clause stands.
        definer = ctx if cte_name is None else (self.ctx_of.get(id(sg.parent)) if sg.parent is not None else None)
        lexical = definer.parent if definer is not None else None
        nested = self._scope(sg, "derived" if cte_name is None else "cte", cte_name or alias, ctx.sid, lexical)
        op_id = ctx.op_id("ref")
        names = [o.name for o in nested.outputs]
        self._add_source(ctx, _Source(alias, op_id, frozenset(names)))
        data = {"alias": alias, "name": cte_name, "scope": nested.sid}
        if cte_name is not None and definer is not None:
            data["declared_in"] = definer.sid  # the scope whose WITH clause declares the CTE
        outer: list[str] = []
        for octx, src in nested.free:  # a correlated body: this use depends on occurrences of enclosing scopes
            self._note(octx, src, ctx, {}, [], outer)
        if outer:
            data["outer_refs"] = outer
        ctx.from_entries.append(SideEntry(op_id, "ref", self._sql(node), data))
        unique = _unique(names)
        for o in nested.outputs:
            if o.identity is not None and o.name in unique:
                self.facts.keeps.append(Keep(op_id, o.name, o.identity))
        return self._leaf(op_id, Op1(nested.plan, op_id), [alias])

    @staticmethod
    def _add_source(ctx: _Ctx, source: _Source) -> None:
        if source.alias in ctx.sources:
            raise UnsupportedSQL("name resolution", f"FROM alias {source.alias!r} is used twice in one SELECT")
        ctx.sources[source.alias] = source

    def _barrier(self, left: _Region, right: _Region, join: exp.Join, side: str, kind: str, ctx: _Ctx) -> _Region:
        """An outer / semi / anti join: both sides are closed, and the Op2 is one input of the region above."""
        if kind in ("SEMI", "ANTI") and side not in ("", "LEFT"):
            raise UnsupportedSQL("unsupported join type", f"{side} {kind} JOIN")
        join_type = kind if kind in ("SEMI", "ANTI") else side
        both = {**left.alias_map, **right.alias_map}
        left_plan, right_plan = self._close(left, ctx), self._close(right, ctx)
        on = join.args.get("on")
        conds = list(_conjuncts(on)) if on is not None else []
        refs: list[str] = []
        outer: list[str] = []
        subs: list[str] = []
        for c in conds:
            self._analyze(c, ctx, both, refs, outer, subs)
        op_id = ctx.op_id("outer_join")
        on_sql = [self._conj_sql(c) for c in conds]
        visible = list(left.alias_map) + ([] if join_type in ("SEMI", "ANTI") else list(right.alias_map))
        # `exposes`: the occurrences whose columns are visible above this join. Conjuncts above name the Op2 as their
        # input (refs, edges, owners); column facts name the occurrence (equates, keeps). This list connects the two.
        data = {"join_type": join_type, "on": on_sql, "refs": refs, "exposes": [ctx.sources[a].id for a in visible]}
        if outer:
            data["outer_refs"] = outer
        if subs:
            data["subqueries"] = subs
        sql = f"{join_type} JOIN" + (" ON " + " AND ".join(on_sql) if on_sql else "")
        ctx.from_entries.append(SideEntry(op_id, "outer_join", sql, data))
        return self._leaf(op_id, Op2(op_id, left_plan, right_plan), visible)

    def _close(self, region: _Region, ctx: _Ctx) -> Plan:
        """Classify the region's conjuncts and return its plan: the join tree with every join conjunct on the lowest
        join that covers its inputs, under one Filter per remaining conjunct."""
        filters: list[str] = []
        seen: set[str] = set()
        for e, source in region.pool:
            sql = self._conj_sql(e)
            if sql in seen and _repeatable(e):  # a conjunction is a SET of conjuncts
                continue
            seen.add(sql)
            refs: list[str] = []
            outer: list[str] = []
            subs: list[str] = []
            self._analyze(e, ctx, region.alias_map, refs, outer, subs)
            cid = ctx.conj_id()
            ctx.conj_entries.append(SideEntry(cid, "conjunct", sql, {"source": source, "refs": refs, "outer_refs": outer,
                                                                   "subqueries": subs}))
            if len(refs) >= 2:
                self.facts.edges.append(Edge(cid, tuple(refs)))
                self._place(region.tree, cid, frozenset(refs))
                self._equate(cid, e, ctx, region.alias_map)
            else:
                filters.append(cid)
                if refs:
                    self.facts.owners[cid] = refs[0]
        plan = self._plan_of(region.tree)
        for cid in filters:
            plan = Filter(plan, cid)
        return plan

    @staticmethod
    def _place(tree: _Tree, cid: str, needed: frozenset[str]) -> None:
        node = tree
        while True:
            if node.left is not None and needed <= node.left.ids():
                node = node.left
            elif node.right is not None and needed <= node.right.ids():
                node = node.right
            else:
                break
        node.conjuncts.append(cid)

    def _plan_of(self, tree: _Tree) -> Plan:
        if tree.input_id is not None:
            return tree.plan
        return Join(self._plan_of(tree.left), self._plan_of(tree.right), tuple(sorted(tree.conjuncts)))

    def _equate(self, cid: str, e: exp.Expr, ctx: _Ctx, alias_map: dict[str, str]) -> None:
        e = unparen(e)
        if not isinstance(e, exp.EQ):
            return
        left, right = self._column(e.this, ctx, alias_map), self._column(e.expression, ctx, alias_map)
        if left is not None and right is not None and \
                alias_map[unparen(e.this).table] != alias_map[unparen(e.expression).table]:
            self.facts.equates.append(Equate(cid, left, right))

    # ------------------------------------------------------------------------------------------------------------------
    # what an expression references
    # ------------------------------------------------------------------------------------------------------------------

    def _analyze(self, e: exp.Expr, ctx: _Ctx, alias_map: dict[str, str], refs: list[str], outer: list[str],
                 subs: list[str]) -> None:
        """Walk `e` in the order written. `refs` gets the region inputs it depends on, `outer` the occurrences of
        enclosing scopes, `subs` the scopes of the subqueries inside it. A subquery is not entered: its body is a
        scope of its own. What the body references OUTSIDE itself is part of what `e` depends on, though: a filter
        must never move below an input that its subquery is correlated with."""
        for node in e.walk(bfs=False, prune=is_query):
            if is_query(node):
                sub = self._subquery(node, ctx)
                if sub.sid not in subs:
                    subs.append(sub.sid)
                for octx, src in sub.free:
                    self._note(octx, src, ctx, alias_map, refs, outer)
            elif isinstance(node, exp.Column) and node.table:  # without a table: the name of an output (ORDER BY)
                octx, src = self._resolve(node, ctx)
                self._note(octx, src, ctx, alias_map, refs, outer)

    def _resolve(self, column: exp.Column, ctx: _Ctx) -> tuple[_Ctx, _Source]:
        c: _Ctx | None = ctx
        while c is not None:
            src = c.sources.get(column.table)
            if src is not None:
                if src.columns is not None and column.name not in src.columns:
                    raise UnsupportedSQL("name resolution", f"{self._sql(column)} is not an output of {src.alias}")
                return c, src
            c = c.parent
        raise UnsupportedSQL("name resolution", f"{self._sql(column)} does not name a FROM item that is in scope")

    def _note(self, octx: _Ctx, src: _Source, ctx: _Ctx, alias_map: dict[str, str], refs: list[str],
              outer: list[str]) -> None:
        if octx is ctx:
            input_id = alias_map.get(src.alias)
            if input_id is None:
                raise UnsupportedSQL("name resolution", f"FROM item {src.alias!r} is not visible where it is referenced")
            if input_id not in refs:
                refs.append(input_id)
        else:
            if src.id not in outer:
                outer.append(src.id)
            ctx.add_free(octx, src)


_VOLATILE = {"RANDOM", "SETSEED", "NEXTVAL", "CURRVAL", "LASTVAL", "CLOCK_TIMESTAMP", "TIMEOFDAY", "GEN_RANDOM_UUID",
             "UUID_GENERATE_V4", "PG_SLEEP", "TXID_CURRENT", "STATEMENT_TIMESTAMP"}


def _repeatable(e: exp.Expr) -> bool:
    """True when `p AND p` means `p`: no subquery (each one is a scope of its own) and no volatile function."""
    for node in e.walk():
        if is_query(node) or isinstance(node, (exp.Rand, exp.Randn, exp.Uuid)):
            return False
        if isinstance(node, exp.Func) and node.sql_name().upper() in _VOLATILE:
            return False
        if isinstance(node, exp.Anonymous) and node.name.upper() in _VOLATILE:
            return False
    return True


def _has_window(e: exp.Expr) -> bool:
    """True when `e` contains a window function of its own query level."""
    return any(isinstance(node, exp.Window) for node in e.walk(bfs=False, prune=is_query))


def _has_aggregate(e: exp.Expr) -> bool:
    """True when `e` contains an aggregate call of its own query level (not a window call, not inside a subquery)."""
    return bool(_aggregate_calls(e, first=True))


def _aggregate_calls(e: exp.Expr, first: bool = False) -> set[int]:
    """The ids of the aggregate calls of `e`'s own query level (not a window call, not inside a subquery), each taken
    with its FILTER / WITHIN GROUP / IGNORE NULLS wrapper, whose expressions are read per input row as well. `first`:
    stop at the first one."""
    found: set[int] = set()
    for node in e.walk(bfs=False, prune=is_query):
        if not isinstance(node, exp.AggFunc):
            continue
        child, parent = node, node.parent
        while isinstance(parent, (exp.Filter, exp.WithinGroup, exp.IgnoreNulls, exp.RespectNulls)) and child.arg_key == "this":
            child, parent = parent, parent.parent
        if not (isinstance(parent, exp.Window) and child.arg_key == "this"):
            found.add(id(child))
            if first:
                break
    return found


def _distinct_on_exprs(distinct: exp.Distinct | None) -> list[exp.Expr]:
    """The expressions of DISTINCT ON (...); [] for a plain DISTINCT or none."""
    on = distinct.args.get("on") if distinct is not None else None
    return list(on.expressions) if isinstance(on, exp.Tuple) else ([on] if on is not None else [])
