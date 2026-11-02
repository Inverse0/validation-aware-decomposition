"""Plan (or block fragment) -> SQLGlot expression tree.

The SQL is CONSTRUCTED from the skeleton and the side table; nothing is unparsed from an optimizer plan. One scope of
the query becomes one SELECT:

    region (Scan / Join / Filter over region inputs)   -> FROM ... JOIN ... ON ... WHERE ...
    aggregate, HAVING filters, projection, distinct,
    order, limit (Op1 nodes, in SQL evaluation order)  -> GROUP BY / HAVING / select list / DISTINCT / ORDER BY / LIMIT
    ref (Op1)                                          -> a FROM item: `WITH name AS (...)` + `name AS alias`, or a
                                                          derived table `(...) AS alias`
    outer join (Op2)                                   -> LEFT / RIGHT / FULL JOIN inside the FROM clause
    set operation (Op2)                                -> UNION [ALL] / INTERSECT / EXCEPT of two SELECTs

A region prints flat. Its inputs come in the written order of the query, except that an input is postponed while no
join condition connects it to the inputs already listed (so that no JOIN without a condition appears unless the query
itself contains a cross join). Every join conjunct goes into the ON clause of the JOIN that brings in the last input
it mentions; local conjuncts go into WHERE.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from sqlglot import exp

from sqldecomp.model import Filter, Join, Op1, Op2, OutputColumn, Scan, SideEntry
from sqldecomp.printer import sqltext
from sqldecomp.printer.context import BlockInput, NeededColumn, PPlan, PrintContext, children
from sqldecomp.printer.errors import NotABlock, PrintError, UnsupportedBoundary

# hidden alias key -> column key -> the column of the block boundary that replaces `alias.column`
Env = dict[str, dict[str, exp.Column]]

_PIPELINE = ("limit", "order", "distinct", "project")  # peeled from the top of a scope plan, in this order
_SETOPS = {"UNION": exp.Union, "INTERSECT": exp.Intersect, "EXCEPT": exp.Except}


@dataclass
class Fragment:
    """What one block prints: a plan whose leaves may be block inputs, plus the same for every predicate or select-list
    subquery of the block whose body contains a block input (scope id -> plan)."""

    plan: PPlan
    subqueries: dict[str, PPlan] = field(default_factory=dict)


@dataclass
class _Cte:
    ident: exp.Identifier
    scope: str
    body: exp.Expression
    plan: PPlan  # what the body was printed from, block inputs replaced by the blocks that compute them
    used_in: list[str] = field(default_factory=list)  # ids of the scopes whose FROM clause reads this CTE
    declared_in: str | None = None  # id of the scope whose WITH clause declares it, when the import result says so
    reads: set[str] = field(default_factory=set)  # scope ids of the CTEs that the body reads


def boundary_ident(group: str, prefix: str = "b_") -> exp.Identifier:
    """`b_<group id>`, the name of a block at a block boundary. `prefix` is the query's `PrintContext.boundary_prefix`
    (`b_` unless the query has such a name itself)."""
    return sqltext.make_ident(prefix + re.sub(r"[^A-Za-z0-9_]", "_", group))


class Builder:
    """Builds one statement. `resolve(group)` returns the fragment of the block that computes a block input; with it
    block inputs are inlined (derived tables, CTE bodies, subquery bodies), without it they appear as relations named
    `b_<group id>`."""

    def __init__(self, ctx: PrintContext, *, resolve: Callable[[str], Fragment] | None = None,
                 subqueries: dict[str, PPlan] | None = None):
        self.ctx = ctx
        self.resolve = resolve
        self.subquery_plans: dict[str, PPlan] = dict(subqueries or {})
        self.ctes: dict[str, _Cte] = {}  # scope id of the CTE body -> CTE, in the order they are first read
        self._building: list[str] = []  # scope ids of the CTEs whose body is being built, innermost last
        self._pending_reads: dict[str, set[str]] = {}  # the same for bodies that are not finished yet
        self._queries: dict[str, exp.Expression] = {}  # scope id -> the query printed for it (not CTE bodies)
        self.used_inputs: list[str] = []
        self.closed = False  # True when `statement` had to wrap the plan to bind its outer references
        self.boundary_names: set[str] = set()  # folded names b_<group id> of the block boundaries that were printed
        self.input_envs: dict[str, Env] = {}  # block -> the renaming in force where the block is inlined
        self._input_parameters: dict[str, list[exp.Column]] = {}  # block input -> the outer columns it reads
        self._expose_pairs: list[tuple[str, str]] = []  # `expose` of the statement: some may lie inside an input

    # ---------------------------------------------------------------------------------------------------------
    # statement and query
    # ---------------------------------------------------------------------------------------------------------

    def statement(self, plan: PPlan, *, boundary: bool = False, bind_outer: bool = False,
                  env: Env | None = None, expose: list[tuple[str, str]] | None = None) -> exp.Expression:
        """`boundary` only matters for a plan inside a region, whose select list is the columns that the rest of the
        query reads from it: True names them `<alias>__<column>` as a block boundary does, False leaves their names.
        `bind_outer` closes a plan from inside a correlated subquery, see `_closed`. `env` is the renaming that the
        surrounding query forces on the plan's outer references (a block printed as part of a decomposition).
        `expose`: further (alias, column) pairs of the plan's FROM items, its block inputs' included, that the
        boundary exposes beside what this position reads (the union over every occurrence of the block's class,
        design decision on OPT-O1 of the optimizer); a pair inside a block input is read through that input's
        boundary, which exposes it as well; only for a plan inside a region, whose select list the printer chooses."""
        self._expose_pairs = list(expose or [])
        query = self.query(plan, dict(env or {}), boundary=boundary, expose=expose)
        if bind_outer:
            query = self._closed(plan, query)
        self._attach_ctes(query, self.ctx.scope_of(plan).id)
        return sqltext.unquote_safe_identifiers(query)

    def _closed(self, plan: PPlan, query: exp.Expression) -> exp.Expression:
        """A plan from inside a correlated subquery reads columns of the query around it, so its SQL does not stand
        alone. The closed form puts it back where such a plan lives, a subquery over the FROM items it refers to:
        `SELECT 1 FROM <those items> WHERE EXISTS (<the plan>)`. Plans without outer references are returned as is.

        The closed form only BINDS the outer references, so that Postgres can EXPLAIN the plan and the importer can
        read it back (check C1 then looks at the plan of the EXISTS subquery). It does not compute what the plan
        computes: EXISTS ignores the select list and is always true over an ungrouped aggregate. Never execute it for
        its rows or read estimates off it."""
        free: list[str] = []
        for column in sqltext.outer_columns(query):
            alias = sqltext.column_key(column)[0]
            if alias not in free:
                free.append(alias)
        if not free:
            return query
        self.closed = True
        items = [self._item(self.ctx.outer_input(plan, alias), {}, {}) for alias in free]
        closed = exp.Select(expressions=[exp.Literal.number(1)], from_=exp.From(this=items[0]),
                            where=exp.Where(this=exp.Exists(this=query)))
        if items[1:]:
            closed.set("joins", [exp.Join(this=i, kind="CROSS") for i in items[1:]])
        return closed

    def query(self, plan: PPlan, env: Env, *, top: bool | None = None, boundary: bool = False,
              expose: list[tuple[str, str]] | None = None) -> exp.Expression:
        ctx = self.ctx
        if isinstance(plan, BlockInput):
            return self._input_query(plan, env)
        if top is None:
            top = ctx.is_scope_top(plan)
        ops, having, node = self._peel(plan)
        if isinstance(node, Op2) and ctx.kind(node) == "setop":
            if expose:
                raise PrintError("a set operation's select list is its own: no further column can be exposed at it")
            query = self._setop_query(node, ops, having, env)
            self._queries.setdefault(ctx.scope_of(plan).id, query)
            return query
        if isinstance(node, (Op1, Op2)) and ctx.kind(node) not in ("ref", "outer_join"):
            raise PrintError(f"operator {node.op!r} ({ctx.kind(node)}) sits where one SELECT cannot put it")

        hidden = self._region_block_inputs(node)
        defined = ctx.visible_aliases(node)
        outer: Env = {k: v for k, v in env.items() if k not in defined}  # aliases of this SELECT shadow outer ones
        here: Env = dict(outer)
        for bi in hidden:
            here.update(self._expose(bi))

        select = exp.Select()
        if top:
            self._queries.setdefault(ctx.scope_of(plan).id, select)
        if ctx.is_dual(self._below_filters(node)):  # SELECT without FROM
            where = self._filters(node)
        else:
            head, joins, where = self._from_clause(node, here, outer, where_ok=True)
            select.set("from_", exp.From(this=head))
            if joins:
                select.set("joins", joins)
        condition = sqltext.and_all(self._predicate(c, here) for c in ctx.by_written_order(where))
        if condition is not None:
            select.set("where", exp.Where(this=condition))

        aggregate = ops.get("aggregate")
        if aggregate is not None:
            if aggregate.data.get("group_by"):
                select.set("group", self._rewrite(sqltext.parse_group_by(list(aggregate.data["group_by"])), here, ()))
            condition = sqltext.and_all(self._predicate(c, here) for c in ctx.by_written_order(having))
            if condition is not None:
                select.set("having", exp.Having(this=condition))

        select.set("expressions", self._select_list(plan, ops, here, top=top, boundary=boundary, expose=expose))
        if "distinct" in ops:
            d = ops["distinct"].data
            on = list(d.get("on") or ())
            shares = self._share_subqueries(on, list(d.get("subqueries") or ()))
            keys = [self._rewrite(sqltext.parse_item(sql), here, ids) for sql, ids in zip(on, shares)]
            select.set("distinct", exp.Distinct(on=exp.Tuple(expressions=keys)) if keys else exp.Distinct())
        self._order_and_limit(select, ops, here)
        return select

    @staticmethod
    def _below_filters(node: PPlan) -> PPlan:
        while isinstance(node, Filter):
            node = node.input
        return node

    @staticmethod
    def _filters(node: PPlan) -> list[str]:
        found = []
        while isinstance(node, Filter):
            found.append(node.conjunct)
            node = node.input
        return found

    def _peel(self, plan: PPlan) -> tuple[dict[str, SideEntry], list[str], PPlan]:
        """Split a scope plan into its select-list pipeline, its HAVING conjuncts and what is left (a region or a set
        operation). The pipeline order is the evaluation order of SQL, so anything else cannot be one SELECT."""
        ctx = self.ctx
        ops: dict[str, SideEntry] = {}
        node = plan
        for kind in _PIPELINE:
            if isinstance(node, Op1) and ctx.kind(node) == kind:
                ops[kind] = ctx.entry(node.op)
                node = node.input
            # DISTINCT ON keeps the FIRST row of each group in ORDER BY order, so its plan is order, distinct, limit
            if kind == "distinct" and "distinct" in ops and "order" not in ops and ops["distinct"].data.get("on") \
                    and isinstance(node, Op1) and ctx.kind(node) == "order":
                ops["order"] = ctx.entry(node.op)
                node = node.input
        filters: list[str] = []
        probe = node
        while isinstance(probe, Filter):
            filters.append(probe.conjunct)
            probe = probe.input
        having: list[str] = []
        if isinstance(probe, Op1) and ctx.kind(probe) == "aggregate":
            ops["aggregate"] = ctx.entry(probe.op)
            having = filters
            node = probe.input
        if isinstance(node, BlockInput) and not ctx.is_region_level(node):
            raise UnsupportedBoundary(
                f"block input {node.group!r} ends inside the select-list pipeline of scope {ctx.scope_of(node).id!r} "
                "(between aggregate / HAVING / projection / DISTINCT / ORDER BY / LIMIT); the rest of that pipeline "
                "refers to its results by expression, which a derived table cannot offer without changing the operators")
        return ops, having, node

    def _setop_query(self, node: Op2, ops: dict[str, SideEntry], having: list[str], env: Env) -> exp.Expression:
        if having or set(ops) - {"order", "limit"}:
            raise PrintError(f"only ORDER BY and LIMIT can follow the set operation {node.op!r} in one statement")
        e = self.ctx.entry(node.op, "setop")
        words = str(e.data.get("op") or e.sql).upper().split()
        if not words or words[0] not in _SETOPS:
            raise PrintError(f"set operation {node.op!r} has an unknown operator {e.data.get('op')!r}")
        cls = _SETOPS[words[0]]
        left = self._branch(node.left, env, parent=cls, right_side=False)
        right = self._branch(node.right, env, parent=cls, right_side=True)
        query = cls(this=left, expression=right, distinct="ALL" not in words[1:])
        self._order_and_limit(query, ops, env)
        return query

    def _branch(self, plan: PPlan, env: Env, *, parent: type, right_side: bool) -> exp.Expression:
        q = self.query(plan, env, top=True)
        if isinstance(q, exp.SetOperation):
            trailing = q.args.get("order") or q.args.get("limit") or q.args.get("offset")
            if right_side or trailing or type(q) is not parent:
                return exp.Subquery(this=q)
        elif q.args.get("order") or q.args.get("limit") or q.args.get("offset"):
            return exp.Subquery(this=q)
        return q

    def _order_and_limit(self, query: exp.Expression, ops: dict[str, SideEntry], env: Env) -> None:
        order = ops.get("order")
        if order is not None:
            keys = []
            listed = list(order.data.get("keys") or ())
            shares = self._share_subqueries([k["sql"] for k in listed], list(order.data.get("subqueries") or ()))
            for k, ids in zip(listed, shares):
                key = sqltext.parse_order_key(k["sql"], k.get("desc"), k.get("nulls"))
                key.set("this", self._rewrite(key.this, env, ids))
                keys.append(key)
            if not keys:
                raise PrintError(f"order operator {order.id!r} has no keys")
            query.set("order", exp.Order(expressions=keys))
        limit = ops.get("limit")
        if limit is not None:
            if limit.data.get("limit") is not None:
                query.set("limit", exp.Limit(expression=sqltext.parse_item(str(limit.data["limit"]))))
            if limit.data.get("offset") is not None:
                query.set("offset", exp.Offset(expression=sqltext.parse_item(str(limit.data["offset"]))))

    # ---------------------------------------------------------------------------------------------------------
    # select list
    # ---------------------------------------------------------------------------------------------------------

    def _select_list(self, plan: PPlan, ops: dict[str, SideEntry], env: Env, *, top: bool,
                     boundary: bool, expose: list[tuple[str, str]] | None = None) -> list[exp.Expression]:
        ctx = self.ctx
        entry = ops.get("project") or ops.get("aggregate")
        if expose and (entry is not None or top or ops):
            raise PrintError("the select list of a scope is its own: no further column can be exposed at it")
        if entry is not None:
            items = entry.data.get("select") or ()
            if not items:
                # `SELECT COUNT(*) OVER () FROM t GROUP BY x`: every select item is a window function, so the aggregate
                # BELOW that projection outputs nothing by name. The empty select list is valid Postgres.
                if entry.kind == "aggregate":
                    return []
                raise PrintError(f"{entry.kind} {entry.id!r} has an empty select list")
            shares = self._share_subqueries([i["sql"] for i in items], list(entry.data.get("subqueries") or ()))
            return [self._select_item(i["sql"], i.get("alias"), env, ids) for i, ids in zip(items, shares)]
        if top or ops:  # DISTINCT, ORDER BY and LIMIT work on the select list of their scope
            outputs = ctx.scope_of(plan).outputs
            if not outputs:
                raise PrintError(f"scope {ctx.scope_of(plan).id!r} has no output columns")
            return [self._output_item(o, env) for o in outputs]
        # a plan inside a region: the columns that the rest of the query reads from it
        columns = list(ctx.needed_columns(plan))
        if expose:  # ... plus what the other occurrences of the block's class read (in this position's aliases), in
            # the order given, so that every occurrence prints one select list up to renaming
            visible = ctx.visible_aliases(plan)
            by_key = {n.key: n for n in columns}
            ordered: list[NeededColumn] = []
            for alias, column in expose:  # names as the side table holds them: keyed as the FROM aliases are
                key = (sqltext.ident_key(ctx.identifier(alias)), sqltext.ident_key(ctx.identifier(column)))
                if key[0] not in visible:
                    raise PrintError(f"alias {alias!r} is not a FROM item of this block; nothing of it can be exposed")
                if key in by_key:
                    ordered.append(by_key.pop(key))
                elif key not in {n.key for n in ordered}:
                    ordered.append(NeededColumn(key, exp.Column(this=ctx.identifier(column), table=visible[key[0]].copy())))
            columns = ordered + list(by_key.values())
        if not columns:  # nothing of it is read by name (COUNT(*), EXISTS): all columns, which is always legal
            # ... unless there is no FROM item at all (the pseudo scan of a SELECT without FROM): `SELECT *` is an
            # error there, and the empty select list is what Postgres, SQLGlot and the importer all accept. The same
            # when `*` would read a column without a name of its own (`(SELECT x * 2 FROM t) AS w`): the re-import
            # would have to write an invented name into the body, and C1 compares that body with the original's.
            return [exp.Star()] if ctx.visible_aliases(plan) and not ctx.star_reads_an_unnamed_column(plan) else []
        items = []
        for n in columns:
            item = self._rewrite(n.column.copy(), env, ())
            if boundary:
                exposed = self._exposed_name(n)
                if sqltext.natural_name(item) != sqltext.ident_key(exposed):
                    item = exp.Alias(this=item, alias=exposed)
            items.append(item)
        return items

    def _select_item(self, sql: str, alias: str | None, env: Env, subqueries: list[str]) -> exp.Expression:
        tree = sqltext.parse_item(sql)
        if isinstance(tree, exp.Alias) and not alias:
            tree, written = tree.this, tree.args["alias"]
        else:
            written = self.ctx.identifier(alias) if alias else None
        own = tree.this.copy() if isinstance(tree, exp.Column) and isinstance(tree.this, exp.Identifier) else None
        item = self._rewrite(tree, env, subqueries)
        if written is not None:
            return exp.Alias(this=item, alias=written)
        if own is not None and sqltext.natural_name(item) != sqltext.ident_key(own):
            return exp.Alias(this=item, alias=own)  # a renamed column keeps the name the query gave it
        return item

    def _output_item(self, out: OutputColumn, env: Env) -> exp.Expression:
        tree = sqltext.parse_item(out.sql)
        if isinstance(tree, exp.Alias):
            tree = tree.this
        name = out.name if out.name and out.name != "?column?" else None  # ?column?: Postgres' name for "unnamed"
        if name is not None and sqltext.natural_name(tree) == sqltext.ident_key(self.ctx.identifier(name)):
            name = None  # the column already has this name
        return self._select_item(sqltext.to_sql(tree), name, env, [])

    # ---------------------------------------------------------------------------------------------------------
    # expressions: reuse the side-table SQL, print subqueries from their own plans, apply boundary renaming
    # ---------------------------------------------------------------------------------------------------------

    def _share_subqueries(self, sqls: list[str], scope_ids: list[str]) -> list[list[str]]:
        """One side entry can hold several SQL texts (the items of a select list, the ON conjuncts of an outer join) but
        lists its subqueries once, in written order. Give every text the ids of the queries it contains; when the
        counts do not add up every text gets the whole list and the FROM items have to tell the subqueries apart."""
        if not scope_ids:
            return [[] for _ in sqls]
        counts = [len(sqltext.top_level_queries(sqltext.parse_item(sql))) for sql in sqls]
        if sum(counts) != len(scope_ids):
            return [list(scope_ids) for _ in sqls]
        shares, start = [], 0
        for n in counts:
            shares.append(scope_ids[start:start + n])
            start += n
        return shares

    def _predicate(self, conjunct: str, env: Env) -> exp.Expression:
        e = self.ctx.entry(conjunct, "conjunct")
        return self._rewrite(sqltext.parse_predicate(e.sql), env, list(e.data.get("subqueries") or ()))

    def _rewrite(self, tree: exp.Expression, env: Env, subqueries) -> exp.Expression:
        if subqueries:
            holder = exp.Paren(this=tree)  # so that a subquery at the root can be replaced as well
            for node, sid in self._match_subqueries(sqltext.top_level_queries(tree), list(subqueries)):
                node.replace(self.query(self.subquery_plans.get(sid, self.ctx.scopes[sid].plan), env, top=True))
            tree = holder.this.pop()
        if not env:
            return tree

        def rename(column: exp.Column) -> exp.Expression | None:
            alias, name = sqltext.column_key(column)
            if alias not in env:
                return None
            if name not in env[alias]:
                raise PrintError(f"{sqltext.to_sql(column)} is read across a block boundary that does not expose it")
            return env[alias][name].copy()

        return sqltext.rename_columns(tree, rename)

    def _match_subqueries(self, nodes: list[exp.Expression], scope_ids: list[str]) -> list[tuple[exp.Expression, str]]:
        """Pair the subqueries written in a fragment with the scope ids its side entry lists. They are expected in
        the written order; the FROM items of both sides must agree, otherwise the text of the side table is kept."""
        known = [s for s in scope_ids if s in self.ctx.scopes]
        direct = [s for s in known if self.ctx.scopes[s].kind == "subquery"] or known
        for candidates in (direct, known):
            if len(candidates) == len(nodes) and all(
                    self._same_sources(n, s) for n, s in zip(nodes, candidates)):
                return list(zip(nodes, candidates))
        pairs = []
        for n in nodes:
            fits = [s for s in known if self._same_sources(n, s)]
            if len(fits) != 1:
                return []
            pairs.append((n, fits[0]))
        return pairs if len({s for _, s in pairs}) == len(pairs) else []

    def _same_sources(self, node: exp.Expression, scope_id: str) -> bool:
        if not isinstance(node, exp.Select):
            return not isinstance(self._scope_sources(scope_id), list)  # a set operation on both sides
        want = self._scope_sources(scope_id)
        return isinstance(want, list) and sorted(sqltext.source_alias_keys(node)) == want

    def _scope_sources(self, scope_id: str) -> list[str] | None:
        _, _, node = self._peel_plain(self.ctx.scopes[scope_id].plan)
        if isinstance(node, Op2) and self.ctx.kind(node) == "setop":
            return None
        return sorted(self.ctx.visible_aliases(node) | self._hidden_right_sides(node))

    def _peel_plain(self, plan):
        try:
            return self._peel(plan)
        except UnsupportedBoundary:  # pragma: no cover - scope plans have no block inputs
            return {}, [], plan

    def _hidden_right_sides(self, node: PPlan) -> dict[str, exp.Identifier]:
        """Aliases on the right of a semi or anti join: defined in the FROM clause although not visible above it."""
        out: dict[str, exp.Identifier] = {}
        for n in self._region_nodes(node):
            if isinstance(n, Op2):
                out.update(self.ctx.visible_aliases(n.right))
        return out

    def _region_nodes(self, node: PPlan):
        yield node
        if isinstance(node, (Join, Filter)) or (isinstance(node, Op2) and self.ctx.kind(node) == "outer_join"):
            for c in children(node):
                yield from self._region_nodes(c)

    # ---------------------------------------------------------------------------------------------------------
    # block boundaries
    # ---------------------------------------------------------------------------------------------------------

    def _region_block_inputs(self, node: PPlan) -> list[BlockInput]:
        found = [n for n in self._region_nodes(node) if isinstance(n, BlockInput)]
        for bi in found:
            if not self.ctx.is_region_level(bi):
                raise UnsupportedBoundary(f"block input {bi.group!r} is not a FROM item of the block that reads it")
        return found

    @staticmethod
    def _exposed_name(n: NeededColumn) -> exp.Identifier:
        return sqltext.join_names([n.column.args["table"], n.column.this])

    def _expose(self, bi: BlockInput) -> Env:
        """The renaming that a block input forces on its reader: `alias.column` -> `b_<id>.<alias>__<column>`, for
        the columns the query reads from it and for the `expose` pairs of the statement that lie inside it (a union
        column of the reader's class that this input holds: the input exposes it too, OPT-O1)."""
        name = boundary_ident(bi.group, self.ctx.boundary_prefix)
        env: Env = {}
        exposed_as: dict[str, tuple[str, str]] = {}
        self.boundary_names.add(sqltext.ident_key(name))
        for n in list(self.ctx.needed_columns(bi)) + self._extra_needed(bi):
            exposed = self._exposed_name(n)
            clash = exposed_as.setdefault(sqltext.ident_key(exposed), n.key)
            if clash != n.key:
                raise PrintError(f"block {bi.group!r} would expose {clash} and {n.key} under one name {exposed.name!r}")
            env.setdefault(n.key[0], {})[n.key[1]] = exp.Column(this=exposed, table=name.copy())
        for alias in self.ctx.visible_aliases(bi):
            env.setdefault(alias, {})  # hidden even when none of its columns is exposed
        return env

    def _extra_pairs(self, bi: BlockInput) -> list[tuple[str, str]]:
        """The `expose` pairs of the statement whose alias lies inside this input."""
        visible = self.ctx.visible_aliases(bi)
        return [(alias, column) for alias, column in self._expose_pairs
                if sqltext.ident_key(self.ctx.identifier(alias)) in visible]

    def _extra_needed(self, bi: BlockInput) -> list[NeededColumn]:
        visible = self.ctx.visible_aliases(bi)
        out = []
        for alias, column in self._extra_pairs(bi):
            key = (sqltext.ident_key(self.ctx.identifier(alias)), sqltext.ident_key(self.ctx.identifier(column)))
            out.append(NeededColumn(key, exp.Column(this=self.ctx.identifier(column), table=visible[key[0]].copy())))
        return out

    def _input_item(self, bi: BlockInput, outer: Env) -> exp.Expression:
        """A block input as a FROM item."""
        name = boundary_ident(bi.group, self.ctx.boundary_prefix)
        if self.resolve is None:
            self.used_inputs.append(bi.group)
            return self._named_input(bi, outer)
        self.input_envs.setdefault(bi.group, outer)
        body = self.query(self._fragment(bi).plan, outer, top=False, boundary=True, expose=self._extra_pairs(bi) or None)
        return exp.Subquery(this=body, alias=exp.TableAlias(this=name))

    def _input_query(self, bi: BlockInput, env: Env) -> exp.Expression:
        """A block input that is a whole query: the body of a CTE, derived table or subquery, or a set-operation
        branch. Its columns keep the names that the scope gives them."""
        if self.resolve is not None:
            self.input_envs.setdefault(bi.group, env)
            return self.query(self._fragment(bi).plan, env, top=True)
        self.used_inputs.append(bi.group)
        name = boundary_ident(bi.group, self.ctx.boundary_prefix)
        # by name only when the block itself spells the names; never rely on the name Postgres invents for `AVG(x)`
        names = self.ctx.explicit_output_names(self.ctx.scope_of(bi))
        columns = [exp.Column(this=self.ctx.identifier(n), table=name.copy()) for n in names] if names else [exp.Star()]
        return exp.Select(expressions=columns, from_=exp.From(this=self._named_input(bi, env)))

    def _named_input(self, bi: BlockInput, env: Env) -> exp.Table:
        """A block input by name. An input that reads columns of a query AROUND it (a block from inside a correlated
        subquery) is no relation but a function of those columns, and its reader has to say so: written just as
        `b_<id>`, `EXISTS (SELECT * FROM b_g12)` would read as one fixed table, the same for every outer row. Such an
        input is written `b_<id>(<the outer columns it reads>)`, the columns as they read where the input stands
        (through a block boundary that hides their alias: `b_g12(b_g07.so__subject_id_sort)`)."""
        name = boundary_ident(bi.group, self.ctx.boundary_prefix)
        if bi.group not in self._input_parameters:
            self._input_parameters[bi.group] = self.parameters(bi.plan)
        args = [self.through(column, env) for column in self._input_parameters[bi.group]]
        if not args:
            return exp.Table(this=name)
        return exp.Table(this=exp.Anonymous(this=name.name, expressions=args))

    def parameters(self, plan: PPlan) -> list[exp.Column]:
        """The outer columns that a plan reads, as the query writes them, in ONE order (by alias, then column): the
        order of the arguments wherever the plan is called as a block, and of the parameters in its own header."""
        alone = Builder(self.ctx).statement(plan)  # the plan computed from base tables, by itself
        found: dict[tuple[str, str], exp.Column] = {}
        for column in sqltext.outer_columns(alone):
            found.setdefault(sqltext.column_key(column), column)
        return [found[key] for key in sorted(found)]

    @staticmethod
    def through(column: exp.Column, env: Env) -> exp.Expression:
        """An outer column as it reads where `env` is in force (through a block boundary that hides its alias)."""
        alias, name = sqltext.column_key(column)
        return env[alias][name].copy() if name in env.get(alias, {}) else column.copy()

    def _fragment(self, bi: BlockInput) -> Fragment:
        fragment = self.resolve(bi.group)
        self.subquery_plans.update(fragment.subqueries)
        self.used_inputs.append(bi.group)
        return fragment

    # ---------------------------------------------------------------------------------------------------------
    # FROM clause
    # ---------------------------------------------------------------------------------------------------------

    def _from_clause(self, node: PPlan, env: Env, outer: Env, *, where_ok: bool
                     ) -> tuple[exp.Expression, list[exp.Join], list[str]]:
        """(first FROM item, the JOINs after it, conjuncts for WHERE). Under an outer join there is no WHERE of its own
        (`where_ok=False`): local conjuncts then go into the ON clause of an inner join of the same region, which
        means the same, or, for a single input, into a derived table that keeps the input's alias."""
        ctx = self.ctx
        leaves: list[PPlan] = []
        join_conjuncts: list[str] = []
        filters: list[str] = []

        def flatten(n: PPlan) -> None:
            if isinstance(n, Join):
                join_conjuncts.extend(n.conjuncts)
                flatten(n.left), flatten(n.right)
            elif isinstance(n, Filter):
                filters.append(n.conjunct)
                flatten(n.input)
            else:
                leaves.append(n)

        flatten(node)
        leaves, covers = self._from_order(leaves, join_conjuncts)

        on: list[list[str]] = [[] for _ in leaves]
        for c in ctx.by_written_order(join_conjuncts):
            on[self._position(c, covers, floor=1)].append(c)
        if not where_ok and filters:
            if len(leaves) == 1:
                return self._filtered_input(leaves[0], filters, env, outer), [], []
            for c in ctx.by_written_order(filters):
                on[self._position(c, covers, floor=1)].append(c)
            filters = []

        head, joins = self._head(leaves[0], env, outer)
        for leaf, conjuncts in zip(leaves[1:], on[1:]):
            condition = sqltext.and_all(self._predicate(c, env) for c in conjuncts)
            item = self._item(leaf, env, outer)
            joins.append(exp.Join(this=item, on=condition) if condition is not None else exp.Join(this=item, kind="CROSS"))
        return head, joins, filters

    def _from_order(self, leaves: list[PPlan], join_conjuncts: list[str]) -> tuple[list[PPlan], list[frozenset[str]]]:
        ctx = self.ctx
        covers = [ctx.region_input_ids(leaf) for leaf in leaves]
        rest = sorted(range(len(leaves)), key=lambda i: (ctx.written_pos(leaves[i]), i))
        needs = [frozenset(ctx.conjunct_inputs(c)) for c in join_conjuncts]
        order = [rest.pop(0)]
        have = set(covers[order[0]])
        while rest:
            pick = next((i for i in rest if any(n & covers[i] and n <= have | covers[i] for n in needs)), rest[0])
            rest.remove(pick)
            order.append(pick)
            have |= covers[pick]
        return [leaves[i] for i in order], [covers[i] for i in order]

    def _position(self, conjunct: str, covers: list[frozenset[str]], *, floor: int) -> int:
        """Index of the first FROM item at which every input of the conjunct is present."""
        need = set(self.ctx.conjunct_inputs(conjunct))
        owner = self.ctx.result.facts.owners.get(conjunct)
        if owner is not None:
            need = {owner}
        have: set[str] = set()
        for i, cover in enumerate(covers):
            have |= cover
            if need <= have:
                return min(max(i, floor), len(covers) - 1)
        return len(covers) - 1

    def _head(self, leaf: PPlan, env: Env, outer: Env) -> tuple[exp.Expression, list[exp.Join]]:
        if isinstance(leaf, Op2) and self.ctx.kind(leaf) == "outer_join":
            return self._outer_join(leaf, env, outer)
        return self._item(leaf, env, outer), []

    def _item(self, leaf: PPlan, env: Env, outer: Env) -> exp.Expression:
        ctx = self.ctx
        if isinstance(leaf, Scan):
            return ctx.scan_item(leaf)
        if isinstance(leaf, BlockInput):
            return self._input_item(leaf, outer)
        kind = ctx.kind(leaf)
        if kind == "ref":
            return self._ref_item(leaf, outer)
        if kind == "outer_join":  # not the first item of its region: keep it together in parentheses
            head, joins = self._outer_join(leaf, env, outer)
            return self._parenthesized(head, joins)
        raise PrintError(f"operator {leaf.op!r} ({kind}) cannot be a FROM item; a nested scope must be wrapped in a ref")

    @staticmethod
    def _parenthesized(head: exp.Expression, joins: list[exp.Join]) -> exp.Expression:
        if not joins:
            return head
        head.set("joins", joins)
        return exp.Subquery(this=head)

    def _outer_join(self, node: Op2, env: Env, outer: Env) -> tuple[exp.Expression, list[exp.Join]]:
        e = self.ctx.entry(node.op, "outer_join")
        join_type = str(e.data.get("join_type") or "").upper()
        if join_type not in ("LEFT", "RIGHT", "FULL", "SEMI", "ANTI"):
            raise PrintError(f"outer join {node.op!r} has an unknown join_type {e.data.get('join_type')!r}")
        head, joins, _ = self._from_clause(node.left, env, outer, where_ok=False)
        right, right_joins, _ = self._from_clause(node.right, env, outer, where_ok=False)
        on = list(e.data.get("on") or ())
        shares = self._share_subqueries(on, list(e.data.get("subqueries") or ()))
        condition = sqltext.and_all(self._rewrite(sqltext.parse_predicate(sql), env, ids) for sql, ids in zip(on, shares))
        if condition is None:  # `a NATURAL LEFT JOIN b` without a common column: an outer join needs its ON clause
            condition = exp.true()
        args = {"kind": join_type} if join_type in ("SEMI", "ANTI") else {"side": join_type}
        joins.append(exp.Join(this=self._parenthesized(right, right_joins), on=condition, **args))
        return head, joins

    def _filtered_input(self, leaf: PPlan, filters: list[str], env: Env, outer: Env) -> exp.Expression:
        """`(SELECT * FROM <input> WHERE ...) AS <the input's own name>`: the only place SQL has for a filter that
        sits directly below an outer join on a single input. Everything above keeps referring to the same name."""
        if isinstance(leaf, BlockInput):
            name = boundary_ident(leaf.group, self.ctx.boundary_prefix)
        else:
            aliases = list(self.ctx.visible_aliases(leaf).values())
            if len(aliases) != 1:
                raise PrintError("a filter sits directly below an outer join on an input with several aliases; "
                                 "SQL has no place for it")
            name = aliases[0].copy()
        head, joins = self._head(leaf, env, outer)
        condition = sqltext.and_all(self._predicate(c, env) for c in self.ctx.by_written_order(filters))
        body = exp.Select(expressions=[exp.Star()], from_=exp.From(this=head), where=exp.Where(this=condition))
        if joins:
            body.set("joins", joins)
        return exp.Subquery(this=body, alias=exp.TableAlias(this=name))

    # ---------------------------------------------------------------------------------------------------------
    # refs: CTEs and derived tables
    # ---------------------------------------------------------------------------------------------------------

    def _ref_item(self, node: Op1, outer: Env) -> exp.Expression:
        ctx = self.ctx
        e = ctx.entry(node.op, "ref")
        alias = ctx.ref_alias(e)
        scope = ctx.nested_scope(e)
        name = e.data.get("name")
        if isinstance(node.input, BlockInput):
            self.used_inputs.append(node.input.group)  # also when the CTE it stands for is already defined
            if self.resolve is None:
                table = self._named_input(node.input, outer)
                table.set("alias", exp.TableAlias(this=alias))
                return table
        if name:
            ident = self.ctx.identifier(name)
            known = self.ctes.get(scope.id)
            if known is None:
                self._building.append(scope.id)
                self._pending_reads[scope.id] = set()
                try:
                    body = self.query(node.input, {}, top=True)  # a CTE cannot see the FROM items around its uses
                finally:
                    self._building.pop()
                known = self.ctes.setdefault(scope.id, _Cte(ident, scope.id, body, self._effective(node.input),
                                                            reads=self._pending_reads.pop(scope.id),
                                                            declared_in=e.data.get("declared_in")))
            elif known.plan != self._effective(node.input):
                raise NotABlock(f"CTE {name!r} is used in two places that cut it into blocks differently; a group that "
                                "is used in two or more places has to be a block of its own")
            known.used_in.append(ctx.owner.get(node.op, scope.parent or ""))
            if self._building:
                self._pending_reads[self._building[-1]].add(scope.id)
            table = exp.Table(this=ident.copy())
            if sqltext.ident_key(alias) != sqltext.ident_key(ident):
                table.set("alias", exp.TableAlias(this=alias))
            return table
        return exp.Subquery(this=self.query(node.input, outer, top=True), alias=exp.TableAlias(this=alias))

    def _base_tables_written(self, plan: PPlan) -> set[str]:
        """Names (folded) of the base tables that the SQL printed for `plan` names in a FROM clause of its own: its
        scans, those of its derived tables and of its predicate / select-item subqueries, not those of the CTEs it
        reads (they are written in their own definitions)."""
        ctx = self.ctx
        found: set[str] = set()
        todo, seen = [plan], set()
        while todo:
            n = todo.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            if isinstance(n, BlockInput):
                if self.resolve is not None:
                    todo.append(self.resolve(n.group).plan)
                continue
            if isinstance(n, Scan):
                table = ctx.entry(n.occ, "scan").data.get("table")
                if table:
                    found.add(sqltext.ident_key(str(table).split(".")[-1]))
                continue
            ids = list(n.conjuncts) if isinstance(n, Join) else [n.conjunct] if isinstance(n, Filter) else [n.op]
            for id_ in ids:
                for sid in ctx.entry(id_).data.get("subqueries") or ():
                    if sid in ctx.scopes:
                        todo.append(self.subquery_plans.get(sid, ctx.scopes[sid].plan))
            if isinstance(n, Op1) and ctx.kind(n) == "ref" and ctx.entry(n.op).data.get("name"):
                continue  # a use of a CTE: its body is written elsewhere
            todo.extend(children(n))
        return found

    def _effective(self, p: PPlan) -> PPlan:
        """The plan with every block input replaced by the fragment of its block (when blocks are inlined)."""
        if isinstance(p, BlockInput):
            return self._effective(self.resolve(p.group).plan) if self.resolve is not None else p
        if isinstance(p, Scan):
            return p
        if isinstance(p, Join):
            return Join(self._effective(p.left), self._effective(p.right), p.conjuncts)
        if isinstance(p, Filter):
            return Filter(self._effective(p.input), p.conjunct)
        if isinstance(p, Op1):
            return Op1(self._effective(p.input), p.op)
        return Op2(p.op, self._effective(p.left), self._effective(p.right))

    def _attach_ctes(self, statement: exp.Expression, home: str) -> None:
        """WITH clauses. When a ref entry carries `declared_in` (the scope whose WITH clause declares the CTE) and that
        scope is part of the statement and contains every use, the CTE goes exactly there. Otherwise: one WITH clause
        at the top of the statement, the way most queries are written; only when two DIFFERENT CTEs share a name (the
        corpus nests `WITH inner_cte AS ...` inside several CTE bodies) each of them goes to the innermost query that
        contains all of its uses, where it hides the other one exactly as it did in the query. Inside one WITH clause a CTE comes after the CTEs it reads; apart from that the order of
        the scopes in the import result decides."""
        if not self.ctes:
            return
        by_name: dict[str, list[str]] = {}
        for sid, cte in self.ctes.items():
            by_name.setdefault(sqltext.ident_key(cte.ident), []).append(sid)
        place: dict[str, str] = {}

        def target(where: str) -> exp.Expression | None:
            """The printed query of a scope; None for a scope that is not part of this statement (a fragment, or the
            scope that a closed form stands in for)."""
            return statement if where == home else self.ctes[where].body if where in self.ctes else self._queries.get(where)

        def placed(sid: str) -> str:
            if sid not in place:
                place[sid] = home  # while it is being computed; also the answer for a name that is unique
                paths = [path(use) for use in self.ctes[sid].used_in]
                # printed queries that contain every use, innermost first; the statement itself is always one
                shared = [s for s in paths[0] if all(s in p for p in paths[1:]) and target(s) is not None]
                declared = self.ctes[sid].declared_in
                if declared is not None and declared in shared:
                    place[sid] = declared  # where the query declares it (an importer may record that)
                elif len(by_name[sqltext.ident_key(self.ctes[sid].ident)]) > 1 and shared:
                    place[sid] = shared[0]
            return place[sid]

        def path(sid: str) -> list[str]:
            """The scopes whose queries contain the query of scope `sid`, innermost first."""
            out = [sid]
            while out[-1] != home:
                here = out[-1]
                above = placed(here) if here in self.ctes else self.ctx.scopes[here].parent if here in self.ctx.scopes else None
                if above is None or above in out:
                    out.append(home)
                else:
                    out.append(above)
            return out

        at: dict[str, list[str]] = {}
        for sid in self.ctes:
            at.setdefault(placed(sid), []).append(sid)
        rank = {sid: (self.ctx.scope_pos.get(sid, 0), i) for i, sid in enumerate(self.ctes)}
        for where, members in at.items():
            query = target(where)
            # A WITH item sees only the items BEFORE it. So a CTE whose body reads the base table `t` has to stay in
            # front of a sibling CTE that is itself called `t`, or its FROM t would silently read that CTE instead.
            name = {m: sqltext.ident_key(self.ctes[m].ident) for m in members}
            before = {m: {x for x in members if x != m and name[m] in self._base_tables_written(self.ctes[x].plan)}
                      for m in members}
            ordered: list[str] = []
            while len(ordered) < len(members):
                ready = [m for m in members if m not in ordered and self.ctes[m].reads & set(members) <= set(ordered)
                         and before[m] <= set(ordered)]
                if not ready:
                    raise PrintError("the CTEs of one WITH clause cannot be ordered: one reads a base table that has "
                                     "the name of a CTE which it also has to follow")
                ordered.append(min(ready, key=rank.__getitem__))
            query.set("with_", exp.With(expressions=[
                exp.CTE(this=self.ctes[m].body, alias=exp.TableAlias(this=self.ctes[m].ident.copy())) for m in ordered]))
