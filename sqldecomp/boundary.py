"""The boundary information of groups and blocks (D6): what the scoring functions are told about a block's edges.

Everything here is read from the components that own the facts, never re-derived:

- which columns a group exposes is the PRINTER's choice (`PrintContext.needed_columns`: the columns of the aliases
  visible above the plan that the rest of the query reads outside it, named `<alias>__<column>` at a boundary inside a
  FROM clause; at the top of a scope the scope's own output list; `SELECT *` when nothing is read by name);
- which keys and foreign keys hold on a group is component 8's (`derive_constraints`); a boundary keeps exactly the
  keys and FKs whose columns are all exposed;
- a column's type is the schema's, when the column traces back to a base-table column through the importer's `keeps`
  facts (a CTE output that copies an aggregate's grouping column, which copies a scan column); else "unknown";
- the renaming between two occurrences of one signature is `memo.placeholders`, which is how the exposed columns of a
  class are UNIONED over its occurrences and written in the representative's names.
"""
from __future__ import annotations

from typing import Callable

from sqlglot import exp

import hashlib
import json

from sqldecomp.blocks import Blocks
from sqldecomp.memo import placeholders
from sqldecomp.model import ColumnRef, GroupConstraints, GroupId, ImportResult, Memo, OutputColumn, Schema
from sqldecomp.printer import Printer, PrintError, sqltext
from sqldecomp.printer.build import Builder
from sqldecomp.printer.context import BlockInput
from sqldecomp.scoring import ExposedColumn

PRODUCERS = ("aggregate", "project", "setop")


def query_fingerprint(result: ImportResult) -> str:
    """What tells one query's positions from another's: a digest of the SQL text and the side table (the same seed
    as the signature module's `free:` tokens)."""
    seed = json.dumps([result.sql, sorted((e.id, e.kind, e.sql) for e in result.side.values())])
    return hashlib.sha1(seed.encode()).hexdigest()[:12]


class Boundary:
    """Boundary information of one query. Built once; every method is cached per group."""

    def __init__(self, memo: Memo, result: ImportResult, blocks: Blocks, printer: Printer,
                 constraints: dict[GroupId, GroupConstraints] | None = None, schema: Schema | None = None):
        self.memo, self.result, self.blocks, self.printer = memo, result, blocks, printer
        self.constraints = constraints or {}
        self.schema = schema
        self.side = result.side
        self.fingerprint = query_fingerprint(result)
        self.scopes = {s.id: s for s in result.scopes}
        self.keeps: dict[str, list] = {}
        for k in result.facts.keeps:
            self.keeps.setdefault(k.op, []).append(k)
        # (scope id, alias key) -> occurrence id, for every FROM item (scan or ref use)
        self.alias_occ: dict[tuple[str | None, str], str] = {}
        self.occ_alias: dict[str, str] = {}
        for id_, e in self.side.items():
            if e.kind in ("scan", "ref"):
                alias = e.data.get("alias") or e.data.get("name") or e.data.get("table")
                if alias:
                    self.occ_alias[id_] = str(alias)
                    # keyed as the printer keys aliases (a quoted upper-case alias keeps its case)
                    self.alias_occ.setdefault((blocks.owner_scope(id_), sqltext.ident_key(printer.ctx.identifier(str(alias)))), id_)
        self._columns: dict[GroupId, tuple[tuple[ExposedColumn, ...], bool]] = {}
        self._placeholders: dict[GroupId, dict[str, str] | None] = {}
        self._scope_outputs: dict[GroupId, tuple[str, ...] | None] = {}
        self._null_extended: dict[GroupId, frozenset[str]] = {}

    def null_extended_occurrences(self, group: GroupId) -> frozenset[str]:
        """The FROM items (scans and ref uses) that an outer join INSIDE the group NULL-extends: the right side of a
        LEFT join, the left side of a RIGHT join, both sides of a FULL join, over the group's whole expansion from
        base tables, in the join's own scope. A column of such an item that the group exposes is `nullable`.
        """
        if group not in self._null_extended:
            found: set[str] = set()
            try:
                fragment = self.blocks._expand(group, frozenset())
            except Exception:  # noqa: BLE001 - a group that is no block expands to nothing here
                fragment = {}
            for g, e in (fragment or {}).items():
                if e.kind != "op2" or len(e.children) != 2:
                    continue
                entry = self.side[e.arg]
                kind = str(entry.data.get("join_type") or "") if entry.kind == "outer_join" else ""
                if kind not in ("LEFT", "RIGHT", "FULL"):
                    continue
                scope = self.blocks.scope_of(g)
                left, right = e.children
                for side in ([right] if kind == "LEFT" else [left] if kind == "RIGHT" else [left, right]):
                    found.update(occ for occ in self.blocks.occurrences_under(side) if self.blocks.owner_scope(occ) == scope)
            self._null_extended[group] = frozenset(found)
        return self._null_extended[group]

    # ---------------------------------------------------------------------------------------------------------
    # exposed columns of one occurrence
    # ---------------------------------------------------------------------------------------------------------

    def columns(self, group: GroupId) -> tuple[tuple[ExposedColumn, ...], bool]:
        """(the columns the group exposes to the rest of the query, star?). The printer's choice, read from the
        printer: at the top of a scope (the root included) the scope's output list; inside a region the columns the
        query reads outside the group, `<alias>__<column>`; `star` when nothing is read by name (`SELECT *`), and the
        columns are then every column of the visible FROM items, as far as the schema and the nested scopes say."""
        if group not in self._columns:
            self._columns[group] = self._compute_columns(group)
        return self._columns[group]

    def columns_with(self, group: GroupId, expose: list[tuple[str, str]] | None) -> tuple[tuple[ExposedColumn, ...], bool]:
        """The columns `print_block(..., expose=expose)` makes THIS position expose, in the printed order: the `expose`
        pairs (the union of the class's exposed columns written in this position's aliases, `Optimum.expose_at`, the
        columns propagated from its readers' unions included) in the given order, each a column of `columns(group)`
        when this position reads it and a further column otherwise, then the columns of `columns(group)` that
        `expose` does not name; exactly the select list the printer builds (`Builder._select_list`). With no `expose`
        this is `columns(group)`; with one, `star` is False (the printer writes the columns out). A position whose
        record lists only `columns(group)` would describe fewer columns than its text exposes and than its readers
        may use."""
        columns, star = self.columns(group)
        if not expose:
            return columns, star
        ctx = self.printer.ctx
        scope = self.blocks.scope_of(group)
        nullable = self.null_extended_occurrences(group)
        ph = self.placeholder_map(group) or {}
        by_key = {} if star else {(c.alias, c.column): c for c in columns}
        ordered: list[ExposedColumn] = []
        seen: set[tuple[str, str]] = set()
        for alias, column in expose:
            alias_ident, column_ident = ctx.identifier(str(alias)), ctx.identifier(str(column))
            key = (sqltext.ident_key(alias_ident), sqltext.ident_key(column_ident))
            if key in seen:
                continue
            seen.add(key)
            if key in by_key:
                ordered.append(by_key.pop(key))
                continue
            occ = self.alias_occ.get((scope, key[0]))
            p = ph.get(occ or "", "")
            ordered.append(ExposedColumn(sqltext.join_names([alias_ident, column_ident]).name, key[0], key[1], self.type_of(occ, key[1]),
                                         occ or "", p, f"{p or key[0]}.{key[1]}", occ in nullable))
        return tuple(ordered + list(by_key.values())), False

    def _compute_columns(self, group: GroupId) -> tuple[tuple[ExposedColumn, ...], bool]:
        sid = self.blocks.scope_top_of(group)
        plan = self.printer.group_plan(group)
        ph = self.placeholder_map(group)
        if sid is not None or not self.blocks.is_region_level(group):
            scope = self.scopes[sid] if sid is not None else self.scopes[self.printer.ctx.scope_of(plan).id]
            return tuple(self._output_column(o, scope.id, ph, group) for o in scope.outputs), False
        ctx = self.printer.ctx
        try:
            needed = ctx.needed_columns(plan)
        except PrintError:
            needed = []
        nullable = self.null_extended_occurrences(group)
        if needed:
            out = []
            for n in needed:
                occ = self.alias_occ.get((self.blocks.scope_of(group), n.key[0]))
                p = (ph or {}).get(occ or "", "")
                out.append(ExposedColumn(Builder._exposed_name(n).name, n.key[0], n.key[1], self.type_of(occ, n.key[1]),
                                         occ or "", p, f"{p or n.key[0]}.{n.key[1]}", occ in nullable))
            return tuple(out), False
        visible = ctx.visible_aliases(plan)
        if not visible or ctx.star_reads_an_unnamed_column(plan):
            return (), False
        out = []
        for alias_key in visible:  # `SELECT *`: every column of every visible FROM item, in FROM order
            occ = self.alias_occ.get((self.blocks.scope_of(group), alias_key))
            p = (ph or {}).get(occ or "", "")
            for column in self.columns_of_occurrence(occ):
                out.append(ExposedColumn(column, alias_key, column, self.type_of(occ, column), occ or "", p,
                                         f"{p or alias_key}.{column}", occ in nullable))
        return tuple(out), True

    def _output_column(self, o: OutputColumn, sid: str, ph: dict[str, str] | None, group: GroupId | None = None) -> ExposedColumn:
        alias, column, occ, canonical, computed_type = "", o.sql, "", o.sql, "unknown"
        try:
            tree = sqltext.parse_item(o.sql)
            if isinstance(tree, exp.Alias):
                tree = tree.this
            if isinstance(tree, exp.Column) and tree.table:
                alias, column = sqltext.ident_key(tree.args.get("table")), sqltext.ident_key(tree.this)
                occ = self.alias_occ.get((sid, alias), "")
            else:
                computed_type = self._computed_type(tree, sid)
            canonical = self.canonical_text(tree, sid, ph)
        except PrintError:
            pass
        nullable = bool(occ) and group is not None and occ in self.null_extended_occurrences(group)
        return ExposedColumn(o.name, alias, column, self.type_of(occ or None, column) if alias else computed_type, occ,
                             (ph or {}).get(occ, "") if occ else "", canonical, nullable)

    def _computed_type(self, tree: exp.Expression, sid: str) -> str:
        """The type of a computed output where SQL fixes it without looking at the data: COUNT is a bigint; MIN and
        MAX of a column have the column's type. Everything else is "unknown" (a sum or average changes the type)."""
        if isinstance(tree, exp.Count):
            return "BIGINT"
        if isinstance(tree, (exp.Min, exp.Max)) and isinstance(tree.this, exp.Column) and tree.this.table:
            occ = self.alias_occ.get((sid, sqltext.ident_key(tree.this.args.get("table"))))
            return self.type_of(occ, sqltext.ident_key(tree.this.this))
        return "unknown"

    def scope_outputs(self, group: GroupId) -> tuple[str, ...] | None:
        """The alias-free expressions of the outputs of the scope whose whole plan the group is (the root included),
        in order; None for a group that is no scope top. A plain select list is no memo node, so it is not part
        of the group's signature: two branches `SELECT o.o_cust AS k FROM orders o WHERE ...` and `SELECT o2.o_id AS k
        FROM orders o2 WHERE ...` are one signature and two different tasks; the class key adds this list for
        scope-top outputs."""
        if group in self._scope_outputs:
            return self._scope_outputs[group]
        sid = self.blocks.scope_top_of(group)
        if sid is None:
            self._scope_outputs[group] = None
            return None
        ph = self.placeholder_map(group)
        out = []
        for o in self.scopes[sid].outputs:
            try:
                tree = sqltext.parse_item(o.sql)
                if isinstance(tree, exp.Alias):
                    tree = tree.this
                out.append(self.canonical_text(tree, sid, ph))
            except PrintError:
                out.append(o.sql)
        self._scope_outputs[group] = tuple(out)
        return self._scope_outputs[group]

    def canonical_text(self, tree: exp.Expression, sid: str, ph: dict[str, str] | None) -> str:
        """An expression with every FROM alias replaced by its placeholder (the alias-free name of the signature). A
        column of an ENCLOSING query (a correlated select item) is written `free:<query>:<occurrence>`, as the
        signature module writes correlated references: it names that occurrence and no other, so two bodies that
        return outer columns of different relations under one alias never look alike."""
        tree = tree.copy()
        for column in tree.find_all(exp.Column):
            if column.table:
                alias = sqltext.ident_key(column.args.get("table"))
                occ = self.alias_occ.get((sid, alias))
                p = (ph or {}).get(occ or "")
                if p:
                    column.set("table", exp.to_identifier(p))
                    continue
                outer = self.blocks.resolve_alias(sid, alias)
                if outer is not None and self.blocks.owner_scope(outer) != sid:
                    column.set("table", exp.to_identifier(f"free:{self.fingerprint}:{outer}"))
                elif occ is None:  # an alias no scope defines: never shared
                    column.set("table", exp.to_identifier(f"unknown:{self.fingerprint}:{sid}:{alias}"))
        return sqltext.to_sql(tree)

    def columns_of_occurrence(self, occ: str | None) -> list[str]:
        """The column names a FROM item offers: the schema's for a table, the nested scope's outputs for a ref."""
        e = self.side.get(occ or "")
        if e is None:
            return []
        if e.kind == "scan":
            table = e.data.get("table")
            if table and self.schema is not None:
                try:
                    return [c.name for c in self.schema.table(table).columns]
                except KeyError:
                    return []
            return list(e.data.get("columns") or [])
        if e.kind == "ref":
            scope = self.scopes.get(str(e.data.get("scope")))
            return [o.name for o in scope.outputs] if scope is not None else []
        return []

    def type_of(self, origin: str | None, column: str, _seen: frozenset = frozenset()) -> str:
        """The declared type of a column, followed back through the importer's keeps to a base-table column."""
        e = self.side.get(origin or "")
        if e is None or (origin, column.lower()) in _seen:
            return "unknown"
        if e.kind == "scan":
            table = e.data.get("table")
            if not table or self.schema is None:
                return "unknown"
            try:
                t = self.schema.table(table)
            except KeyError:
                return "unknown"
            for c in t.columns:
                if c.name == column.lower():
                    return c.type
            return "unknown"
        for k in self.keeps.get(origin or "", ()):
            if k.name.lower() == column.lower():
                return self.type_of(k.source.origin, k.source.column, _seen | {(origin, column.lower())})
        return "unknown"

    def placeholder_map(self, group: GroupId) -> dict[str, str] | None:
        if group not in self._placeholders:
            self._placeholders[group] = placeholders(self.memo, self.result, group)
        return self._placeholders[group]

    # ---------------------------------------------------------------------------------------------------------
    # constraints through the boundary
    # ---------------------------------------------------------------------------------------------------------

    def exposed_constraints(self, group: GroupId, columns: tuple[ExposedColumn, ...], star: bool
                            ) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...]]:
        """The keys and FKs of the group whose columns are ALL exposed, as tuples of exposed names."""
        c8 = self.constraints.get(group)
        if c8 is None:
            return (), ()
        keys, fks = [], []
        for key in sorted(c8.keys, key=lambda k: sorted((c.origin, c.column) for c in k.columns)):
            names = [self.exposed_name(group, c, columns, star) for c in sorted(key.columns, key=lambda c: (c.origin, c.column))]
            if None not in names:
                keys.append(tuple(names))
        for fk in sorted(c8.fks, key=lambda f: [(c.origin, c.column) for c in f.columns]):
            names = [self.exposed_name(group, c, columns, star) for c in fk.columns]
            if None not in names:
                fks.append((tuple(names), fk.ref_table, tuple(fk.ref_columns)))
        return tuple(keys), tuple(fks)

    def exposed_weak_keys(self, group: GroupId, columns: tuple[ExposedColumn, ...], star: bool) -> tuple[tuple[str, ...], ...]:
        """The weak keys of the group (component 8: keys of the rows that are non-null in their nullable columns) whose
        columns are ALL exposed, as tuples of exposed names, in a fixed order."""
        c8 = self.constraints.get(group)
        if c8 is None:
            return ()
        out = []
        for weak in sorted(c8.weak, key=lambda w: sorted((c.origin, c.column) for c in w.columns)):
            names = [self.exposed_name(group, c, columns, star) for c in sorted(weak.columns, key=lambda c: (c.origin, c.column))]
            if None not in names and tuple(names) not in out:
                out.append(tuple(names))
        return tuple(out)

    def exposed_name(self, group: GroupId, ref: ColumnRef, columns: tuple[ExposedColumn, ...], star: bool) -> str | None:
        """The exposed name that IS the column `ref` of the group, or None when the boundary does not expose it."""
        e = self.side.get(ref.origin)
        if e is None:
            return None
        names = [c.name for c in columns]
        if e.kind in PRODUCERS:  # a column made by an operator keeps its output name above that operator
            return ref.column if names.count(ref.column) == 1 else None
        aliases: list[str] = []
        if e.kind in ("scan", "ref"):
            aliases = [self.occ_alias.get(ref.origin, "")]
        elif e.kind == "outer_join":  # named through the outer join: the one visible FROM item that has the column
            for occ in e.data.get("exposes") or ():
                if ref.column.lower() in {c.lower() for c in self.columns_of_occurrence(occ)}:
                    aliases.append(self.occ_alias.get(occ, ""))
            if len(aliases) != 1:
                return None
        found = [c.name for c in columns
                 if c.alias and any(c.alias == sqltext.ident_key(a) for a in aliases if a) and c.column.lower() == ref.column.lower()]
        return found[0] if len(found) == 1 and names.count(found[0]) == 1 else None

    # ---------------------------------------------------------------------------------------------------------
    # the union over the occurrences of a class (D6)
    # ---------------------------------------------------------------------------------------------------------

    def union_columns(self, representative: GroupId, others: list[GroupId]) -> tuple[tuple[ExposedColumn, ...], bool, list[str]]:
        """The exposed columns of the representative, plus every column another occurrence of the same signature
        exposes and the representative does not, renamed into the representative's aliases through the signature's
        placeholders. Returns (columns, star, notes); a note names a column that could not be carried over."""
        columns, star = self.columns(representative)
        out = list(columns)
        seen = {self._union_key(c) for c in out}
        notes: list[str] = []
        rep_map = self.placeholder_map(representative)
        back = {v: k for k, v in (rep_map or {}).items()}  # placeholder -> the representative's occurrence
        for g in others:
            more, star_g = self.columns(g)
            star = star or star_g
            for c in more:
                key = self._union_key(c)
                if key in seen:
                    continue
                if c.placeholder and c.placeholder in back:
                    occ = back[c.placeholder]
                    alias = self.occ_alias.get(occ, c.alias)
                    name = c.name if self.blocks.scope_top_of(representative) is not None else f"{alias}__{c.column}"
                    carried = ExposedColumn(name, sqltext.ident_key(alias), c.column, c.type, occ, c.placeholder, c.canonical)
                elif not c.alias:  # a computed scope output: the same expression, under its own name
                    carried = ExposedColumn(c.name, "", c.column, c.type, "", "", c.canonical)
                else:
                    notes.append(f"column {c.name} of occurrence {g} has no counterpart in {representative}")
                    continue
                if self._union_key(carried) not in seen:
                    seen.add(self._union_key(carried))
                    out.append(carried)
        return tuple(out), star, notes

    def columns_at(self, group: GroupId, union: tuple[ExposedColumn, ...]) -> list[tuple[str, str]]:
        """The union of a class's exposed columns, written in the aliases of THIS occurrence: (alias, column) pairs to
        hand `print_block(..., expose=...)`, so that every occurrence prints one text up to renaming (OPT-O1). A column
        whose placeholder this occurrence does not bind (never for two occurrences of one signature) is left out;
        computed scope outputs have no alias and are never exposed this way (a scope's select list is its own)."""
        return [(alias, column) for _, alias, column in self.exposures_at(group, union)]

    def exposures_at(self, group: GroupId, union: tuple[ExposedColumn, ...]) -> list[tuple[str, str, str]]:
        """`columns_at` with the occurrence that exposes each column at this position: (occurrence, alias, column).
        Whoever prints a decomposition needs the occurrence to see whether the column lies inside one of the block's
        inputs, which must then expose it too (`Optimum.expose_at`)."""
        ph = self.placeholder_map(group) or {}
        back = {v: k for k, v in ph.items()}
        under = self.blocks.occurrences_under(group)
        out: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for c in union:
            if not c.alias:
                continue
            occ = back.get(c.placeholder) if c.placeholder else (c.occurrence if c.occurrence in under else None)
            alias = self.occ_alias.get(occ) if occ else None
            if alias and (alias.lower(), c.column.lower()) not in seen:
                seen.add((alias.lower(), c.column.lower()))
                out.append((occ, alias, c.column))
        return out

    @staticmethod
    def _union_key(c: ExposedColumn) -> str:
        """Two exposed columns of two occurrences are the same column when their alias-free forms are equal."""
        return c.canonical.lower() if c.canonical else f"{c.alias}.{c.column}".lower()
