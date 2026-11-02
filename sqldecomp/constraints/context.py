"""Read-only indexes over one (memo, import result, schema) triple.

The propagation rules ask this object about ids; they never read the side table, the facts or SQLGlot themselves.
Every lookup is conservative: when the answer is not certain it is None, False or empty, and the calling rule then
derives nothing. A missing answer costs completeness, never soundness.

Column namespaces. The columns of a group are named as follows, and every ColumnRef that the rules emit follows it:

- scan, filter, inner join, outer join: `ColumnRef(occurrence id, column)` of the FROM items (the LEAVES) below;
- aggregate, computed projection, `ref`, set operation: `ColumnRef(operator id, output name)`;
- distinct, order, limit: the columns of their input (they rename nothing).

Facts may name a column one level too deep (a `ref` over a grouped CTE whose `Keep.source` is the base column and not
the aggregate's output column). `same_column` therefore compares columns after following the `keeps` of aggregates
and projections down to where the value comes from. It never follows a `ref`: two uses of one CTE share everything
below their `ref` operators, so below a `ref` a column no longer identifies one value per row.
"""
from __future__ import annotations

import re
from itertools import product
from typing import Iterable

import sqlglot
from sqlglot import exp

from sqldecomp.model import (ColumnRef, Equate, Filter, GroupId, ImportResult, Join, Keep, Memo, Op1, Op2, Scan, Schema,
                             ScopeInfo, Table, plan_children, walk)

PRODUCERS = ("aggregate", "project")  # Op1 kinds whose output columns are ColumnRef(operator id, output name)
TRANSPARENT = ("distinct", "order", "limit")  # Op1 kinds that keep the columns of their input
OUTER_TYPES = ("LEFT", "RIGHT", "FULL")
HALF_TYPES = ("SEMI", "ANTI")  # only the left side's columns come out

MAX_RENAMINGS = 16  # a key whose columns are kept under several names yields at most this many renamed copies

_GROUPING_SETS = re.compile(r"\b(ROLLUP|CUBE|GROUPING\s+SETS)\b", re.IGNORECASE)
_LATERAL = re.compile(r"\b(LATERAL|APPLY)\b", re.IGNORECASE)
# A set-returning function multiplies rows wherever Postgres evaluates it with the select list: in the select list
# itself, in ORDER BY and in DISTINCT ON. SQLGlot gives a few of them classes of their own (Explode,
# ExplodingGenerateSeries, ...), recognised by name; every other one arrives as an Anonymous function, exactly like a
# user-defined function, which may return a set as well. Both count as "not certainly one value per row".
_SET_RETURNING_MARKERS = ("explod", "unnest", "generateseries", "generate_series", "generate_subscripts", "flatten",
                          "inline")
# Column types between which `=` is as fine as the equality that makes a key unique; any other type is comparable
# with itself only. bigint = double precision is NOT in one family: both are compared as doubles, so two different
# bigint keys can equal one double. Neither is char = varchar: Postgres compares them as char, ignoring trailing
# blanks, so the two varchar keys 'a' and 'a ' equal one char value (both checked on Postgres 16).
_TYPE_FAMILIES = {
    "exact": {"SMALLINT", "INT", "INTEGER", "BIGINT", "TINYINT", "INT2", "INT4", "INT8", "SERIAL", "BIGSERIAL",
              "DECIMAL", "NUMERIC"},
    "float": {"REAL", "FLOAT", "DOUBLE", "DOUBLE PRECISION", "FLOAT4", "FLOAT8"},
    "text": {"VARCHAR", "TEXT", "CHARACTER VARYING", "STRING", "NVARCHAR"},
    "char": {"CHAR", "CHARACTER", "BPCHAR", "NCHAR"},
}


def _returns_set(f: exp.Func) -> bool:
    if isinstance(f, (exp.UDTF, exp.Anonymous)):
        return True
    names = (type(f).__name__.lower(), str(f.sql_name()).lower())
    return any(m in n for n in names for m in _SET_RETURNING_MARKERS)


def _type_family(declared: str | None) -> str | None:
    """The comparison family of a declared column type; None when the type is not known."""
    name = " ".join(re.sub(r"\(.*?\)", "", declared or "").upper().split())
    if not name or name == "UNKNOWN":
        return None
    return next((family for family, names in _TYPE_FAMILIES.items() if name in names), name)


# Functions SQLGlot gives a class of its own that are strict in their arguments (NULL in, NULL out), so a comparison
# over them is never TRUE when the column inside is NULL. Anything not listed (COALESCE, CASE, user-defined
# functions, ...) is not assumed strict.
_STRICT_FUNCS = (exp.Cast, exp.Lower, exp.Upper, exp.Trim, exp.Substring, exp.Length, exp.Abs, exp.Round, exp.Floor, exp.Ceil,
                 exp.Extract, exp.DateTrunc, exp.Date, exp.TimeToStr, exp.StrToTime, exp.StrToDate, exp.ToChar, exp.Neg)
_COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.GT, exp.LTE, exp.GTE, exp.Like, exp.ILike)
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)
_QUANTIFIERS = {"ALL", "ANY", "SOME"}


def _unparen(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _quantifier(node: exp.Expression) -> str | None:
    """`any` / `all` when the operand of a comparison is a quantified set (ANY / SOME / ALL over a subquery or an array,
    which SQLGlot gives as `Any`, `All` or an Anonymous function of that name), else None. `x op ANY (S)` is FALSE for
    an empty S and NULL otherwise when x is NULL (never TRUE, not strict); `x op ALL (S)` is TRUE for an empty S."""
    node = _unparen(node)
    if isinstance(node, exp.Any):
        return "any"
    if isinstance(node, exp.All):
        return "all"
    if isinstance(node, exp.Anonymous) and str(node.this).upper() in _QUANTIFIERS:
        return "all" if str(node.this).upper() == "ALL" else "any"
    return None


class Context:
    def __init__(self, memo: Memo, result: ImportResult, schema: Schema, *, constant_binding: bool = True):
        self.memo, self.result, self.schema = memo, result, schema
        self.constant_binding = constant_binding  # the constant-binding rule (rules.bind_constants)
        self.side = result.side
        self.dialect = result.dialect or "postgres"
        self._tables: dict[str, Table] = {t.name: t for t in schema.tables}
        self._table_of = self._index_scans()
        self._scopes: dict[str, ScopeInfo] = {s.id: s for s in result.scopes}
        self._equates: dict[str, list[Equate]] = {}
        for q in result.facts.equates:
            self._equates.setdefault(q.conjunct, []).append(q)
        self._keeps: dict[str, list[Keep]] = {}
        for k in result.facts.keeps:
            self._keeps.setdefault(k.op, []).append(k)
        self._group_of_op = self._index_operators()
        self._scope_of: dict[str, ScopeInfo] = {}  # distinct, top producer or set operation -> the scope it outputs
        self._producer_below: dict[str, str | None] = {}  # distinct -> the producer of its scope, if any
        self._index_scopes()
        (self.has_lateral, self.has_distinct_on, self.distinct_on_multiplies, self.windows_multiply,
         self.has_grouping_sets) = self._scan_sql_text()
        self._from_items, self._owner_of_ref = self._index_from_items()
        self._leaves: dict[GroupId, frozenset[str]] = {}
        self._open: set[GroupId] = set()
        self._alias_maps: dict[GroupId, dict[str, str | None]] = {}
        self._parsed: dict[str, exp.Expression | None] = {}
        self._ambiguous: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------------------------------------ ids

    def kind(self, op: str | None) -> str | None:
        entry = self.side.get(op) if op is not None else None
        return entry.kind if entry is not None else None

    def data(self, op: str) -> dict:
        entry = self.side.get(op)
        return entry.data if entry is not None and isinstance(entry.data, dict) else {}

    def table(self, occ: str) -> Table | None:
        name = self._table_of.get(occ)
        return self._tables.get(name) if name else None

    def _index_scans(self) -> dict[str, str | None]:
        plans = [self.result.plan] + [s.plan for s in self.result.scopes] + [i.plan for i in self.result.facts.inputs]
        found: dict[str, str | None] = {}
        for plan in plans:
            for node in walk(plan):
                if isinstance(node, Scan):
                    name = node.table.lower()
                    found[node.occ] = name if found.get(node.occ, name) == name else None  # two tables: not sure
        for occ, name in list(found.items()):
            declared = self.data(occ).get("table")
            if name and isinstance(declared, str) and declared.lower() != name:
                found[occ] = None
        return found

    def _index_operators(self) -> dict[str, GroupId | None]:
        found: dict[str, GroupId | None] = {}
        for g, exprs in self.memo.groups.items():
            for e in exprs:
                if e.kind in ("op1", "op2") and e.arg is not None:
                    found[e.arg] = g if found.get(e.arg, g) == g else None
        return found

    # --------------------------------------------------------------------------------------------- scopes

    def _index_scopes(self) -> None:
        """Walk the operator spine of every scope, top down: limit, order, distinct, projection, HAVING filters,
        aggregate. The scope's output list belongs to its topmost producer; without one, to the region itself."""
        claimed: dict[str, str] = {}
        for scope in self.result.scopes:
            node, spine, only_order_limit = scope.plan, [], True
            while True:
                if isinstance(node, Op1) and self.kind(node.op) in PRODUCERS + TRANSPARENT:
                    spine.append(node.op)
                    only_order_limit = only_order_limit and self.kind(node.op) in ("order", "limit")
                    node = node.input
                elif isinstance(node, Filter):
                    only_order_limit = False
                    node = node.input
                else:
                    break
            producers = [op for op in spine if self.kind(op) in PRODUCERS]
            owners = [op for op in spine if self.kind(op) == "distinct"] + producers[:1]
            if isinstance(node, Op2) and self.kind(node.op) == "setop" and only_order_limit:
                owners.append(node.op)
            for i, op in enumerate(spine):
                if self.kind(op) == "distinct":
                    below = [p for p in spine[i + 1:] if self.kind(p) in PRODUCERS]
                    self._producer_below[op] = below[0] if below else None
            for op in owners:
                if claimed.setdefault(op, scope.id) == scope.id:
                    self._scope_of[op] = scope
                else:
                    self._scope_of.pop(op, None)  # one operator on the spines of two scopes: not sure whose it is

    def _scan_sql_text(self) -> tuple[bool, bool, bool, bool, bool]:
        """What only the query TEXT shows: a LATERAL item may occur; a DISTINCT ON occurs; a DISTINCT ON expression
        may return a set; a NAMED window (WINDOW w AS ...) may return a set, which the select item `... OVER w` does
        not show; grouping sets occur (`GROUPING SETS ((), ())` has an empty grouping list and yet two rows). None of
        them can be tied to one operator from here, so each switches its guard for the whole query. Text that does
        not parse counts as all of them."""
        try:
            tree = sqlglot.parse_one(self.result.sql, dialect=self.dialect)
        except Exception:  # noqa: BLE001 - any parser failure means "not sure"
            return True, True, True, True, True
        if tree is None:
            return True, True, True, True, True
        lateral = tree.find(exp.Lateral) is not None or _LATERAL.search(self.result.sql) is not None
        on = [d.args["on"] for d in tree.find_all(exp.Distinct) if d.args.get("on")]
        windows = [w for sel in tree.find_all(exp.Select) for w in sel.args.get("windows") or ()]
        return (lateral, bool(on), any(_returns_set(f) for node in on for f in node.find_all(exp.Func)),
                any(_returns_set(f) for w in windows for f in w.find_all(exp.Func)),
                _GROUPING_SETS.search(self.result.sql) is not None)

    def _index_from_items(self) -> tuple[dict[str, set[str]], dict[str, str]]:
        """Per scope the ids of its FROM items (scans and `ref`s, not what lies below a `ref`), and for every `ref` the
        scope whose FROM clause it stands in."""
        items: dict[str, set[str]] = {}
        owner: dict[str, str] = {}
        for scope in self.result.scopes:
            todo = [scope.plan]
            while todo:
                node = todo.pop()
                if isinstance(node, Scan):
                    items.setdefault(scope.id, set()).add(node.occ)
                elif isinstance(node, Op1) and self.kind(node.op) == "ref":
                    items.setdefault(scope.id, set()).add(node.op)
                    owner[node.op] = scope.id
                else:
                    todo += plan_children(node)
        return items, owner

    def _outer_refs_below(self, scope_id: str | None, seen: set[str]) -> set[str]:
        """Every occurrence of an outer scope that a conjunct of this scope refers to, nested scopes included."""
        scope = self._scopes.get(scope_id)
        if scope is None or scope_id in seen:
            return set()
        seen.add(scope_id)
        out: set[str] = set()
        for node in walk(scope.plan):
            for conjunct in ([node.conjunct] if isinstance(node, Filter) else node.conjuncts if isinstance(node, Join) else ()):
                data = self.data(conjunct)
                out |= {r for r in data.get("outer_refs") or () if isinstance(r, str)}
                for sub in data.get("subqueries") or ():
                    out |= self._outer_refs_below(sub, seen)
        return out

    def lateral_suspect(self, ref_op: str) -> bool:
        """A derived table may be LATERAL: its rows then depend on the row of a sibling FROM item, and a key of one
        evaluation is not a key of the `ref`. Two signs, either is enough: the query text says LATERAL / APPLY (a CTE
        cannot be lateral), or a conjunct below the `ref` refers to a FROM item of the scope the `ref` stands in."""
        nested = self.data(ref_op).get("scope")
        scope = self._scopes.get(nested)
        if self.has_lateral and (scope is None or scope.kind != "cte"):
            return True
        siblings = self._from_items.get(self._owner_of_ref.get(ref_op), set())
        return bool(self._outer_refs_below(nested, set()) & siblings)

    # ------------------------------------------------------------------------------- leaves and aliases

    def leaves(self, g: GroupId) -> frozenset[str]:
        """Ids whose columns are visible in a group: the FROM items below it, down to the next operator that renames
        its output, plus the id of every outer join passed on the way. The same for every expression (M1)."""
        if g in self._leaves:
            return self._leaves[g]
        if g in self._open or g not in self.memo.groups:
            return frozenset()
        self._open.add(g)
        out: set[str] = set()
        for e in self.memo.groups[g]:
            kind = self.kind(e.arg)
            if e.kind == "scan":
                out.add(e.arg)
            elif e.kind in ("filter", "join") or (e.kind == "op1" and kind in TRANSPARENT):
                for c in e.children:
                    out |= self.leaves(c)
            elif e.kind == "op2" and kind == "outer_join" and len(e.children) == 2:
                out.add(e.arg)
                join_type = self.join_type(e.arg)
                visible = e.children if join_type in OUTER_TYPES else e.children[:1] if join_type in HALF_TYPES else ()
                for c in visible:
                    out |= self.leaves(c)
            elif e.arg is not None:
                out.add(e.arg)
        self._open.discard(g)
        self._leaves[g] = frozenset(out)
        return self._leaves[g]

    def join_type(self, op: str) -> str | None:
        value = self.data(op).get("join_type")
        return value.strip().upper() if isinstance(value, str) else None

    def _leaf_columns(self, leaf: str) -> dict[str, str | None] | None:
        """Lower-case name -> the column name as declared (None when two columns differ only by case or repeat), for
        a base table or for one use of a nested scope. None when the columns of the leaf are not known."""
        if self.kind(leaf) == "ref":
            scope = self._scopes.get(self.data(leaf).get("scope"))
            names = [o.name for o in scope.outputs] if scope is not None else None
        else:
            table = self.table(leaf)
            names = [c.name for c in table.columns] if table is not None and table.columns else None
        if names is None:
            return None
        lowered = [n.lower() for n in names]
        return {n.lower(): n if lowered.count(n.lower()) == 1 else None for n in names}

    def _aliases(self, g: GroupId) -> dict[str, str | None]:
        """FROM alias (lower case) -> leaf id; None when two leaves answer to the same alias."""
        if g not in self._alias_maps:
            out: dict[str, str | None] = {}
            for leaf in sorted(self.leaves(g)):
                if self.kind(leaf) not in ("scan", "ref"):
                    continue
                alias = self.data(leaf).get("alias") or self.data(leaf).get("table")
                if isinstance(alias, str) and alias:
                    out[alias.lower()] = None if alias.lower() in out else leaf
            self._alias_maps[g] = out
        return self._alias_maps[g]

    def _resolve(self, column: tuple[str, str] | None, g: GroupId) -> ColumnRef | None:
        """A parsed `alias.column` as a column of group g, or None."""
        if column is None:
            return None
        leaf = self._aliases(g).get(column[0])
        names = self._leaf_columns(leaf) if leaf is not None else None
        name = names.get(column[1]) if names is not None else None
        return ColumnRef(leaf, name) if name is not None else None

    # ---------------------------------------------------------------------------------------- SQL text

    def _parse(self, sql: str | None) -> exp.Expression | None:
        if not isinstance(sql, str):
            return None
        if sql not in self._parsed:
            try:
                self._parsed[sql] = sqlglot.parse_one(sql, dialect=self.dialect)
            except Exception:  # noqa: BLE001
                self._parsed[sql] = None
        return self._parsed[sql]

    @staticmethod
    def _plain_column(node: exp.Expression | None) -> tuple[str, str] | None:
        """(alias, column) in lower case when the node is exactly `alias.column`."""
        if not isinstance(node, exp.Column) or node.args.get("db") or node.args.get("catalog"):
            return None
        parts = (node.args.get("table"), node.this)
        if not all(isinstance(p, exp.Identifier) and p.name for p in parts):
            return None
        if any(p.args.get("quoted") and p.name != p.name.lower() for p in parts):
            return None  # a quoted mixed-case name is not the lower-case name the schema uses
        return parts[0].name.lower(), parts[1].name.lower()

    # ------------------------------------------------------------------------- NULL rejection, constants

    @staticmethod
    def _column_key(node: exp.Expression) -> tuple[str, str] | None:
        if isinstance(node, exp.Column) and node.table and isinstance(node.this, exp.Identifier):
            return node.table.lower(), node.name.lower()
        return None

    @classmethod
    def _strict_in(cls, node: exp.Expression, key: tuple[str, str]) -> bool:
        """The expression is NULL whenever column `key` is NULL (STRICT: NULL in, NULL out): the column itself, a
        strict function, an arithmetic operation or a comparison with such an operand, `x IN (a, b, ...)` over a
        non-empty value list when x is strict, `x BETWEEN lo AND hi` when x is strict. NOT strict, although never
        TRUE for a NULL column (that is `_rejects`' weaker notion): `x IN (subquery)` (FALSE for an empty result),
        a column as a BETWEEN bound (`5 BETWEEN NULL AND 3` is FALSE), a comparison with ANY / SOME (FALSE for an
        empty set) or ALL (`NULL <> ALL (empty)` is TRUE). The difference matters under NOT, `= FALSE`, `IS [NOT]
        NULL` and inside a CAST, where a FALSE turns into a TRUE. Nothing else is assumed."""
        node = _unparen(node)
        if cls._column_key(node) == key:
            return True
        if isinstance(node, _STRICT_FUNCS):
            return any(cls._strict_in(c, key) for c in node.args.values() if isinstance(c, exp.Expression))
        if isinstance(node, _COMPARISONS):
            if _quantifier(node.this) or _quantifier(node.expression):
                return False
            return cls._strict_in(node.this, key) or cls._strict_in(node.expression, key)
        if isinstance(node, _ARITHMETIC):
            return cls._strict_in(node.this, key) or cls._strict_in(node.expression, key)
        if isinstance(node, exp.Between):
            return cls._strict_in(node.this, key)
        if isinstance(node, exp.In):
            if node.args.get("query") is not None or node.args.get("unnest") is not None or not node.expressions:
                return False
            return cls._strict_in(node.this, key)
        return False

    @classmethod
    def _rejects(cls, node: exp.Expression, key: tuple[str, str]) -> bool:
        """The predicate is not TRUE whenever column `key` is NULL (so a filter keeping the rows where it is TRUE keeps
        no row with a NULL there): a strict comparison (NULL), `x IS NOT NULL` (FALSE), AND when either conjunct
        rejects, OR when both do, NOT of a strict expression (NOT NULL is NULL); and, at the top level only, the
        forms that are never TRUE for a NULL column without being strict: `x IN (subquery)` and `x = ANY (S)` when x
        is strict (NULL or FALSE), `BETWEEN` with a strict value in any of its three parts (NULL or FALSE). Never:
        `x op ALL (S)` (TRUE for an empty S), `IS NULL`, `IS [NOT] DISTINCT FROM`, COALESCE, CASE, anything else."""
        node = _unparen(node)
        if isinstance(node, exp.In):
            return cls._strict_in(node.this, key)
        if isinstance(node, exp.Between):
            return any(cls._strict_in(node.args[k], key) for k in ("this", "low", "high") if node.args.get(k) is not None)
        if isinstance(node, _COMPARISONS):
            q = _quantifier(node.expression) or _quantifier(node.this)
            if q == "all":
                return False
            if q == "any":
                side = node.this if _quantifier(node.expression) else node.expression
                return cls._strict_in(side, key)
        if isinstance(node, exp.And):
            return cls._rejects(node.this, key) or cls._rejects(node.expression, key)
        if isinstance(node, exp.Or):
            return cls._rejects(node.this, key) and cls._rejects(node.expression, key)
        if isinstance(node, exp.Not):
            inner = node.this
            while isinstance(inner, exp.Paren):
                inner = inner.this
            if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null) and not inner.args.get("negate"):
                return cls._strict_in(inner.this, key)  # NOT (x IS NULL) is x IS NOT NULL
            return cls._strict_in(inner, key)
        if isinstance(node, exp.Is):  # sqlglot 30: `x IS NOT NULL` is Is(x, Null, negate=True); `x IS NULL` is Is(x, Null)
            return bool(node.args.get("negate")) and isinstance(node.expression, exp.Null) and cls._strict_in(node.this, key)
        if isinstance(node, (exp.NullSafeEQ, exp.NullSafeNEQ)):
            return False
        return cls._strict_in(node, key)

    def null_rejected(self, conjunct: str, g: GroupId) -> frozenset[ColumnRef]:
        """The columns of group `g` (named as the rules name them: the FROM item that exposes them) that the conjunct's
        text proves non-null on every row that satisfies it (`_rejects`). A name the conjunct qualifies with an alias
        that is not a FROM item of `g`, or a text that does not parse, yields nothing."""
        tree = self._parse(self.side[conjunct].sql if conjunct in self.side else None)
        if tree is None:
            return frozenset()
        out: set[ColumnRef] = set()
        for key in {self._column_key(c) for c in tree.find_all(exp.Column)} - {None}:
            if self._rejects(tree, key):
                resolved = self._resolve(key, g)
                if resolved is not None:
                    out.add(self.norm(resolved))
        return frozenset(out)

    def constant_bound(self, conjunct: str, g: GroupId) -> frozenset[ColumnRef]:
        """The columns of group `g` that a top-level conjunct of the text equates with a literal of the column's own
        comparison family (`alias.column = literal`, a numeric literal against an exact or float column, a string
        literal against a column of any other known type, TRUE / FALSE against a boolean column; a CAST of a literal
        counts with the cast's type): the rows of the occurrence all carry that value. An unknown column type, a
        literal of another family, or anything below OR, NOT or a function yields nothing."""
        out: set[ColumnRef] = set()
        todo = [self._parse(self.side[conjunct].sql if conjunct in self.side else None)]
        while todo:
            node = todo.pop()
            while isinstance(node, exp.Paren):
                node = node.this
            if isinstance(node, exp.And):
                todo += [node.this, node.expression]
            elif isinstance(node, exp.EQ):
                for column, other in ((node.this, node.expression), (node.expression, node.this)):
                    key = self._column_key(column)
                    if key is None:
                        continue
                    literal_family = self._literal_family(other)
                    resolved = self._resolve(key, g)
                    if literal_family is None or resolved is None:
                        continue
                    family = _type_family(self._type_of(resolved))
                    if family is not None and family in literal_family:
                        out.add(self.norm(resolved))
        return frozenset(out)

    @staticmethod
    def _literal_family(node: exp.Expression) -> set[str] | None:
        """The comparison families a literal may be compared with as itself: a number with the exact numerics and the
        floats (the literal is numeric, the comparison is exact in both), a string with every non-numeric type (the
        string is coerced to the column's type), a boolean with a boolean column; a CAST of a literal: the cast's
        family. None for anything else (a column, a function, a parameter)."""
        while isinstance(node, exp.Paren):
            node = node.this
        if isinstance(node, exp.Neg):
            node = node.this
        if isinstance(node, exp.Cast) and isinstance(node.this, (exp.Literal, exp.Boolean)):
            family = _type_family(node.to.sql(dialect="postgres"))
            return {family} if family else None
        if isinstance(node, exp.Boolean):
            return {"BOOLEAN", "BOOL"}
        if isinstance(node, exp.Literal):
            if node.is_string:
                return {f for f in ("text", "char", "DATE", "TIMESTAMP", "TIMESTAMP WITHOUT TIME ZONE", "TIMESTAMPTZ", "TIME", "BOOLEAN", "BOOL", "UUID", "INTERVAL")}
            return {"exact", "float"}
        return None

    def _plain_equalities(self, sql: str | None) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        """Every top-level conjunct of the text that is exactly `alias.column = alias.column`. Splitting a top-level
        AND is sound (a match satisfies each of its conjuncts); nothing below an OR, a NOT or a function counts."""
        todo, out = [self._parse(sql)], []
        while todo:
            node = todo.pop()
            while isinstance(node, exp.Paren):
                node = node.this
            if isinstance(node, exp.And):
                todo += [node.this, node.expression]
            elif isinstance(node, exp.EQ):
                left, right = self._plain_column(node.this), self._plain_column(node.expression)
                if left is not None and right is not None:
                    out.append((left, right))
        return out

    # -------------------------------------------------------------------------------- column identity

    def norm(self, c: ColumnRef) -> ColumnRef:
        """A column named through an outer join's id is renamed to the one FROM item below that join that has it."""
        if self.kind(c.origin) != "outer_join":
            return c
        g = self._group_of_op.get(c.origin)
        if g is None:
            return c
        owners = []
        for leaf in sorted(self.leaves(g)):
            if self.kind(leaf) == "outer_join":
                continue
            names = self._leaf_columns(leaf)
            if names is None:
                return c  # an unknown column list could hide a second owner
            if c.column.lower() in names:
                owners.append(ColumnRef(leaf, names[c.column.lower()] or c.column))
        return owners[0] if len(owners) == 1 else c

    def _source(self, c: ColumnRef) -> ColumnRef:
        """Where a value comes from inside its scope: follows the keeps of aggregates and projections, never a ref."""
        seen: set[ColumnRef] = set()
        while True:
            c = self.norm(c)
            if c in seen or self.kind(c.origin) not in PRODUCERS or c.column in self._ambiguous_names(c.origin):
                return c
            seen.add(c)
            sources = {k.source for k in self._keeps.get(c.origin, ()) if k.name == c.column}
            if len(sources) != 1:
                return c
            c = next(iter(sources))

    def same_column(self, a: ColumnRef, b: ColumnRef) -> bool:
        return self.norm(a) == self.norm(b) or self._source(a) == self._source(b)

    # ------------------------------------------------------------------------------------------ keeps

    def _declared_names(self, op: str) -> list[str] | None:
        """Every output name of a renaming operator, duplicates included, when known."""
        if self.kind(op) == "ref":
            scope = self._scopes.get(self.data(op).get("scope"))
            return [o.name for o in scope.outputs] if scope is not None else None
        if op in self._scope_of:
            return [o.name for o in self._scope_of[op].outputs]
        select = self.data(op).get("select")
        if isinstance(select, list):
            return [s["alias"] for s in select if isinstance(s, dict) and isinstance(s.get("alias"), str)]
        return None

    def _ambiguous_names(self, op: str) -> frozenset[str]:
        """Output names of `op` that do not identify one column: declared twice, or kept from two different sources."""
        if op not in self._ambiguous:
            self._ambiguous[op] = frozenset()  # re-entrancy guard for malformed, cyclic keeps
            declared = [n.lower() for n in self._declared_names(op) or ()]
            by_name: dict[str, list[ColumnRef]] = {}
            for k in self._keeps.get(op, ()):
                by_name.setdefault(k.name, []).append(k.source)
            self._ambiguous[op] = frozenset(
                name for name, sources in by_name.items()
                if declared.count(name.lower()) > 1 or any(not self.same_column(sources[0], s) for s in sources[1:]))
        return self._ambiguous[op]

    def names_for(self, op: str, column: ColumnRef) -> tuple[str, ...]:
        """The output names under which operator `op` keeps an unmodified copy of `column` of its input."""
        bad = self._ambiguous_names(op)
        return tuple(sorted({k.name for k in self._keeps.get(op, ())
                             if k.name not in bad and self.same_column(k.source, column)}))

    def renamings(self, op: str, columns: Iterable[ColumnRef], allowed: Iterable[ColumnRef] | None = None
                  ) -> list[tuple[ColumnRef, ...]]:
        """Every way to name all of `columns` in the output of `op`, in the given order. Empty when one of them is
        not kept or, with `allowed`, is not one of those columns. No columns at all: one empty renaming."""
        allowed = None if allowed is None else list(allowed)
        options = []
        for c in columns:
            if allowed is not None and not any(self.same_column(c, a) for a in allowed):
                return []
            names = self.names_for(op, c)
            if not names:
                return []
            options.append(names)
        size = 1
        for names in options:
            size *= len(names)
        if size > MAX_RENAMINGS:
            options = [names[:1] for names in options]
        return [tuple(ColumnRef(op, n) for n in combo) for combo in product(*options)]

    # -------------------------------------------------------------------------------------- operators

    def join_equalities(self, conjuncts: Iterable[str], a: GroupId, b: GroupId) -> list[tuple[ColumnRef, ColumnRef]]:
        """(column of a, column of b) for every `column = column` fact among THESE conjuncts that spans a and b."""
        in_a, in_b = self.leaves(a), self.leaves(b)
        out = []
        for conjunct in conjuncts:
            for q in self._equates.get(conjunct, ()):
                x, y = self.norm(q.left), self.norm(q.right)
                if not self.comparable(x, y):
                    continue
                if x.origin in in_a and y.origin in in_b and x.origin not in in_b and y.origin not in in_a:
                    out.append((x, y))
                elif x.origin in in_b and y.origin in in_a and x.origin not in in_a and y.origin not in in_b:
                    out.append((y, x))
        return out

    def on_equalities(self, op: str, left: GroupId, right: GroupId) -> list[tuple[ColumnRef, ColumnRef]]:
        """(left column, right column) for every ON conjunct of an outer join that is a plain `column = column`."""
        on = self.data(op).get("on")
        out = []
        for sql in on if isinstance(on, list) else ():
            for pair in self._plain_equalities(sql):
                for x, y in (pair, pair[::-1]):
                    lcol, rcol = self._resolve(x, left), self._resolve(y, right)
                    if lcol is not None and rcol is not None and self._resolve(x, right) is None \
                            and self._resolve(y, left) is None and self.comparable(lcol, rcol):
                        out.append((lcol, rcol))
        return out

    def grouping(self, op: str) -> tuple[int, list[ColumnRef]] | None:
        """(number of grouping expressions, those that are plain columns). None when the GROUP BY is unknown or uses
        grouping sets, whose subtotal rows repeat grouping values."""
        entry = self.side.get(op)
        group_by = self.data(op).get("group_by")
        if entry is None or not isinstance(group_by, list) or self.has_grouping_sets:
            return None
        if _GROUPING_SETS.search(entry.sql or ""):
            return None
        if any(not isinstance(s, str) or _GROUPING_SETS.search(s) for s in group_by):
            return None
        return len(group_by), [self.norm(c) for c in self.result.facts.group_cols.get(op, ())]

    def _single_valued(self, items: object, field: str = "sql") -> bool:
        """Every `{field: sql}` of a side-table list parses and calls nothing that may return a set of rows."""
        if self.windows_multiply or not isinstance(items, list) or not items:
            return False
        for item in items:
            node = self._parse(item.get(field)) if isinstance(item, dict) else None
            if node is None or any(_returns_set(f) for f in node.find_all(exp.Func)):
                return False
        return True

    def row_preserving(self, op: str) -> bool:
        """The select list of a projection or an aggregate returns one row per input row (per group): Postgres expands
        a set-returning function of the select list AFTER grouping, so it repeats groups, and even the one row of a
        global aggregate."""
        return self._single_valued(self.data(op).get("select"))

    def order_keeps_rows(self, op: str) -> bool:
        """ORDER BY only sorts, unless a key calls a set-returning function: Postgres evaluates the keys with the
        select list, so such a key repeats every row."""
        return self._single_valued(self.data(op).get("keys"))

    def _type_of(self, c: ColumnRef) -> str | None:
        """The declared type behind a column: follows unmodified copies through aggregates, projections and `ref`s
        down to a base column. NEVER through a set operation: `SELECT k FROM t UNION SELECT 1.5` turns a bigint k into
        double precision, so its output has no known type even if a fact calls it a copy."""
        seen: set[ColumnRef] = set()
        while c not in seen:
            seen.add(c)
            c = self.norm(c)
            table = self.table(c.origin)
            if table is not None:
                return next((col.type for col in table.columns if col.name == c.column), None)
            if self.kind(c.origin) not in PRODUCERS + ("ref",):
                return None
            sources = {k.source for k in self._keeps.get(c.origin, ()) if k.name == c.column}
            if len(sources) != 1 or c.column in self._ambiguous_names(c.origin):
                return None
            c = next(iter(sources))
        return None

    def comparable(self, a: ColumnRef, b: ColumnRef) -> bool:
        """`a = b` tells the values of a key column apart exactly as the key does: both types are known and of one
        family. Otherwise the equality says nothing about how many rows of the keyed side one row can meet."""
        fa, fb = _type_family(self._type_of(a)), _type_family(self._type_of(b))
        return fa is not None and fa == fb

    def distinct_outputs(self, op: str, child: GroupId) -> frozenset[ColumnRef] | None:
        """The columns a DISTINCT returns: the select list of its scope, named as columns of its input."""
        scope = self._scope_of.get(op)
        if scope is None or not scope.outputs or op not in self._producer_below:
            return None
        producer = self._producer_below[op]
        if producer is not None:
            return self._named_outputs(producer, scope)
        columns = [self._resolve(self._plain_column(self._parse(o.sql)), child) for o in scope.outputs]
        return frozenset(columns) if all(c is not None for c in columns) else None

    def setop_outputs(self, op: str) -> frozenset[ColumnRef] | None:
        scope = self._scope_of.get(op)
        return self._named_outputs(op, scope) if scope is not None and scope.outputs else None

    def setop_kind(self, op: str) -> str | None:
        value = self.data(op).get("op")
        return " ".join(value.upper().split()) if isinstance(value, str) else None

    def _named_outputs(self, op: str, scope: ScopeInfo) -> frozenset[ColumnRef] | None:
        """The outputs of a scope as columns of the operator that produces them. None when two outputs share a name
        or the operator's own select list disagrees with the scope."""
        names = [o.name for o in scope.outputs]
        if len({n.lower() for n in names}) != len(names):
            return None
        select = self.data(op).get("select")
        if isinstance(select, list):
            if len(select) != len(names):
                return None
            for item, name in zip(select, names):
                alias = item.get("alias") if isinstance(item, dict) else None
                if isinstance(alias, str) and alias.lower() != name.lower():
                    return None
        return frozenset(ColumnRef(op, n) for n in names)
