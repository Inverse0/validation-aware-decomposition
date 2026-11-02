"""The propagation rules, one function per operator.

Each rule takes the constraints already derived for the child occurrences and returns what THIS expression proves
about its own occurrence. The engine takes the union over all expressions of an occurrence (sound by M1).

Definitions. A KEY is a set K of output columns such that no two output rows agree on all columns of K, where NULL
agrees with NULL (the test is `GROUP BY K HAVING COUNT(*) > 1`). The empty key says "at most one row". An FK is a
tuple of output columns c with a base table T and columns k such that every row whose c values are all non-null has
a matching row in T.k. A WEAK KEY (the declared UNIQUE constraints with a nullable column) is a pair
(K, N), N a non-empty subset of K, such that no two output rows that are non-null in every column of N agree on all
columns of K: a key of the rows that are non-null in N. Every rule is valid under bag semantics, and an operator
that no rule names derives nothing.

Weak keys follow the keys with three differences, each proved where it is used: (a) a weak key may serve as the
EQUATED key of the other side of a join (the join condition being TRUE, the equated values are non-null); (b) a
weak key is PROMOTED where its nullable columns are null-rejected for the rows of the occurrence (a conjunct that is
never TRUE when the column is NULL: `IS NOT NULL`, a comparison, an inner-join equality), its N shrinking to the
columns not yet rejected and the key becoming a `Key` when N is empty; (c) it never survives where the keys are
dropped, and never crosses the null-extended side of an outer join. The CONSTANT-BINDING rule (on by default,
switched by `ctx.constant_binding`) removes a column equated with a literal of the column's own comparison family
from a key or a weak key.
"""
from __future__ import annotations

from typing import Callable, Iterable

from sqldecomp.constraints.context import HALF_TYPES, OUTER_TYPES, Context
from sqldecomp.model import ColumnRef, FKey, GroupConstraints, GroupId, Key, MExpr, WeakKey


def apply(ctx: Context, e: MExpr, constraints_of: Callable[[GroupId], GroupConstraints]) -> GroupConstraints:
    """What one expression proves about its occurrence. Anything unexpected derives nothing."""
    kids = [constraints_of(c) for c in e.children]
    if e.kind == "scan" and e.arg is not None and not kids:
        return scan(ctx, e.arg)
    if e.kind == "filter" and len(kids) == 1:
        return filter_(ctx, e.arg, e.children[0], kids[0])
    if e.kind == "join" and len(kids) == 2:
        return inner_join(ctx, e.conjuncts, e.children[0], e.children[1], kids[0], kids[1])
    if e.kind == "op1" and e.arg is not None and len(kids) == 1:
        return op1(ctx, e.arg, e.children[0], kids[0])
    if e.kind == "op2" and e.arg is not None and len(kids) == 2:
        return op2(ctx, e.arg, e.children[0], e.children[1], kids[0], kids[1])
    return GroupConstraints()


# ------------------------------------------------------------------------------------------------ helpers


def keep_all(child: GroupConstraints) -> GroupConstraints:
    """A copy: the keys, FKs and weak keys of the input, for an operator that keeps every row and column."""
    return GroupConstraints(set(child.keys), set(child.fks), set(child.weak))


def _covers(ctx: Context, column: ColumnRef, columns: Iterable[ColumnRef]) -> bool:
    return any(ctx.same_column(column, c) for c in columns)


def _weak_minus(ctx: Context, w: WeakKey, rejected: Iterable[ColumnRef]) -> Key | WeakKey:
    """`w` with every null-rejected column taken out of its nullable set: a `Key` when nothing nullable is left."""
    rejected = list(rejected)
    nullable = frozenset(c for c in w.nullable if not _covers(ctx, c, rejected))
    return Key(w.columns) if not nullable else WeakKey(w.columns, nullable)


def promote(ctx: Context, c: GroupConstraints, rejected: Iterable[ColumnRef]) -> GroupConstraints:
    """`c` with its weak keys promoted by the null-rejected columns `rejected` (the rows of the occurrence are known
    to be non-null there): a weak key whose nullable columns are all rejected is a key of those rows, one with some
    of them rejected keeps the rest as its nullable set. Sound: the weak key speaks about the rows non-null in N; where
    every row is non-null in a column of N, that column need not be demanded."""
    rejected = list(rejected)
    out = GroupConstraints(set(c.keys), set(c.fks), set())
    for w in c.weak:
        k = _weak_minus(ctx, w, rejected) if rejected else w
        (out.keys if isinstance(k, Key) else out.weak).add(k)
    return out


def bind_constants(ctx: Context, c: GroupConstraints, bound: Iterable[ColumnRef]) -> GroupConstraints:
    """The constant-binding rule: a column equated with a literal (`col = constant`, every output row satisfies it,
    the literal of the column's own comparison family) is taken out of every key and weak key that contains it.

    Sound: two output rows agree on `col` (both equal the constant under the column type's own equality, which is
    the equality GROUP BY uses; the family guard keeps out `bigint = float8`, where two different bigints equal one
    double, and `char = varchar`), so two rows agreeing on K minus col agree on K. For a weak key, `col` is also
    non-null on every row (the equality is never TRUE for NULL), so it leaves the nullable set as well; the rest of
    the nullable set still has to be demanded. K minus col empty: at most one row."""
    bound = list(bound)
    if not bound:
        return c
    out = GroupConstraints(set(c.keys), set(c.fks), set(c.weak))
    for k in c.keys:
        rest = frozenset(x for x in k.columns if not _covers(ctx, x, bound))
        if len(rest) < len(k.columns):
            out.keys.add(Key(rest))
    for w in c.weak:
        rest = frozenset(x for x in w.columns if not _covers(ctx, x, bound))
        if len(rest) < len(w.columns):
            nullable = frozenset(x for x in w.nullable if x in rest)
            (out.keys.add(Key(rest)) if not nullable else out.weak.add(WeakKey(rest, nullable)))
    return out


def _union(x: Key | WeakKey, y: Key | WeakKey) -> Key | WeakKey:
    """The union of a key of each side of a join: a key when both are keys, else a weak key whose nullable set is the
    union of the nullable sets (two rows non-null in both agree on both column sets, so they share both rows)."""
    nullable = frozenset(getattr(x, "nullable", frozenset()) | getattr(y, "nullable", frozenset()))
    columns = x.columns | y.columns
    return Key(columns) if not nullable else WeakKey(columns, nullable)


def _union_pairs(a: GroupConstraints, b: GroupConstraints) -> Iterable[tuple[Key | WeakKey, Key | WeakKey]]:
    """The pairs the union rule combines: a key of each side, and a key of one side with a weak key of the other. Two
    weak keys are not combined (a minimal representation: their union would demand both nullable sets, and every
    table with a primary key already gives the key-with-weak unions; with every pair combined, a 10-way join took
    0.70 s against 0.06 s without UNIQUE keys). Dropping a derivation is always sound."""
    for x in a.keys:
        for y in list(b.keys) + list(b.weak):
            yield x, y
    for x in a.weak:
        for y in b.keys:
            yield x, y


def _equated(keys: Iterable[Key], weak: Iterable[WeakKey], equated: set[ColumnRef]) -> bool:
    """Whether some key, or some weak key, of a side has all its columns equated by the join's own condition."""
    return any(k.columns <= equated for k in keys) or any(w.columns <= equated for w in weak)


# ------------------------------------------------------------------------------------------------- the rules


def scan(ctx: Context, occ: str) -> GroupConstraints:
    """The declared PK and FKs of the table, on this occurrence's columns; a declared UNIQUE whose columns are all
    NOT NULL as a key (level 1); a UNIQUE with a nullable column as a weak key whose nullable set is the columns not
    declared NOT NULL (level 2). Sound when the database really satisfies what the schema declares: a PK column is
    never NULL, so no two rows agree on the PK even when NULL agrees with NULL, and the same holds for a UNIQUE over
    NOT NULL columns; a UNIQUE with a nullable column rejects two rows that are non-null in every column and agree
    on all of them, which is the weak key's statement (SQL's NULLS DISTINCT: NULLs never conflict); an FK check skips
    exactly the rows that have a NULL in an FK column (MATCH SIMPLE). A column whose nullability the schema does not
    state counts as nullable."""
    table = ctx.table(occ)
    if table is None:
        return GroupConstraints()
    known = {c.name for c in table.columns}
    not_null = {c.name for c in table.columns if not c.nullable}
    out = GroupConstraints()
    pk = table.primary_key
    if pk and len(set(pk)) == len(pk) and (not known or set(pk) <= known):
        out.keys.add(Key(frozenset(ColumnRef(occ, c) for c in pk)))
    for u in table.unique_keys:
        if not u or len(set(u)) != len(u) or (known and not set(u) <= known):
            continue
        nullable = frozenset(ColumnRef(occ, c) for c in u if c not in not_null)
        columns = frozenset(ColumnRef(occ, c) for c in u)
        (out.keys.add(Key(columns)) if not nullable else out.weak.add(WeakKey(columns, nullable)))
    for fk in table.foreign_keys:
        if fk.columns and len(fk.columns) == len(fk.ref_columns) and (not known or set(fk.columns) <= known):
            out.fks.add(FKey(tuple(ColumnRef(occ, c) for c in fk.columns), fk.ref_table, tuple(fk.ref_columns)))
    return out


def filter_(ctx: Context, conjunct: str, child_group: GroupId, child: GroupConstraints) -> GroupConstraints:
    """Keeps everything (sound: the output is a subset of the input rows, with the same columns); promotes the weak
    keys by the columns the conjunct null-rejects (every output row satisfies the conjunct, which is never TRUE
    when such a column is NULL: `promote`); and applies the constant-binding rule to the columns the conjunct
    equates with a literal (`bind_constants`)."""
    out = promote(ctx, keep_all(child), ctx.null_rejected(conjunct, child_group))
    if ctx.constant_binding:
        out = bind_constants(ctx, out, ctx.constant_bound(conjunct, child_group))
    return out


def inner_join(ctx: Context, conjuncts: tuple[str, ...], a_group: GroupId, b_group: GroupId,
               a: GroupConstraints, b: GroupConstraints) -> GroupConstraints:
    """FKs of both sides; the union of a key of each side; the keys of one side when THIS join's own condition
    equates every column of some key, or some weak key, of the other side (a cross join has no condition: union rule
    only). The weak keys of a side are first promoted by the columns the join's own conjuncts null-reject (an
    equality `a.x = b.y` is never TRUE when a.x is NULL, so every output row has a non-null a.x).

    Sound: an output row is one pair (row of a, row of b), produced at most once. FKs: the pair contains whole rows of
    a and of b. Union: two pairs that agree on Ka and Kb have the same a-row and the same b-row (for a weak key, two
    pairs non-null in its nullable columns). Kept: if two pairs agree on Ka they share their a-row; the condition is
    TRUE on both, so both b-rows carry, in every column of Kb, the same non-null value of that a-row; they agree on
    Kb and are non-null there, so they are the same b-row, whether Kb is a key or a weak key. A conjunct of another
    join says nothing about the pairs formed here, so only `conjuncts` count."""
    rejected_a = {c for x in conjuncts for c in ctx.null_rejected(x, a_group)}
    rejected_b = {c for x in conjuncts for c in ctx.null_rejected(x, b_group)}
    a, b = promote(ctx, a, rejected_a), promote(ctx, b, rejected_b)
    out = GroupConstraints(set(), a.fks | b.fks)
    for x, y in _union_pairs(a, b):
        u = _union(x, y)
        (out.keys if isinstance(u, Key) else out.weak).add(u)
    pairs = ctx.join_equalities(conjuncts, a_group, b_group)
    a_equated, b_equated = {x for x, _ in pairs}, {y for _, y in pairs}
    if _equated(b.keys, b.weak, b_equated):
        out.keys |= a.keys
        out.weak |= a.weak
    if _equated(a.keys, a.weak, a_equated):
        out.keys |= b.keys
        out.weak |= b.weak
    return out


def op1(ctx: Context, op: str, child_group: GroupId, child: GroupConstraints) -> GroupConstraints:
    kind = ctx.kind(op)
    if kind == "aggregate":
        return aggregate(ctx, op, child)
    if kind == "project":
        return renamed(ctx, op, child, keep_keys=ctx.row_preserving(op))
    if kind == "ref":
        return renamed(ctx, op, child, keep_keys=not ctx.lateral_suspect(op))
    if kind == "distinct":
        return distinct(ctx, op, child_group, child)
    if kind == "limit":
        return keep_all(child)  # sound: a subset of the rows
    if kind == "order":
        return order(ctx, op, child)
    return GroupConstraints()


def order(ctx: Context, op: str, child: GroupConstraints) -> GroupConstraints:
    """Keeps everything: the same rows in another order. Except that Postgres evaluates the ORDER BY keys together
    with the select list, so a key that calls a set-returning function REPEATS every row: then no key and no weak key
    survives (the FKs do, the repeated rows are exact copies). A function SQLGlot does not know counts as possibly
    set-returning."""
    return keep_all(child) if ctx.order_keeps_rows(op) else GroupConstraints(set(), set(child.fks))


def aggregate(ctx: Context, op: str, child: GroupConstraints) -> GroupConstraints:
    """No GROUP BY: the empty key. Otherwise the grouping columns form a key when every grouping expression is a
    plain column that the output keeps; an FK survives on kept grouping columns only. Weak keys of the input are
    dropped (a group's row is not an input row).

    Sound: without GROUP BY there is one output row (none under a failing HAVING). With it there is one row per
    combination of grouping values, and GROUP BY puts NULLs together exactly as the key definition does. A kept
    grouping column carries the value of at least one input row of its group, so that row's FK match still exists.
    Grouping sets (ROLLUP, CUBE) repeat grouping values in subtotal rows, so they derive nothing. A set-returning
    function in the select list is expanded AFTER grouping and repeats every group (also the single row of a global
    aggregate): then no key is derived; the FKs stay, the repeated rows are exact copies."""
    grouping = ctx.grouping(op)
    if grouping is None:
        return GroupConstraints()
    count, columns = grouping
    one_row_per_group = ctx.row_preserving(op)  # the select list is part of the aggregate operator
    if count == 0:
        return GroupConstraints({Key(frozenset())}, set()) if one_row_per_group and not columns else GroupConstraints()
    out = GroupConstraints()
    if len(columns) == count and one_row_per_group:
        out.keys |= {Key(frozenset(names)) for names in ctx.renamings(op, columns)}
    for f in child.fks:
        out.fks |= {FKey(names, f.ref_table, f.ref_columns) for names in ctx.renamings(op, f.columns, allowed=columns)}
    return out


def renamed(ctx: Context, op: str, child: GroupConstraints, *, keep_keys: bool) -> GroupConstraints:
    """`ref` and computed projection: a key, a weak key or an FK survives when the operator keeps all of its columns,
    under the output names. This is how constraints cross a CTE or derived-table boundary.

    Sound: one output row per input row, and a kept column is an unmodified copy, so rows agree on the renamed
    columns exactly when the input rows agreed on the originals (and are NULL exactly where the originals are: a
    weak key's nullable set is renamed with it). Keys and weak keys are dropped (FKs stay, they speak about single
    rows) where one row per input row is not certain: a select list with a set-returning function, and a derived
    table that may be LATERAL, whose keys hold per evaluation only."""
    out = GroupConstraints()
    if keep_keys:
        for k in child.keys:
            columns = sorted(k.columns, key=lambda c: (c.origin, c.column))
            out.keys |= {Key(frozenset(names)) for names in ctx.renamings(op, columns)}
        for w in child.weak:
            columns = sorted(w.columns, key=lambda c: (c.origin, c.column))
            nullable_at = [i for i, c in enumerate(columns) if c in w.nullable]
            for names in ctx.renamings(op, columns):
                out.weak.add(WeakKey(frozenset(names), frozenset(names[i] for i in nullable_at)))
    for f in child.fks:
        out.fks |= {FKey(names, f.ref_table, f.ref_columns) for names in ctx.renamings(op, f.columns)}
    return out


def distinct(ctx: Context, op: str, child_group: GroupId, child: GroupConstraints) -> GroupConstraints:
    """All output columns together form a key; keys, weak keys and FKs of the input stay when all their columns are
    output.

    Sound: DISTINCT returns the select list of its scope without duplicate rows, NULLs compared as equal. Its rows
    are input rows cut down to the select list, so a constraint on columns that survive still holds; a constraint
    on a dropped column means nothing afterwards. DISTINCT ON keeps duplicates of the select list: no new key; and
    when one of its ON expressions may return a set, one input row lands in several ON groups: no key at all."""
    outputs = ctx.distinct_outputs(op, child_group)
    if outputs is None:
        return GroupConstraints()
    out = GroupConstraints(set(), {f for f in child.fks if set(f.columns) <= outputs})
    if not (ctx.has_distinct_on and ctx.distinct_on_multiplies) and not ctx.windows_multiply:
        out.keys |= {k for k in child.keys if k.columns <= outputs}
        out.weak |= {w for w in child.weak if w.columns <= outputs}
    if not ctx.has_distinct_on:
        out.keys.add(Key(outputs))
    return out


def op2(ctx: Context, op: str, left_group: GroupId, right_group: GroupId,
        left: GroupConstraints, right: GroupConstraints) -> GroupConstraints:
    kind = ctx.kind(op)
    if kind == "outer_join":
        return outer_join(ctx, op, left_group, right_group, left, right)
    if kind == "setop":
        return setop(ctx, op)
    return GroupConstraints()


def outer_join(ctx: Context, op: str, left_group: GroupId, right_group: GroupId,
               left: GroupConstraints, right: GroupConstraints) -> GroupConstraints:
    """SEMI / ANTI: the left side's keys, weak keys and FKs (the output is a subset of the left rows). LEFT / RIGHT /
    FULL: the FKs of both sides. LEFT / RIGHT also: the keys and weak keys of the preserved side when the ON
    condition equates every column of a key, or of a weak key, of the other side, and the union of a key (or weak
    key) of each side. The keys and weak keys of a null-supplying side are dropped, and nothing is promoted by the ON
    condition (a padded row did not satisfy it).

    Sound: a padded row is NULL in every column of the padded side, so its FK columns are exempt. Kept: a preserved
    row matches at most one row (as for the inner join, the matched values being non-null even against a weak key),
    so it comes out exactly once, matched or padded; a weak key of the preserved side holds among the rows non-null
    in its nullable set, which are preserved rows as before. Union: two output rows with the same preserved row are
    both matched rows (a row is padded only when it has no match), with two different rows of the other side, which
    disagree on that side's key; with a weak key of the other side, two rows non-null in its nullable columns are
    matched rows (padded rows are NULL there), and the same argument applies. Dropped: two padded rows agree on the
    padded side's key, NULL with NULL. FULL pads both sides, so even the union can repeat (all-NULL keys)."""
    join_type = ctx.join_type(op)
    if join_type in HALF_TYPES:
        return keep_all(left)
    if join_type not in OUTER_TYPES:
        return GroupConstraints()
    out = GroupConstraints(set(), left.fks | right.fks)
    if join_type == "FULL":
        return out
    pairs = ctx.on_equalities(op, left_group, right_group)
    preserved, other = (left, right) if join_type == "LEFT" else (right, left)
    other_equated = {r for _, r in pairs} if join_type == "LEFT" else {l for l, _ in pairs}
    if _equated(other.keys, other.weak, other_equated):
        out.keys |= preserved.keys
        out.weak |= preserved.weak
    for x, y in _union_pairs(left, right):
        u = _union(x, y)
        (out.keys if isinstance(u, Key) else out.weak).add(u)
    return out


def setop(ctx: Context, op: str) -> GroupConstraints:
    """UNION (distinct): all output columns together form a key, because it removes duplicate rows with NULLs
    compared as equal. Every other set operation derives nothing."""
    if ctx.setop_kind(op) not in ("UNION", "UNION DISTINCT"):
        return GroupConstraints()
    outputs = ctx.setop_outputs(op)
    return GroupConstraints({Key(outputs)}, set()) if outputs else GroupConstraints()
