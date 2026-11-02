"""Runs the rules over the memo read-out: once per occurrence, children first, union over all expressions."""
from __future__ import annotations

from typing import Iterable

from sqldecomp.constraints import rules
from sqldecomp.constraints.context import Context
from sqldecomp.model import GroupConstraints, GroupId, ImportResult, Key, Memo, Schema, WeakKey

MAX_KEYS = 64  # per occurrence; the smallest keys win. Dropping a key (or a weak key) is always sound.
MAX_WEAK = 32  # weak keys per occurrence (the join unions grow with the square of the count)


def derive_constraints(memo: Memo, result: ImportResult, schema: Schema, *, constant_binding: bool = True) -> dict[GroupId, GroupConstraints]:
    """The keys (minimal ones only), the weak keys (minimal ones) and FKs that hold on the output of every entry of
    `memo.groups`. `constant_binding=False` switches the constant-binding rule off.

    All expressions of an occurrence return the same rows (M1), so whatever one expression proves holds for the
    occurrence: each occurrence is derived once, from the union over its expressions, after all of its children. The
    result therefore does not depend on the order of the expressions, of the groups or of the join inputs.

    Inside a correlated scope (a subquery that refers to an outer row) the constraints hold for every single
    evaluation of that scope, which is how a block that stays inside the subquery sees it.

    The rules always run. A schema that declares no PK and no FK (such as the beaver_dw load) gives the scans nothing,
    so nothing is derived FROM DECLARATIONS and no FK exists; but GROUP BY, a global aggregate, DISTINCT and UNION make
    keys by themselves, and those are shown and carried on (through CTE boundaries, filters, ORDER BY / LIMIT and
    joins) exactly as under any other schema.
    """
    ctx = Context(memo, result, schema, constant_binding=constant_binding)
    order, cyclic = _children_first(memo)
    done: dict[GroupId, GroupConstraints] = {}
    for g in order:
        done[g] = _derive(ctx, g, done)
    # A memo with a cycle (a group below itself): the loop above read an empty set for the back edge, which is sound.
    # Iterating to the least fixpoint makes the result independent of where the traversal entered the cycle.
    for _ in range(len(order) if cyclic else 0):
        changed = False
        for g in order:
            new = _derive(ctx, g, done)
            keys = _smallest(minimal_keys(done[g].keys | new.keys))
            merged = GroupConstraints(keys, done[g].fks | new.fks, _smallest_weak(minimal_weak(done[g].weak | new.weak, keys)))
            if merged != done[g]:
                done[g], changed = merged, True
        if not changed:
            break
    return done


def minimal_keys(keys: Iterable[Key]) -> set[Key]:
    """Drops every key that contains another key: a superset of a key is a key and adds no information."""
    kept: list[Key] = []
    for k in sorted(set(keys), key=_key_order):
        if not any(m.columns <= k.columns for m in kept):
            kept.append(k)
    return set(kept)


def minimal_weak(weak: Iterable[WeakKey], keys: Iterable[Key]) -> set[WeakKey]:
    """Drops every weak key that a key implies (a key inside its columns: no two rows agree there at all) and every
    weak key that another weak key implies (fewer or the same columns, and fewer or the same nullable columns to
    demand)."""
    keys = list(keys)
    kept: list[WeakKey] = []
    for w in sorted(set(weak), key=_weak_order):
        if any(k.columns <= w.columns for k in keys):
            continue
        if not any(m.columns <= w.columns and m.nullable <= w.nullable for m in kept):
            kept.append(w)
    return set(kept)


def constraints_to_json(constraints: dict[GroupId, GroupConstraints]) -> dict:
    """The canonical JSON form of the constraints: everything sorted. `weak` (the weak keys: `[columns,
    nullable]`) is written only when a group has some."""
    out = {}
    for g, c8 in constraints.items():
        out[g] = {"keys": sorted(sorted([c.origin, c.column] for c in k.columns) for k in c8.keys),
                  "fks": sorted([[[c.origin, c.column] for c in f.columns], f.ref_table, list(f.ref_columns)] for f in c8.fks)}
        if c8.weak:
            out[g]["weak"] = sorted([sorted([c.origin, c.column] for c in w.columns), sorted([c.origin, c.column] for c in w.nullable)]
                                    for w in c8.weak)
    return out


def _derive(ctx: Context, g: GroupId, done: dict[GroupId, GroupConstraints]) -> GroupConstraints:
    out = GroupConstraints()
    for e in ctx.memo.groups[g]:
        proved = rules.apply(ctx, e, lambda child: done.get(child) or GroupConstraints())
        out.keys |= proved.keys
        out.fks |= proved.fks
        out.weak |= proved.weak
    out.keys = _smallest(minimal_keys(out.keys))
    out.weak = _smallest_weak(minimal_weak(out.weak, out.keys))
    return out


def _smallest(keys: set[Key]) -> set[Key]:
    return keys if len(keys) <= MAX_KEYS else set(sorted(keys, key=_key_order)[:MAX_KEYS])


def _smallest_weak(weak: set[WeakKey]) -> set[WeakKey]:
    return weak if len(weak) <= MAX_WEAK else set(sorted(weak, key=_weak_order)[:MAX_WEAK])


def _key_order(k: Key) -> tuple:
    return len(k.columns), sorted((c.origin, c.column) for c in k.columns)


def _weak_order(w: WeakKey) -> tuple:
    return len(w.columns), len(w.nullable), sorted((c.origin, c.column) for c in w.columns), sorted((c.origin, c.column) for c in w.nullable)


def _children_first(memo: Memo) -> tuple[list[GroupId], bool]:
    """Every group after all of its child groups (iterative depth-first search), and whether a cycle was met."""
    children = {g: sorted({c for e in exprs for c in e.children if c in memo.groups})
                for g, exprs in memo.groups.items()}
    order: list[GroupId] = []
    state: dict[GroupId, int] = {}  # 1 = open, 2 = finished
    cyclic = False
    for start in sorted(memo.groups):
        if start in state:
            continue
        state[start] = 1
        stack = [(start, iter(children[start]))]
        while stack:
            g, rest = stack[-1]
            for c in rest:
                if c not in state:
                    state[c] = 1
                    stack.append((c, iter(children[c])))
                    break
                cyclic = cyclic or state[c] == 1
            else:
                stack.pop()
                state[g] = 2
                order.append(g)
    return order, cyclic
