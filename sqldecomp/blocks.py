"""Blocks and decompositions: the legality rules that the checks, the tests and the optimizer (component 6) share.

No scoring and no optimization here. `Blocks(memo, result)` answers, for ONE query:

- what the CHILDREN of a group are. The memo's child groups, PLUS the root group of every predicate / select-item
  subquery that the expression's conjuncts or operator name in `data["subqueries"]`. The memo registers such a body
  as a plan of its own; the side-table link is the only edge from its consumer, so whoever walks the memo without it
  never sees a subquery body.
- where a block boundary may be (`can_cut`): not at a bare scan, not at a `ref` (cut at the body of the CTE or
  derived table instead), and only at a plan that lives in a FROM clause or is the top of a scope. There is no
  boundary INSIDE one SELECT's pipeline (aggregate / HAVING / select list / DISTINCT / ORDER BY / LIMIT): those
  operators refer to their input by expression (`HAVING COUNT(*) > 1`), which a derived table cannot offer.
- which pairs (output, inputs) are blocks, all of them (`legal_blocks`), and which sets of blocks are decompositions
  (`problems`): one block for the root, every input the output of exactly one block, every block reachable, every
  FORCED cut made, and one way of cutting per group (a shared body that two blocks expand is cut the same way in both).
- which groups every decomposition must cut at (`forced_cuts`): a group that is read from two or more places, the
  body of a CTE under two `ref` operators above all, is always a block of its own and is counted once. Only a group that can be a block is forced; a shared body that cannot (a bare scan, a rename-only
  select list, another ref) is expanded by every reader, and nothing cuttable lies inside it.
- the coarsest (one block plus the forced cuts), the finest, random and targeted decompositions.
- which blocks are PARAMETERIZED: something computed at or below the block's output refers to a FROM item of an
  enclosing query (`outer_refs`) that does not lie below the output. Such a block is not a statement by itself; its
  SQL names the outer column as a parameter (`o.o_cust = c.c_id`).
- how a block relates to subqueries (`describe`): part of a body alone, outer part with a whole body, outer part
  with the rest of a body.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from sqldecomp.model import (Block, Decomposition, Filter, GroupId, ImportResult, Join, Memo, MExpr, Op1, Plan, Scan,
                             plan_children, walk)
from sqlglot import exp

from sqldecomp.clock import NO_DEADLINE, Deadline, DecompositionTimeout
from sqldecomp.printer import sqltext

# Bound the number of candidate input sets retained during enumeration.
# This also bounds blocks, classes and scoring contexts when a search grows
# exponentially; the deadline independently bounds enumeration time.
MEMORY_VALVE = 1_000_000
CLOCK_EVERY = 4_096  # units of enumeration work (candidate sets kept; unions tried inside a merge) between two looks at the clock


class TooManyBlocks(DecompositionTimeout):
    """The memory safety valve: `legal_blocks` would hold more than `MEMORY_VALVE` candidate input sets. A
    timeout-class finding (stage `blocks`), never an answer."""

    def __init__(self, held: int, elapsed_s: float = 0.0, budget_s: float | None = None):
        self.held = held
        super().__init__("blocks", budget_s, elapsed_s, f"the memory safety valve ({held:,} candidate blocks held, at most {MEMORY_VALVE:,})")


class TooManyDecompositions(TooManyBlocks):
    """`decompositions`, the exhaustive enumeration the tests use on small queries, stops beyond its limit."""

    def __init__(self, limit: int):
        self.held = limit
        DecompositionTimeout.__init__(self, "blocks", None, 0.0, f"more than {limit:,} candidate decompositions")


@dataclass(frozen=True)
class Parameter:
    """One reference to an enclosing query that the result of a block depends on."""

    entry: str  # id of the conjunct or operator whose SQL holds the reference
    occurrence: str  # occurrence id of the FROM item of the enclosing query
    columns: tuple[str, ...]  # the references as the side table writes them, e.g. ("c.c_id",)
    own: bool  # True: applied by this block's own fragment; False: applied inside one of its input blocks
    alias: str = ""  # the FROM alias of that occurrence (what a reader sees; the occurrence id is internal)


@dataclass
class BlockInfo:
    """How one block of a decomposition relates to subqueries and to the rest of the query."""

    block: Block
    scope: str  # id of the scope the output group belongs to
    inside_body: str | None  # id of the predicate / select-item subquery scope the output lies in (innermost), or None
    parameters: list[Parameter] = field(default_factory=list)
    whole_bodies: list[str] = field(default_factory=list)  # subquery scopes whose whole body is inside this block
    partial_bodies: list[str] = field(default_factory=list)  # ... the block holds the consumer and the REST of the body
    input_bodies: list[str] = field(default_factory=list)  # ... the body is an input block of its own

    @property
    def parameterized(self) -> bool:
        return bool(self.parameters)

    @property
    def kinds(self) -> list[str]:
        """The ways a block can hold part of a subquery ("plain" when it holds none). A block can have several: it may
        hold one subquery whole and the rest of another. The fourth way of the design, "outer part + rest of the
        subquery", is split by whether any of the body is left in this block or the whole body is a block of its own."""
        out = []
        if self.inside_body is not None and self.parameterized:
            # the block applies the correlated condition itself, or only reads an input block that does
            out.append("correlated part (parameterized)" if any(p.own for p in self.parameters)
                       else "parameterized through an input block")
        elif self.inside_body is not None:
            out.append("part of a body alone")
        if self.whole_bodies:
            out.append("outer part + whole subquery")
        if self.partial_bodies:
            out.append("outer part + rest of the subquery")
        if self.input_bodies:
            out.append("outer part + subquery body as an input block")
        return out or ["plain"]


class Blocks:
    def __init__(self, memo: Memo, result: ImportResult):
        self.memo, self.result = memo, result
        self.side = result.side
        self.scopes = {s.id: s for s in result.scopes}
        self._scope_top: dict[GroupId, str] = {}
        for s in result.scopes:
            g = memo.group_of(s.plan)
            if g is not None:
                self._scope_top.setdefault(g, s.id)
        self._owner = self._owners()
        self._under: dict[GroupId, frozenset[str]] = {}
        self._cuttable: dict[GroupId, bool] = {}
        self._inside_cache: dict[GroupId, frozenset[GroupId]] = {}
        self._written = self._written_expressions()
        self._forced: list[GroupId] | None = None
        self._readers: dict[GroupId, int] = {}
        self._applied_cache: dict[GroupId, list[str]] = {}
        self._children_cache: dict[MExpr, tuple[GroupId, ...]] = {}
        self._preferred_cache: dict[GroupId, list[MExpr]] = {}
        self._cover_memo: dict[tuple[GroupId, frozenset[GroupId]], tuple[frozenset[GroupId], MExpr | None]] = {}
        self.rejected: list[tuple[Block, str]] = []  # candidates that `legal_blocks` found not to be blocks

    # ---------------------------------------------------------------------------------------------------------
    # structure: children (with the subquery link), scopes, where a boundary may be
    # ---------------------------------------------------------------------------------------------------------

    def applied(self, expr: MExpr) -> tuple[str, ...]:
        """Ids of the side entries that one expression applies: its conjunct, its join condition or its operator."""
        if expr.kind == "join":
            return tuple(expr.conjuncts)
        return () if expr.kind == "scan" else (expr.arg,)

    def subquery_bodies(self, expr: MExpr) -> tuple[tuple[str, GroupId], ...]:
        """(scope id, root group) of every subquery that the expression's conjuncts / operator hold."""
        out = []
        for id_ in self.applied(expr):
            for sid in self.side[id_].data.get("subqueries") or ():
                scope = self.scopes.get(sid)
                g = None if scope is None else self.memo.group_of(scope.plan)
                if g is not None and (sid, g) not in out:
                    out.append((sid, g))
        return tuple(out)

    def children(self, expr: MExpr) -> tuple[GroupId, ...]:
        """The memo's child groups, then the root groups of the subqueries the expression holds."""
        found = self._children_cache.get(expr)
        if found is None:
            out = list(expr.children)
            out += [g for _, g in self.subquery_bodies(expr) if g not in out]
            found = self._children_cache[expr] = tuple(out)
        return found

    def kind(self, group: GroupId) -> str:
        """scan, join, filter, or the side-table kind of the group's operator (the same for all of its expressions
        that are operators; a region group may hold joins and filters, then the first expression decides)."""
        return self.kind_of_expression(self.memo.groups[group][0])

    def kind_of_expression(self, e: MExpr) -> str:
        """scan, join, filter, or the side-table kind of an operator expression (aggregate, ref, setop, ...)."""
        return self.side[e.arg].kind if e.kind in ("op1", "op2") else e.kind

    def scope_of(self, group: GroupId) -> str | None:
        seen = set()
        while group not in seen:
            seen.add(group)
            e = self.memo.groups[group][0]
            ids = self.applied(e) or ((e.arg,) if e.kind == "scan" else ())
            if ids:
                return self._owner.get(ids[0])
            group = e.children[0]  # a join without a condition: the scope of its first input
        return None

    def is_scope_top(self, group: GroupId) -> bool:
        """The group of a whole scope: the query, a CTE / derived-table body, a subquery body, a set-operation branch."""
        return group in self._scope_top

    def scope_top_of(self, group: GroupId) -> str | None:
        """The id of the (first) scope whose whole plan the group is, or None."""
        return self._scope_top.get(group)

    def owner_scope(self, id_: str) -> str | None:
        """The scope that a scan, conjunct or operator id belongs to."""
        return self._owner.get(id_)

    def is_region_level(self, group: GroupId) -> bool:
        """The group lives in a FROM clause: region inputs (scan, ref, outer join) and the joins and WHERE filters over
        them. False for the operators of a select-list pipeline, for HAVING filters and for a set operation."""
        seen = set()
        while group not in seen:
            seen.add(group)
            e = self.memo.groups[group][0]
            if e.kind in ("scan", "join"):
                return True
            if e.kind != "filter":
                return self.side[e.arg].kind in ("ref", "outer_join")
            group = e.children[0]
        return False

    def can_cut(self, group: GroupId) -> bool:
        """May `group` be the output of a block that is an INPUT of another block?"""
        if group not in self._cuttable:
            e = self.memo.groups[group][0]
            ok = e.kind != "scan" and self.kind(group) != "ref"
            ok = ok and (self.is_region_level(group) or self.is_scope_top(group))
            # a SELECT without FROM has nothing to expose: its pseudo scan is no table and has no SQL
            scans = self.memo.scans(group)
            ok = ok and not (scans and all(self.side[o].data.get("table") is None and not self.side[o].sql for o in scans))
            # ... and it has to DO something: `SELECT c.c_id AS id FROM customers c` (a set-operation branch that only
            # renames a column of a bare table) holds no operator at all, so it is no more a block than the scan is
            ok = ok and any(self._is_operator(e) for e in self._expand(group, frozenset()).values())
            self._cuttable[group] = ok
        return self._cuttable[group]

    def reachable(self, start: GroupId | None = None) -> list[GroupId]:
        """Groups reachable from `start` (default: the root) through children, subquery links included."""
        seen, todo = [], [self.memo.root if start is None else start]
        mark = set()
        while todo:
            g = todo.pop()
            if g in mark:
                continue
            mark.add(g)
            seen.append(g)
            for e in self.memo.groups[g]:
                todo.extend(self.children(e))
        return sorted(seen)

    def cut_points(self) -> list[GroupId]:
        return [g for g in self.reachable() if g != self.memo.root and self.can_cut(g)]

    def forced_cuts(self) -> list[GroupId]:
        """Groups that EVERY decomposition cuts at (design decision D2): a group that is read from two or more places
        is always a block of its own, so that it is verified once and its readers see one input.

        Where the memo is a DAG. Sharing by value only happens below a `ref`: a CTE read by several `ref` operators is
        ONE nested plan. The number of readers of a scope is therefore counted through the refs: the root is read once;
        a CTE as often as the scopes holding its (reachable) refs are read, where a scope that is itself forced counts
        once, because its own block reads the CTE one time; a derived table, a subquery body and a set-operation
        branch are read as often as the scope around them. A scope read twice or more whose plan can be a block is
        forced. (A CTE read twice whose body is a bare scan, a rename-only select list or just another ref cannot be a
        block and is expanded by both readers; nothing cuttable lies inside such a body, only nested scopes below it,
        which are counted on their own and forced when they are read twice through it.)"""
        if self._forced is None:
            self._forced = self._forced_cuts()
        return list(self._forced)

    def readers(self, group: GroupId) -> int:
        """How many places read a scope-top group (see `forced_cuts`); 1 for any other group."""
        self.forced_cuts()
        return self._readers.get(group, 1)

    def _forced_cuts(self) -> list[GroupId]:
        """The fixpoint behind `forced_cuts`. A scope is forced when it is read from two or more places AND it can be a
        block once the other forced groups are its leaves: `WITH t AS (...), u AS (SELECT t.o_cust, 1 AS one FROM t)`
        with u read twice and t read by u and by the query: t is forced (three readers once u's two readers expand
        it), and u is not, because above t its body is a plain projection that asks nothing. Scopes are tried from the
        outer ones inward, so that a reader is preferred over what it reads when both could be forced; a forced scope
        that turns out to have no block (its only operators lie below a forced group decided later) is rejected and the
        counts are recomputed without it, until nothing changes."""
        reachable = set(self.reachable())
        refs_of: dict[str, list[str]] = {}  # nested scope id -> reachable ref operators that read it
        for g in sorted(reachable):
            for e in self.memo.groups[g]:
                if e.kind == "op1" and self.side[e.arg].kind == "ref":
                    refs_of.setdefault(str(self.side[e.arg].data.get("scope")), []).append(e.arg)
        group_of_scope = {s.id: self.memo.group_of(s.plan) for s in self.result.scopes}
        candidates = [s.id for s in self.result.scopes if s.kind != "root" and group_of_scope[s.id] is not None
                      and group_of_scope[s.id] != self.memo.root and group_of_scope[s.id] in reachable
                      and self.can_cut(group_of_scope[s.id])]
        # outer scopes first: breadth first over "scope s reads scope s'" from the root
        reads_of: dict[str, list[str]] = {s.id: [] for s in self.result.scopes}
        for sid, refs in refs_of.items():
            for ref in refs:
                owner = self._owner.get(ref)
                if owner in reads_of and sid in self.scopes:
                    reads_of[owner].append(sid)
        for s in self.result.scopes:
            if s.kind != "cte" and s.parent in reads_of:
                reads_of[s.parent].append(s.id)
        order: list[str] = []
        todo = [s.id for s in self.result.scopes if s.kind == "root"]
        while todo:
            sid = todo.pop(0)
            if sid in order:
                continue
            order.append(sid)
            todo.extend(sorted(set(reads_of.get(sid, ())), key=lambda x: (len(x), x)))
        rank = {sid: i for i, sid in enumerate(order)}
        candidates.sort(key=lambda sid: (rank.get(sid, len(rank)), sid))

        def counts(forced: set[str]) -> dict[str, int]:
            memo: dict[str, int] = {}

            def read(sid: str) -> int:
                if sid in memo:
                    return memo[sid]
                memo[sid] = 1  # a cycle (a recursive CTE) is not imported; this only stops the recursion
                s = self.scopes[sid]
                if s.kind == "root":
                    n = 1
                elif s.kind == "cte":
                    n = 0
                    for ref in refs_of.get(sid, ()):
                        owner = self._owner.get(ref)
                        n += 1 if owner is None or owner not in self.scopes or owner in forced else read(owner)
                else:  # derived table, subquery body, set-operation branch: read as often as the scope around it
                    parent = s.parent
                    n = 1 if parent is None or parent not in self.scopes or parent in forced else read(parent)
                memo[sid] = n
                return n

            for sid in self.scopes:
                read(sid)
            return memo

        def blockable(sid: str, forced: set[str]) -> bool:
            """Does the scope's group hold an operator above the other forced groups (a block of its own)?"""
            g = group_of_scope[sid]
            leaves = frozenset(group_of_scope[f] for f in forced if f != sid) - {g}
            covered, _ = self._cover(g, leaves, {})
            return any(self._is_operator(e) for e in self._expand(g, covered).values())

        rejected: set[str] = set()
        while True:
            forced: set[str] = set()
            for sid in candidates:
                if sid in rejected:
                    continue
                if counts(forced)[sid] >= 2 and blockable(sid, forced):
                    forced.add(sid)
            hollow = [sid for sid in forced if not blockable(sid, forced)]
            if not hollow:
                break
            rejected |= set(hollow)
        final = counts(forced)
        for s in self.result.scopes:
            g = group_of_scope[s.id]
            if g is not None and g in reachable:
                self._readers[g] = max(self._readers.get(g, 1), final[s.id])
        return sorted(group_of_scope[sid] for sid in forced)

    def pipeline_groups(self) -> list[GroupId]:
        """Groups that end inside a select-list pipeline (an aggregate below its HAVING, a DISTINCT below its ORDER
        BY): no block boundary there."""
        return [g for g in sorted(self.memo.groups) if self.memo.groups[g][0].kind != "scan" and self.kind(g) != "ref"
                and not self.is_region_level(g) and not self.is_scope_top(g)]

    def body_of(self, group: GroupId) -> str | None:
        """The innermost predicate / select-item subquery scope that contains the group's scope, or None."""
        sid, seen = self.scope_of(group), set()
        while sid is not None and sid in self.scopes and sid not in seen:
            seen.add(sid)
            if self.scopes[sid].kind == "subquery":
                return sid
            sid = self.scopes[sid].parent
        return None

    def occurrences_under(self, group: GroupId) -> frozenset[str]:
        """Occurrence ids (scans and refs) at or below a group, subquery bodies included."""
        if group not in self._under:
            self._under[group] = frozenset()  # a cyclic memo is not expected; this only stops the recursion
            e = self.memo.groups[group][0]
            out = {e.arg} if e.kind == "scan" or (e.kind == "op1" and self.side[e.arg].kind == "ref") else set()
            for child in self.children(e):
                out |= self.occurrences_under(child)
            self._under[group] = frozenset(out)
        return self._under[group]

    # ---------------------------------------------------------------------------------------------------------
    # blocks
    # ---------------------------------------------------------------------------------------------------------

    def _cover(self, group: GroupId, leaves: frozenset[GroupId], cache: dict) -> tuple[frozenset[GroupId], MExpr | None]:
        """The members of `leaves` that a fragment rooted at `group` reaches when it stops at every leaf, and the
        expression that reaches them. Greedy: at every group the expression reaching MOST leaves is taken (ties: the
        written one), which is the fragment whenever every leaf lies below the group on one route only. Where a
        leaf is reachable by several routes (a CTE body read from two places, reached through both readers) the
        greedy choice can miss a fragment that cuts one route higher up and reaches the leaf on the other;
        `_reach` / `_expand_exact` decide such a candidate exactly, and `_expand` falls back to them."""
        if group in leaves:
            return frozenset({group}), None
        found = cache.get(group)
        if found is not None:
            return found
        # the answer depends only on the leaves that lie below the group: shared across every block of the query
        key = (group, leaves & self._inside(group))
        found = self._cover_memo.get(key)
        if found is None:
            self._cover_memo[key] = (frozenset(), None)  # while it is being computed (a cyclic memo)
            best: tuple[frozenset[GroupId], MExpr | None] | None = None
            for e in self._preferred(group):
                covered = frozenset().union(*(self._cover(c, leaves, cache)[0] for c in self.children(e)))
                if best is None or len(covered) > len(best[0]):  # ties: the expression the query wrote comes first
                    best = (covered, e)
            found = self._cover_memo[key] = best
        cache[group] = found
        return found

    def _preferred(self, group: GroupId) -> list[MExpr]:
        """The expressions of a group, the one the query wrote first (it breaks ties in `_cover`)."""
        found = self._preferred_cache.get(group)
        if found is None:
            written = self._written.get(group)
            rest = [e for e in self.memo.groups[group] if e != written]
            found = self._preferred_cache[group] = ([written] if written is not None else []) + rest
        return found

    def fragment(self, block: Block) -> dict[GroupId, MExpr]:
        """group -> expression, for every group that the block expands (its output down to its inputs and to base
        tables, subquery bodies included). Raises ValueError when (output, inputs) is not a block."""
        why = self.why_not_a_block(block)
        if why:
            raise ValueError(why)
        return self._expand(block.output, frozenset(block.inputs))

    def _expand(self, output: GroupId, leaves: frozenset[GroupId]) -> dict[GroupId, MExpr]:
        """group -> expression of a fragment from `output` that stops at `leaves`: the greedy one when it reaches every
        leaf, else the exact one (`_expand_exact`), else the greedy one (a caller that checked legality first never
        gets there)."""
        cache: dict = {}
        covered, _ = self._cover(output, leaves, cache)
        out: dict[GroupId, MExpr] = {}
        todo = [output]
        while todo:
            g = todo.pop()
            if g in out or g in leaves:
                continue
            e = cache[g][1] if g in cache else self._cover(g, leaves, cache)[1]
            if e is None:
                continue
            out[g] = e
            todo.extend(self.children(e))
        if covered != leaves and leaves:
            exact = self._expand_exact(output, leaves)
            if exact is not None:
                return exact
        return out

    def _reach(self, group: GroupId, leaves: frozenset[GroupId], cache: dict) -> frozenset[frozenset[GroupId]]:
        """Every subset of `leaves` that some fragment rooted at `group` reaches when it stops at every leaf (the
        exact counterpart of `_cover`; a subset can be reached by several fragments)."""
        if group in leaves:
            return frozenset({frozenset({group})})
        found = cache.get(group)
        if found is not None:
            return found
        cache[group] = frozenset()  # a cyclic memo: this way of expanding the group never ends
        out: set[frozenset[GroupId]] = set()
        for e in self.memo.groups[group]:
            partial: set[frozenset[GroupId]] = {frozenset()}
            for c in self.children(e):
                below = self._reach(c, leaves, cache)
                partial = {a | b for a in partial for b in below}
            out |= partial
        cache[group] = frozenset(out)
        return cache[group]

    def _expand_exact(self, output: GroupId, leaves: frozenset[GroupId]) -> dict[GroupId, MExpr] | None:
        """A fragment from `output` whose leaves are exactly `leaves`, found by choosing at every group an expression
        whose children can reach the parts of the wanted set (the first such choice in the written-first order); None
        when there is none, or when a group reached on two routes would have to be cut differently on them."""
        reach_cache: dict = {}
        if leaves not in self._reach(output, leaves, reach_cache):
            return None
        out: dict[GroupId, MExpr] = {}
        wanted: dict[GroupId, frozenset[GroupId]] = {}
        todo: list[tuple[GroupId, frozenset[GroupId]]] = [(output, leaves)]
        while todo:
            g, w = todo.pop()
            if g in leaves:
                continue
            if g in wanted:
                if wanted[g] != w:
                    return None  # the same group, cut differently on two routes
                continue
            wanted[g] = w
            chosen = None
            for e in self._preferred(g):
                kids = self.children(e)
                options = [sorted(self._reach(c, leaves, reach_cache), key=lambda s: (len(s), sorted(s))) for c in kids]
                if any(not o for o in options):
                    continue
                for combo in _combinations(options, w):
                    chosen = (e, list(zip(kids, combo)))
                    break
                if chosen is not None:
                    break
            if chosen is None:
                return None
            out[g] = chosen[0]
            todo.extend(chosen[1])
        return out

    def why_not_a_block(self, block: Block) -> str | None:
        """None when (output, inputs) is a block, else the reason."""
        memo = self.memo
        if block.output not in memo.groups:
            return f"memo has no group {block.output!r}"
        for g in sorted(block.inputs):
            if g not in memo.groups:
                return f"memo has no group {g!r}"
            if not self.can_cut(g):
                return f"input {g} cannot be a block: {self.why_no_cut(g)}"
        if block.output in block.inputs:
            return f"group {block.output} is its own input"
        if block.output != memo.root and not self.can_cut(block.output):
            return f"output {block.output} cannot be a block: {self.why_no_cut(block.output)}"
        covered, _ = self._cover(block.output, frozenset(block.inputs), {})
        if covered != block.inputs and (not block.inputs or self._expand_exact(block.output, frozenset(block.inputs)) is None):
            missing = ", ".join(sorted(block.inputs - covered))
            return f"no fragment computes {block.output} from exactly these inputs: {missing} cannot be a leaf of it"
        if block.output != memo.root or block.inputs:  # the whole query as one block is always legal
            expanded = self._expand(block.output, frozenset(block.inputs))
            if not any(self._is_operator(e) for e in expanded.values()):
                return f"the only operator of block {block.output} is a ref or a bare scan"
        return None

    def is_block(self, block: Block) -> bool:
        return self.why_not_a_block(block) is None

    def why_no_cut(self, group: GroupId) -> str:
        if self.memo.groups[group][0].kind == "scan":
            return "a bare scan"
        if self.kind(group) == "ref":
            return "a ref (cut at the body of the CTE or derived table instead)"
        if not (self.is_region_level(group) or self.is_scope_top(group)):
            return "it ends inside the select-list pipeline of one SELECT"
        if not any(self._is_operator(e) for e in self._expand(group, frozenset()).values()):
            return "it only renames columns of a table"
        return "it reads no table"

    def _is_operator(self, e: MExpr) -> bool:
        """Does this expression DO something? A scan and a `ref` do not. Neither does a select list that only picks or
        renames columns or lists constants: it is a plan node only where the listed columns decide the rows (under
        DISTINCT, in a row-comparing set operation) or because a constant counts as computed (`SELECT 1` in EXISTS).
        A block whose only operator is one of these (`SELECT 1 FROM b_g9`, `SELECT b_g3.c__c_id AS id FROM b_g3`) asks
        nothing, so it is no block."""
        if e.kind in ("join", "filter", "op2"):
            return True
        if e.kind != "op1":
            return False
        entry = self.side[e.arg]
        if entry.kind == "ref":
            return False
        if entry.kind == "project" and not entry.data.get("has_window"):
            return not all(self._is_plain(i.get("sql", "")) for i in entry.data.get("select") or ())
        return True

    @staticmethod
    def _is_plain(sql: str) -> bool:
        """A select item that is one column, a constant or a star (possibly renamed)."""
        try:
            tree = sqltext.parse_item(sql)
        except Exception:
            return False
        if isinstance(tree, exp.Alias):
            tree = tree.this
        return isinstance(tree, (exp.Column, exp.Literal, exp.Null, exp.Boolean, exp.Star))

    def legal_blocks(self, *, forced: bool = False, validate: bool | None = None, deadline: Deadline | None = None
                     ) -> Iterator[Block]:
        """Every block of the query: every output (the root and the cut points reachable from it) with every set of
        inputs from which a fragment computes it. Raises `DecompositionTimeout` (stage `blocks`) when `deadline` runs
        out and `TooManyBlocks` (the memory safety valve) beyond `MEMORY_VALVE` held candidates; `self.enumeration`
        holds the work done (candidate sets formed, the measure of the old fixed limit of 200,000) and the most held.

        With `forced=True` a group that every decomposition must cut at (`forced_cuts`) is only ever an INPUT, never
        expanded inside a block: every block yielded can then be part of a legal decomposition, which is what the
        optimizer enumerates. Without it the pairs are still blocks (a fragment computes them), but a decomposition
        holding one that expands a forced group is refused by `problems`.

        Every candidate is a frontier of some fragment tree. Whether a fragment from the output to a frontier holds an
        operator is a property of the pair (every fragment of (g, I) applies the same conjuncts and operators), so it
        is tracked while the frontiers are built. Without forced cuts a candidate can still fail to be a block: a
        shared group (a CTE body read twice) cut on one path and expanded on the other has no single fragment, so
        every candidate is validated with `why_not_a_block` then (`validate` defaults to `not forced`). With forced
        cuts every shared group that can be cut IS cut, nothing cuttable lies inside a shared group that cannot, and
        the tracked flag decides alone; `validate=True` re-checks each candidate all the same (the tests do). In that
        mode a cut point is offered as an INPUT only when it has a block of its own under the forced cuts: the branch
        `SELECT t.o_cust FROM t` of a UNION over a forced CTE t is a cut point (its full expansion holds t's
        operators) but has no block once t is a leaf, so a block that reads it could never be completed."""
        clock = deadline or NO_DEADLINE
        work, held, next_look = [0], [0], [CLOCK_EVERY]
        steps, next_step = [0], [CLOCK_EVERY]  # the unions tried inside the merges (`work` counts their results)
        self.enumeration = {"work": 0, "held": 0}
        frontiers: dict[GroupId, dict[frozenset[GroupId], bool]] = {}
        must_cut = frozenset(self.forced_cuts()) if forced else frozenset()
        validate = (not forced) if validate is None else validate

        def options(g: GroupId) -> dict[frozenset[GroupId], bool]:
            """Input sets of the fragments rooted at g (g itself not cut) -> whether such a fragment has an operator."""
            if g not in frontiers:
                frontiers[g] = {}  # a cyclic memo
                found: dict[frozenset[GroupId], bool] = {}
                for e in self.memo.groups[g]:
                    partial: dict[frozenset[GroupId], bool] = {frozenset(): self._is_operator(e)}
                    for child in self.children(e):
                        if child in must_cut:
                            here: dict[frozenset[GroupId], bool] = {frozenset({child}): False}
                        else:
                            here = dict(options(child))
                            if self.can_cut(child) and (not must_cut or any(here.values())):
                                here[frozenset({child})] = False
                        merged: dict[frozenset[GroupId], bool] = {}
                        for a, op_a in partial.items():
                            for b, op_b in here.items():
                                key = a | b
                                merged[key] = merged.get(key, False) or op_a or op_b
                            # one merge can take minutes while its result stays small (the products collapse): its own
                            # steps count for the clock and the valve, not only its result
                            steps[0] += len(here)
                            if steps[0] >= next_step[0]:
                                next_step[0] = steps[0] + CLOCK_EVERY
                                clock.check("blocks")
                                if held[0] + len(merged) > MEMORY_VALVE:
                                    raise TooManyBlocks(held[0] + len(merged), clock.elapsed(), clock.seconds)
                        partial = merged
                        work[0] += len(partial)
                        if held[0] + len(partial) > MEMORY_VALVE:
                            raise TooManyBlocks(held[0] + len(partial), clock.elapsed(), clock.seconds)
                        if work[0] >= next_look[0]:
                            next_look[0] = work[0] + CLOCK_EVERY
                            clock.check("blocks")
                    for f, op in partial.items():
                        found[f] = found.get(f, False) or op
                frontiers[g] = found
                held[0] += len(found)
                self.enumeration = {"work": work[0], "held": max(self.enumeration["held"], held[0])}
            return frontiers[g]

        self.rejected = []
        root = self.memo.root
        for output in [root] + self.cut_points():
            choices = options(output)
            for inputs in sorted(choices, key=lambda s: (len(s), sorted(s))):
                block = Block(output, inputs)
                if validate:
                    clock.check("blocks")  # validating a candidate costs more than forming it (half a million: minutes)
                    why = self.why_not_a_block(block)
                elif choices[inputs] or (output == root and not inputs):
                    why = None
                else:
                    why = f"the only operator of block {output} is a ref or a bare scan"
                if why is None:
                    yield block
                else:  # a block without an operator of its own, or a shared group cut on one path and expanded on another
                    self.rejected.append((block, why))

    def count_blocks(self, *, deadline: Deadline | None = None) -> tuple[int, bool]:
        """(number of legal blocks, complete?): incomplete when the clock or the memory valve stopped the count."""
        n = 0
        try:
            for _ in self.legal_blocks(deadline=deadline):
                n += 1
        except DecompositionTimeout:
            return n, False
        return n, True

    def _has_operator_above(self, block: Block) -> bool:
        return any(self._is_operator(e) for e in self._expand(block.output, frozenset(block.inputs)).values())

    # ---------------------------------------------------------------------------------------------------------
    # decompositions
    # ---------------------------------------------------------------------------------------------------------

    def one_block(self) -> Decomposition:
        """The whole query as ONE block. A legal decomposition only when nothing is forced (`forced_cuts` is empty);
        `coarsest` is the decomposition with the fewest blocks that is always legal."""
        return Decomposition([Block(self.memo.root)])

    def coarsest(self) -> Decomposition:
        """The fewest blocks: the whole query as one block, plus a block for every forced cut (the one-block
        decomposition when nothing is forced)."""
        return self.with_cuts(())

    def with_cuts(self, cuts: Iterable[GroupId]) -> Decomposition:
        """The decomposition that cuts at every group of `cuts` it can reach, and at every forced cut: each block
        stops at the cut groups below it. Cut groups that would leave a block without an operator of its own
        (`SELECT ... FROM cte`) are given up, largest first, until every block is a block; a forced cut is never given
        up (a reader of a shared body always has an operator of its own, see `forced_cuts`)."""
        forced = frozenset(self.forced_cuts())
        leaves = frozenset(g for g in cuts if g in self.memo.groups and g != self.memo.root and self.can_cut(g)) | forced
        while True:
            blocks: dict[GroupId, Block] = {}
            todo = [self.memo.root]
            while todo:
                output = todo.pop(0)
                if output in blocks:
                    continue
                inputs, _ = self._cover(output, leaves - {output}, {})
                blocks[output] = Block(output, inputs)
                todo.extend(sorted(inputs))
            hollow = [b for b in blocks.values() if b.inputs and not self._has_operator_above(b)]
            if not hollow:
                return Decomposition([blocks[g] for g in self._block_order(blocks)])
            candidates = [g for b in hollow for g in b.inputs if g not in forced]
            if not candidates:
                # a hollow block whose inputs are all forced (`SELECT t.o_cust FROM t` over a forced CTE t): the cut at
                # its own output is given up, so that its reader expands it
                candidates = [b.output for b in hollow if b.output in leaves and b.output not in forced]
            if not candidates:  # not expected: a forced group holds an operator above the other forced groups
                raise ValueError(f"no legal decomposition: block {hollow[0].output} has no operator above its forced "
                                 f"inputs {sorted(hollow[0].inputs)}")
            give_up = max(candidates, key=lambda g: (len(self.occurrences_under(g)), g))
            leaves = leaves - {give_up}

    def finest(self) -> Decomposition:
        """Every operator its own block, as far as the rules allow: a cut at every cut point. (A select-list pipeline
        stays one block, a ref stays with its reader, a bare scan is no block.)"""
        return self.with_cuts(self.cut_points())

    def sample(self, rng: random.Random, p_cut: float = 0.4) -> Decomposition | None:
        """One random decomposition: a random expression per group (ONE per group for the whole decomposition, so that
        a CTE body which two blocks expand is cut the same way in both) and a random decision per cut point. None when
        the draw is not a decomposition (it leaves a block without an operator of its own, `SELECT ... FROM cte`)."""
        decided: dict[GroupId, bool] = {}
        chosen: dict[GroupId, MExpr] = {}
        blocks: dict[GroupId, Block] = {}
        todo = [self.memo.root]
        forced = frozenset(self.forced_cuts())

        def cut(group: GroupId) -> bool:
            if group not in decided:
                decided[group] = self.can_cut(group) and (group in forced or rng.random() < p_cut)
            return decided[group]

        def expand(group: GroupId, inputs: set[GroupId], seen: set[GroupId]) -> None:
            e = chosen.setdefault(group, rng.choice(self.memo.groups[group]))
            for child in self.children(e):
                if cut(child):
                    inputs.add(child)
                    if child not in blocks and child not in todo:
                        todo.append(child)
                elif child not in seen:
                    seen.add(child)
                    expand(child, inputs, seen)

        while todo:
            output = todo.pop(0)
            inputs: set[GroupId] = set()
            expand(output, inputs, {output})
            blocks[output] = Block(output, frozenset(inputs))
        d = Decomposition([blocks[g] for g in self._block_order(blocks)])
        return None if self.problems(d) else d

    def samples(self, seed: int, wanted: int = 12, tries: int = 80, p_cut: float = 0.4) -> list[Decomposition]:
        """Distinct random decompositions with at least two blocks, in the order they were drawn. Same seed, same
        list."""
        rng = random.Random(seed)
        found: dict[tuple, Decomposition] = {}
        for _ in range(tries):
            if len(found) >= wanted:
                break
            d = self.sample(rng, p_cut)
            if d is not None and len(d.blocks) > 1:
                found.setdefault(decomposition_key(d), d)
        return list(found.values())

    def decompositions(self, *, limit: int = 2_000) -> list[Decomposition]:
        """EVERY legal decomposition, the coarsest one first. Raises TooManyDecompositions (a TooManyBlocks) beyond
        `limit` candidates: the number grows exponentially, so this is for small queries (exhaustive C2 in the tests)."""
        by_output: dict[GroupId, list[Block]] = {}
        for b in self.legal_blocks(forced=True):
            by_output.setdefault(b.output, []).append(b)
        found: dict[tuple, Decomposition] = {}
        budget = [limit]

        def grow(todo: tuple[GroupId, ...], chosen: dict[GroupId, Block]) -> None:
            if not todo:
                budget[0] -= 1
                if budget[0] < 0:
                    raise TooManyDecompositions(limit)
                d = Decomposition([chosen[g] for g in self._block_order(chosen)])
                if not self.problems(d):
                    found.setdefault(decomposition_key(d), d)
                return
            g, rest = todo[0], todo[1:]
            if g in chosen:
                grow(rest, chosen)
                return
            for b in by_output.get(g, ()):
                grow(rest + tuple(sorted(b.inputs)), {**chosen, g: b})

        grow((self.memo.root,), {})
        return sorted(found.values(), key=lambda d: (len(d.blocks), decomposition_key(d)))

    def _block_order(self, blocks: dict[GroupId, Block]) -> list[GroupId]:
        """Blocks reachable from the root block, the root first, then breadth first by group id."""
        order, todo = [], [self.memo.root]
        while todo:
            g = todo.pop(0)
            if g in order or g not in blocks:
                continue
            order.append(g)
            todo.extend(sorted(blocks[g].inputs))
        return order

    def problems(self, decomposition: Decomposition) -> list[str]:
        """Why a set of blocks is not a legal decomposition; empty when it is one."""
        bad: list[str] = []
        by_output: dict[GroupId, Block] = {}
        for b in decomposition.blocks:
            if b.output in by_output:
                bad.append(f"two blocks have the output {b.output}")
            by_output[b.output] = b
        if self.memo.root not in by_output:
            bad.append(f"no block has the root group {self.memo.root} as its output")
        for b in decomposition.blocks:
            why = self.why_not_a_block(b)
            if why:
                bad.append(why)
            bad += [f"input {g} of block {b.output} is not the output of a block" for g in sorted(b.inputs) if g not in by_output]
        if bad:
            return bad
        unreachable = sorted(set(by_output) - set(self._block_order(by_output)))
        bad += [f"block {g} is not reachable from the root block" for g in unreachable]
        # every forced cut is made: a group read from two or more places is a block of its own (D2)
        bad += [f"group {g} is read from {self.readers(g)} places and must be a block of its own"
                for g in self.forced_cuts() if g not in by_output]
        # One way of cutting per group. (a) A group that is a block is a block wherever it is reached: no other
        # block expands it. (b) A group that two blocks expand (the body of a CTE used twice) reaches the same cut
        # groups in both, so it is expanded the same way and the CTE has ONE body in the recomposed query.
        outputs = frozenset(by_output)
        first_use: dict[GroupId, Block] = {}
        for b in decomposition.blocks:
            leaves = frozenset(b.inputs)
            expanded = self._expand(b.output, leaves)
            inside = sorted(g for g in expanded if g in outputs and g != b.output)
            if inside:
                bad.append(f"block {b.output} expands {', '.join(inside)}, which the decomposition makes a block of its own")
            for g in expanded:
                first = first_use.setdefault(g, b)
                if first is not b and self._cover(g, frozenset(first.inputs), {})[0] != self._cover(g, leaves, {})[0]:
                    bad.append(f"group {g} is cut differently inside block {first.output} and inside block {b.output}")
        return bad

    # ---------------------------------------------------------------------------------------------------------
    # parameters and subqueries
    # ---------------------------------------------------------------------------------------------------------

    def parameters(self, block: Block, expanded: dict[GroupId, MExpr] | None = None) -> list[Parameter]:
        """References to an enclosing query that the result of the block depends on: `outer_refs` of everything
        applied at or below the output (inside its input blocks too) that name an occurrence which does NOT lie below
        the output. Empty for a block that is a statement by itself. `expanded` may pass the block's fragment
        (`fragment(block)`) when the caller has it already."""
        bound = self.occurrences_under(block.output)
        if expanded is None:
            expanded = self._expand(block.output, frozenset(block.inputs))
        own_ids = {id_ for e in expanded.values() for id_ in self.applied(e)}
        own_ids |= {e.arg for e in expanded.values() if e.kind == "scan"}
        out: list[Parameter] = []
        # a VALUES list in FROM is a scan that may read an enclosing query through its rows (`outer_refs` on the scan
        # entry); it is no applied entry, so it is listed beside them
        value_scans = [e.arg for g in self.reachable(block.output) for e in self.memo.groups[g]
                       if e.kind == "scan" and self.side[e.arg].data.get("outer_refs")]
        for id_ in self._applied_below(block.output) + sorted(set(value_scans)):
            entry = self.side[id_]
            for occ in entry.data.get("outer_refs") or ():
                if occ in bound:
                    continue
                p = Parameter(id_, occ, self._columns_of(entry, occ), id_ in own_ids, self._alias_of(occ))
                if p not in out:
                    out.append(p)
        # A plain select list is no side entry, yet its items can read an enclosing query (`SELECT c.c_id FROM orders o
        # WHERE ...` inside a subquery of a query over customers c): the outputs of every scope at or below the output
        # are read as well; the pseudo id `outputs@<scope>` names the entry.
        for g in sorted(self._inside(block.output)):
            sid = self.scope_top_of(g)
            if sid is None:
                continue
            own = g in expanded  # the scope's top is expanded by this block itself, not by one of its inputs
            for o in self.scopes[sid].outputs:
                for occ, columns in self._outer_columns_of_text(o.sql, sid).items():
                    if occ in bound:
                        continue
                    p = Parameter(f"outputs@{sid}", occ, columns, own, self._alias_of(occ))
                    if p not in out:
                        out.append(p)
        # A derived table or CTE inside a correlated body depends on the outer query because its BODY does; the entry
        # of the body that reads the column is listed as well, and only that one names a column.
        named = {p.occurrence for p in out if p.columns}
        return [p for p in out if p.columns or p.occurrence not in named]

    def resolve_alias(self, scope_id: str | None, alias: str) -> str | None:
        """The occurrence (scan or ref use) that a FROM alias means inside a scope: the scope's own FROM items first,
        then the scopes around it (a CTE body sees only itself). None when no scope defines it, or two do."""
        if not hasattr(self, "_alias_index"):
            index: dict[tuple[str | None, str], list[str]] = {}
            for id_, e in self.side.items():
                if e.kind in ("scan", "ref"):
                    name = e.data.get("alias") or e.data.get("name") or e.data.get("table")
                    if name:
                        index.setdefault((self._owner.get(id_), sqltext.ident_key(str(name))), []).append(id_)
            self._alias_index = index
        key = sqltext.ident_key(alias)
        sid, seen = scope_id, set()
        while sid is not None and sid in self.scopes and sid not in seen:
            seen.add(sid)
            found = self._alias_index.get((sid, key))
            if found:
                return found[0] if len(found) == 1 else None
            s = self.scopes[sid]
            sid = s.parent if s.kind != "cte" else None
        return None

    def _outer_columns_of_text(self, sql: str, scope_id: str) -> dict[str, tuple[str, ...]]:
        """occurrence -> the columns of it that an expression of scope `scope_id` reads, for occurrences the scope
        does not define itself (references to an enclosing query), as written (`c.c_id`). Cached per text."""
        cache = self.__dict__.setdefault("_outer_text_cache", {})
        key = (scope_id, sql)
        if key in cache:
            return cache[key]
        cache[key] = {}
        try:
            tree = sqltext.parse_item(sql)
        except Exception:  # unreadable text: nothing to report
            return {}
        out: dict[str, list[str]] = {}
        for column in sqltext.outer_columns(tree):
            alias = sqltext.column_key(column)[0]
            occ = self.resolve_alias(scope_id, alias)
            if occ is None or self._owner.get(occ) == scope_id:
                continue  # the scope's own FROM item, or unknown
            name = sqltext.to_sql(column)
            if name not in out.setdefault(occ, []):
                out[occ].append(name)
        cache[key] = {occ: tuple(cols) for occ, cols in out.items()}
        return cache[key]

    def _applied_below(self, group: GroupId) -> list[str]:
        """Ids applied at or below a group, subquery bodies included, in side-table order (cached per group)."""
        if group not in self._applied_cache:
            found: set[str] = set()
            for g in self.reachable(group):
                for e in self.memo.groups[g]:
                    found.update(self.applied(e))
            order = {id_: i for i, id_ in enumerate(self.side)}
            self._applied_cache[group] = sorted(found, key=lambda i: (order.get(i, len(order)), i))
        return list(self._applied_cache[group])

    def _alias_of(self, occurrence: str) -> str:
        target = self.side.get(occurrence)
        return "" if target is None else str(target.data.get("alias") or target.data.get("name") or target.data.get("table") or "")

    def _columns_of(self, entry, occurrence: str) -> tuple[str, ...]:
        """The columns of one outer occurrence that an entry's SQL reads, as written (`c.c_id`)."""
        alias = self._alias_of(occurrence)
        if not alias:
            return ()
        texts = [entry.sql] if entry.kind == "conjunct" else []
        d = entry.data
        texts += [i["sql"] for i in d.get("select") or ()] + list(d.get("group_by") or ()) + list(d.get("on") or ())
        texts += [k["sql"] for k in d.get("keys") or ()]
        found: list[str] = []
        parse = sqltext.parse_item
        if entry.kind == "scan" and d.get("columns") is not None and entry.sql:  # a VALUES list: its rows' expressions
            texts, parse = [entry.sql], sqltext.parse_from_item
        for text in texts:
            try:
                tree = parse(text)
            except Exception:  # unreadable text: the parameter is still reported, without its columns
                continue
            for column in sqltext.outer_columns(tree):
                if sqltext.column_key(column)[0] == sqltext.ident_key(alias):
                    name = sqltext.to_sql(column)
                    if name not in found:
                        found.append(name)
        return tuple(found)

    def describe(self, block: Block, expanded: dict[GroupId, MExpr] | None = None) -> BlockInfo:
        """`expanded` may pass the block's fragment when the caller has it already."""
        if expanded is None:
            expanded = self._expand(block.output, frozenset(block.inputs))
        info = BlockInfo(block, self.scope_of(block.output) or "", self.body_of(block.output),
                         self.parameters(block, expanded))
        for g, e in expanded.items():
            for sid, body in self.subquery_bodies(e):
                if body in block.inputs:
                    info.input_bodies.append(sid)
                elif any(i in self._inside(body) for i in block.inputs):
                    info.partial_bodies.append(sid)
                else:
                    info.whole_bodies.append(sid)
        for name in ("whole_bodies", "partial_bodies", "input_bodies"):
            setattr(info, name, sorted(set(getattr(info, name))))
        return info

    def _inside(self, group: GroupId) -> frozenset[GroupId]:
        if group not in self._inside_cache:
            self._inside_cache[group] = frozenset(self.reachable(group))
        return self._inside_cache[group]

    # ---------------------------------------------------------------------------------------------------------
    # indexes
    # ---------------------------------------------------------------------------------------------------------

    def _owners(self) -> dict[str, str]:
        """id -> the scope it belongs to: what a scope's plan reaches without entering a nested scope (the body below
        a ref, a set-operation branch). Subquery bodies are scopes of their own and are not part of any plan."""
        nested = {s.plan for s in self.result.scopes}
        owner: dict[str, str] = {}

        def visit(p: Plan, sid: str, top: bool) -> None:
            if not top and p in nested:
                return  # another scope starts here
            if isinstance(p, Scan):
                owner.setdefault(p.occ, sid)
                return
            if isinstance(p, Join):
                for c in p.conjuncts:
                    owner.setdefault(c, sid)
            elif isinstance(p, Filter):
                owner.setdefault(p.conjunct, sid)
            else:
                owner.setdefault(p.op, sid)
            for child in plan_children(p):
                visit(child, sid, False)

        for s in sorted(self.result.scopes, key=lambda s: len(list(walk(s.plan)))):  # innermost first
            visit(s.plan, s.id, True)
        return owner

    def _written_expressions(self) -> dict[GroupId, MExpr]:
        """group -> the expression the query itself wrote for it (P0 and every scope plan)."""
        by_key: dict[GroupId, dict[tuple, MExpr]] = {}
        for g, exprs in self.memo.groups.items():
            for e in exprs:
                kids = tuple(sorted(e.children)) if e.kind == "join" else e.children
                by_key.setdefault(g, {})[(e.kind, kids, e.arg, tuple(sorted(e.conjuncts)))] = e
        written: dict[GroupId, MExpr] = {}
        for plan in [self.result.plan] + [s.plan for s in self.result.scopes]:
            for node in walk(plan):
                g = self.memo.group_of(node)
                if g is None or g in written:
                    continue
                kids = tuple(self.memo.group_of(c) for c in plan_children(node))
                if isinstance(node, Scan):
                    key = ("scan", (), node.occ, ())
                elif isinstance(node, Join):
                    key = ("join", tuple(sorted(kids)), None, tuple(sorted(node.conjuncts)))
                elif isinstance(node, Filter):
                    key = ("filter", kids, node.conjunct, ())
                else:
                    key = ("op1" if isinstance(node, Op1) else "op2", kids, node.op, ())
                if key in by_key.get(g, {}):
                    written[g] = by_key[g][key]
        return written


def decomposition_key(d: Decomposition) -> tuple:
    """A canonical, hashable form of a decomposition (for de-duplication and deterministic output)."""
    return tuple(sorted((b.output, tuple(sorted(b.inputs))) for b in d.blocks))


def _combinations(options: list[list[frozenset]], wanted: frozenset) -> Iterator[tuple[frozenset, ...]]:
    """Every choice of one set per child whose union is exactly `wanted` (a leaf may be reached by two children)."""
    def go(i: int, so_far: frozenset, chosen: tuple) -> Iterator[tuple[frozenset, ...]]:
        if i == len(options):
            if so_far == wanted:
                yield chosen
            return
        for s in options[i]:
            if s <= wanted:
                yield from go(i + 1, so_far | s, chosen + (s,))
    return go(0, frozenset(), ())
