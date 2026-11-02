"""Component 6: the optimizer. Finds the decomposition that minimises the objective, exactly.

    optimize(memo, result, blocks, scoring, ...) -> Optimum          python -m sqldecomp.optimize --sql ... --schema ...

THE OBJECTIVE. A decomposition D is a set of blocks (sqldecomp.blocks). Its UNIQUE blocks B(D) are the block CLASSES
it uses (D3): two blocks are the same block when their output signatures are equal and their inputs are the same up
to the renaming behind those signatures. The score of D is the product over its unique blocks of a per-class score
F(theta, C(c)), times exp(Omega(|B(D)|)) for an arbitrary function Omega of the number of unique blocks. Minimised as

    cost(D) = sum_{c in B(D)} w(c) - Omega(|B(D)|),        w(c) = -log F(theta, C(c)),

where every F is evaluated ONCE per class (D7) on a context that carries the block's boundary information (D6) and
the semantic score of its output group (D1, once per output signature). A group read from two or more places is
always a block of its own (D2, `Blocks.forced_cuts`), so it is counted once as well.

WHY A PLAIN RECURSION IS NOT EXACT ANY MORE. The design document's best(g) = min over blocks (g, I) of c(g, I) + sum of
best(i) charges a block at every position; the objective charges a class once however many positions it occupies,
and Omega is a function of the unique count. So the cost of a block depends on whether its class was paid elsewhere
in the SAME decomposition, and the cost of the whole depends on how many distinct classes it holds.

THE EXACT ALGORITHM. Let R be the REPEATABLE classes: those offered by two or more distinct output groups (only such a
class can be paid twice in one decomposition, because a decomposition has one block per output group, and two blocks
of one output with the same key have the same inputs).

1. No repeatable class (R empty; 323 of the 415 distinct corpus queries): every block's class is unique in the
   decomposition, so the sum over classes IS the sum over blocks. If Omega is linear on the feasible range of counts,
   Omega(k) = Omega(k0) + (k - k0) s, then cost(D) = sum over blocks of (w(c) - s) + constant, and the memoized recursion
   over groups is exact for any sign of the terms. Otherwise a COUNT DIMENSION makes it exact for ANY Omega:
   best(g, k) = the least sum of w over the subtree of g with exactly k blocks, a (min, +) convolution over the inputs;
   the answer is min over k of best(root, k) - Omega(k). Forced cuts are independent subproblems whose count
   profiles are convolved at the top.

2. Otherwise a best-first search (A*) over FRONTIER STATES: (the groups still to be expanded, the repeatable classes
   paid so far that some pending group can still use, the unique count so far). A move expands the first pending
   group (in a fixed order) with one of its blocks, paying w(c) unless c is repeatable and already paid. Classes that
   no pending group can use any more are dropped from the state, which merges paths that differ only in dead
   history: the future cost of a state depends on nothing else. A pending group whose subtree offers no repeatable
   class that is live (unpaid and offered by another pending group, or offered twice inside the subtree) is not
   searched but solved by the recursion of point 1 with the paid classes at cost 0 ("DP leaf"), which is exact
   there. The heuristic is the recursion with every repeatable class free (a relaxation, hence admissible) plus, for a
   non-linear Omega, the least -Omega(k') over the counts k' the state can still reach (a nested range, hence
   consistent). Costs are made non-negative first: with shift = max(0, -min w), w' = w + shift and
   Omega'(k) = Omega(k) + shift k give the same objective; a linear Omega whose slope leaves every w(c) - s >= 0 is
   folded into the class costs and the count is dropped from the state. The first goal state popped is optimal
   (A* with a consistent heuristic), and the first optimum in the deterministic move order is returned; ties are
   not enumerated.

THE CLOCK. The optimizer has no budget of its own: it takes the query's `Deadline` (the decomposition timeout,
`sqldecomp.clock`) and looks at it while it enumerates the blocks (`Blocks.legal_blocks`), while it scores the classes,
while the recursion builds new subproblems and while A* expands states; when it is gone, `DecompositionTimeout` names
the stage (`blocks`, `scoring`, `optimizer`) and the counts, never an approximation. The enumeration's memory safety
valve (`TooManyBlocks`) is the only other stop.

WHAT IS NOT SHARED. A class shares scores and verdicts only. Legality is per position (`Blocks`), the cached
recursion is per group of THIS query, and the block SQL is printed per occurrence (the aliases differ).
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from sqldecomp.blocks import Blocks, TooManyBlocks  # noqa: F401 - TooManyBlocks: what `optimize` raises at the valve
from sqldecomp.boundary import Boundary
from sqldecomp.boundary import query_fingerprint as _query_fingerprint
from sqldecomp.clock import NO_DEADLINE, Deadline, DecompositionTimeout
from sqldecomp.constraints import derive_constraints
from sqldecomp.memo import placeholders, signature
from sqldecomp.memo.signature import is_unique
from sqldecomp.model import Block, Decomposition, GroupConstraints, GroupId, ImportResult, Memo, Schema
from sqldecomp.printer import Printer, PrintError
from sqldecomp.scoring import BlockContext, GroupContext, InputContext, Scoring

ClassKey = tuple
INF = float("inf")
EPS = 1e-9
ASTAR_CLOCK_EVERY = 1  # expansions between two looks at the clock in the A* search (a look costs far less than an expansion)


# --------------------------------------------------------------------------------------------------------------
# block classes (D3)
# --------------------------------------------------------------------------------------------------------------


def class_key(block: Block, signatures: dict[GroupId, str], placeholder_of, fingerprint: str = "",
              outputs_of=None) -> ClassKey:
    """The class of a block: ("class", output signature, scope outputs, sorted ((input signature, where the input
    sits), ...)) or ("unique", fingerprint of the query, output group, inputs) for a block that never shares. A
    "class" key names content only (two queries holding the same task get the same key and class id), a "unique" key
    names one position of one query: the fingerprint keeps `g7` of one query apart from `g7` of another. The CONTEXT
    a class is scored on (its boundary) still depends on the query around it, so a cache of F across queries has to
    be keyed by the class id AND the context, never by the class id alone. `outputs_of(group)` gives the alias-free
    output expressions of a scope-top group (None otherwise): a plain select list is no memo node and so no part of
    the signature, yet what a scope-top block returns is that list (`Boundary.scope_outputs`).

    Two blocks are the same block when their fragments are equal up to a renaming of aliases. The output signature
    gives the renaming of the output's occurrences (`placeholders`: occurrence -> t0, t1, ...); an input with the same
    signature at the same PLACEHOLDER POSITIONS then covers the same part of the output's content, and what the block
    applies above it is the rest, which is the same as well. "Where the input sits" is the tuple of the output's
    placeholders for the input's occurrences in the input's own placeholder order, so that the input's content is
    laid onto the output's content the same way in both blocks. The simpler formulation (output signature plus the
    MULTISET of input signatures) is coarser and merges blocks that are not the same: in `t a JOIN t b ON a.x = b.y`
    the blocks that cut out `a` and `b` have equal input signatures but sit at different positions and print
    differently; in dw_4527 the same filtered table stands in three CTEs and the coarser key merges the blocks that cut
    out each of them (34,677 classes instead of 71,303).

    A signature that is unique (a conjunct with a subquery, a fragment that cannot be read) never shares, for the
    output or for an input. Occurrences of the output's signature that are tied (a symmetric self-join) get one of
    several equally good renamings; then equal blocks can get different keys, which only loses sharing."""
    g, inputs = block.output, tuple(sorted(block.inputs))
    sg = signatures[g]
    if is_unique(sg):
        return ("unique", fingerprint, g, inputs)
    pg = placeholder_of(g)
    if pg is None:
        return ("unique", fingerprint, g, inputs)
    parts = []
    for i in inputs:
        si = signatures[i]
        pi = placeholder_of(i)
        if is_unique(si) or pi is None:
            return ("unique", fingerprint, g, inputs)
        try:
            order = sorted(pi, key=lambda o: int(pi[o][1:]))
            parts.append((si, tuple(pg[o] for o in order)))
        except (KeyError, ValueError):  # an occurrence the output's signature does not bind: never share
            return ("unique", fingerprint, g, inputs)
    outputs = outputs_of(g) if outputs_of is not None else None
    return ("class", sg, outputs, tuple(sorted(parts)))


def query_fingerprint(result: ImportResult) -> str:
    """What tells one query's positions from another's in a unique class key (`boundary.query_fingerprint`)."""
    return _query_fingerprint(result)


def class_id(key: ClassKey) -> str:
    """A 64-bit digest of the key (40 bits gave a few per cent chance of one collision over a corpus's 238,000
    classes)."""
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16]


@dataclass
class ClassInfo:
    key: ClassKey
    id: str
    blocks: list[Block]  # every admissible block of the class, in enumeration order; the first is the representative
    outputs: set[GroupId] = field(default_factory=set)

    @property
    def representative(self) -> Block:
        return self.blocks[0]

    @property
    def repeatable(self) -> bool:
        return len(self.outputs) >= 2


# --------------------------------------------------------------------------------------------------------------
# contexts (D6): what a scoring function is told
# --------------------------------------------------------------------------------------------------------------


class ContextBuilder:
    def __init__(self, memo: Memo, result: ImportResult, blocks: Blocks, signatures: dict[GroupId, str],
                 printer: Printer | None, boundary: Boundary, scoring: Scoring):
        self.memo, self.result, self.blocks, self.signatures = memo, result, blocks, signatures
        self.printer, self.boundary, self.scoring = printer, boundary, scoring
        self._groups: dict[str, GroupContext] = {}  # per signature
        self._meaning: dict[str, float] = {}
        self._inputs: dict[GroupId, InputContext] = {}
        self.union_notes: list[str] = []
        occurrences: dict[str, list[GroupId]] = {}
        for g in sorted(memo.groups):
            occurrences.setdefault(signatures[g], []).append(g)
        self._occurrences = occurrences

    def group_context(self, group: GroupId) -> GroupContext:
        sig = self.signatures[group]
        if sig not in self._groups:
            occs = self._occurrences[sig]
            rep = occs[0]
            columns, star, notes = self.boundary.union_columns(rep, occs[1:])
            self.union_notes += notes
            keys, fks = self.boundary.exposed_constraints(rep, columns, star)
            c8 = self.boundary.constraints.get(rep) or GroupConstraints()
            printer = self.printer
            self._groups[sig] = GroupContext(
                sig, rep, tuple(occs), self.blocks.kind(rep), self.blocks.is_scope_top(rep) or rep == self.memo.root,
                columns, star, keys, fks, (len(c8.keys), len(c8.fks)),
                (lambda: printer.group_sql(rep, bind_outer=True)) if printer is not None else None)
        return self._groups[sig]

    def meaningfulness(self, group: GroupId) -> float:
        sig = self.signatures[group]
        if sig not in self._meaning:
            value = float(self.scoring.meaningfulness(self.group_context(group)))
            if not (value > 0.0) or math.isinf(value):
                raise ValueError(f"meaningfulness must be a positive finite number, got {value!r}")
            self._meaning[sig] = value
        return self._meaning[sig]

    def input_context(self, group: GroupId) -> InputContext:
        if group not in self._inputs:
            columns, star = self.boundary.columns(group)
            keys, fks = self.boundary.exposed_constraints(group, columns, star)
            self._inputs[group] = InputContext(group, self.signatures[group], columns, star,
                                               bool(self.blocks.parameters(Block(group))), keys, fks)
        return self._inputs[group]

    def block_context(self, info: ClassInfo) -> BlockContext:
        rep = info.representative
        blocks = self.blocks
        fragment = blocks._expand(rep.output, frozenset(rep.inputs))  # `fragment(rep)` would validate the pair again
        operators: dict[str, int] = {}
        for e in fragment.values():
            kind = "join" if e.kind == "join" else "filter" if e.kind == "filter" else blocks.kind_of_expression(e)
            if kind not in ("scan", "ref"):
                operators[kind] = operators.get(kind, 0) + 1
        describe = blocks.describe(rep, fragment)
        others = [b.output for b in info.blocks[1:] if b.output != rep.output]
        exposed, star, notes = self.boundary.union_columns(rep.output, sorted(set(others), key=others.index))
        self.union_notes += notes
        printer = self.printer
        return BlockContext(
            info.id, info.key, rep, tuple(info.blocks), self.group_context(rep.output), exposed, star,
            tuple(self.input_context(i) for i in sorted(rep.inputs)), tuple(describe.parameters),
            {"inside_body": describe.inside_body, "whole_bodies": list(describe.whole_bodies),
             "partial_bodies": list(describe.partial_bodies), "input_bodies": list(describe.input_bodies)},
            dict(sorted(operators.items())), self._depth(rep, fragment), self.meaningfulness(rep.output),
            lambda: fragment,  # already built for the operators and the depth: no second expansion
            (lambda: printer.block_fragment(rep)) if printer is not None else None,
            (lambda: printer.block_sql(rep, inputs="name")) if printer is not None else None,
            self.result.side)

    def _depth(self, block: Block, fragment: dict) -> int:
        """Operators on the longest path of the fragment (scans and refs not counted)."""
        seen: dict[GroupId, int] = {}

        def depth(g: GroupId) -> int:
            if g in seen:
                return seen[g]
            seen[g] = 0
            e = fragment.get(g)
            if e is None:
                return 0
            below = max((depth(c) for c in self.blocks.children(e)), default=0)
            own = 0 if e.kind == "scan" or (e.kind == "op1" and self.blocks.kind_of_expression(e) == "ref") else 1
            seen[g] = below + own
            return seen[g]

        return depth(block.output)


# --------------------------------------------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------------------------------------------


@dataclass
class Optimum:
    decomposition: Decomposition
    cost: float  # sum over the unique blocks of -log F, minus Omega(|B(D)|)
    score: float  # the product form, exp(-cost)
    omega: float  # Omega(|B(D)|)
    unique_blocks: int
    cost_by_class: dict[str, float]  # class id -> -log F, for the classes of the decomposition
    class_of_block: dict[GroupId, str]  # output group -> class id
    classes_shared: dict[str, list[GroupId]]  # class id -> the outputs of the decomposition that share it
    contexts: dict[str, BlockContext]  # class id -> the context F was evaluated on, for the classes of the decomposition
    stats: dict[str, Any]
    all_classes: dict[str, ClassInfo] = field(default_factory=dict, repr=False)  # every class of the query, by id
    boundary: Any = field(default=None, repr=False)  # the `Boundary` of the query: per-position exposed columns

    def expose_at(self, block: Block) -> list[tuple[str, str]]:
        """The (alias, column) pairs that `print_block(block, inputs="name", within=decomposition, expose=...)` needs
        so that this position's text exposes the UNION of its class's exposed columns (OPT-O1): what a verifier sees
        once for the class. The union is the class's over every admissible block of the query (what the context F
        saw), whether or not the decomposition uses more than one of them. Empty for a block whose output is a scope
        top (its select list is its own) or the root, for a class with one admissible block, and for a block that is
        not in the decomposition.

        Two details: the scope-top test is on the BLOCK's own output (the signature's representative can be a scope top
        while this occurrence is not, and the other way round); and a union column that lies INSIDE one of the block's
        inputs (a sub-join of its region that the decomposition cuts out) is exposed by that input as well, down to the
        block that holds the column, so that the reader's text can expose it. Those propagated columns are part of the
        input's text verified once, but not of the context F saw for the input's class (contexts are per class, the
        propagation is per decomposition): open question OPT-O12."""
        if block not in self.decomposition.blocks:
            return []
        return list(self._exposures().get(block.output, ()))

    def _exposures(self) -> dict[GroupId, list[tuple[str, str]]]:
        found = getattr(self, "_exposure_map", None)
        if found is not None:
            return found
        found = {}
        if self.boundary is None:
            self._exposure_map = found
            return found
        blocks = self.boundary.blocks
        root = blocks.memo.root
        by_output = {b.output: b for b in self.decomposition.blocks}
        pairs: dict[GroupId, dict[tuple[str, str], tuple[str, str, str]]] = {g: {} for g in by_output}

        def add(g: GroupId, occ: str, alias: str, column: str) -> bool:
            key = (alias.lower(), column.lower())
            if key in pairs[g]:
                return False
            pairs[g][key] = (occ, alias, column)
            return True

        todo: list[tuple[GroupId, str, str, str]] = []
        for g, b in by_output.items():
            ctx = self.contexts.get(self.class_of_block.get(g, ""))
            if ctx is None or len(ctx.occurrences) < 2 or g == root or blocks.is_scope_top(g):
                continue
            for occ, alias, column in self.boundary.exposures_at(g, ctx.exposed):
                if add(g, occ, alias, column):
                    todo.append((g, occ, alias, column))
        while todo:  # a column inside an input of the block is exposed by that input too, all the way down
            g, occ, alias, column = todo.pop()
            for i in by_output[g].inputs:
                if i != root and not blocks.is_scope_top(i) and occ in blocks.occurrences_under(i) and add(i, occ, alias, column):
                    todo.append((i, occ, alias, column))
        self._exposure_map = {g: [(alias, column) for _, alias, column in v.values()] for g, v in pairs.items() if v}
        return self._exposure_map


# --------------------------------------------------------------------------------------------------------------
# the optimizer
# --------------------------------------------------------------------------------------------------------------


class Optimizer:
    def __init__(self, memo: Memo, result: ImportResult, blocks: Blocks, scoring: Scoring, *,
                 schema: Schema | None = None, constraints: dict[GroupId, GroupConstraints] | None = None,
                 signatures: dict[GroupId, str] | None = None, printer: Printer | None = None,
                 boundary: Boundary | None = None, deadline: Deadline | None = None):
        """`boundary`: the query's `Boundary` when the caller built one already (for the scoring: `make_scoring` reads
        it), so that the contexts and the scoring see one object; else it is built here. `deadline`: the query's
        decomposition timeout (none: no limit)."""
        self.memo, self.result, self.blocks, self.scoring = memo, result, blocks, scoring
        self.schema = schema
        self.printer = printer if printer is not None else Printer(result, memo)
        self.signatures = signatures if signatures is not None else {g: signature(memo, result, g) for g in memo.groups}
        if constraints is None and schema is not None:
            constraints = derive_constraints(memo, result, schema)
        self.boundary = boundary if boundary is not None else Boundary(memo, result, blocks, self.printer, constraints, schema)
        self.builder = ContextBuilder(memo, result, blocks, self.signatures, self.printer, self.boundary, scoring)
        self.deadline = deadline or NO_DEADLINE
        self._ticks = 0
        self._ph: dict[GroupId, dict[str, str] | None] = {}
        self.fingerprint = query_fingerprint(result)
        self.stats: dict[str, Any] = {"deadline_s": self.deadline.seconds}

    def _tick(self, stage: str, every: int = 1024) -> None:
        """Look at the clock every `every` calls."""
        self._ticks += 1
        if self._ticks % every == 0:
            self.deadline.check(stage)

    def _placeholders(self, g: GroupId):
        if g not in self._ph:
            self._ph[g] = placeholders(self.memo, self.result, g)
        return self._ph[g]

    # ---------------------------------------------------------------------------------------------------------
    # enumeration, classes, costs
    # ---------------------------------------------------------------------------------------------------------

    def prepare(self) -> None:
        self.deadline.check("blocks")
        t0 = time.perf_counter()
        blocks = self.blocks
        self.forced = frozenset(blocks.forced_cuts())
        self.root = self.memo.root
        self.roots = [self.root] + sorted(self.forced)  # the independent subproblems
        # every admissible block, by output, in enumeration order
        self.options: dict[GroupId, list[tuple[Block, ClassKey]]] = {}
        self.classes: dict[ClassKey, ClassInfo] = {}
        n = 0
        for b in blocks.legal_blocks(forced=True, deadline=self.deadline):
            n += 1
            self._tick("blocks", 4096)  # the classes of the enumerated blocks are built here, between the enumeration's own looks
            key = class_key(b, self.signatures, self._placeholders, self.fingerprint, self.boundary.scope_outputs)
            info = self.classes.get(key)
            if info is None:
                info = self.classes[key] = ClassInfo(key, class_id(key), [])
            info.blocks.append(b)
            info.outputs.add(b.output)
            self.options.setdefault(b.output, []).append((b, key))
        t1 = time.perf_counter()
        self.stats["enumeration_ms"] = round((t1 - t0) * 1000, 2)  # known even when the clock stops the scoring
        self.deadline.check("scoring")
        self.repeatable = frozenset(k for k, c in self.classes.items() if c.repeatable)
        # the inputs of a block that are subproblems of their own are leaves for the reader
        self.kids: dict[tuple[GroupId, int], tuple[GroupId, ...]] = {}
        for g, opts in self.options.items():
            for idx, (b, _) in enumerate(opts):
                self.kids[(g, idx)] = tuple(i for i in sorted(b.inputs) if i not in self.forced)
        # costs: F once per class, in first-seen order
        self.contexts: dict[ClassKey, BlockContext] = {}
        self.w: dict[ClassKey, float] = {}
        for key, info in self.classes.items():
            self._tick("scoring", 1)  # one class can take tens of ms to score: a look per class bounds the overshoot
            ctx = self.builder.block_context(info)
            self.contexts[key] = ctx
            self.w[key] = self.scoring.block_cost(ctx)
        t2 = time.perf_counter()
        enumeration = getattr(blocks, "enumeration", {})
        self.stats.update(candidate_work=enumeration.get("work"), candidates_held=enumeration.get("held"))
        self.stats.update(blocks_evaluated=n, classes=len(self.classes), f_evaluations=len(self.classes),
                          repeatable_classes=len(self.repeatable), forced_cuts=len(self.forced),
                          memo_groups=len(self.memo.groups), memo_complete=bool(self.memo.complete),
                          output_groups=len(self.options), enumeration_ms=round((t1 - t0) * 1000, 2),
                          scoring_ms=round((t2 - t1) * 1000, 2),
                          repeatable_outputs=sorted({g for k in self.repeatable for g in self.classes[k].outputs}))

    # ---------------------------------------------------------------------------------------------------------
    # recursions over one subtree (exact when no class of the subtree can be paid twice)
    # ---------------------------------------------------------------------------------------------------------

    def _scalar(self, g: GroupId, cost: dict[ClassKey, float], cache: dict) -> tuple[float, int]:
        """(least cost of the subtree of g, index of the block chosen at g); ties keep the first block."""
        if g in cache:
            return cache[g]
        self._tick("optimizer")
        best: tuple[float, int] | None = None
        for idx, (b, key) in enumerate(self.options[g]):
            c = cost[key]
            for i in self.kids[(g, idx)]:
                c += self._scalar(i, cost, cache)[0]
            if best is None or c < best[0] - EPS:
                best = (c, idx)
        if best is None:
            raise ValueError(f"group {g} has no admissible block")
        cache[g] = best
        return best

    def _profile(self, g: GroupId, cost: dict[ClassKey, float], count: dict[ClassKey, int], cache: dict
                 ) -> dict[int, tuple[float, int, tuple[int, ...]]]:
        """k -> (least cost of the subtree of g with exactly k counted blocks, block index, counts of the kids).
        A block counts `count[key]` (1, or 0 for a class already paid). (min, +) convolution over the kids."""
        if g in cache:
            return cache[g]
        self._tick("optimizer")
        best: dict[int, tuple[float, int, tuple[int, ...]]] = {}
        for idx, (b, key) in enumerate(self.options[g]):
            combo: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
            for i in self.kids[(g, idx)]:
                below = self._profile(i, cost, count, cache)
                merged: dict[int, tuple[float, tuple[int, ...]]] = {}
                for k1, (c1, ks) in combo.items():
                    for k2, (c2, _, _) in below.items():
                        k, c = k1 + k2, c1 + c2
                        if k not in merged or c < merged[k][0] - EPS:
                            merged[k] = (c, ks + (k2,))
                combo = merged
            for k, (c, ks) in combo.items():
                kk, total = k + count[key], c + cost[key]
                if kk not in best or total < best[kk][0] - EPS:
                    best[kk] = (total, idx, ks)
        if not best:
            raise ValueError(f"group {g} has no admissible block")
        cache[g] = dict(sorted(best.items()))
        return cache[g]

    def _rebuild_scalar(self, g: GroupId, cache: dict, out: dict[GroupId, Block]) -> None:
        _, idx = cache[g]
        out[g] = self.options[g][idx][0]
        for i in self.kids[(g, idx)]:
            self._rebuild_scalar(i, cache, out)

    def _rebuild_profile(self, g: GroupId, k: int, cache: dict, out: dict[GroupId, Block]) -> None:
        _, idx, ks = cache[g][k]
        out[g] = self.options[g][idx][0]
        for i, ki in zip(self.kids[(g, idx)], ks):
            self._rebuild_profile(i, ki, cache, out)

    def _min_max_blocks(self, g: GroupId, count: dict[ClassKey, int], cache: dict, pick) -> int:
        if g in cache:
            return cache[g]
        self._tick("optimizer")
        cache[g] = 0
        values = []
        for idx, (b, key) in enumerate(self.options[g]):
            values.append(count[key] + sum(self._min_max_blocks(i, count, cache, pick) for i in self.kids[(g, idx)]))
        cache[g] = pick(values)
        return cache[g]

    # ---------------------------------------------------------------------------------------------------------
    # Omega on the feasible range
    # ---------------------------------------------------------------------------------------------------------

    def _omega_table(self) -> None:
        """Omega on the feasible range of UNIQUE counts. The most blocks any decomposition has (kmax) bounds the unique
        count from above; from below the least number of blocks whose classes are not repeatable (every repeatable
        class at 0, at least one block in all) bounds it, because the coarsest decomposition can hold fewer unique
        classes than blocks when two forced bodies are one class (the range of block counts is not enough). Omega is
        never evaluated at 0: no decomposition has no block."""
        ones = {k: 1 for k in self.classes}
        nonrep = {k: (0 if k in self.repeatable else 1) for k in self.classes}
        self.kmin_total = sum(self._min_max_blocks(r, ones, {}, min) for r in self.roots)
        self.kmax_total = sum(self._min_max_blocks(r, ones, {}, max) for r in self.roots)
        self.kmin_unique = max(1, sum(self._min_max_blocks(r, nonrep, {}, min) for r in self.roots))
        lo, hi = self.kmin_unique, self.kmax_total
        self.omega_values = {k: self.scoring.omega(k) for k in range(lo, hi + 1)}
        self.slope = (self.omega_values[lo + 1] - self.omega_values[lo]) if hi > lo else 0.0
        self.omega_linear = all(abs(self.omega_values[k] - self.omega_values[lo] - (k - lo) * self.slope)
                                <= 1e-9 * (1.0 + abs(self.omega_values[k])) for k in range(lo, hi + 1))

    # ---------------------------------------------------------------------------------------------------------
    # the search
    # ---------------------------------------------------------------------------------------------------------

    def optimize(self) -> Optimum:
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))
        started = time.perf_counter()
        self.prepare()
        self._omega_table()
        self.deadline.check("optimizer")
        t_search = time.perf_counter()
        wmin = min(self.w.values())
        if not self.repeatable:
            chosen, claimed, method = self._tree() + ("tree",)
        else:
            chosen, claimed, method = self._astar(wmin) + ("astar",)
        search_ms = (time.perf_counter() - t_search) * 1000
        order = self.blocks._block_order(chosen)
        d = Decomposition([chosen[g] for g in order])
        self.stats.update(method=self.stats.get("method", method), search_ms=round(search_ms, 2),
                          time_ms=round((time.perf_counter() - started) * 1000, 2),
                          omega_linear=self.omega_linear, kmin=self.kmin_total, kmax=self.kmax_total,
                          kmin_unique=self.kmin_unique, shift=round(max(0.0, -wmin), 6))
        self.stats.setdefault("states_expanded", 0)
        self.stats.setdefault("states_generated", 0)
        found = self.optimum_of(d)
        # the cost of what was found, from the definition: it must equal what the search claims (a self-check that
        # turns a wrong fold or bound into an error instead of a wrong answer)
        if abs(claimed - found.cost) > 1e-6 * (1.0 + abs(found.cost)):
            raise AssertionError(f"the search's value {claimed!r} differs from the cost {found.cost!r} of what it returned")
        return found

    def optimum_of(self, d: Decomposition) -> Optimum:
        """The bookkeeping of `optimize()` for a legal decomposition of the caller's choice: its cost from the
        definition (the sum over its distinct classes of -log F, minus Omega of their number), its classes, the
        contexts F saw, its exposures (`Optimum.expose_at`). What component 9 needs when it evaluates or prints a
        decomposition of its own; `optimize()` ends here too. An illegal decomposition is an AssertionError."""
        if not hasattr(self, "classes"):
            self.prepare()
        problems = self.blocks.problems(d)
        if problems:
            raise AssertionError("not a legal decomposition: " + "; ".join(problems))
        order = self.blocks._block_order({b.output: b for b in d.blocks})
        by_block = {b.output: b for b in d.blocks}
        by_key: dict[ClassKey, list[GroupId]] = {}
        for g in order:
            by_key.setdefault(self._key_of(by_block[g]), []).append(g)
        k = len(by_key)
        omega = self.scoring.omega(k)
        cost = sum(self.w[key] for key in by_key) - omega
        try:
            score = math.exp(-cost)
        except OverflowError:  # the product form of a huge F: the cost is the number to use
            score = INF
        stats = dict(self.stats, blocks=len(order), unique_blocks=k)
        return Optimum(Decomposition([by_block[g] for g in order]), cost, score, omega, k,
                       {self.classes[key].id: self.w[key] for key in by_key},
                       {g: self.classes[self._key_of(by_block[g])].id for g in order},
                       {self.classes[key].id: gs for key, gs in by_key.items() if len(gs) > 1},
                       {self.classes[key].id: self.contexts[key] for key in by_key}, stats,
                       {c.id: c for c in self.classes.values()}, self.boundary)

    def _key_of(self, block: Block) -> ClassKey:
        for b, key in self.options[block.output]:
            if b == block:
                return key
        raise KeyError(block)

    def _tree(self) -> tuple[dict[GroupId, Block], float]:
        """No repeatable class: the memoized recursion, with the count dimension when Omega is not linear. Returns the
        blocks and the value the recursion claims for them."""
        chosen: dict[GroupId, Block] = {}
        lo = self.kmin_unique
        if self.omega_linear:
            cost = {key: w - self.slope for key, w in self.w.items()}
            cache: dict = {}
            total = 0.0
            for r in self.roots:
                total += self._scalar(r, cost, cache)[0]
                self._rebuild_scalar(r, cache, chosen)
            self.stats["method"] = "tree-dp-linear"
            return chosen, total + self.slope * lo - self.omega_values[lo]
        ones = {k: 1 for k in self.classes}
        cache = {}
        profiles = [self._profile(r, self.w, ones, cache) for r in self.roots]
        # convolve the subproblems, then pick the count that minimises cost - Omega(k)
        combo: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
        for p in profiles:
            merged: dict[int, tuple[float, tuple[int, ...]]] = {}
            for k1, (c1, ks) in combo.items():
                for k2, (c2, _, _) in p.items():
                    k, c = k1 + k2, c1 + c2
                    if k not in merged or c < merged[k][0] - EPS:
                        merged[k] = (c, ks + (k2,))
            combo = merged
        best_k, best = None, INF
        for k in sorted(combo):
            value = combo[k][0] - self.omega_values[k]
            if value < best - EPS:
                best_k, best = k, value
        for r, kr in zip(self.roots, combo[best_k][1]):
            self._rebuild_profile(r, kr, cache, chosen)
        self.stats["method"] = "tree-dp-count"
        return chosen, best

    def _astar(self, wmin: float) -> tuple[dict[GroupId, Block], float]:
        R = self.repeatable
        shift = max(0.0, -wmin)
        # variant A: the linear part of Omega is folded into the class costs, which stay non-negative; no count
        linear_fold = self.omega_linear and wmin - self.slope >= -EPS
        lo_unique = self.kmin_unique
        if linear_fold:
            cost = {key: w - self.slope for key, w in self.w.items()}
            omega_prime = None
        else:
            cost = {key: w + shift for key, w in self.w.items()}
            omega_prime = {k: v + shift * k for k, v in self.omega_values.items()}
        self.stats["method"] = "astar-linear" if linear_fold else "astar-count"
        relaxed = {key: (0.0 if key in R else c) for key, c in cost.items()}
        ones = {k: 1 for k in self.classes}
        nonrep = {k: (0 if k in R else 1) for k in self.classes}
        h_cache: dict = {}
        kmin_cache: dict = {}
        kmax_cache: dict = {}

        def h_free(g: GroupId) -> float:
            return self._scalar(g, relaxed, h_cache)[0]

        def kmin_nonrep(g: GroupId) -> int:
            return self._min_max_blocks(g, nonrep, kmin_cache, min)

        def kmax(g: GroupId) -> int:
            return self._min_max_blocks(g, ones, kmax_cache, max)

        # which repeatable classes a subtree offers, and how many of its output groups offer each
        sub_outputs: dict[GroupId, frozenset[GroupId]] = {}

        def outputs_below(g: GroupId) -> frozenset[GroupId]:
            if g in sub_outputs:
                return sub_outputs[g]
            self._tick("optimizer")
            sub_outputs[g] = frozenset({g})
            found = {g}
            for idx in range(len(self.options[g])):
                for i in self.kids[(g, idx)]:
                    found |= outputs_below(i)
            sub_outputs[g] = frozenset(found)
            return sub_outputs[g]

        offered: dict[GroupId, frozenset[ClassKey]] = {g: frozenset(k for _, k in opts if k in R) for g, opts in self.options.items()}
        below: dict[GroupId, frozenset[ClassKey]] = {}
        twice: dict[GroupId, frozenset[ClassKey]] = {}
        for g in self.options:
            self._tick("optimizer", 64)
            counts: dict[ClassKey, int] = {}
            for h in outputs_below(g):
                for key in offered[h]:
                    counts[key] = counts.get(key, 0) + 1
            below[g] = frozenset(counts)
            twice[g] = frozenset(k for k, n in counts.items() if n >= 2)

        leaf_cache: dict[tuple[GroupId, frozenset], dict] = {}

        def leaf_ok(g: GroupId, paid: frozenset, others: frozenset) -> bool:
            for key in below[g]:
                if key in paid:
                    continue
                if key in others or key in twice[g]:
                    return False
            return True

        def leaf_solve(g: GroupId, paid: frozenset):
            """The exact recursion for a subtree none of whose repeatable classes can be paid twice: paid ones free."""
            local = frozenset(paid & below[g])
            key = (g, local)
            if key not in leaf_cache:
                c = {k: (0.0 if k in local else v) for k, v in cost.items()}
                n = {k: (0 if k in local else 1) for k in self.classes}
                if linear_fold:
                    cache: dict = {}
                    self._scalar(g, c, cache)
                    leaf_cache[key] = ("scalar", cache)
                else:
                    cache = {}
                    self._profile(g, c, n, cache)
                    leaf_cache[key] = ("profile", cache)
            return leaf_cache[key]

        rank = {g: i for i, g in enumerate([self.root] + sorted(self.forced) + sorted(g for g in self.options if g != self.root and g not in self.forced))}

        def fold(groups: list[GroupId], paid: frozenset, k: int, g_cost: float):
            """Solve every pending group that needs no search; branch over the count for the others. Yields
            (pending tuple, paid, k, g_cost, folded moves)."""
            pending = sorted(set(groups), key=rank.__getitem__)
            moves: list[tuple] = []
            changed = True
            while changed:
                changed = False
                for g in list(pending):
                    self._tick("optimizer", 64)
                    others = frozenset().union(*(below[o] for o in pending if o != g)) if len(pending) > 1 else frozenset()
                    if leaf_ok(g, paid, others):
                        pending.remove(g)
                        moves.append(("leaf", g, paid & below[g]))
                        changed = True
            live = frozenset().union(*(below[p] for p in pending)) if pending else frozenset()
            paid = paid & live
            if linear_fold:
                total = g_cost
                for _, g, local in moves:
                    total += leaf_solve(g, local)[1][g][0]
                yield tuple(pending), paid, k, total, tuple((kind, g, local, None) for kind, g, local in moves)
                return
            combo: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
            for _, g, local in moves:
                prof = leaf_solve(g, local)[1][g]
                merged: dict[int, tuple[float, tuple[int, ...]]] = {}
                for k1, (c1, ks) in combo.items():
                    for k2, (c2, _, _) in prof.items():
                        kk, c = k1 + k2, c1 + c2
                        if kk not in merged or c < merged[kk][0] - EPS:
                            merged[kk] = (c, ks + (k2,))
                combo = merged
            for kk in sorted(combo):
                c, ks = combo[kk]
                yield (tuple(pending), paid, k + kk, g_cost + c,
                       tuple((kind, g, local, kg) for (kind, g, local), kg in zip(moves, ks)))

        def heuristic(pending: tuple[GroupId, ...], k: int) -> float:
            h = sum(h_free(p) for p in pending)
            if omega_prime is not None:
                lo = max(lo_unique, k + sum(kmin_nonrep(p) for p in pending))
                hi = max(lo, k + sum(kmax(p) for p in pending))
                h += min(-omega_prime[j] for j in range(lo, hi + 1))
            return h

        # states: key -> (g_cost, parent key, move); open: (f, seq, key)
        best_g: dict[tuple, float] = {}
        parent: dict[tuple, tuple] = {}
        move_of: dict[tuple, tuple] = {}
        closed: set[tuple] = set()
        heap: list[tuple[float, int, tuple]] = []
        seq = 0
        generated = expanded = 0
        for pending, paid, k, g_cost, moves in fold(list(self.roots), frozenset(), 0, 0.0):
            state = (pending, paid, k if not linear_fold else 0)
            if g_cost < best_g.get(state, INF) - EPS:
                best_g[state] = g_cost
                parent[state] = None
                move_of[state] = ("start", None, None, moves)
                heapq.heappush(heap, (g_cost + heuristic(pending, k), seq, state))
                seq += 1
                generated += 1
        goal = None
        while heap:
            f, _, state = heapq.heappop(heap)
            if state in closed:
                continue
            closed.add(state)
            pending, paid, k = state
            if not pending:
                goal = state
                break
            expanded += 1
            if expanded % ASTAR_CLOCK_EVERY == 0 and self.deadline.expired():
                self.stats.update(states_expanded=expanded, states_generated=generated, open_states=len(heap))
                raise DecompositionTimeout("optimizer", self.deadline.seconds, self.deadline.elapsed(),
                                           f"the decomposition timeout ({expanded} states expanded, {generated} generated, {len(heap)} open)")
            g_cost = best_g[state]
            p = pending[0]
            rest = list(pending[1:])
            for idx, (b, key) in enumerate(self.options[p]):
                rep = key in R
                new_cost = g_cost + (0.0 if (rep and key in paid) else cost[key])
                new_k = k + (0 if (rep and key in paid) else 1)
                new_paid = paid | {key} if rep else paid
                for pending2, paid2, k2, cost2, moves in fold(rest + list(self.kids[(p, idx)]), new_paid, new_k, new_cost):
                    state2 = (pending2, paid2, k2 if not linear_fold else 0)
                    if state2 in closed or cost2 >= best_g.get(state2, INF) - EPS:
                        continue
                    best_g[state2] = cost2
                    parent[state2] = state
                    move_of[state2] = ("block", p, idx, moves)
                    heapq.heappush(heap, (cost2 + heuristic(pending2, k2), seq, state2))
                    seq += 1
                    generated += 1
        self.stats.update(states_expanded=expanded, states_generated=generated, open_states=len(heap))
        if goal is None:
            raise AssertionError("the search ran out of states without a goal")
        goal_cost = best_g[goal]
        claimed = (goal_cost + self.slope * lo_unique - self.omega_values[lo_unique]) if linear_fold \
            else goal_cost - omega_prime[goal[2]]
        # rebuild the decomposition along the path
        chosen: dict[GroupId, Block] = {}
        state = goal
        while state is not None:
            kind, p, idx, moves = move_of[state]
            if kind == "block":
                chosen[p] = self.options[p][idx][0]
            for leaf_kind, g, local, kg in moves:
                what, cache = leaf_solve(g, local)
                if what == "scalar":
                    self._rebuild_scalar(g, cache, chosen)
                else:
                    self._rebuild_profile(g, kg, cache, chosen)
            state = parent[state]
        return chosen, claimed


def optimize(memo: Memo, result: ImportResult, blocks: Blocks, scoring: Scoring, *, schema: Schema | None = None,
             constraints: dict[GroupId, GroupConstraints] | None = None, signatures: dict[GroupId, str] | None = None,
             printer: Printer | None = None, deadline: Deadline | None = None) -> Optimum:
    """The optimal decomposition of one query under `scoring`. Exact (see the module docstring); raises
    DecompositionTimeout when `deadline` runs out first (TooManyBlocks at the enumeration's memory valve)."""
    return Optimizer(memo, result, blocks, scoring, schema=schema, constraints=constraints, signatures=signatures,
                     printer=printer, deadline=deadline).optimize()


def decomposition_cost(blocks: Blocks, d: Decomposition, w_of, omega) -> tuple[float, int]:
    """cost(D) from the definition, for a decomposition given a per-BLOCK class cost function `w_of(block) -> (key,
    w)`: the sum over the distinct classes minus Omega of their number. Used by the tests' brute force."""
    seen: dict = {}
    for b in d.blocks:
        key, w = w_of(b)
        seen.setdefault(key, w)
    k = len(seen)
    return sum(seen.values()) - omega(k), k
