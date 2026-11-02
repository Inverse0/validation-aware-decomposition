"""Baseline decompositions for RQ2, evaluated in place of the optimizer's choice:

    syntactic   a block at every CTE, derived table and subquery as the query writes them, where the cut is legal,
                plus the forced cuts (`Blocks.with_cuts` of the scopes' top groups)
    random-k    a seeded random legal decomposition with k distinct blocks (k: the optimum's of another run), or
                the nearest number that has one: every legal decomposition (`Blocks.decompositions`) when there are
                at most `EXHAUSTIVE_LIMIT`, drawn uniformly among those with k; else the distinct ones of `DRAWS`
                random draws (`Blocks.sample`, the cut probability running over `P_GRID`), drawn among those with k

Either is evaluated by the optimizer's own bookkeeping (`Optimizer.optimum_of`: its cost, classes and contexts under
the run's scoring) and checked by C2 (`printer.check_recomposition`) before its blocks are printed. The number of
distinct blocks is the optimum's: block classes, a class shared by two blocks counted once.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Callable

from sqldecomp.blocks import Blocks, TooManyBlocks, decomposition_key
from sqldecomp.model import Decomposition, GroupId, ImportResult, Memo

STRATEGIES = ("optimum", "syntactic", "random-k")
SYNTACTIC_KINDS = ("cte", "derived", "subquery")  # the scopes the query writes as relations of their own
EXHAUSTIVE_LIMIT = 2_000
DRAWS = 4_000
P_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def query_seed(seed: int, key: str) -> int:
    """The seed of one query: the run's seed and the query's key, hashed (the same query draws the same in any order)."""
    return int(hashlib.sha256(f"{seed}\n{key}".encode()).hexdigest()[:16], 16)


def syntactic_cuts(memo: Memo, result: ImportResult) -> list[GroupId]:
    """The top groups of the scopes the query writes as relations of their own (CTEs, derived tables, subqueries; not
    the root, not a branch of a set operation), where the memo holds them."""
    out = []
    for scope in result.scopes:
        if scope.kind in SYNTACTIC_KINDS:
            g = memo.group_of(scope.plan)
            if g is not None and g != memo.root and g not in out:
                out.append(g)
    return out


def syntactic(blocks: Blocks, memo: Memo, result: ImportResult) -> tuple[Decomposition, dict[str, Any]]:
    """The syntactic decomposition and what it was made of: the scopes' groups, those cut (legal and kept), the
    forced cuts."""
    cuts = syntactic_cuts(memo, result)
    d = blocks.with_cuts(cuts)
    made = {b.output for b in d.blocks} - {memo.root}
    return d, {"strategy": "syntactic", "scope_groups": len(cuts), "cut": sorted(g for g in cuts if g in made),
               "not_cut": sorted(g for g in cuts if g not in made), "forced": sorted(blocks.forced_cuts())}


def random_k(blocks: Blocks, unique_count: Callable[[Decomposition], int], k: int, seed: int, *, exhaustive_limit: int = EXHAUSTIVE_LIMIT,
             draws: int = DRAWS, tick: Callable[[], None] | None = None) -> tuple[Decomposition, dict[str, Any]]:
    """A seeded random legal decomposition with `k` distinct blocks (`unique_count`), or with the nearest number that
    has one (ties: the smaller); and how it was drawn. `tick`: looked at every 100 draws (the decomposition clock)."""
    rng = random.Random(seed)
    try:
        candidates = blocks.decompositions(limit=exhaustive_limit)
        method = "exhaustive"
    except TooManyBlocks:
        found: dict[tuple, Decomposition] = {}
        sampler = random.Random(rng.random())
        for i in range(draws):
            if tick is not None and i % 100 == 0:
                tick()
            d = blocks.sample(sampler, P_GRID[i % len(P_GRID)])
            if d is not None:
                found.setdefault(decomposition_key(d), d)
        candidates = list(found.values())
        method = "sampled"
    by_count: dict[int, list[Decomposition]] = {}
    for d in candidates:
        by_count.setdefault(unique_count(d), []).append(d)
    if not by_count:
        raise ValueError("no legal decomposition was found")
    reached = k if k in by_count else min(by_count, key=lambda c: (abs(c - k), c))
    pool = sorted(by_count[reached], key=decomposition_key)
    chosen = pool[rng.randrange(len(pool))]
    return chosen, {"strategy": "random-k", "seed": seed, "method": method, "candidates": len(candidates), "k_wanted": k, "k_reached": reached,
                    "pool": len(pool), "counts": {str(c): len(ds) for c, ds in sorted(by_count.items())}}
