"""Build an e-graph from ImportResult, then read it into the plain Memo data model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

from sqldecomp.model import (Filter, ImportResult, Join, Memo, MExpr, Op1, Plan, Scan, region_inputs_under,
                             validate_import_result, walk)

from . import egraph as eg


class InvalidImportResult(ValueError):
    """build_memo refuses an ImportResult that breaks the contract; `problems` lists every violation."""

    def __init__(self, problems: list[str]):
        super().__init__("invalid ImportResult: " + "; ".join(problems))
        self.problems = problems


class InvalidMemo(AssertionError):
    """Raised when the e-graph read-out violates memo soundness invariants.
    Every expression of a class must cover the same region inputs and conjuncts;
    join inputs must be disjoint; no conjunct may be applied twice on one path.
    A filter must cover its conjunct's owner, while an unowned conjunct must remain
    where P0 places it. Every registered root/scope plan must remain in the graph.

    An invalid memo is never returned. Memo.complete=False instead denotes a sound
    but incomplete search. problems lists the violated invariants. These checks
    supplement the engine registration/read-back checks in sqldecomp.memo.egraph.
    """

    def __init__(self, problems: list[str]):
        super().__init__("unsound memo: " + "; ".join(problems))
        self.problems = problems


@dataclass
class MemoStats:
    """What one build did. Kept beside the Memo (the Memo itself is the plain contract type)."""

    groups: int = 0
    expressions: int = 0
    iterations: int = 0
    saturate_s: float = 0.0
    readout_s: float = 0.0
    readout: str = ""
    complete: bool = True
    extra: dict[str, float] = field(default_factory=dict)


def build_memo(result: ImportResult, *, budget_s: float = 5.0, readout: str = "auto") -> Memo:
    return build_memo_with_stats(result, budget_s=budget_s, readout=readout)[0]


def build_memo_with_stats(result: ImportResult, *, budget_s: float = 5.0, readout: str = "auto") -> tuple[Memo, MemoStats]:
    problems = check_import_result(result)
    if problems:
        raise InvalidImportResult(problems)

    inputs = {i.id: i.plan for i in result.facts.inputs}
    input_idx = {iid: n for n, iid in enumerate(inputs)}
    conj_idx: dict[str, int] = {}
    op_idx: dict[str, int] = {}
    cache: dict[Plan, eg.Term] = {}

    def term(p: Plan) -> eg.Term:
        if p in cache:
            return cache[p]
        if isinstance(p, Scan):
            t = ("scan", input_idx[p.occ])
        elif isinstance(p, Join):
            t = ("join", term(p.left), term(p.right))
        elif isinstance(p, Filter):
            t = ("filter", term(p.input), conj_idx.setdefault(p.conjunct, len(conj_idx)))
        elif isinstance(p, Op1):
            t = ("op1", term(p.input), op_idx.setdefault(p.op, len(op_idx)))
        else:
            t = ("op2", op_idx.setdefault(p.op, len(op_idx)), term(p.left), term(p.right))
        cache[p] = t
        return t

    plans = [result.plan] + [s.plan for s in result.scopes]
    terms = [term(p) for p in plans]
    present = {n for p in plans for n in walk(p)}
    # covers facts only for inputs that occur in a registered plan: a stray fact would add a group of its own
    cover_facts = [(term(p), input_idx[iid]) for iid, p in inputs.items() if p in present]
    edges = [tuple(input_idx[x] for x in dict.fromkeys(e.inputs)) for e in result.facts.edges]
    owners = [(conj_idx[c], input_idx[o]) for c, o in result.facts.owners.items() if c in conj_idx]

    raw = eg.saturate(terms, cover_facts, edges, owners, budget_s=budget_s, readout=readout)
    memo = _read_out(raw, result, terms, list(inputs), list(conj_idx), list(op_idx), result.facts.owners,
                     _p0_filter_covers(result))
    stats = MemoStats(len(memo.groups), sum(len(v) for v in memo.groups.values()), raw.iterations, raw.seconds,
                      raw.stats.get("readout_s", 0.0), raw.readout, raw.complete)
    return memo, stats


def check_import_result(result: ImportResult) -> list[str]:
    """The contract's invariants plus the four this implementation relies on: a region input is named after its scan
    or operator, an occurrence id names one table, the two sides of a join cover disjoint inputs (every region input
    is ONE occurrence), and an owned conjunct sits above its owner (otherwise R4 would move it to a side it does not
    belong to)."""
    bad = list(validate_import_result(result))
    if bad:
        return bad
    by_plan = {i.plan: i.id for i in result.facts.inputs}
    for i in result.facts.inputs:  # the read-out recognises a region leaf by this id
        if i.id != (i.plan.occ if isinstance(i.plan, Scan) else getattr(i.plan, "op", None)):
            bad.append(f"region input {i.id} must be named after its scan occurrence or its operator")
    nodes = [n for p in [result.plan] + [s.plan for s in result.scopes] for n in walk(p)]
    tables: dict[str, str] = {}
    for n in nodes:
        if isinstance(n, Scan) and tables.setdefault(n.occ, n.table) != n.table:
            bad.append(f"occurrence {n.occ} is used for two tables ({tables[n.occ]} and {n.table})")
    for n in set(nodes):
        if isinstance(n, Join):
            left, right = region_inputs_under(n.left, by_plan), region_inputs_under(n.right, by_plan)
            if left is not None and right is not None and left & right:
                bad.append(f"the two sides of a join share the inputs {sorted(left & right)}")
        if isinstance(n, Filter) and n.conjunct in result.facts.owners:
            under = region_inputs_under(n.input, by_plan)
            if under is not None and result.facts.owners[n.conjunct] not in under:
                bad.append(f"conjunct {n.conjunct} is applied below its owner {result.facts.owners[n.conjunct]}")
    return bad


def _p0_filter_covers(result: ImportResult) -> dict[str, frozenset[str]]:
    """conjunct -> the region inputs under its Filter node in P0 (the root and every scope plan). A conjunct without an
    owner never moves, so every memo expression that applies it must sit over exactly these inputs."""
    by_plan = {i.plan: i.id for i in result.facts.inputs}
    out: dict[str, frozenset[str]] = {}
    for plan in [result.plan] + [s.plan for s in result.scopes]:
        for n in walk(plan):
            if isinstance(n, Filter):
                under = region_inputs_under(n.input, by_plan)
                if under is not None:
                    out.setdefault(n.conjunct, under)
    return out


def _read_out(raw: eg.RawEGraph, result: ImportResult, terms: list[eg.Term], inputs: list[str], conjs: list[str],
              ops: list[str], owners: dict[str, str] | None = None,
              p0_covers: dict[str, frozenset[str]] | None = None) -> Memo:
    names = {"scan": inputs, "filter": conjs, "op1": ops, "op2": ops}
    input_ids = set(inputs)
    root_term = terms[0]
    owners = owners or {}
    p0_covers = p0_covers or {}
    problems: list[str] = []

    # 1. e-nodes per class, with commuted join duplicates removed (children as a sorted pair of class keys).
    #    "Every expression belongs to exactly one group": at saturation that holds by construction. When the budget
    #    stopped the run early, join(a, b) and join(b, a) can still sit in two classes, so classes that hold the same
    #    expression are merged here (R1 plus congruence, repeated until nothing changes); a no-op when saturated.
    parent: dict[Hashable, Hashable] = {}

    def find(x: Hashable) -> Hashable:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    changed = True
    while changed:
        changed = False
        owner_of: dict[tuple, Hashable] = {}
        for r in raw.rows:
            kids = tuple(find(k) for k in r.children)
            key = (r.kind, r.arg, tuple(sorted(kids, key=_order)) if r.kind == "join" else kids)
            cls = find(r.cls)
            if owner_of.setdefault(key, cls) != cls:
                parent[cls] = owner_of[key]
                changed = True
    members: dict[Hashable, set[tuple]] = {}
    for r in raw.rows:
        arg = None if r.kind == "join" else names[r.kind][r.arg]
        kids = tuple(find(k) for k in r.children)
        kids = tuple(sorted(kids, key=_order)) if r.kind == "join" else kids
        members.setdefault(find(r.cls), set()).add((r.kind, arg, kids))

    # 2. per class: the region inputs it covers, the conjuncts applied under it, its size. All three are the same
    #    for every expression of a class (M1); a disagreement means the e-graph merged what it must not merge. The
    #    soundness checks of `InvalidMemo` run here, on every build: they cost one pass over the e-nodes.
    info: dict[Hashable, tuple[frozenset, frozenset, int]] = {}

    def describe(cls: Hashable) -> tuple[frozenset, frozenset, int]:
        if cls in info:
            return info[cls]
        info[cls] = (frozenset(), frozenset(), 0)  # a cyclic e-graph (never expected): this stops the recursion
        if cls not in members:  # a child class without any expression: a torn read-out
            problems.append("a class referenced as a child holds no expression")
            return info[cls]
        seen = None
        for kind, arg, kids in sorted(members[cls], key=lambda m: (m[0], m[1] or "")):
            sub = [describe(k) for k in kids]
            if kind in ("join", "filter"):
                cov = frozenset().union(*[s[0] for s in sub])
            else:  # a scan or an operator is a leaf of the region above it, if it is a registered input
                cov = frozenset({arg}) if arg in input_ids else frozenset()
            if kind == "join" and sub[0][0] & sub[1][0]:
                problems.append(f"join sides overlap on {sorted(sub[0][0] & sub[1][0])}")
            if kind == "filter":
                below = sub[0]
                if arg in below[1]:
                    problems.append(f"conjunct {arg} is applied twice on one path")
                owner = owners.get(arg)
                if owner is not None and owner not in below[0]:
                    problems.append(f"filter {arg} sits over {sorted(below[0])}, which does not hold its owner {owner}")
                if owner is None and arg in p0_covers and below[0] != p0_covers[arg]:
                    problems.append(f"conjunct {arg} has no owner and sits over {sorted(below[0])} instead of "
                                    f"{sorted(p0_covers[arg])}, where the query put it")
            applied = frozenset().union(*[s[1] for s in sub]) | ({arg} if kind == "filter" else frozenset())
            here = (cov, applied if kind in ("join", "filter") else frozenset(), 1 + sum(s[2] for s in sub))
            if seen is not None and here != seen:
                problems.append(f"M1 violated: one class holds {seen} and {here}")
            seen = here
        info[cls] = seen
        return seen

    for cls in members:
        describe(cls)
    for n, term in enumerate(terms):
        if raw.class_of(term) is None:
            problems.append("the e-graph lost the root plan" if n == 0 else f"the e-graph lost the plan of scope {n}")
    if problems:
        raise InvalidMemo(sorted(set(problems)))

    # 3. deterministic opaque ids: order classes by size, then by a canonical smallest-plan text
    canon: dict[Hashable, str] = {}

    def text(cls: Hashable) -> str:
        if cls not in canon:
            canon[cls] = min(f"{kind}[{arg or ''}]({','.join(sorted(text(k) for k in kids) if kind == 'join' else [text(k) for k in kids])})"
                             for kind, arg, kids in members[cls])
        return canon[cls]

    order = sorted(members, key=lambda c: (info[c][2], text(c)))
    width = len(str(len(order) - 1))
    gid = {cls: f"g{n:0{width}d}" for n, cls in enumerate(order)}

    # 4. expressions, every join with its condition: the edges that span its two sides
    edges = [(e.conjunct, frozenset(e.inputs)) for e in result.facts.edges]
    groups: dict[str, list[MExpr]] = {}
    for cls in order:
        exprs = []
        for kind, arg, kids in members[cls]:
            cond: tuple[str, ...] = ()
            if kind == "join":
                left, right = info[kids[0]][0], info[kids[1]][0]
                cond = tuple(sorted(c for c, ins in edges if ins <= left | right and not ins <= left and not ins <= right))
            children = tuple(sorted(gid[k] for k in kids)) if kind == "join" else tuple(gid[k] for k in kids)
            exprs.append(MExpr(kind, children, arg, cond))
        groups[gid[cls]] = sorted(exprs, key=lambda e: (e.kind, e.children, e.arg or ""))

    root = raw.class_of(root_term)
    return Memo(groups, gid[find(root)], raw.complete)


def _order(key: Hashable):
    """Any total order on class keys (only used to drop commuted duplicates)."""
    return key if isinstance(key, str) else (hash(key), repr(key))
