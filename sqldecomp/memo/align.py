"""Component 5, second half: comparing plans that come from two different imports.

Ids are opaque and never match across imports, so a plan of `other` is first translated onto the ids of `onto`:
scans by (table, alias), every other side entry by (kind, canonical sql). Then component 5 is a lookup:
`memo_of_onto.group_of(translated plan)`.

WHAT THE KEYS MEAN. Both keys keep the FROM aliases: a scan of `other` maps only onto a scan of `onto` with the same
table AND the same alias, and a conjunct or operator only onto one with the same canonical text, which names its
columns by those aliases. So everything in this module is ALIAS-BOUND. It is made for a re-import of a print of the
SAME query, whose aliases the printer keeps (checks C1 / C2). Two queries that differ in their aliases (`customers c`
against `customers k`) have no alignment at all, and `align` / `groups_in` / `equivalent` answer None / empty / False
for them WITHOUT saying so, although the design document counts alpha-equivalent plans as equivalent. The alias-free
comparison of two queries is `memo.same_task_across` (built on `signature`, conservative).

WHY THIS IS SOUND, whatever mapping is used: a mapping that preserves those keys sends every piece of `other` to a
piece of `onto` with the same text, and group_of only answers for plans that the memo of `onto` really holds, with
every conjunct above the inputs it references and every join carrying exactly its spanning conjuncts. Inside one
region aliases are unique, so equal text over equally named inputs is the same predicate. One thing the skeleton does
not hold is a nested scope's OUTPUT LIST (a rename-only projection is not a node), and it decides what `d.k` means
above a `ref` and what a set operation lines up, so the mapping also requires equal output lists there.
A second thing the compared plan does not hold is the BODY of a predicate subquery: the outer plan only has the opaque
conjunct text, and that text names a CTE without saying what the CTE computes. So a conjunct or operator that names
subquery scopes (`data.subqueries`) only maps to a partner whose subquery scopes hold, position by position, the
translated plans of its own subquery scopes. `align` enforces this for every such entry (it needs the memo of `onto`
for that and builds it unless the caller passes it), so the contract's path `align` + `Memo.group_of` is safe; the
tolerant search enforces it while it binds ids.
The output list of the ROOT scope is NOT compared by default: two queries that differ only in their final
rename-or-drop projection are equivalent PLANS by the design document's normalization (iii). That is right for a printed
block (check C1), whose columns are not the query's, and wrong for a whole query (check C2): `SELECT x.v FROM ... x, ... y`
and `SELECT y.v FROM ...` have one plan and return different data. `equivalent(..., compare_outputs=True)` adds the
ordered comparison of the two root output lists (`same_root_outputs`).

`align` is the contract's strict mapping: any ambiguity (the same alias in two scopes, the same conjunct text twice)
or any id without a partner gives None, never a guess. `groups_in(..., strict=False)` is the tolerant variant: it
searches the key-preserving mappings under which the plan IS in the memo and returns every group found that way; by
the argument above each of them is a true answer. `alignments` is the same search handing out each (group, mapping)
it finds: the printer's checks C1 / C2 are built on it (they need the mapping to compare the output lists of nested
scopes), so it is the ONE aligner of the product path.
"""
from __future__ import annotations

from typing import Iterator

from sqldecomp.model import Filter, GroupId, ImportResult, Join, Memo, Op1, Op2, Plan, Scan, SideEntry, walk

from .build import build_memo

SEARCH_LIMIT = 50_000  # states the tolerant search may visit before it gives up (an empty answer, never a guess)


class SearchLimit(Exception):
    """The tolerant search visited SEARCH_LIMIT states without finishing. Raised by `alignments` only; a caller that
    certifies something must treat it as 'cannot tell', never as a yes."""


def align(other: ImportResult, onto: ImportResult, *, memo: Memo | None = None) -> dict[str, str] | None:
    """Map the ids of `other` onto the ids of `onto` so that plans from two imports can be compared: scans by
    (table, alias), conjuncts and operators by (kind, canonical sql). Returns None when the mapping is ambiguous
    or incomplete.

    The mapping is ALIAS-BOUND: it exists only for two imports of the same query under the same aliases (a print of
    the query re-imported, checks C1 / C2). None therefore means one of: a key that occurs twice on either side
    (`FROM orders o` in two scopes, one conjunct text written twice: ambiguous, even for a query against itself); an
    id of `other` without a partner, which is what a renamed alias (`customers k` for `customers c`), a changed
    literal or an extra conjunct all look like (incomplete); different nested output lists or subquery bodies. None
    never means 'not equivalent under R' by itself; `explain_alignment` says which of these it was.

    Two things that those keys cannot see are required as well, because without them `align` + `Memo.group_of` could
    call two different queries equal: a `ref` / set operation only maps onto one whose nested scopes have the same
    output lists, and an entry that names predicate subqueries only maps onto one whose subquery bodies are the same
    under R. The second needs the memo of `onto`: pass it as `memo` if you have it, otherwise it is built here (only
    when the queries have predicate subqueries at all)."""
    mine, theirs = _by_key(other), _by_key(onto)
    mapping: dict[str, str] = {}
    for key, ids in mine.items():
        if len(ids) != 1 or len(theirs.get(key, ())) != 1:
            return None
        mapping[ids[0]] = theirs[key][0]
    a, b = _Shape(other), _Shape(onto)
    for src, dst in mapping.items():
        if not _same_outputs(a, src, b, dst):
            return None
    if not _subqueries_match(memo, onto, other, mapping):
        return None
    return mapping


def translate_plan(plan: Plan, mapping: dict[str, str]) -> Plan:
    """The same plan with every id replaced through `mapping` (KeyError when an id has no partner)."""
    if isinstance(plan, Scan):
        return Scan(plan.table, mapping[plan.occ])
    if isinstance(plan, Join):
        return Join(translate_plan(plan.left, mapping), translate_plan(plan.right, mapping),
                    tuple(sorted(mapping[c] for c in plan.conjuncts)))
    if isinstance(plan, Filter):
        return Filter(translate_plan(plan.input, mapping), mapping[plan.conjunct])
    if isinstance(plan, Op1):
        return Op1(translate_plan(plan.input, mapping), mapping[plan.op])
    return Op2(mapping[plan.op], translate_plan(plan.left, mapping), translate_plan(plan.right, mapping))


def groups_in(memo: Memo, onto: ImportResult, other: ImportResult, plan: Plan | None = None, *,
              strict: bool = True) -> frozenset[GroupId]:
    """The groups of `memo` (the memo of `onto`) that hold `plan`, a plan of `other` (default: other.plan).
    strict=True goes through `align` and answers with at most one group. strict=False searches instead and may
    answer with several groups when `onto` writes the same thing at several places.

    Empty means 'not found', which covers two things it does not tell apart: the plan is in no group of the memo
    under any alignment (not equivalent under R), or there is no alignment at all, because the keys are alias-bound
    (see `align`: other aliases, and under strict=True an ambiguous key). Meant for a re-import of a print of the
    same query; for two queries with other aliases use `same_task_across`."""
    plan = other.plan if plan is None else plan
    if strict:
        mapping = align(other, onto, memo=memo)
        try:
            g = None if mapping is None else memo.group_of(translate_plan(plan, mapping))
        except KeyError:
            g = None
        return frozenset() if g is None else frozenset({g})
    try:
        return frozenset(g for g, _ in alignments(memo, onto, other, plan))
    except SearchLimit:
        return frozenset()


def alignments(memo: Memo, onto: ImportResult, other: ImportResult, plan: Plan | None = None, *,
               limit: int = SEARCH_LIMIT) -> Iterator[tuple[GroupId, dict[str, str]]]:
    """Every (group, mapping) such that `plan` (a plan of `other`, default other.plan), written with the ids of
    `onto` through the key-preserving, one-to-one `mapping`, is a plan of that group of `memo` (the memo of `onto`).
    A mapping covers the ids of the plan, of the nested scopes below its refs, and of every predicate / select-item
    subquery that its conjuncts and operators name; each such subquery body is, translated, in the group of the
    subquery scope its partner names at the same position. Lazy: a caller that needs one good mapping stops early.
    Raises SearchLimit after `limit` states."""
    return _search(memo, onto, other, other.plan if plan is None else plan, limit)


def equivalent(a: ImportResult, b: ImportResult, *, budget_s: float = 5.0, strict: bool = True,
               compare_outputs: bool = False) -> bool:
    """Component 5 for two imports of the SAME query, the original and a print or recomposition of it: True iff b's
    plan, aligned onto a, lies in the root group of a's memo. A True is a proof (the aligned plan IS in the memo).

    False means one of THREE things, which only `explain_alignment(b, a)` tells apart:
    - not equivalent under R1-R4;
    - no alignment: ids are aligned by (table, alias) and by canonical text over those aliases, so a query written
      with other aliases (`customers k` for `customers c`) is never aligned, and under strict=True neither is one
      with an alias reused in two scopes or a conjunct text written twice (see `align`). The design document counts
      alpha-equivalent plans as equivalent; this function cannot see that. Use `same_task_across` for it;
    - the one direction: a's memo holds what R reaches from a, and R2 removes a cross join but never creates one, so
      around a removable cross join `equivalent(a, b)` can be True while `equivalent(b, a)` is False. A caller that wants a symmetric answer asks both ways.
    For two sub-plans of ONE query use Memo.group_of directly.

    `a` is the reference (the original query), `b` the candidate (a re-printed or recomposed query). strict=False
    searches for an alignment instead of demanding the unique one (see groups_in). compare_outputs=True also demands
    equal root output lists.
    """
    if compare_outputs and not same_root_outputs(a, b):
        return False
    memo = build_memo(a, budget_s=budget_s)
    return memo.root in groups_in(memo, a, b, strict=strict)


def same_root_outputs(a: ImportResult, b: ImportResult) -> bool:
    """The two root scopes expose the same columns: the same (name, sql) pairs in the same order. Alignment keeps
    aliases, so the texts compare directly. Not part of plan equivalence (a rename-or-drop projection is not a node);
    a caller that certifies a whole query wants both."""
    roots = [[s for s in r.scopes if s.kind == "root"] for r in (a, b)]
    if any(len(r) != 1 for r in roots):
        return False
    return [(o.name, o.sql) for o in roots[0][0].outputs] == [(o.name, o.sql) for o in roots[1][0].outputs]


def explain_alignment(other: ImportResult, onto: ImportResult) -> list[str]:
    """Why `align(other, onto)` is None, in words; empty when it succeeds. 'Cannot tell' (an ambiguous key) and
    'different' (an id without a partner) are both None for align; a caller that reports a failed check wants to
    know which."""
    mine, theirs = _by_key(other), _by_key(onto)
    why = []
    for key, ids in mine.items():
        if len(ids) > 1:
            why.append(f"ambiguous: {key} occurs {len(ids)} times in `other` ({', '.join(ids)})")
        elif len(theirs.get(key, ())) > 1:
            why.append(f"ambiguous: {key} occurs {len(theirs[key])} times in `onto` ({', '.join(theirs[key])})")
        elif not theirs.get(key):
            why.append(f"incomplete: {ids[0]} {key} has no partner in `onto`")
    if not why:
        a, b = _Shape(other), _Shape(onto)
        mapping = {src: theirs[_key(entry)][0] for src, entry in other.side.items()}
        why += [f"different: the output lists under {src} -> {dst} differ" for src, dst in mapping.items()
                if not _same_outputs(a, src, b, dst)]
        if not why and not _subqueries_match(None, onto, other, mapping):
            why.append("different: a predicate subquery has the same text but another body (it names a CTE that differs)")
    return why


# --------------------------------------------------------------------------------------------------------------


def _key(entry: SideEntry) -> tuple:
    if entry.kind == "scan":
        return ("scan", entry.data.get("table"), entry.data.get("alias"))
    return (entry.kind, entry.sql)


def _by_key(result: ImportResult) -> dict[tuple, list[str]]:
    out: dict[tuple, list[str]] = {}
    for entry in result.side.values():
        out.setdefault(_key(entry), []).append(entry.id)
    return out


class _Shape:
    """What the output-list guard needs from one import: its scopes by id and by plan, its Op2 nodes by id."""

    def __init__(self, result: ImportResult):
        self.side = result.side
        self.scopes = {s.id: s for s in result.scopes}
        self.by_plan = {s.plan: s for s in result.scopes if s.kind != "root"}
        self.op2 = {n.op: n for p in [result.plan] + [s.plan for s in result.scopes] for n in walk(p) if isinstance(n, Op2)}

    def outputs_under(self, op: str) -> list[tuple | None] | None:
        """The output lists an operator depends on: a ref's nested scope; the two inputs of an op2 where they are
        scope plans (set-operation branches). None for every other operator."""
        entry = self.side.get(op)
        if entry is None:
            return None
        if entry.kind == "ref":
            scope = self.scopes.get(entry.data.get("scope"))
            return [None if scope is None else tuple((o.name, o.sql) for o in scope.outputs)]
        if op in self.op2:
            sides = [self.by_plan.get(self.op2[op].left), self.by_plan.get(self.op2[op].right)]
            return [None if s is None else tuple((o.name, o.sql) for o in s.outputs) for s in sides]
        return None


def _subqueries_match(memo: Memo | None, onto: ImportResult, other: ImportResult, mapping: dict[str, str]) -> bool:
    """Every subquery scope named by an entry of `other` must be, translated, in the group of the subquery scope that
    its partner names at the same position. All entries are checked, so subqueries inside subqueries are covered."""
    mine, theirs = {s.id: s for s in other.scopes}, {s.id: s for s in onto.scopes}
    for src, entry in other.side.items():
        a_subs = entry.data.get("subqueries") or []
        b_subs = onto.side[mapping[src]].data.get("subqueries") or []
        if len(a_subs) != len(b_subs):
            return False
        for sa, sb in zip(a_subs, b_subs):
            if sa not in mine or sb not in theirs:
                return False
            if memo is None:  # built only when some entry really names a subquery
                memo = build_memo(onto)
            try:
                g = memo.group_of(translate_plan(mine[sa].plan, mapping))
            except KeyError:
                return False
            if g is None or g != memo.group_of(theirs[sb].plan):
                return False
    return True


def _same_outputs(a: _Shape, src: str, b: _Shape, dst: str) -> bool:
    mine, theirs = a.outputs_under(src), b.outputs_under(dst)
    if mine is None and theirs is None:
        return True
    if mine is None or theirs is None or len(mine) != len(theirs):
        return False
    if a.side[src].kind == "ref" and None in mine + theirs:
        return False  # a ref whose scope is unknown cannot be compared
    return mine == theirs


def _search(memo: Memo, onto: ImportResult, other: ImportResult, plan: Plan,
            limit: int = SEARCH_LIMIT) -> Iterator[tuple[GroupId, dict[str, str]]]:
    """Every group that holds `plan` under SOME key-preserving, one-to-one id mapping, with that mapping. Bottom-up
    over the plan; each node yields (group, mapping so far); the memo's index prunes every combination that is not
    really there."""
    theirs = _by_key(onto)
    index = memo._index()
    a, b = _Shape(other), _Shape(onto)
    budget = [limit]

    def candidates(src: str, mapping: dict[str, str]) -> list[str]:
        if src in mapping:
            return [mapping[src]]
        entry = other.side.get(src)
        taken = set(mapping.values())
        return [] if entry is None else [d for d in theirs.get(_key(entry), ()) if d not in taken and _same_outputs(a, src, b, d)]

    def bind(src: str, d: str, mapping: dict[str, str]):
        """Mappings that send src to d AND place every subquery scope src names into its partner's group."""
        a_subs = other.side[src].data.get("subqueries") or []
        b_subs = onto.side[d].data.get("subqueries") or []
        if len(a_subs) != len(b_subs):
            return

        def rest(k: int, m: dict[str, str]):
            if k == len(a_subs):
                yield m
                return
            sa, sb = a.scopes.get(a_subs[k]), b.scopes.get(b_subs[k])
            target = None if sa is None or sb is None else memo.group_of(sb.plan)
            if target is None:
                return
            for g, m2 in solve(sa.plan, m):
                if g == target:
                    yield from rest(k + 1, m2)

        yield from rest(0, {**mapping, src: d})

    def solve(node: Plan, mapping: dict[str, str]):
        budget[0] -= 1
        if budget[0] < 0:
            raise SearchLimit(f"the alignment search gave up after {limit} states")
        if isinstance(node, Scan):
            for d in candidates(node.occ, mapping):
                g = index.get(("scan", (), d, ()))
                if g is not None:
                    yield g, {**mapping, node.occ: d}
        elif isinstance(node, (Filter, Op1)):
            kind, src = ("filter", node.conjunct) if isinstance(node, Filter) else ("op1", node.op)
            for gc, m in solve(node.input, mapping):
                for d in candidates(src, m):
                    g = index.get((kind, (gc,), d, ()))
                    if g is not None:
                        for m2 in bind(src, d, m):
                            yield g, m2
        elif isinstance(node, Op2):
            for gl, m1 in solve(node.left, mapping):
                for gr, m2 in solve(node.right, m1):
                    for d in candidates(node.op, m2):
                        g = index.get(("op2", (gl, gr), d, ()))
                        if g is not None:
                            for m3 in bind(node.op, d, m2):
                                yield g, m3
        else:
            for gl, m1 in solve(node.left, mapping):
                for gr, m2 in solve(node.right, m1):
                    for m3 in conditions(list(node.conjuncts), m2):
                        g = index.get(("join", tuple(sorted((gl, gr))), None, tuple(sorted(m3[c] for c in node.conjuncts))))
                        if g is not None:
                            yield g, m3

    def conditions(conjuncts: list[str], mapping: dict[str, str]):
        if not conjuncts:
            yield mapping
            return
        for d in candidates(conjuncts[0], mapping):
            for m in bind(conjuncts[0], d, mapping):
                yield from conditions(conjuncts[1:], m)

    yield from solve(plan, {})
