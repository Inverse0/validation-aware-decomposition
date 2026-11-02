"""Memo -> plans: a representative plan of a group, and the fragment of a block (output group, input groups).

All expressions of a group are equivalent, so which one is printed only affects readability. The choice prefers the
shape the query was written in: filters above the joins, a left-deep join tree, inputs in written order. A block
fragment additionally has to reach every input group as a leaf."""
from __future__ import annotations

from sqldecomp.model import Block, Filter, GroupId, Join, MExpr, Memo, Op1, Op2, Plan, Scan
from sqldecomp.printer.build import Fragment
from sqldecomp.printer.context import BlockInput, PPlan, PrintContext, walk
from sqldecomp.printer.errors import NotABlock, PrintError

Cost = tuple[int, int, int]  # (filters below a join, joins whose later side is a join itself, inputs out of written order)
_ZERO: Cost = (0, 0, 0)


def _add(*costs: Cost) -> Cost:
    return tuple(sum(c[i] for c in costs) for i in range(3))  # type: ignore[return-value]


class FragmentChooser:
    """Chooses plans from one memo. `inputs` are the groups that must appear as leaves instead of being expanded."""

    def __init__(self, memo: Memo, ctx: PrintContext, inputs: frozenset[GroupId] = frozenset(),
                 base: "FragmentChooser | None" = None):
        self.memo = memo
        self.ctx = ctx
        self.inputs = inputs
        self.base = base if inputs else self  # expands input groups from base tables, to learn their aliases
        self._cache: dict[GroupId, dict[frozenset[GroupId], tuple[tuple[Cost, int], PPlan]]] = {}
        self._visiting: set[GroupId] = set()
        self._cut_short = 0

    # ---------------------------------------------------------------------------------------------------------

    def plan(self, group: GroupId) -> Plan:
        """The most readable plan of a group, from base tables."""
        if self.inputs:
            return self.base.plan(group)
        options = self._solve(group)
        if frozenset() not in options:
            raise PrintError(f"memo group {group!r} has no finite plan")
        return options[frozenset()][1]  # type: ignore[return-value]

    def reaching(self, output: GroupId) -> Fragment:
        """The fragment that computes `output` and stops at every group of `self.inputs` it meets. Unlike `fragment`
        it does not have to meet all of them, and it may consist of a ref alone. This is how an input of a block is
        expanded when the block is printed by itself: the block's other inputs stay leaves INSIDE it, so that a CTE
        body which both the block and its input read is cut the same way in both."""
        options = self._solve(output)
        if not options:
            raise NotABlock(f"memo group {output!r} has no finite plan")
        _, plan = self._widest(options)
        subqueries: dict[str, PPlan] = {}
        self._subquery_fragments(plan, subqueries)
        return Fragment(plan, subqueries)

    def fragment(self, output: GroupId) -> Fragment:
        """The fragment of the block (output, self.inputs)."""
        if output in self.inputs:
            raise NotABlock(f"group {output!r} is its own input")
        options = self._solve(output)
        if not options:
            raise NotABlock(f"memo group {output!r} has no finite plan")
        covered, plan = self._widest(options)
        subqueries: dict[str, PPlan] = {}
        covered |= self._subquery_fragments(plan, subqueries)
        if covered != self.inputs:
            missing = ", ".join(sorted(self.inputs - covered))
            raise NotABlock(f"no fragment computes {output!r} from the inputs given: {missing} cannot be a leaf of it")
        fragment = Fragment(plan, subqueries)
        if not _has_operator(fragment, self.ctx):
            raise NotABlock(f"the only operator of block {output!r} is a ref or a bare scan")
        return fragment

    # ---------------------------------------------------------------------------------------------------------

    @staticmethod
    def _widest(options) -> tuple[frozenset[GroupId], PPlan]:
        covered = max(options, key=lambda c: (len(c), sorted(c)))
        return covered, options[covered][1]

    def _subquery_fragments(self, plan: PPlan, out: dict[str, PPlan]) -> frozenset[GroupId]:
        """Fragments for the predicate and select-list subqueries of a plan, including those of its nested scopes;
        returns the inputs found inside them. Only a subquery whose body holds an input needs a fragment."""
        ctx = self.ctx
        ids: list[str] = []
        for node in walk(plan):
            if isinstance(node, Join):
                ids.extend(node.conjuncts)
            elif isinstance(node, Filter):
                ids.append(node.conjunct)
            elif isinstance(node, (Op1, Op2)):
                ids.append(node.op)
        covered: frozenset[GroupId] = frozenset()
        for id_ in sorted(set(ids), key=lambda i: (ctx.side_pos.get(i, 0), i)):
            for sid in ctx.entry(id_).data.get("subqueries") or ():
                scope = ctx.scopes.get(sid)
                if scope is None or sid in out:
                    continue
                group = self.memo.group_of(scope.plan)
                if group is None:
                    continue
                options = self._solve(group)
                if not options:
                    continue
                inside, sub = self._widest(options)
                if inside:
                    out[sid] = sub
                covered |= inside | self._subquery_fragments(sub, out)
        return covered

    def _solve(self, group: GroupId):
        """inputs reached -> (cost, plan), over every way to expand `group` that stops exactly at input groups."""
        if group in self._cache:
            return self._cache[group]
        if group in self.inputs:
            found = {frozenset({group}): ((_ZERO, 0), BlockInput(group, self.base.plan(group)))}
            self._cache[group] = found
            return found
        if group in self._visiting:  # a cyclic memo: this way of expanding the group never ends
            self._cut_short += 1
            return {}
        if group not in self.memo.groups:
            raise PrintError(f"memo has no group {group!r}")
        self._visiting.add(group)
        cut_before = self._cut_short
        found: dict = {}
        for index, e in enumerate(self.memo.groups[group]):
            for covered, (cost, plan) in self._expand(e).items():
                if covered not in found or (cost, index) < found[covered][0]:
                    found[covered] = ((cost, index), plan)
        self._visiting.discard(group)
        if self._cut_short == cut_before:
            self._cache[group] = found
        return found

    def _expand(self, e: MExpr) -> dict[frozenset[GroupId], tuple[Cost, PPlan]]:
        """One expression, over the WIDEST option of each child. Every input that lies below a group has to be a leaf
        of the fragment, so a narrower option of a child can never be part of a legal fragment; and taking the same
        option wherever a group occurs keeps the two uses of a shared CTE body cut the same way."""
        if e.kind == "scan":
            entry = self.ctx.entry(e.arg, "scan")
            # the importer's own node: a pseudo scan is Scan("$dual") / Scan("$values") while its side entry has no
            # table, and a plan rebuilt with another table name would not EQUAL the scope plan it is part of
            return {frozenset(): (_ZERO, self.ctx.scans.get(e.arg) or Scan(entry.data.get("table") or "", e.arg))}
        kids = []
        for child in e.children:
            options = self._solve(child)
            if not options:
                return {}
            covered = max(options, key=lambda c: (len(c), sorted(c)))
            (cost, _), plan = options[covered]
            kids.append((covered, cost, plan))
        covered = frozenset().union(*(k[0] for k in kids))
        if e.kind == "filter":
            return {covered: (kids[0][1], Filter(kids[0][2], e.arg))}
        if e.kind == "op1":
            return {covered: (kids[0][1], Op1(kids[0][2], e.arg))}
        if len(kids) != 2:
            raise PrintError(f"memo expression of kind {e.kind!r} needs two children, got {len(kids)}")
        (_, cost1, p1), (_, cost2, p2) = kids
        if e.kind == "op2":
            return {covered: (_add(cost1, cost2), Op2(e.arg, p1, p2))}
        return {covered: self._join(p1, cost1, p2, cost2, e)}

    def _join(self, p1: PPlan, cost1: Cost, p2: PPlan, cost2: Cost, e: MExpr) -> tuple[Cost, PPlan]:
        ctx = self.ctx
        left, right = sorted((p1, p2), key=ctx.written_pos)  # stable: equal positions keep the memo's order
        pushed = sum(isinstance(p, Filter) for p in (left, right))
        bushy = int(isinstance(_below_filters(right), Join))
        last_left = max((ctx.written.get(i, 0) for i in ctx.region_input_ids(left)), default=0)
        disorder = int(last_left > ctx.written_pos(right))
        return _add(cost1, cost2, (pushed, bushy, disorder)), Join(left, right, tuple(sorted(e.conjuncts)))


def _below_filters(p: PPlan) -> PPlan:
    while isinstance(p, Filter):
        p = p.input
    return p


def _has_operator(fragment: Fragment, ctx: PrintContext) -> bool:
    """A block needs an operator that is neither a scan nor a ref."""
    for plan in (fragment.plan, *fragment.subqueries.values()):
        for node in walk(plan):
            if isinstance(node, (Join, Filter, Op2)):
                return True
            if isinstance(node, Op1) and ctx.kind(node) != "ref":
                return True
    return False


def block_fragment(memo: Memo, block: Block, ctx: PrintContext, base: FragmentChooser | None = None) -> Fragment:
    base = base or FragmentChooser(memo, ctx)
    if not block.inputs:
        plan = base.plan(block.output)
        fragment = Fragment(plan)
        # the whole query as one block is always a legal decomposition, even `SELECT * FROM t`
        if not _has_operator(fragment, ctx) and block.output != memo.root:
            raise NotABlock(f"the only operator of block {block.output!r} is a ref or a bare scan")
        return fragment
    return FragmentChooser(memo, ctx, frozenset(block.inputs), base).fragment(block.output)
