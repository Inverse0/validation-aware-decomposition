"""The printer proper: plans, memo groups, blocks and whole decompositions to SQL text."""
from __future__ import annotations

from sqlglot import exp

from sqldecomp.model import Block, Decomposition, GroupId, ImportResult, Memo, Plan
from sqldecomp.printer import sqltext
from sqldecomp.printer.build import Builder, Fragment
from sqldecomp.printer.context import PrintContext
from sqldecomp.printer.errors import NotABlock, PrintError
from sqldecomp.printer.fragment import FragmentChooser, block_fragment


class Printer:
    """Prints everything of ONE import result (and, for groups and blocks, its memo). Building it once and reusing it
    saves the indexing of the import result; the module-level functions of the package build one per call."""

    def __init__(self, result: ImportResult, memo: Memo | None = None):
        self.result = result
        self.memo = memo
        self.ctx = PrintContext(result)
        self._chooser = FragmentChooser(memo, self.ctx) if memo is not None else None

    # ---------------------------------------------------------------------------------------------------------

    def plan_sql(self, plan: Plan, *, pretty: bool = True, bind_outer: bool = False) -> str:
        """`bind_outer`: a plan from inside a correlated subquery refers to columns of the query around it and is not
        a statement of its own. With bind_outer it is printed in closed form, as the EXISTS subquery of
        `SELECT 1 FROM <the FROM items it refers to> WHERE EXISTS (...)`, which Postgres and the importer accept."""
        return _text(Builder(self.ctx).statement(plan, bind_outer=bind_outer), pretty)

    def group_plan(self, group: GroupId) -> Plan:
        """The plan that `group_sql` prints: the most readable expression of the group, expanded to base tables."""
        return self._need_memo().plan(group)

    def group_sql(self, group: GroupId, *, pretty: bool = True, bind_outer: bool = False) -> str:
        return self.plan_sql(self.group_plan(group), pretty=pretty, bind_outer=bind_outer)

    def block_fragment(self, block: Block) -> Fragment:
        self._need_memo()
        return block_fragment(self.memo, block, self.ctx, self._chooser)

    def block_sql(self, block: Block, *, inputs: str = "inline", pretty: bool = True, bind_outer: bool = False,
                  within: Decomposition | None = None, expose: list[tuple[str, str]] | None = None) -> str:
        """SQL of one block. `inputs="inline"`: every input is a derived table (or the body of the CTE / subquery it
        stands for) computed from base tables, so the statement runs as it is. `inputs="name"`: every input is just
        the relation `b_<group id>`, which is how a reader sees the block when its inputs are given.

        The SQL depends on the block alone. That matters for a block inside a correlated subquery: its reference to
        the outer query is written with the alias of the query (`o.o_cust = c.c_id`), whatever the other blocks are.
        `within=<decomposition>` prints the block as it appears in that decomposition instead: if another block hides
        `c`, the reference becomes `b_<id>.c__c_id`, exactly as in the recomposed query.

        `expose`: further (alias, column) pairs of the block's FROM items, its inputs' included, that its output
        exposes beside what this position reads: the union over every occurrence of the block's class, so that ONE
        text is verified for the class. A pair inside an
        input is read through that input's boundary, which exposes it too. Only for a block whose output lies inside
        a region (the printer chooses that select list); a scope's select list is its own.
        """
        if inputs not in ("inline", "name"):
            raise ValueError("inputs must be 'inline' or 'name'")
        if within is not None and bind_outer:
            raise ValueError("a block is bound either by its decomposition (within) or by the closed form (bind_outer)")
        fragment = self.block_fragment(block)
        env = self._recompose(within)[1].input_envs.get(block.output, {}) if within is not None else {}
        resolve = self._input_resolver(block, within) if inputs == "inline" else None
        builder = Builder(self.ctx, resolve=resolve, subqueries=fragment.subqueries)
        tree = builder.statement(fragment.plan, boundary=block.output != self.memo.root, bind_outer=bind_outer, env=env,
                                 expose=list(expose) if expose else None)
        self._check_inputs_used(block, builder)
        return _text(tree, pretty)

    def block_parameters(self, block: Block, *, within: Decomposition | None = None) -> list[str]:
        """The parameters of a block from inside a correlated subquery, as its text names them (see `block_sql`), in
        the order in which a reader passes them: `b_g6(c.c_id, c.c_name)` calls the block `g6(c.c_id, c.c_name)`.
        Empty for a block that is a statement by itself."""
        env = self._recompose(within)[1].input_envs.get(block.output, {}) if within is not None else {}
        builder = Builder(self.ctx)
        return [sqltext.to_sql(sqltext.unquote_safe_identifiers(builder.through(c, env)))
                for c in builder.parameters(self.group_plan(block.output))]

    def _input_resolver(self, block: Block, within: Decomposition | None):
        """How the inputs of a block are expanded when it is printed with `inputs="inline"`. Inside a decomposition an
        input is the block that computes it. By itself an input is computed from base tables, except that the block's
        other inputs stay leaves inside it: a CTE body that the block cuts and that one of its inputs reads as well
        must have ONE definition in the statement."""
        if within is not None:
            by_output = {b.output: b for b in within.blocks}
            return lambda g: self.block_fragment(by_output[g]) if g in by_output else Fragment(self.group_plan(g))
        leaves = frozenset(block.inputs)

        def resolve(group: GroupId) -> Fragment:
            rest = leaves - {group}
            if not rest:
                return Fragment(self.group_plan(group))
            return FragmentChooser(self.memo, self.ctx, rest, self._chooser).reaching(group)

        return resolve

    def decomposition_sql(self, decomposition: Decomposition, *, pretty: bool = True) -> str:
        return _text(self._recompose(decomposition)[0], pretty)

    def recompose(self, decomposition: Decomposition) -> tuple[exp.Expression, set[str]]:
        """The query that results from inlining every block into its consumers, and the (folded) names `b_<group id>`
        of the derived tables that are block boundaries in it."""
        tree, builder = self._recompose(decomposition)
        return tree, builder.boundary_names

    def _recompose(self, decomposition: Decomposition) -> tuple[exp.Expression, Builder]:
        self._need_memo()
        by_output: dict[GroupId, Block] = {}
        for b in decomposition.blocks:
            if b.output in by_output:
                raise NotABlock(f"two blocks of the decomposition have the output {b.output!r}")
            by_output[b.output] = b
        if self.memo.root not in by_output:
            raise NotABlock(f"no block of the decomposition has the root group {self.memo.root!r} as its output")
        for b in decomposition.blocks:
            for g in sorted(b.inputs):
                if g not in by_output:
                    raise NotABlock(f"input {g!r} of block {b.output!r} is not the output of a block")
        fragments: dict[GroupId, Fragment] = {}

        def resolve(group: GroupId) -> Fragment:
            if group not in fragments:
                fragments[group] = self.block_fragment(by_output[group])
            return fragments[group]

        root = resolve(self.memo.root)
        builder = Builder(self.ctx, resolve=resolve, subqueries=root.subqueries)
        tree = builder.statement(root.plan)
        unused = sorted(set(by_output) - set(builder.used_inputs) - {self.memo.root})
        if unused:
            raise NotABlock(f"blocks {', '.join(unused)} are not reachable from the root block")
        return tree, builder

    # ---------------------------------------------------------------------------------------------------------

    def _need_memo(self) -> FragmentChooser:
        if self._chooser is None:
            raise PrintError("this Printer was built without a memo")
        return self._chooser

    @staticmethod
    def _check_inputs_used(block: Block, builder: Builder) -> None:
        missing = sorted(set(block.inputs) - set(builder.used_inputs))
        if missing:
            raise PrintError(f"inputs {', '.join(missing)} of block {block.output!r} did not make it into the SQL "
                             "(a subquery of the side table could not be matched to its scope)")


def _text(tree: exp.Expression, pretty: bool) -> str:
    # normalize_functions=False: a parameterized block input is written like a function call, `b_g12(c.c_id)`, and must
    # keep its lower-case name. Every other function name comes from canonical side-table SQL and is upper case already.
    return tree.sql(dialect=sqltext.DIALECT, pretty=pretty, normalize_functions=False)
