"""Components 3 and 4 as the optimizer sees them: black-box callables and the context objects they read.

The optimizer (component 6, `sqldecomp.optimizer`) never looks inside a scoring function. It hands every function a
context object and reads back a positive finite number:

    F(theta, block_context) -> float > 0     the score of one UNIQUE block class, from its features
    Omega(k) -> float                        the penalty for k unique blocks, an arbitrary function of k alone,
                                             defined for every integer k >= 1 (the optimizer evaluates it on a range
                                             that starts at a LOWER BOUND of the unique count, which need not be reached)
    meaningfulness(group_context) -> float   the semantic score of an OUTPUT GROUP (per signature, D1), which F may read

Objective: D* = argmax_D prod_{c in B(D)} F(theta, C(c)) * exp(Omega(|B(D)|)), where B(D) is the set of unique block
classes of the decomposition; minimised as cost(D) = sum_{c in B(D)} -log F(theta, C(c)) - Omega(|B(D)|).

What a context carries (D6): the boundary information of the block. Its output's exposed columns (what its readers
read; the printer names them `<alias>__<column>` inside a FROM clause and keeps the scope's own output names at the
top of a scope), their declared types where a column traces back to a base table, the keys and foreign keys that hold
on the output and whose columns are all exposed (component 8), the number of inputs and each input's exposed columns,
the parameters of a parameterized block, the subquery links, the operators of the fragment, and the printed SQL ON
DEMAND (a lazy property: nothing is printed unless a function asks). For a class with several occurrences the exposed
columns are the UNION over the occurrences, written in the names of the representative occurrence: a block that is
verified once and reused at every occurrence must expose every column any of them reads.

The scoring of the pipeline implements this interface: the human-facing model of component 3
(`sqldecomp.complexity.human_scoring`), with the semantic score of component 4 (the LLM judge of
`sqldecomp.semantic_llm`) behind `meaningfulness` when it is switched on; `sqldecomp.scoring_args` builds it from a
JSON-able spec. Any other callables with these signatures plug in the same
way (the optimizer's tests use synthetic ones, among them an F > 1 and non-linear Omegas).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from sqldecomp.blocks import Parameter
from sqldecomp.model import Block, GroupId


# --------------------------------------------------------------------------------------------------------------
# context objects
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposedColumn:
    """One column a block exposes at its output (or reads from an input)."""

    name: str  # what a reader sees: `<alias>__<column>` at a boundary inside a FROM clause, the output name at a scope top
    alias: str  # the FROM alias inside the block that the column comes from; "" for a computed scope output
    column: str  # the column name inside the block, or the output expression of a scope output
    type: str = "unknown"  # the declared type when the column traces back to a base-table column
    occurrence: str = ""  # occurrence id (scan or ref) that exposes it; "" when an operator produces it
    placeholder: str = ""  # the alias-free name (t0, t1, ...) of that occurrence in the group's signature, if any
    canonical: str = ""  # the column in alias-free form (`t0.c_id`, `SUM(t1.o_total)`): equal across occurrences
    nullable: bool = False  # from the NULL-extended side of an outer join INSIDE the group that exposes it (CPX-O2): a reader's use is NULL-sensitive


@dataclass
class GroupContext:
    """An output GROUP in the design document's sense: one per signature (D1). `meaningfulness` reads this."""

    signature: str
    group: GroupId  # the representative occurrence (the first in memo order)
    occurrences: tuple[GroupId, ...]
    kind: str  # scan, join, filter, aggregate, setop, ... (`Blocks.kind`)
    scope_top: bool  # the group is the whole of a scope (a CTE body, a subquery body, a set-operation branch, the root)
    columns: tuple[ExposedColumn, ...]  # the union over the occurrences, in the representative's names
    star: bool  # the boundary exposes every column (`SELECT *`): nothing of it is read by name
    keys: tuple[tuple[str, ...], ...]  # keys whose columns are all exposed, as exposed names (the empty key: one row)
    fks: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...]  # (exposed names, referenced table, its columns)
    constraint_count: tuple[int, int]  # keys and FKs the group has in all, exposed or not
    _sql: Callable[[], str] | None = field(default=None, repr=False)

    @property
    def sql(self) -> str:
        """The group printed from base tables (`print_group`), on demand."""
        if self._sql is None:
            raise RuntimeError("this context was built without a printer")
        return self._sql()


@dataclass
class InputContext:
    group: GroupId
    signature: str
    columns: tuple[ExposedColumn, ...]  # what the input exposes to its readers
    star: bool
    parameterized: bool  # the input is a function of columns of an enclosing query
    keys: tuple[tuple[str, ...], ...]
    fks: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...]


@dataclass
class BlockContext:
    """One block CLASS (D3): what F is evaluated on, once per class."""

    class_id: str  # short, stable digest of the class key
    key: tuple  # the class key itself (see sqldecomp.optimizer.class_key)
    block: Block  # the representative occurrence (the first in enumeration order)
    occurrences: tuple[Block, ...]
    output: GroupContext
    exposed: tuple[ExposedColumn, ...]  # the union over THIS class's occurrences, in the representative's names
    star: bool
    inputs: tuple[InputContext, ...]
    parameters: tuple[Parameter, ...]  # of the representative (the same up to renaming for every occurrence)
    subqueries: dict[str, Any]  # inside_body, whole_bodies, partial_bodies, input_bodies (`Blocks.describe`)
    operators: dict[str, int]  # operator kinds in the fragment, scans and refs apart
    depth: int  # longest path of the fragment, in operators
    meaningfulness: float  # the semantic score of the output group (D1)
    _fragment: Callable[[], dict] | None = field(default=None, repr=False)
    _plan: Callable[[], Any] | None = field(default=None, repr=False)
    _sql: Callable[[], str] | None = field(default=None, repr=False)
    side: Any = field(default=None, repr=False)  # the query's side table (id -> SideEntry): the texts the fragment applies, so that F is a function of the context alone

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def n_operators(self) -> int:
        return sum(self.operators.values())

    @property
    def fragment(self) -> dict:
        """The block's logical plan as the memo holds it: group -> expression (`MExpr`: kind, children, the conjunct or
        operator id) for every group the block expands, its inputs as leaves. The SQL text of every conjunct and
        operator is in the side table (`result.side[id].sql`); reading the plan this way costs microseconds per
        class, the plan being built for every class anyway."""
        if self._fragment is None:
            raise RuntimeError("this context was built without the legality rules")
        return self._fragment()

    @property
    def plan(self):
        """The block's logical plan as a TREE in the printer's readable shape (`printer.Fragment`: a plan whose leaves
        are `BlockInput`s, plus the plans of the subqueries that hold an input), on demand: about 0.4 ms per class."""
        if self._plan is None:
            raise RuntimeError("this context was built without a printer")
        return self._plan()

    @property
    def sql(self) -> str:
        """The block in the reader form, `print_block(inputs="name")`, on demand: about 1-2 ms per print, so a scoring
        function that reads it pays that once per class."""
        if self._sql is None:
            raise RuntimeError("this context was built without a printer")
        return self._sql()


# --------------------------------------------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------------------------------------------


@dataclass
class Scoring:
    """The three black boxes and their parameters. `name` only labels reports."""

    F: Callable[[Any, BlockContext], float]
    Omega: Callable[[int], float]
    meaningfulness: Callable[[GroupContext], float]
    theta: Any = None
    name: str = ""

    def block_cost(self, ctx: BlockContext) -> float:
        """-log F(theta, C(block)). F must return a positive finite number."""
        f = float(self.F(self.theta, ctx))
        if not (f > 0.0) or math.isinf(f) or math.isnan(f):
            raise ValueError(f"F must return a positive finite number, got {f!r} for class {ctx.class_id}")
        return -math.log(f)

    def omega(self, k: int) -> float:
        value = float(self.Omega(k))
        if math.isinf(value) or math.isnan(value):
            raise ValueError(f"Omega({k}) must be finite, got {value!r}")
        return value
