"""Component 7: skeleton plan -> human-readable SQL, plus checks C1-C3.

    print_plan(plan, result)                 SQL of any plan of the query (whole query, scope, region fragment)
    print_group(memo, group, result)         SQL of one memo occurrence, computed from base tables
    print_block(memo, block, result)         SQL of one block; its inputs are derived tables named b_<group id>
    print_decomposition(memo, d, result)     the query with every block of a decomposition inlined into its consumers
    check_round_trip / check_recomposition / check_valid      checks C1 / C2 / C3

`Printer(result, memo)` offers the same as methods and indexes the import result only once.
"""
from __future__ import annotations

from sqldecomp.model import Block, Decomposition, GroupId, ImportResult, Memo, Plan
from sqldecomp.printer.checks import CheckResult, check_recomposition, check_round_trip, check_valid
from sqldecomp.printer.errors import NotABlock, PrintError, UnsupportedBoundary
from sqldecomp.printer.printer import Printer

__all__ = ["CheckResult", "NotABlock", "PrintError", "Printer", "UnsupportedBoundary", "check_recomposition",
           "check_round_trip", "check_valid", "print_block", "print_decomposition", "print_group", "print_plan"]


def print_plan(plan: Plan, result: ImportResult, *, pretty: bool = True, bind_outer: bool = False) -> str:
    """Build the SQL with SQLGlot from the skeleton and the side table: original expressions, aliases and CTE names
    are reused unchanged. Never unparse an optimizer plan. `bind_outer`: see Printer.plan_sql."""
    return Printer(result).plan_sql(plan, pretty=pretty, bind_outer=bind_outer)


def print_group(memo: Memo, group: GroupId, result: ImportResult, *, pretty: bool = True,
                bind_outer: bool = False) -> str:
    """SQL of one memo group computed from base tables; picks the most readable of the group's fragments."""
    return Printer(result, memo).group_sql(group, pretty=pretty, bind_outer=bind_outer)


def print_block(memo: Memo, block: Block, result: ImportResult, *, inputs: str = "inline", pretty: bool = True,
                bind_outer: bool = False, within: Decomposition | None = None,
                expose: list[tuple[str, str]] | None = None) -> str:
    """SQL of the block (output, inputs). Each input appears as a derived table named `b_<group id>` that exposes
    every column the rest of the query reads from it, named `<input alias>__<column>`; with `inputs="name"` only the
    name appears. Raises NotABlock when the pair is not a block. `within`, `expose`: see Printer.block_sql."""
    return Printer(result, memo).block_sql(block, inputs=inputs, pretty=pretty, bind_outer=bind_outer, within=within,
                                          expose=expose)


def print_decomposition(memo: Memo, decomposition: Decomposition, result: ImportResult, *, pretty: bool = True) -> str:
    """The recomposed query Q' of check C2: every block inlined into its consumers."""
    return Printer(result, memo).decomposition_sql(decomposition, pretty=pretty)
