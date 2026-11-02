"""Errors of component 7. Every failure of the printer is one of these; nothing else escapes on purpose."""
from __future__ import annotations


class PrintError(Exception):
    """The plan cannot be printed as SQL. The message says which node and why."""


class UnsupportedBoundary(PrintError):
    """A block boundary sits at a place where the printer cannot express it without changing an operator
    (for example between ORDER BY and LIMIT). The decomposition rules should not produce such a cut."""


class NotABlock(PrintError):
    """The pair (output, inputs) is not a block: no fragment computes the output from exactly these inputs, or the
    only operator of the fragment is a `ref` or a bare `Scan`."""
