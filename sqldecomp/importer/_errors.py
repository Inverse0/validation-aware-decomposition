"""The one exception the importer raises."""
from __future__ import annotations


class UnsupportedSQL(Exception):
    """Raised, with a reason, for SQL the importer deliberately does not handle. Never crash with anything else.

    `reason` is a short, stable category ("LATERAL", "parse error", ...) that reports can count; `detail` says what
    was found in this query. `str(exc)` gives both."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)
