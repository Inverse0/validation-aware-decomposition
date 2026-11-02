"""The ONE time limit of the pipeline: the DECOMPOSITION TIMEOUT, a wall-clock budget per query over the decomposition
side (import, memo, constraints, legality and the enumeration of blocks, features and scoring, the exact search,
printing and the task writer's model calls). Verification is never under it.

    deadline = Deadline(300.0)          started when it is made; Deadline(None) never expires
    deadline.check("blocks")            raises DecompositionTimeout(stage="blocks") once the budget is gone
    deadline.remaining()                seconds left (inf without a budget), for a step that takes a budget itself

The stages that loop (the memo's saturation, the enumeration of legal blocks, the scoring of the classes, the exact
search) check the clock as they go and stop when it is gone; what they leave is a `timeout` FINDING with the stage
reached, never an answer: the search stays exact or gives none."""
from __future__ import annotations

import math
import time
from typing import Any


class DecompositionTimeout(Exception):
    """The decomposition side of a query stopped at `stage` without an answer: its budget is gone, or (`reason`) the
    memory safety valve of the block enumeration refused to hold more candidates."""

    def __init__(self, stage: str, budget_s: float | None, elapsed_s: float, reason: str = ""):
        self.stage, self.budget_s, self.elapsed_s = stage, budget_s, elapsed_s
        self.reason = reason or "the decomposition timeout"
        super().__init__(f"{self.reason} at stage {stage} after {elapsed_s:.1f} s"
                         + (f" (budget {budget_s:g} s)" if budget_s is not None else ""))

    def finding(self) -> dict[str, Any]:
        return {"kind": "timeout", "stage": self.stage, "detail": str(self), "reason": self.reason,
                "budget_s": self.budget_s, "elapsed_s": round(self.elapsed_s, 3)}


class Deadline:
    """A wall-clock budget, started when the object is made."""

    def __init__(self, seconds: float | None = None):
        if seconds is not None and (math.isnan(seconds) or seconds < 0):
            raise ValueError("a decomposition timeout is a number of seconds >= 0, or None")
        self.seconds = seconds
        self.start = time.perf_counter()
        self.end = None if seconds is None or math.isinf(seconds) else self.start + seconds

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def remaining(self) -> float:
        return math.inf if self.end is None else max(0.0, self.end - time.perf_counter())

    def expired(self) -> bool:
        return self.end is not None and time.perf_counter() >= self.end

    def check(self, stage: str) -> None:
        if self.expired():
            raise DecompositionTimeout(stage, self.seconds, self.elapsed())


NO_DEADLINE = Deadline(None)
DECOMPOSITION_TIMEOUT_S = 300.0  # the default budget per query (seconds) of the deterministic tools (no model call)
# Per-query decomposition budget including task-writing model calls.
TASKS_TIMEOUT_S = 3600.0


def timeout_seconds(text: str) -> float:
    """The `--timeout` of the command lines: a finite number of seconds above 0 (argparse refuses anything else)."""
    import argparse
    import math
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number of seconds: {text!r}") from None
    if not (value > 0) or math.isinf(value):
        raise argparse.ArgumentTypeError(f"the decomposition timeout must be a finite number of seconds above 0, not {text!r}")
    return value
MEMO_BUDGET_S = 5.0  # the memo's saturation budget when no deadline is given (a library call outside a run)


def memo_budget(deadline: Deadline | None) -> float:
    """The saturation budget of the memo: what the run's clock has left, or the library default without a clock."""
    return MEMO_BUDGET_S if deadline is None else deadline.remaining()


def memo_stopped(memo_complete: bool, deadline: Deadline | None) -> None:
    """A memo whose saturation stopped because the clock ran out is a timeout at stage `memo`."""
    if not memo_complete and deadline is not None and deadline.end is not None and deadline.remaining() <= 0.0:
        raise DecompositionTimeout("memo", deadline.seconds, deadline.elapsed())
