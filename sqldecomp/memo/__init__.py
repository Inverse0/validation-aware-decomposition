"""Components 2 and 5: the memo (an e-graph built with egglog) and plan equivalence.

    build_memo   sqldecomp.memo.build      ImportResult -> Memo (rules R1-R4, every join with its condition; an
                                           unsound e-graph is refused with InvalidMemo, never returned)
    signature    sqldecomp.memo.signature  the alias-free name of an occurrence's group; same_task compares them
                                           inside one memo, same_task_across between two imports (conservative)
    align        sqldecomp.memo.align      ids of one import mapped onto another, ALIAS-BOUND (scans by (table,
                                           alias), the rest by canonical text): for a re-import of a print of the
                                           same query. equivalent decides p =R q under that alignment; groups_in
                                           finds a plan of another import in this memo; alignments hands out the
                                           mappings too (what checks C1 / C2 are built on)
"""
from __future__ import annotations

from sqldecomp.model import ImportResult, Memo

from .align import (SearchLimit, align, alignments, equivalent, explain_alignment, groups_in, same_root_outputs,
                    translate_plan)
from .build import InvalidImportResult, InvalidMemo, MemoStats, build_memo_with_stats, check_import_result
from .build import build_memo as _build_memo
from .signature import placeholders, same_task, same_task_across, signature

__all__ = ["build_memo", "build_memo_with_stats", "check_import_result", "InvalidImportResult", "InvalidMemo", "MemoStats",
           "signature", "same_task", "same_task_across", "placeholders", "align", "alignments", "SearchLimit", "equivalent",
           "translate_plan", "groups_in", "same_root_outputs", "explain_alignment"]


def build_memo(result: ImportResult, *, budget_s: float = 5.0) -> Memo:
    """Saturate rules R1-R4 over result.plan (and every scope plan) and read the e-graph out into a plain Memo.
    Every join expression of the read-out carries its condition (MExpr.conjuncts): the join conjuncts that span its
    two sides. When the time budget is hit the memo is returned with complete=False (sound, not complete).
    Raises InvalidImportResult when `result` breaks the contract."""
    return _build_memo(result, budget_s=budget_s)
