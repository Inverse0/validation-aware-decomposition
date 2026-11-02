"""Component 8: PK / FK / UNIQUE propagation to every memo group (keys, weak keys and FKs). Plain Python.

`derive_constraints(memo, result, schema, *, constant_binding=True)` is the entry point. The rules are in `rules.py`
(one function per operator, each with its soundness argument), the lookups they rely on in `context.py`, the
traversal in `engine.py`.
"""
from __future__ import annotations

from sqldecomp.constraints.engine import constraints_to_json, derive_constraints, minimal_keys, minimal_weak

__all__ = ["constraints_to_json", "derive_constraints", "minimal_keys", "minimal_weak"]
