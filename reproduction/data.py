"""Output-directory creation."""
from __future__ import annotations


from pathlib import Path


def new_dir(path: str | Path) -> Path:
    out = Path(path)
    if out.exists():
        raise SystemExit(f"refusing to write into an existing directory: {out}")
    out.mkdir(parents=True)
    return out
