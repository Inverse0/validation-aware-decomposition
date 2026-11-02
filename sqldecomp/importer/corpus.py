"""Run the importer over a directory of `{"sql": ...}` JSON files and summarize the outcome. Read-only.

    python -m sqldecomp.importer.corpus <corpus dir> <beaver schema json>          # a summary as text
    python -m sqldecomp.importer.corpus <corpus dir> <beaver schema json> --json   # the same as JSON
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sqldecomp.importer import UnsupportedSQL, import_sql
from sqldecomp.importer.render import regions
from sqldecomp.model import Join, Schema, validate_import_result, walk
from sqldecomp.schema import load_beaver_schema


def load_queries(corpus_dir: str | Path) -> list[tuple[str, str]]:
    """(file name, sql) for every JSON file in the directory that carries a non-empty `sql` field, sorted by name."""
    out = []
    for path in sorted(Path(corpus_dir).glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("sql"), str) and d["sql"].strip():
            out.append((path.name, d["sql"]))
    return out


@dataclass
class Summary:
    total: int = 0
    imported: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)  # (file, reason, detail)
    invalid: list[tuple[str, list[str]]] = field(default_factory=list)
    other_exceptions: list[tuple[str, str]] = field(default_factory=list)
    millis: list[float] = field(default_factory=list)
    largest_region: list[int] = field(default_factory=list)  # per imported query
    scopes: list[int] = field(default_factory=list)
    conjuncts: list[int] = field(default_factory=list)
    kinds: Counter = field(default_factory=Counter)  # side-entry kinds, counted once per query
    cross_joins: int = 0  # queries with a Join that carries no condition
    shared_ctes: int = 0  # queries in which one nested plan is used by two or more refs

    @property
    def rate(self) -> float:
        return self.imported / self.total if self.total else 0.0

    def reasons(self) -> Counter:
        return Counter(reason for _, reason, _ in self.failures)


def run(queries: list[tuple[str, str]], schema: Schema) -> Summary:
    s = Summary(total=len(queries))
    for name, sql in queries:
        start = time.perf_counter()
        try:
            result = import_sql(sql, schema)
        except UnsupportedSQL as e:
            s.failures.append((name, e.reason, e.detail))
            continue
        except Exception as e:  # must never happen: import_sql promises UnsupportedSQL only
            s.other_exceptions.append((name, f"{type(e).__name__}: {e}"))
            continue
        finally:
            s.millis.append((time.perf_counter() - start) * 1000)
        violations = validate_import_result(result)
        if violations:
            s.invalid.append((name, violations))
            continue
        s.imported += 1
        s.largest_region.append(max(len(r) for r in regions(result)))
        s.scopes.append(len(result.scopes))
        s.conjuncts.append(sum(1 for e in result.side.values() if e.kind == "conjunct"))
        s.kinds.update({e.kind for e in result.side.values()})
        nodes = [n for sc in result.scopes for n in walk(sc.plan)]
        s.cross_joins += any(isinstance(n, Join) and not n.conjuncts for n in nodes)
        uses = Counter(e.data["scope"] for e in result.side.values() if e.kind == "ref")
        s.shared_ctes += any(v > 1 for v in uses.values())
    return s


def _dist(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    return {"min": v[0], "median": statistics.median(v), "p90": v[int(0.9 * (len(v) - 1))], "max": v[-1]}


def to_json(s: Summary) -> dict:
    return {
        "total": s.total, "imported_and_valid": s.imported, "rate": round(s.rate, 4),
        "unsupported": dict(s.reasons()), "invalid": len(s.invalid), "other_exceptions": len(s.other_exceptions),
        "failures": [{"file": f, "reason": r, "detail": d} for f, r, d in s.failures],
        "millis_per_query": {k: round(x, 2) for k, x in _dist(s.millis).items()},
        "largest_region_inputs": _dist(s.largest_region), "largest_region_histogram": dict(sorted(Counter(s.largest_region).items())),
        "scopes_per_query": _dist(s.scopes), "conjuncts_per_query": _dist(s.conjuncts),
        "queries_with_kind": dict(sorted(s.kinds.items())), "queries_with_cross_join": s.cross_joins,
        "queries_with_shared_cte": s.shared_ctes,
    }


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    summary = run(load_queries(args[0]), load_beaver_schema(args[1]))
    report = to_json(summary)
    if "--json" in argv:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            if key != "failures":
                print(f"{key}: {value}")
        for f in report["failures"]:
            print(f"  {f['file']}: {f['reason']}: {f['detail']}")
        for name, violations in summary.invalid:
            print(f"  INVALID {name}: {violations}")
        for name, error in summary.other_exceptions:
            print(f"  EXCEPTION {name}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
