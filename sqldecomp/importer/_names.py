"""The name Postgres gives a select item that has no alias (FigureColname in the Postgres parser), best effort.

SQLGlot's `qualify` invents `_col_<n>` for such items. Those invented names must not leak into the side table, so the
importer drops them and reports the name Postgres would use: `SELECT SUM(x)` has the output column `sum`.
"""
from __future__ import annotations

import re

from sqlglot import exp

UNKNOWN = "?column?"
OUTPUT_NAME = "sqldecomp.output_name"  # mark on a select item without alias: the name of its output column

# CAST(<not a column> AS <type>) is named after the type's internal Postgres name.
_TYPE_NAMES = {
    "INT": "int4", "INTEGER": "int4", "BIGINT": "int8", "SMALLINT": "int2", "DECIMAL": "numeric", "NUMERIC": "numeric",
    "REAL": "float4", "FLOAT": "float8", "DOUBLE": "float8", "DOUBLE PRECISION": "float8", "BOOLEAN": "bool",
    "VARCHAR": "varchar", "CHAR": "bpchar", "TEXT": "text", "DATE": "date", "TIMESTAMP": "timestamp",
    "TIMESTAMPTZ": "timestamptz", "TIME": "time", "INTERVAL": "interval",
}
_CALL = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def pg_default_name(e: exp.Expr, dialect: str = "postgres") -> str:
    while isinstance(e, exp.Paren):
        e = e.this
    if isinstance(e, exp.Column):
        return e.name
    if isinstance(e, exp.Cast):
        inner = pg_default_name(e.this, dialect)
        if inner != UNKNOWN:
            return inner
        kind = re.sub(r"\(.*\)", "", e.to.sql(dialect=dialect)).strip().upper()
        return _TYPE_NAMES.get(kind, kind.lower())
    if isinstance(e, exp.Collate):  # FigureColname looks through COLLATE
        return pg_default_name(e.this, dialect)
    if isinstance(e, exp.Dot) and isinstance(e.expression, exp.Func):  # pg_catalog.upper(x): the last name part
        return pg_default_name(e.expression, dialect)
    if isinstance(e, exp.Case):
        default = e.args.get("default")
        inner = pg_default_name(default, dialect) if default is not None else UNKNOWN
        return inner if inner != UNKNOWN else "case"
    if isinstance(e, exp.Exists):
        return "exists"
    if isinstance(e, exp.Subquery):
        query = e.this
        if isinstance(query, exp.Query) and query.selects:  # a scalar subquery is named after its one column
            first = query.selects[0]
            return first.alias or first.meta.get(OUTPUT_NAME) or pg_default_name(first, dialect)
        return UNKNOWN
    if isinstance(e, exp.Alias):
        return e.alias
    if isinstance(e, (exp.Window, exp.Filter, exp.WithinGroup, exp.IgnoreNulls, exp.RespectNulls)):
        return pg_default_name(e.this, dialect)
    if isinstance(e, exp.Array):
        return "array"
    if isinstance(e, exp.Func):
        # The name as it is printed for Postgres, e.g. TO_CHAR(...) -> to_char, COUNT(*) -> count.
        m = _CALL.match(e.sql(dialect=dialect, comments=False))
        if m:
            return m.group(1).lower()
    return UNKNOWN
