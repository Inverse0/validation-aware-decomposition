"""Import SQL into the shared ImportResult data model using SQLGlot.
import_sql(sql, schema) returns the result or raises UnsupportedSQL.
render(result) produces a readable representation.

_prepare parses SQL, screens unsupported constructs and qualifies names.
_build constructs skeleton plans, side-table entries and facts. The output
data types and invariants are defined in sqldecomp.model.
"""
from __future__ import annotations

from sqldecomp.importer._build import DUAL_TABLE, VALUES_TABLE, Builder
from sqldecomp.importer._errors import UnsupportedSQL
from sqldecomp.importer._prepare import prepare
from sqldecomp.importer.render import render
from sqldecomp.model import ImportResult, Schema, validate_import_result

__all__ = ["DUAL_TABLE", "VALUES_TABLE", "UnsupportedSQL", "import_sql", "render"]


def import_sql(sql: str, schema: Schema, *, dialect: str = "postgres") -> ImportResult:
    """Import one SELECT statement. The result always satisfies `validate_import_result`, and importing the same
    text twice gives the same ids. Anything that cannot be imported raises UnsupportedSQL with a reason."""
    try:
        tree = prepare(sql, schema, dialect)
        result = Builder(sql, tree, schema, dialect).run()
        violations = validate_import_result(result)
    except UnsupportedSQL:
        raise
    except RecursionError as e:
        raise UnsupportedSQL("query too deeply nested", "recursion limit reached") from e
    except Exception as e:  # a bug in the importer or in SQLGlot; the caller still gets the one documented exception
        raise UnsupportedSQL("internal error", f"{type(e).__name__}: {e}") from e
    if violations:
        raise UnsupportedSQL("internal error", "the import result violates the contract: " + "; ".join(violations[:3]))
    return result
