"""Shared identifier spelling for imported side-table texts and printed SQL.
PostgreSQL treats a quoted lowercase plain identifier like its unquoted form.
Reserved words remain quoted, including column qualifiers such as
"order".account_id. canonical_quotes normalizes the imported tree, and
sqldecomp.printer.sqltext applies the same rule when composing SQL.
"""
from __future__ import annotations

import re

from sqlglot import exp

# Words that Postgres 16 does not accept as a bare column or table name: pg_get_keywords() with catcode R, T or C.
# C words are legal in some positions; they stay quoted because the printer cannot tell positions apart cheaply.
PG_KEYWORDS = frozenset("""
all analyse analyze and any array as asc asymmetric authorization between bigint binary bit boolean both case cast
char character check coalesce collate collation column concurrently constraint create cross current_catalog
current_date current_role current_schema current_time current_timestamp current_user dec decimal default deferrable
desc distinct do else end except exists extract false fetch float for foreign freeze from full grant greatest group
grouping having ilike in initially inner inout int integer intersect interval into is isnull join json_array
json_arrayagg json_object json_objectagg lateral leading least left like limit localtime localtimestamp national
natural nchar none normalize not notnull null nullif numeric offset on only or order out outer overlaps overlay
placing position precision primary real references returning right row select session_user setof similar smallint
some substring symmetric system_user table tablesample then time timestamp to trailing treat trim true union unique
user using values varchar variadic verbose when where window with xmlattributes xmlconcat xmlelement xmlexists
xmlforest xmlnamespaces xmlparse xmlpi xmlroot xmlserialize xmltable
""".split())

PLAIN = re.compile(r"[a-z_][a-z0-9_]*\Z")
PLAIN_ANY_CASE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def is_safe_unquoted(name: str) -> bool:
    """True when Postgres reads the bare word `name` as exactly the identifier `name`."""
    return bool(PLAIN.match(name)) and name not in PG_KEYWORDS


def canonical_quotes(tree: exp.Expression) -> exp.Expression:
    """Every identifier spelled the one way Postgres and the printer agree on: a quoted lower-case plain word that is
    no keyword loses its quotes, a bare keyword (which `qualify` produces as the qualifier of a column of `FROM
    "order"`) gets them. Unquoted identifiers are already folded to lower case by `qualify`."""
    for ident in tree.find_all(exp.Identifier):
        quoted = bool(ident.args.get("quoted"))
        if quoted and is_safe_unquoted(ident.name):
            ident.set("quoted", False)
        elif not quoted and ident.name.lower() in PG_KEYWORDS:
            ident.set("quoted", True)
    return tree
