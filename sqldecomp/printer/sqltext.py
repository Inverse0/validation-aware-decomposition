"""SQLGlot helpers of the printer: parse side-table fragments, handle identifiers, find and rename column references.

All parsing happens in the Postgres dialect and inside a wrapper statement (`SELECT <item>`, `SELECT 1 WHERE <predicate>`,
...), so that a fragment can never be mistaken for a statement of its own and dialect details such as the default
position of NULLs in ORDER BY are handled by SQLGlot itself."""
from __future__ import annotations

import re
from typing import Callable, Iterable, Iterator

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from sqldecomp.identifiers import PG_KEYWORDS, PLAIN, PLAIN_ANY_CASE, is_safe_unquoted
from sqldecomp.printer.errors import PrintError

DIALECT = "postgres"

# the one rule for identifier quotes, shared with the importer (sqldecomp.identifiers)
_PG_KEYWORDS, _PLAIN, _PLAIN_ANY_CASE = PG_KEYWORDS, PLAIN, PLAIN_ANY_CASE


# --------------------------------------------------------------------------------------------------------------
# Parsing side-table fragments
# --------------------------------------------------------------------------------------------------------------


def _parse(sql: str, what: str) -> exp.Expression:
    try:
        tree = sqlglot.parse_one(sql, dialect=DIALECT)
    except SqlglotError as e:  # pragma: no cover - the message is what matters
        raise PrintError(f"cannot parse {what}: {e}") from e
    if tree is None:
        raise PrintError(f"cannot parse {what}: empty")
    return tree


def parse_item(sql: str) -> exp.Expression:
    """A select item or any scalar expression: `c.c_id`, `SUM(o.o_total) AS total`, `(SELECT ...)`."""
    tree = _parse(f"SELECT {sql}", f"expression {sql!r}")
    if not isinstance(tree, exp.Select) or len(tree.expressions) != 1:
        raise PrintError(f"expected one expression, got {sql!r}")
    return tree.expressions[0].pop()


def parse_predicate(sql: str) -> exp.Expression:
    """One conjunct."""
    tree = _parse(f"SELECT 1 WHERE {sql}", f"predicate {sql!r}")
    where = tree.args.get("where") if isinstance(tree, exp.Select) else None
    if where is None:
        raise PrintError(f"expected a predicate, got {sql!r}")
    return where.this.pop()


def parse_from_item(sql: str) -> exp.Expression:
    """A FROM item as the side table stores it: `customers AS c`."""
    tree = _parse(f"SELECT * FROM {sql}", f"FROM item {sql!r}")
    from_ = tree.args.get("from_") if isinstance(tree, exp.Select) else None
    if from_ is None or tree.args.get("joins"):
        raise PrintError(f"expected one FROM item, got {sql!r}")
    return from_.this.pop()


def parse_group_by(items: list[str]) -> exp.Group:
    """A GROUP BY clause. ROLLUP (...), CUBE (...) and GROUPING SETS (...) only parse inside one."""
    tree = _parse("SELECT 1 GROUP BY " + ", ".join(items), f"GROUP BY {items!r}")
    group = tree.args.get("group") if isinstance(tree, exp.Select) else None
    if group is None:
        raise PrintError(f"expected GROUP BY items, got {items!r}")
    return group.pop()


def parse_group_item(sql: str) -> exp.Group:
    """ONE item of a GROUP BY clause, as the clause that holds only that item: `ROLLUP (a, b)` is no expression on
    its own. Walk or rename its columns like any tree; `group_item_sql` gives the item back."""
    return parse_group_by([sql])


def group_item_sql(group: exp.Group) -> str:
    text = to_sql(group)
    if not text.upper().startswith("GROUP BY "):
        raise PrintError(f"expected a GROUP BY clause, got {text!r}")
    return text[len("GROUP BY "):]


def parse_order_key(sql: str, desc: bool | None, nulls: object) -> exp.Ordered:
    """An ORDER BY key. It is parsed as text so that the dialect decides what an absent NULLS clause means."""
    text = sql + (" DESC" if desc else "")
    where = _nulls_position(nulls)
    if where:
        text += f" NULLS {where}"
    tree = _parse(f"SELECT 1 ORDER BY {text}", f"ORDER BY key {text!r}")
    order = tree.args.get("order") if isinstance(tree, exp.Select) else None
    if order is None or len(order.expressions) != 1:
        raise PrintError(f"expected one ORDER BY key, got {text!r}")
    return order.expressions[0].pop()


def _nulls_position(nulls: object) -> str | None:
    if nulls is None or nulls == "":
        return None
    word = str(nulls).strip().upper()
    if word.endswith("FIRST"):
        return "FIRST"
    if word.endswith("LAST"):
        return "LAST"
    raise PrintError(f"ORDER BY key has an unknown NULLS position {nulls!r}")


def and_all(parts: Iterable[exp.Expression]) -> exp.Expression | None:
    parts = list(parts)
    if not parts:
        return None
    # exp.and_ wraps OR operands in parentheses and leaves the operands themselves untouched
    return exp.and_(*parts, dialect=DIALECT, copy=False) if len(parts) > 1 else parts[0]


# --------------------------------------------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------------------------------------------


def ident_key(node: exp.Identifier | str | None) -> str:
    """The name Postgres resolves an identifier to: unquoted identifiers fold to lower case."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node.lower()
    return node.name if node.args.get("quoted") else node.name.lower()


def make_ident(name: str) -> exp.Identifier:
    """An identifier for a name the printer itself composes. Quoted only when Postgres needs it."""
    plain = bool(_PLAIN_ANY_CASE.match(name)) and name.lower() not in _PG_KEYWORDS
    return exp.Identifier(this=name, quoted=not plain)


def join_names(parts: Iterable[exp.Identifier], sep: str = "__") -> exp.Identifier:
    """`<alias>__<column>`. Stays unquoted (and keeps the written case) when every part is unquoted; otherwise the
    unquoted parts are folded the way Postgres folds them and the whole name is quoted."""
    parts = list(parts)
    if all(not p.args.get("quoted") or is_safe_unquoted(p.name) for p in parts):
        return make_ident(sep.join(p.name for p in parts))
    return exp.Identifier(this=sep.join(ident_key(p) for p in parts), quoted=True)


def unquote_safe_identifiers(tree: exp.Expression) -> exp.Expression:
    """Drop the quotes of every identifier that means the same without them."""
    for node in tree.find_all(exp.Identifier):
        if node.args.get("quoted") and is_safe_unquoted(node.name):
            node.set("quoted", False)
    return tree


# --------------------------------------------------------------------------------------------------------------
# Column references, with correct shadowing inside subqueries
# --------------------------------------------------------------------------------------------------------------


def source_alias_keys(query: exp.Expression) -> frozenset[str]:
    """Alias keys that the FROM clause of one SELECT defines (tables, derived tables, joined items)."""
    out: set[str] = set()
    if not isinstance(query, exp.Select):
        return frozenset()
    items: list[exp.Expression] = []
    from_ = query.args.get("from_")
    if from_ is not None:
        items.append(from_.this)
    items += [j.this for j in query.args.get("joins") or ()]
    while items:
        item = items.pop()
        if isinstance(item, exp.Table):
            alias = item.args.get("alias")
            out.add(ident_key(alias.this) if alias is not None and alias.this is not None else ident_key(item.this))
            items += [j.this for j in item.args.get("joins") or ()]
        elif isinstance(item, exp.Subquery):
            alias = item.args.get("alias")
            if alias is not None and alias.this is not None:
                out.add(ident_key(alias.this))
            elif isinstance(item.this, (exp.Table, exp.Subquery)):  # a parenthesized join, its first item a table or a derived table
                items.append(item.this)
            items += [j.this for j in item.args.get("joins") or ()]
        else:
            alias = item.args.get("alias")
            if isinstance(alias, exp.TableAlias) and alias.this is not None:
                out.add(ident_key(alias.this))
    return frozenset(out)


def outer_columns(tree: exp.Expression, shadowed: frozenset[str] = frozenset()) -> Iterator[exp.Column]:
    """Every qualified column of `tree` whose table alias is NOT defined by a SELECT between the column and the root of
    `tree`: exactly the references that the enclosing query has to satisfy. Unqualified columns are skipped (the side
    table qualifies every real column, so what is left are references to select aliases)."""
    if isinstance(tree, exp.Select):
        shadowed = shadowed | source_alias_keys(tree)
    if isinstance(tree, exp.Column):
        table = tree.args.get("table")
        if table is not None and ident_key(table) not in shadowed:
            yield tree
        return
    for child in tree.iter_expressions():
        yield from outer_columns(child, shadowed)


def rename_columns(tree: exp.Expression, rename: Callable[[exp.Column], exp.Expression | None]) -> exp.Expression:
    """Replace outer column references in place. `rename` returns the replacement or None to keep the column."""
    for column in list(outer_columns(tree)):
        new = rename(column)
        if new is not None:
            if column is tree:
                tree = new
            else:
                column.replace(new)
    return tree


def top_level_queries(tree: exp.Expression) -> list[exp.Expression]:
    """The outermost SELECT / set-operation nodes inside an expression, in the order they are written."""
    found: list[exp.Expression] = []

    def visit(node: exp.Expression) -> None:
        if isinstance(node, (exp.Select, exp.SetOperation)):
            found.append(node)
            return
        for child in node.iter_expressions():
            visit(child)

    visit(tree)
    return found


def column_key(column: exp.Column) -> tuple[str, str]:
    return ident_key(column.args.get("table")), ident_key(column.this) if isinstance(column.this, exp.Identifier) else "*"


def natural_name(item: exp.Expression) -> str | None:
    """The column name Postgres gives an unaliased select item, when the printer can tell."""
    if isinstance(item, exp.Alias):
        return ident_key(item.args.get("alias"))
    if isinstance(item, exp.Column) and isinstance(item.this, exp.Identifier):
        return ident_key(item.this)
    return None


def to_sql(tree: exp.Expression, *, pretty: bool = False) -> str:
    return tree.sql(dialect=DIALECT, pretty=pretty)
