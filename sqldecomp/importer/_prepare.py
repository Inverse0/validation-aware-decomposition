"""Step 1 of the import: parse one statement, refuse what we do not support, and let SQLGlot resolve every name.

The result is a qualified SQLGlot tree: identifiers lower-cased, every column qualified by its FROM alias, stars,
USING, NATURAL and positional GROUP BY / ORDER BY expanded. Around `qualify` we correct what SQLGlot gets wrong for
Postgres or invents, so that the side table shows what the author wrote:

before  set operations get their SQL precedence (`_set_operation_precedence`); unnamed select items get the name
        Postgres gives them as a placeholder alias (`_screen_select`); what the author wrote is marked.
after   aliases that qualify or we added are dropped again unless the query refers to them (`_drop_added_aliases`);
        ORDER BY keys become what the author wrote (`_normalize_order_keys`); the merged column of an inner / LEFT
        JOIN ... USING is a plain column again (`_simplify_merged_columns`); named windows are inlined.
"""
from __future__ import annotations

import functools
import itertools
import re

import sqlglot
from sqlglot import exp
from sqlglot.dialects.postgres import Postgres
from sqlglot.errors import OptimizeError, SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import MappingSchema

from sqldecomp.importer._errors import UnsupportedSQL
from sqldecomp.identifiers import canonical_quotes
from sqldecomp.importer._names import OUTPUT_NAME, pg_default_name
from sqldecomp.model import Schema

# Marks left on the tree before `qualify` runs, so that afterwards we can tell what the query's author wrote from
# what SQLGlot added.
ORIG_ALIAS = "sqldecomp.orig_alias"  # an Alias node the author wrote
UNALIASED = "sqldecomp.unaliased"  # a select item the author left without an alias
PLACEHOLDER = "sqldecomp.placeholder"  # an Alias we put around an unnamed select item, see _screen_select
BARE_KEY = "sqldecomp.bare_key"  # on an ORDER BY key: the author wrote a bare name or a position, not an expression
ORIG_KEY = "sqldecomp.orig_key"  # on an ORDER BY key the author wrote as an expression: a copy of it, before qualify
ORIG_COALESCE = "sqldecomp.orig_coalesce"  # a COALESCE the author wrote (qualify adds its own for USING columns)
COMMA_JOIN = "sqldecomp.comma_join"  # a FROM item introduced by a comma
MERGING_JOIN = "sqldecomp.merging_join"  # a join written with USING or NATURAL

_SELECT_CLAUSES = {"into": "SELECT INTO", "locks": "locking clause (FOR UPDATE / FOR SHARE)", "qualify": "QUALIFY",
                   "connect": "CONNECT BY", "match": "MATCH_RECOGNIZE", "laterals": "LATERAL VIEW", "sample": "TABLESAMPLE",
                   "cluster": "CLUSTER BY", "distribute": "DISTRIBUTE BY", "sort": "SORT BY", "kind": "SELECT AS STRUCT / VALUE"}
_SAFE_NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")  # a placeholder alias must not need quotes ...
_KEYWORDS = frozenset(k for k in Postgres.Tokenizer.KEYWORDS if k.isalpha())  # ... and must not be a keyword (case, exists)


def is_query(node: exp.Expr) -> bool:
    return isinstance(node, (exp.Select, exp.SetOperation))


def unparen(node: exp.Expr) -> exp.Expr:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def contains_query(node: exp.Expr) -> bool:
    return any(is_query(n) for n in node.walk())


def prepare(sql: str, schema: Schema, dialect: str) -> exp.Expr:
    """The qualified tree of the one SELECT statement in `sql`."""
    tree = _set_operation_precedence(_parse(sql, dialect))
    _refuse_natural_joins_over_unnamed_columns(tree)
    naming = _Naming(schema, tree)
    _unwrap_bare_keys(tree)
    _screen(tree, naming)
    _known_tables(tree, schema)
    _stars_in_from_order(tree, naming)
    _positions_over_shared_names(tree)
    _rename_bare_keys_under_alias_lists(tree)
    distinct_marks = [_distinct_marks(select) for select in tree.find_all(exp.Select)]
    tree = canonical_quotes(_qualify(tree, schema, dialect, sql))  # "molecule_id" -> molecule_id; order.x -> "order".x
    selects = list(tree.find_all(exp.Select))
    if len(selects) != len(distinct_marks):  # never seen; the marks are then unknown
        distinct_marks = [None] * len(selects)
    for select in selects:
        if any(item.is_star for item in select.selects):
            raise UnsupportedSQL("name resolution", "SQLGlot could not expand a * (a FROM item with duplicate column names?)")
        _refuse_outer_aggregates(select)
        _inline_named_windows(select)
        _repair_merged_join_conditions(select)
        _simplify_merged_columns(select)
        _normalize_order_keys(select)
    for select, marks in zip(selects, distinct_marks):
        _normalize_distinct_keys(select, marks)
    referenced = _referenced_outputs(tree)
    for select in reversed(selects):  # innermost first: a scalar subquery is named after its body's column
        _drop_added_aliases(select, referenced, dialect)
    return tree


# ----------------------------------------------------------------------------------------------------------------------
# parse
# ----------------------------------------------------------------------------------------------------------------------


def _refuse_literals_the_parser_misreads(sql: str, dialect: str) -> None:
    """Postgres 16 reads `0x1F`, `0o17`, `0b101` and `1_000` as integers. SQLGlot 30.18 reads `0x1F` / `0b101` as the
    bit strings x'1F' / b'101', and `0o17` / `1_000` as the number 0 / 1 followed by an ALIAS (`SELECT 0 AS o17`):
    another value, silently. Such a literal is refused (re-check when the pin moves)."""
    try:
        tokens = list(sqlglot.tokenize(sql, read=dialect))
    except SqlglotError:
        return  # the parser will say what is wrong
    for n, (token, following) in enumerate(zip(tokens, tokens[1:] + [None])):
        kind = token.token_type.name
        if kind in ("HEX_STRING", "BIT_STRING") and sql[token.start:token.start + 2].lower() in ("0x", "0b"):
            raise UnsupportedSQL("numeric literal", f"{sql[token.start:token.end + 1]}: a non-decimal integer literal")
        if kind == "NUMBER" and following is not None and following.token_type.name == "VAR" and following.start == token.end + 1:
            raise UnsupportedSQL("numeric literal", f"{token.text}{following.text}: a non-decimal integer or one with underscores")
        if kind == "PLUS" and not _ends_an_operand(tokens, n - 1):
            # SQLGlot 30.18 drops a unary plus while parsing: `ORDER BY +1` becomes the position 1 and `ORDER BY +k`
            # the bare output name k, where Postgres reads a constant and the input column k
            raise UnsupportedSQL("unary plus", f"the parser drops the unary + before {following.text if following else 'the end'!r}")


_OPERAND_END = {"VAR", "IDENTIFIER", "NUMBER", "STRING", "R_PAREN", "R_BRACKET", "R_BRACE", "STAR", "NULL", "TRUE",
                "FALSE", "PARAMETER", "PLACEHOLDER", "HEX_STRING", "BIT_STRING", "NATIONAL_STRING", "BYTE_STRING",
                "RAW_STRING", "HEREDOC_STRING", "UNICODE_STRING", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
                "CURRENT_USER", "SESSION_USER", "LOCALTIME", "LOCALTIMESTAMP", "DEFAULT"}


def _ends_an_operand(tokens: list, i: int) -> bool:
    """Whether the token at `i` ends an operand, so that a following `+` is binary: a name, a literal, a closing
    bracket, or the type of a cast (`x::int + 1`)."""
    if i < 0:
        return False
    if tokens[i].token_type.name in _OPERAND_END:
        return True
    return i >= 1 and tokens[i - 1].token_type.name == "DCOLON"


def _parse(sql: str, dialect: str) -> exp.Expr:
    _refuse_literals_the_parser_misreads(sql, dialect)
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as e:
        raise UnsupportedSQL("parse error", str(e).splitlines()[0] if str(e) else type(e).__name__) from e
    if not statements:
        raise UnsupportedSQL("not a query", "empty statement")
    if len(statements) > 1:
        raise UnsupportedSQL("not a query", f"{len(statements)} statements; exactly one SELECT is expected")
    tree = statements[0]
    while isinstance(tree, exp.Subquery) and not tree.alias:  # (SELECT ...) written in parentheses
        if any(tree.args.get(k) for k in ("order", "limit", "offset", "with_")):
            raise UnsupportedSQL("modifiers on a parenthesized query", "ORDER BY / LIMIT / WITH outside the parentheses")
        tree = tree.this.pop()
    if not is_query(tree):
        raise UnsupportedSQL("not a query", f"{tree.key.upper()}: only a SELECT or a set operation can be imported")
    return tree


_MODIFIERS = ("with_", "order", "limit", "offset")


def _set_operation_precedence(tree: exp.Expr) -> exp.Expr:
    """SQLGlot parses a chain of set operations strictly left to right. In SQL, INTERSECT binds tighter than UNION and
    EXCEPT: `a UNION b INTERSECT c` is `a UNION (b INTERSECT c)`. Rebuild every unparenthesized chain accordingly
    (a parenthesized operand is a Subquery node, so it is an operand here and never part of the chain)."""
    for top in [n for n in tree.find_all(exp.SetOperation)
                if not (isinstance(n.parent, exp.SetOperation) and n.arg_key == "this")]:
        operators: list[exp.SetOperation] = []
        node: exp.Expr = top
        while isinstance(node, exp.SetOperation):
            operators.append(node)
            node = node.this
        operators.reverse()
        kinds = [isinstance(o, exp.Intersect) for o in operators]
        if not any(tight and not all(kinds[:i]) for i, tight in enumerate(kinds)):
            continue  # no INTERSECT to the right of a UNION / EXCEPT: left to right is what SQL means
        if any(o.args.get(k) for o in operators[:-1] for k in _MODIFIERS):
            raise UnsupportedSQL("unsupported set operation", "ORDER BY / LIMIT inside a chain of set operations")
        modifiers = {k: top.args.get(k) for k in _MODIFIERS if top.args.get(k)}
        operands = [operators[0].this] + [o.expression for o in operators]

        def combine(op: exp.SetOperation, left: exp.Expr, right: exp.Expr) -> exp.SetOperation:
            if isinstance(right, exp.SetOperation):
                right = exp.Subquery(this=right)  # printed with parentheses, so that SQLGlot reads it back correctly
            keep = {k: v for k, v in op.args.items() if k not in ("this", "expression", *_MODIFIERS)}
            return type(op)(this=left, expression=right, **keep)

        loose_ops: list[exp.SetOperation] = []
        items: list[exp.Expr] = [operands[0]]
        for op, operand in zip(operators, operands[1:]):
            if isinstance(op, exp.Intersect):
                items.append(combine(op, items.pop(), operand))
            else:
                loose_ops.append(op)
                items.append(operand)
        result = items[0]
        for op, operand in zip(loose_ops, items[1:]):
            result = combine(op, result, operand)
        for key, value in modifiers.items():
            result.set(key, value)
        if top is tree:
            tree = result
        else:
            top.replace(result)
    return tree


# ----------------------------------------------------------------------------------------------------------------------
# screen: refuse the unsupported, mark what the author wrote
# ----------------------------------------------------------------------------------------------------------------------


def _refuse_outer_aggregates(select: exp.Select) -> None:
    """`(SELECT COUNT(*) FROM orders o WHERE o.o_cust = MIN(c.c_id))`: an aggregate whose arguments are all columns of
    an OUTER query is evaluated by that outer query and handed in as a constant. The plan language has no place for
    that (the conjunct would carry an aggregate into a WHERE clause as soon as its body is printed by itself)."""
    local = {i.alias_or_name.lower() for i in _from_items(select) if i.alias_or_name}
    for call in select.find_all(exp.AggFunc):
        if call.find_ancestor(exp.Select) is not select:
            continue
        columns = [c for c in call.find_all(exp.Column) if c.table]
        if columns and all(c.table.lower() not in local for c in columns) and select.parent_select is not None:
            raise UnsupportedSQL("outer aggregate", f"{call.sql()} aggregates columns of an enclosing query")


def _output_ident(item: exp.Expr) -> tuple[str, bool]:
    """(name, quoted) of a select item's output name as written."""
    ident = item.args.get("alias") if isinstance(item, exp.Alias) else item.this if isinstance(item, exp.Column) else None
    if isinstance(ident, exp.Identifier):
        return ident.name, bool(ident.args.get("quoted"))
    return item.alias_or_name, False


def _folded(name: str, quoted: bool) -> str:
    """The name Postgres resolves an identifier to: unquoted names fold to lower case."""
    return name if quoted else name.lower()


def _folded_key(column: exp.Column) -> str:
    return _folded(column.name, bool(column.this.args.get("quoted")))


def _branches(query: exp.Expr) -> list[exp.Select]:
    """The SELECTs whose select lists form the output of a query: itself, or the branches of its set operations."""
    while isinstance(query, (exp.Subquery, exp.Paren)):
        query = query.this
    if isinstance(query, exp.SetOperation):
        return _branches(query.this) + _branches(query.expression)
    return [query] if isinstance(query, exp.Select) else []


def _set_operations(query: exp.Expr) -> list[exp.SetOperation]:
    """The set operations that make up a query's own tree (not those inside its FROM items)."""
    while isinstance(query, (exp.Subquery, exp.Paren)):
        query = query.this
    if isinstance(query, exp.SetOperation):
        return [query] + _set_operations(query.this) + _set_operations(query.expression)
    return []


def _alias_list_over(query: exp.Expr) -> list[tuple[str, bool]] | None:
    """The column alias list of the derived table or CTE whose body `query` (a SELECT or a set operation) is part of,
    as (name, quoted) pairs; None when there is none."""
    node = query
    while node is not None:
        parent = node.parent
        if isinstance(parent, (exp.Subquery, exp.CTE)) and parent.this is node:
            alias = parent.args.get("alias")
            columns = alias.args.get("columns") if isinstance(alias, exp.TableAlias) else None
            return [(c.name, bool(c.args.get("quoted"))) for c in columns] if columns else None
        if isinstance(parent, (exp.SetOperation, exp.Paren)):
            node = parent
            continue
        return None
    return None


def _effective_names(select: exp.Select) -> list[str] | None:
    """The folded output names of a SELECT as they will be after qualify: renamed by an enclosing column alias list
    where there is one. None when the select list has a star (positions unknown)."""
    if any(item.is_star for item in select.selects):
        return None
    names = [_folded(*_output_ident(item)) for item in select.selects]
    listed = _alias_list_over(select)
    if listed:
        names = [_folded(*x) for x in listed] + names[len(listed):]
    return names


def _order_keys(query: exp.Expr) -> list[exp.Ordered]:
    """The ORDER BY items of a SELECT or of a set operation (the ones written at its own level)."""
    order = query.args.get("order")
    return list(order.expressions) if order is not None else []


def _distinct_on_keys(select: exp.Select) -> list[exp.Expr]:
    distinct = select.args.get("distinct")
    on = distinct.args.get("on") if isinstance(distinct, exp.Distinct) else None
    if on is None:
        return []
    return list(on.expressions) if isinstance(on, exp.Tuple) else [on]


def _set_distinct_on_keys(select: exp.Select, keys: list[exp.Expr]) -> None:
    on = select.args["distinct"].args["on"]
    if isinstance(on, exp.Tuple):
        on.set("expressions", keys)
    else:
        select.args["distinct"].set("on", keys[0])


def _unwrap_bare_keys(tree: exp.Expr) -> None:
    """Postgres reads `ORDER BY (k)`, `ORDER BY (2)` and `DISTINCT ON ((k))` exactly like the bare key: parentheses around a
    name or a number produce no node in its grammar, so the name still resolves to an OUTPUT column first and the number
    is still a position. SQLGlot keeps a Paren node, which `_input_names` then read as an expression over an INPUT column
    and qualify never expanded as a position: `c.c_region AS c_name ... ORDER BY (c_name)` sorted by the input column
    (another row), and `ORDER BY (1)` stayed the constant 1."""

    def unparen(key: exp.Expr) -> exp.Expr:
        inner = key
        while isinstance(inner, exp.Paren):
            inner = inner.this
        if (isinstance(inner, exp.Column) and not inner.table) or (isinstance(inner, exp.Literal) and inner.is_int):
            return inner
        return key

    for node in list(tree.walk()):
        if isinstance(node, exp.Ordered):
            node.set("this", unparen(node.this))
        elif isinstance(node, exp.Select) and _distinct_on_keys(node):
            _set_distinct_on_keys(node, [unparen(k) for k in _distinct_on_keys(node)])


def _positions_over_shared_names(tree: exp.Expr) -> None:
    """`SELECT c_id AS a, c_name AS a ... ORDER BY 2`: qualify turns the position into the item's alias, and `ORDER BY a`
    is ambiguous (Postgres rejects it) and no longer says WHICH item. Where the item's output name is shared, the
    position is replaced by the item's expression before qualify runs; in DISTINCT ON the same; at the ORDER BY of a
    set operation, whose keys can only be output names, such a position is refused."""
    for select in tree.find_all(exp.Select):
        names = _effective_names(select)  # folded, and as an enclosing column alias list will rename them
        if names is None:
            continue

        def expanded(key: exp.Expr) -> exp.Expr:
            if isinstance(key, exp.Literal) and key.is_int and 1 <= int(key.name) <= len(names):
                item = select.selects[int(key.name) - 1]
                if names.count(names[int(key.name) - 1]) > 1 and not contains_query(item):
                    return item.unalias().copy()
            return key

        for ordered in _order_keys(select):
            ordered.set("this", expanded(ordered.this))
        if _distinct_on_keys(select):
            _set_distinct_on_keys(select, [expanded(k) for k in _distinct_on_keys(select)])
    for setop in tree.find_all(exp.SetOperation):
        first = _first_select(setop)
        if not isinstance(first, exp.Select):
            continue
        names = _effective_names(first)
        for ordered in _order_keys(setop):
            key = ordered.this
            if not (isinstance(key, exp.Literal) and key.is_int):
                continue
            if names is None:
                raise UnsupportedSQL("name resolution", f"ORDER BY {key.name} of a set operation whose first branch "
                                     "selects *: the position cannot be written by name")
            if 1 <= int(key.name) <= len(names) and names.count(names[int(key.name) - 1]) > 1:
                raise UnsupportedSQL("name resolution", f"ORDER BY {key.name} of a set operation names an output "
                                     f"({names[int(key.name) - 1]}) that two items carry; it cannot be written by name")


def _rename_bare_keys_under_alias_lists(tree: exp.Expr) -> None:
    """`(SELECT c.c_name AS k, c.c_region AS j FROM customers c ORDER BY k) AS x(j, k)`: the column alias list renames
    the body's outputs as seen from outside, and qualify writes the new names onto the body's select items; the body's
    own bare ORDER BY / DISTINCT ON keys keep naming the body's OWN outputs, so `k` must follow the output it named
    (`c.c_name`, now called `j`), or qualify reads it as the other column. A name that two outputs of the body carry
    is ambiguous (Postgres rejects it); a star in the select list leaves the positions unknown, so a bare key next to
    it is refused."""
    for node in list(tree.find_all(exp.Subquery, exp.CTE)):
        alias = node.args.get("alias")
        columns = alias.args.get("columns") if isinstance(alias, exp.TableAlias) else None
        if not columns or not is_query(node.this):
            continue
        body = node.this
        listed = [(c.name, bool(c.args.get("quoted"))) for c in columns]
        new_folded = [_folded(*x) for x in listed]
        # qualify writes the list onto EVERY branch's select items, so the bare keys of every branch and of every set
        # operation of the body's own tree (which name the first branch's outputs) follow their output
        targets: list[tuple[exp.Select, list[exp.Expr], exp.Expr]] = []  # (the select whose names apply, keys, key holder)
        for branch in _branches(body):
            targets.append((branch, _order_keys(branch) + _distinct_on_keys(branch), branch))
        for setop in _set_operations(body):
            first = _first_select(setop)
            if isinstance(first, exp.Select):
                targets.append((first, _order_keys(setop), setop))
        for select, keys, holder in targets:
            bare = [k for k in keys if isinstance(k.this if isinstance(k, exp.Ordered) else k, exp.Column)
                    and not (k.this if isinstance(k, exp.Ordered) else k).table]
            if not bare:
                continue
            if any(item.is_star for item in select.selects):
                raise UnsupportedSQL("name resolution", "a bare ORDER BY / DISTINCT ON key under a column alias list over a select list with *")
            old = [_folded(*_output_ident(item)) for item in select.selects]
            new = new_folded + old[len(new_folded):]
            for k in bare:
                column = k.this if isinstance(k, exp.Ordered) else k
                matches = [i for i, name in enumerate(old) if name == _folded_key(column)]
                if len(matches) > 1:
                    raise UnsupportedSQL("name resolution", f"ORDER BY / DISTINCT ON {column.name} is ambiguous: {len(matches)} outputs carry the name")
                if len(matches) != 1 or new[matches[0]] == old[matches[0]]:
                    continue
                i = matches[0]
                if new.count(new[i]) > 1:  # the new name is shared: the key becomes the item's expression
                    item = select.selects[i]
                    if contains_query(item):
                        raise UnsupportedSQL("name resolution", f"ORDER BY / DISTINCT ON {column.name} names an output that the alias list renames to a name two outputs carry")
                    renamed: exp.Expr = item.unalias().copy()
                else:
                    renamed = exp.Column(this=exp.to_identifier(listed[i][0], quoted=listed[i][1] or None))
                if isinstance(k, exp.Ordered):
                    k.set("this", renamed)
                    k.meta[ORIG_KEY] = renamed.copy()  # the renamed key is what the author meant, for _the_item
                else:
                    _set_distinct_on_keys(select, [renamed if x is k else x for x in _distinct_on_keys(select)])


def _refuse_natural_joins_over_unnamed_columns(tree: exp.Expr) -> None:
    """NATURAL JOIN matches the columns of its two sides BY NAME. For a select item without an alias that is no plain
    column Postgres derives the name from the expression (`?column?`, `numeric`, `case`, `btrim`, ...), SQLGlot invents
    another one, and `pg_default_name` is a best effort. A misjudged name there is a column that silently joins, or
    silently does not: other rows, which no check that re-imports the print can see. So a NATURAL join is accepted
    only when the author named every column of every derived table and CTE next to it."""
    ctes: dict[str, list[exp.Expr]] = {}
    for cte in tree.find_all(exp.CTE):
        ctes.setdefault(cte.alias.lower(), []).append(cte)

    def named(source: exp.Expr, seen: frozenset[int]) -> bool:
        if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier):
            definitions = ctes.get(source.name.lower(), []) if not source.db else []
            if not definitions:
                return True  # a table of the schema
            if len(definitions) > 1 or id(definitions[0]) in seen:
                return False  # which WITH clause is meant depends on the scope: not decided here
            cte = definitions[0]
            return _lists_its_columns(cte.args.get("alias"), cte.this) or query_named(cte.this, seen | {id(cte)})
        if isinstance(source, exp.Subquery):
            if not is_query(source.this):
                return _lists_its_columns(source.args.get("alias"), None) if isinstance(source.this, exp.Values) else True
            return _lists_its_columns(source.args.get("alias"), source.this) or query_named(source.this, seen)
        return _lists_its_columns(source.args.get("alias"), None)  # VALUES and whatever else has columns

    def query_named(query: exp.Expr, seen: frozenset[int]) -> bool:
        select = _first_select(query)
        if not isinstance(select, exp.Select):
            return False
        for item in select.selects:
            if item.is_star:
                qualifier = item.table.lower() if isinstance(item, exp.Column) and item.table else None
                if not all(named(i, seen) for i in _from_items(select)
                           if qualifier is None or i.alias_or_name.lower() == qualifier):
                    return False
            elif not isinstance(item, (exp.Alias, exp.Column)):
                return False
        return True

    for join in tree.find_all(exp.Join):
        if (join.method or "").upper() != "NATURAL":
            continue
        select = join.find_ancestor(exp.Select)
        if select is not None and not all(named(i, frozenset()) for i in _from_items(select)):
            raise UnsupportedSQL("NATURAL join", "next to a derived table or CTE with a select item that has no name of "
                                 "its own: Postgres matches such a column by a name it derives from the expression")


def _lists_its_columns(alias: exp.Expr | None, query: exp.Expr | None) -> bool:
    """`AS z(a, b)` names every column of the FROM item."""
    columns = alias.args.get("columns") if isinstance(alias, exp.TableAlias) else None
    if not columns:
        return False
    return query is None or len(columns) == len(_first_select(query).selects)


def _stars_in_from_order(tree: exp.Expr, naming: _Naming) -> None:
    """`SELECT *` lists the columns of the FROM items in the order they are written. SQLGlot 30.18's qualify expands a
    bare `*` with the base tables FIRST and the derived tables, CTE uses and VALUES lists after them, whatever the
    FROM order: another column order (and other ORDER BY positions, other UNION alignment), silently. So a bare `*`
    over several FROM items is spelled `a.*, b.*, ...` in FROM order before qualify runs.

    With USING / NATURAL the merged columns come FIRST and once (in the order of the USING list, or of the left
    table for NATURAL), then the rest of the left side, then the rest of the right side; qualify keeps the left
    side's order and only drops the right side's copies, which is the same thing exactly when the merged columns
    lead the left side. Over base tables the star is therefore left alone when the two orders agree, spelled out in
    Postgres's order when they differ (inner and LEFT joins, whose merged column is the left side's), and refused
    otherwise (RIGHT / FULL, or a parenthesized join). Over derived tables or CTEs it is refused."""
    for select in list(tree.find_all(exp.Select)):
        bare = [item for item in select.selects if isinstance(item, exp.Star)]
        items = list(_from_items(select))
        if not bare or len(items) < 2:
            continue
        joins = [j for j in select.find_all(exp.Join) if j.find_ancestor(exp.Select) is select]
        if any(j.args.get("using") or j.method == "NATURAL" for j in joins):
            if all(isinstance(i, exp.Table) and isinstance(i.this, exp.Identifier) for i in items) and not any(
                    cte.alias.lower() in {i.name.lower() for i in items} for cte in tree.find_all(exp.CTE)):
                spelled = _merged_star(select, naming)
                if spelled is not None:
                    select.set("expressions", [new for item in select.selects for new in (
                        [c.copy() for c in spelled] if item in bare else [item])])
                continue
            raise UnsupportedSQL("name resolution", "SELECT * over a USING / NATURAL join of derived tables or CTEs")
        if any(not i.alias_or_name for i in items):
            continue
        def per_item() -> list[exp.Expr]:
            return [exp.Column(this=exp.Star(), table=exp.to_identifier(i.alias_or_name, quoted=bool(
                isinstance(i.args.get("alias"), exp.TableAlias) and i.args["alias"].this.args.get("quoted")))) for i in items]
        select.set("expressions", [new for item in select.selects for new in (per_item() if item in bare else [item])])


def _merged_star(select: exp.Select, naming: _Naming) -> list[exp.Column] | None:
    """What a bare `*` lists over a chain of joins of base tables with USING / NATURAL among them, in Postgres's order,
    as qualified columns; None when qualify's order is the same (the star may stay). Raises UnsupportedSQL when the
    order differs and cannot be spelled out: a RIGHT / FULL join (its merged column is COALESCE of both sides), a
    parenthesized join, or a column name that would need quoting."""
    from_ = select.args.get("from_")
    joins = list(select.args.get("joins") or [])
    chain = ([from_.this] if from_ is not None else []) + [j.this for j in joins]
    flat = all(isinstance(t, exp.Table) and not t.args.get("joins") for t in chain)

    def columns(table: exp.Table) -> list[tuple[exp.Identifier, str]]:
        qualifier = table.args["alias"].this if isinstance(table.args.get("alias"), exp.TableAlias) else table.this
        return [(qualifier, name) for name in naming.columns.get(table.name.lower(), [])]

    def refuse(why: str) -> None:
        raise UnsupportedSQL("name resolution", f"SELECT * over a USING / NATURAL join: {why}")

    if not flat:
        refuse("a parenthesized join; the merged column order is not reconstructed")
    output = columns(chain[0])  # (qualifier of the source table, column name), in Postgres's order so far
    differs = False
    for join, table in zip(joins, chain[1:]):
        right = columns(table)
        using = join.args.get("using")
        if using:
            merged = [u.name.lower() for u in using]
        elif (join.method or "").upper() == "NATURAL":
            right_names = {n.lower() for _, n in right}
            merged = [n.lower() for _, n in output if n.lower() in right_names]
        else:
            output = output + right
            continue
        left_names = [n.lower() for _, n in output]
        if left_names[:len(merged)] != merged:
            differs = True
            if (join.side or "").upper() in ("RIGHT", "FULL"):
                refuse("the merged column of a RIGHT / FULL join is not the leading column of its left side")
        by_name = {n.lower(): (q, n) for q, n in reversed(output)}  # the FIRST source of a name wins
        output = ([by_name[m] for m in merged] + [(q, n) for q, n in output if n.lower() not in merged]
                  + [(q, n) for q, n in right if n.lower() not in merged])
    if not differs:
        return None
    if any(not _SAFE_NAME.match(n) or n.upper() in _KEYWORDS for _, n in output):
        refuse("a column name that needs quoting")
    return [exp.Column(this=exp.to_identifier(n), table=q.copy()) for q, n in output]


def _from_items(select: exp.Select):
    """The FROM items of a SELECT as written, looking through parentheses around joins."""
    pending = [select.args["from_"].this] if select.args.get("from_") else []
    pending += [j.this for j in select.args.get("joins") or []]
    while pending:
        node = pending.pop(0)
        if isinstance(node, exp.Subquery) and not node.alias and isinstance(node.this, (exp.Table, exp.Subquery)):
            pending.insert(0, node.this)  # parentheses
            continue
        if isinstance(node, (exp.Table, exp.Subquery, exp.Values)):  # a chain hanging off the first item of parentheses
            pending[0:0] = [j.this for j in node.args.get("joins") or []]
        yield node


class _Naming:
    """What `_screen_select` needs to give unnamed select items their Postgres names."""

    def __init__(self, schema: Schema, tree: exp.Expr):
        self.columns = {t.name: [c.name for c in t.columns] for t in schema.tables}
        self.ctes = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
        self.numbers = itertools.count()
        self.identifiers = {i.name.lower() for i in tree.find_all(exp.Identifier)}

    def invented(self, stem: str) -> str:
        """A name for a select item that has none, which no identifier that the AUTHOR wrote can be mistaken for. (Two
        select lists may invent the same name, as qualify's `_col_<n>` always did: nobody can write a reference to
        it, and the same subquery written twice must read the same.)"""
        name, n = stem, 0
        while name.lower() in self.identifiers:
            n += 1
            name = f"{stem}_{n}"
        return name

    def star(self, select: exp.Select, item: exp.Expr) -> list[str] | None:
        """The column names a `*` or `t.*` expands to, if every FROM item it covers is a table of the schema."""
        qualifier = item.table.lower() if isinstance(item, exp.Column) and item.table else None
        names: list[str] = []
        for source in _from_items(select):
            if qualifier is not None and source.alias_or_name.lower() != qualifier:
                continue
            table = source.name.lower() if isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier) else None
            if table not in self.columns or table in self.ctes:
                return None  # a CTE, a derived table, VALUES: its column names are not known before qualify
            names += self.columns[table]
        return names or None


def _screen(tree: exp.Expr, naming: _Naming) -> None:
    for node in list(tree.walk()):
        if isinstance(node, exp.Lateral):
            raise UnsupportedSQL("LATERAL", "a lateral FROM item sees the items on its left; regions cannot express that")
        if isinstance(node, exp.Unnest):
            raise UnsupportedSQL("UNNEST", "set-returning FROM items are not supported")
        if isinstance(node, (exp.Pivot, exp.TableSample)):
            raise UnsupportedSQL(node.key.upper(), "not supported")
        if isinstance(node, exp.Table):
            _screen_table(node)
        elif isinstance(node, exp.With):
            _screen_with(node)
        elif isinstance(node, exp.Select):
            _screen_select(node, naming)
        elif isinstance(node, exp.Subquery):
            if isinstance(node.this, exp.Table) and node.alias:
                raise UnsupportedSQL("aliased parenthesized join", f"({node.this.sql()} ...) AS {node.alias} hides the aliases inside")
            if not is_query(node.this) and not isinstance(node.this, (exp.Table, exp.Values, exp.Subquery)):
                raise UnsupportedSQL("unsupported FROM item", node.this.key.upper())
        elif isinstance(node, exp.Alias):
            node.meta[ORIG_ALIAS] = True
        elif isinstance(node, exp.Coalesce):
            node.meta[ORIG_COALESCE] = True
        elif isinstance(node, exp.Ordered):
            key = node.this
            node.meta[BARE_KEY] = (isinstance(key, exp.Column) and not key.table) or (isinstance(key, exp.Literal) and key.is_int)
            node.meta[ORIG_KEY] = key.copy()
        elif isinstance(node, exp.Join):
            if not any(node.args.get(k) for k in ("side", "kind", "method", "on", "using")):
                node.meta[COMMA_JOIN] = True
            if node.args.get("using") or (node.method or "").upper() == "NATURAL":
                node.meta[MERGING_JOIN] = True
        elif isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Command)):
            raise UnsupportedSQL("not a query", f"contains a {node.key.upper()} statement")


def _known_tables(tree: exp.Expr, schema: Schema) -> None:
    """qualify passes silently over a table it does not know; say so ourselves. (A name that some WITH clause of the
    statement defines is left to the scope analysis.)"""
    known = {t.name for t in schema.tables} | {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Identifier) and table.name.lower() not in known:
            raise UnsupportedSQL("unknown table", table.name)


def _screen_table(table: exp.Table) -> None:
    if not isinstance(table.this, exp.Identifier):
        what = table.this.sql() if isinstance(table.this, exp.Expr) else repr(table.this)
        raise UnsupportedSQL("table function", f"{what} in FROM")
    if table.args.get("sample"):
        raise UnsupportedSQL("TABLESAMPLE", table.name)
    if table.args.get("pivots"):
        raise UnsupportedSQL("PIVOT", table.name)
    alias = table.args.get("alias")
    if alias is not None and alias.args.get("columns"):
        raise UnsupportedSQL("column aliases on a table reference", f"{table.name} AS {alias.sql()}")
    if any(table.args.get(k) for k in ("ordinality", "when", "version", "changes")):
        raise UnsupportedSQL("unsupported FROM item", table.sql())


def _screen_with(with_: exp.With) -> None:
    if not with_.args.get("recursive"):
        return
    for cte in with_.expressions:  # WITH RECURSIVE is fine as long as no CTE actually refers to itself
        for t in cte.this.find_all(exp.Table):
            if t.name.lower() == cte.alias.lower() and not t.db:
                raise UnsupportedSQL("recursive CTE", cte.alias)


def _screen_select(select: exp.Select, naming: _Naming) -> None:
    for key, what in _SELECT_CLAUSES.items():
        if select.args.get(key):
            raise UnsupportedSQL(what, "not supported")
    items = list(select.expressions)
    taken: list[str | None] = []  # the output names of the select list as far as they are known; None: unknown
    for item in items:
        taken += (naming.star(select, item) or [None]) if item.is_star else [_planned_name(item)]
    for index, item in enumerate(items):
        name = None if item.is_star else _planned_name(item)
        if isinstance(item, exp.Alias) or item.is_star:
            continue
        item.meta[UNALIASED] = True
        if isinstance(item, exp.Column):
            continue
        # An unnamed expression. Postgres calls its column `sum`, `max`, ...; qualify would call it `_col_<n>`. Name it
        # the Postgres way up front, so that `SELECT x.sum FROM (SELECT SUM(a) FROM t) x` resolves and `x.*` expands to
        # the names Postgres shows. The alias is a placeholder: it is dropped again unless the query refers to it.
        usable = name is not None and None not in taken and taken.count(name) == 1 and _SAFE_NAME.match(name) \
            and name.upper() not in _KEYWORDS
        if not usable:
            # qualify would invent `_col_<n>`, or the text of a literal (`1`, `lit`), and either can be a name the
            # author uses for something else: `SELECT z._col_0 FROM (SELECT x * 2, x AS _col_0 ...) z` then had two
            # columns of one name. (An unnamed scalar subquery it names with a TABLE alias, and then fails on
            # `ORDER BY <its position>`.) The name is the same for the same select list, so that two occurrences of
            # one subquery stay the same task.
            name = naming.invented(f"_sq_{next(naming.numbers)}" if isinstance(item, exp.Subquery) else f"_col_{index}")
        wrapper = exp.Alias(alias=exp.to_identifier(name))
        wrapper.meta[PLACEHOLDER] = True
        item.replace(wrapper)
        wrapper.set("this", item)


def _planned_name(item: exp.Expr) -> str:
    if isinstance(item, (exp.Alias, exp.Column)):
        return item.alias_or_name.lower()
    return pg_default_name(item)


# ----------------------------------------------------------------------------------------------------------------------
# qualify
# ----------------------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _mapping_schema(schema: Schema, dialect: str) -> MappingSchema:
    return MappingSchema(schema.as_sqlglot(), dialect=dialect)


def _qualify(tree: exp.Expr, schema: Schema, dialect: str, sql: str) -> exp.Expr:
    try:
        return qualify(tree, schema=_mapping_schema(schema, dialect), dialect=dialect, quote_identifiers=True,
                       identify=False, sql=sql)
    except OptimizeError as e:
        message = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", str(e)).split())  # SQLGlot underlines the text for terminals
        raise UnsupportedSQL("name resolution", message[:300] + _unnamed_item_hint(tree, message)) from e
    except Exception as e:  # not ours: whatever else goes wrong inside SQLGlot's qualify
        raise UnsupportedSQL("qualify failed", f"SQLGlot could not qualify the query ({type(e).__name__}: {e})"[:300]) from e


def _unnamed_item_hint(tree: exp.Expr, message: str) -> str:
    """Postgres lets a query read a select item without an alias by the name it derives from the expression
    (`z."?column?"`, `z.numeric`, `z.case`). The importer can offer that name only when it is a plain word, no keyword
    and unique in its select list (`z.count`, `z.max`); say so when the column that was not found is such a name."""
    if not re.search(r"unknown column|could not be resolved", message, re.I):
        return ""
    derived = {pg_default_name(i.unalias()).lower() for select in tree.find_all(exp.Select) for i in select.selects
               if not i.is_star and (i.meta.get(PLACEHOLDER) or not isinstance(i, (exp.Alias, exp.Column)))}
    if any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", message, re.I) for name in derived):
        return (" (Postgres derives this name from a select item that has no alias; the importer reads such a column "
                "by name only when the name is a plain word, no keyword and unique in its select list)")
    return ""


# ----------------------------------------------------------------------------------------------------------------------
# what SQLGlot added, and what we take back out
# ----------------------------------------------------------------------------------------------------------------------


def alias_status(item: exp.Expr, index: int) -> str:
    """How select item number `index` of a qualified SELECT got its alias.

    'explicit'   the author wrote it (or it is needed: a column list `AS t(a, b)`, a merged USING column); kept.
    'natural'    `c.c_id AS c_id`, added by qualify; the alias says nothing, so it is dropped.
    'generated'  our placeholder, or `_col_<n>` / a literal's text invented by qualify, for an unnamed expression. It is
                 dropped and the output gets the name Postgres would use, unless the query refers to the column by
                 that name (`x.sum`, `SELECT x.*` over a derived table, `ORDER BY 2` over a scalar subquery)."""
    if isinstance(item, exp.Alias):
        if item.meta.get(ORIG_ALIAS):
            return "explicit"
        if item.meta.get(PLACEHOLDER):
            return "generated"
        inner = item.this
        if isinstance(inner, exp.Column) and inner.name == item.alias:
            return "natural"
        if inner.meta.get(UNALIASED) and item.alias.lower() == (inner.output_name or f"_col_{index}").lower():
            return "generated"
        return "explicit"
    return "none"


def _inline_named_windows(select: exp.Select) -> None:
    """`SUM(x) OVER w ... WINDOW w AS (...)` becomes `SUM(x) OVER (...)`, so that a projection carries its own SQL."""
    named = {w.name: w for w in select.args.get("windows") or []}
    if not named:
        return
    for w in named.values():
        if w.args.get("alias"):  # WINDOW w2 AS (w1 ORDER BY ...): inlining w2 alone would drop what w1 says
            raise UnsupportedSQL("named window", f"WINDOW {w.name} extends the window {w.alias}")
    for w in list(select.find_all(exp.Window)):
        if w.find_ancestor(exp.Select) is not select or not w.args.get("alias") or isinstance(w.this, exp.Identifier):
            continue
        definition = named.get(w.alias)
        if definition is None or any(w.args.get(k) for k in ("partition_by", "order", "spec")):
            raise UnsupportedSQL("named window", f"OVER {w.alias} extends or misses its WINDOW definition")
        for key in ("partition_by", "order", "spec"):
            value = definition.args.get(key)
            if value:
                w.set(key, [v.copy() for v in value] if isinstance(value, list) else value.copy())
        w.set("alias", None)
    select.set("windows", None)


def _own_joins(select: exp.Select) -> list[exp.Join]:
    return [j for j in select.find_all(exp.Join) if j.find_ancestor(exp.Select) is select]


def _repair_merged_join_conditions(select: exp.Select) -> None:
    """For `a RIGHT JOIN b USING (x) JOIN c USING (x)` qualify writes the second condition as `a.x = c.x`. The merged
    column of the first join is COALESCE(a.x, b.x), and `a.x` is NULL exactly in the rows a RIGHT join adds. (qualify
    takes the first table's column whenever USING has ONE column; that is right for inner and LEFT joins only.)
    With a RIGHT or FULL join in the FROM clause, rebuild the left side of every USING / NATURAL condition from the
    tables merged so far."""
    joins = _own_joins(select)
    merging = [j for j in joins if j.meta.get(MERGING_JOIN) and j.args.get("on") is not None]
    if not merging or not any((j.side or "").upper() in ("RIGHT", "FULL") for j in joins):
        return
    flat = select.args.get("joins") or []
    if any(all(j is not f for f in flat) for j in merging):
        raise UnsupportedSQL("USING next to a RIGHT or FULL join", "inside a parenthesized join: the merged column is not reconstructed")
    merged: dict[str, list[exp.Identifier]] = {}  # USING column -> the FROM aliases merged so far, in order
    for join in flat:
        if all(join is not m for m in merging):
            continue
        conditions = list(join.args["on"].flatten()) if isinstance(join.args["on"], exp.And) else [join.args["on"]]
        for condition in conditions:
            right = condition.expression if isinstance(condition, exp.EQ) else None
            if not isinstance(right, exp.Column) or right.name == "" or right.table != join.alias_or_name:
                raise UnsupportedSQL("name resolution", f"unexpected USING condition {condition.sql()}")
            left = condition.this
            tables = merged.get(right.name) or [c.args["table"] for c in left.find_all(exp.Column)]
            columns = [exp.Column(this=right.this.copy(), table=t.copy()) for t in tables]
            condition.set("this", columns[0] if len(columns) == 1 else exp.func("coalesce", *columns))
            if not join.is_semi_or_anti_join:
                merged[right.name] = tables + [right.args["table"]]


def _simplify_merged_columns(select: exp.Select) -> None:
    """qualify writes the merged column of `JOIN ... USING (x)` / NATURAL JOIN as COALESCE(a.x, b.x). That is needed
    for RIGHT and FULL joins only: in a FROM clause of inner and LEFT joins the merged column always equals the first
    table's column, so put that back. `SELECT x` then stays what it is, a plain column."""
    if any((j.side or "").upper() in ("RIGHT", "FULL") for j in _own_joins(select)):
        return
    for node in list(select.find_all(exp.Coalesce)):
        if node.meta.get(ORIG_COALESCE) or node.find_ancestor(exp.Select) is not select:
            continue
        args = [node.this, *node.expressions]
        if len(args) >= 2 and all(isinstance(a, exp.Column) and a.table for a in args) \
                and len({a.name for a in args}) == 1 and len({a.table for a in args}) == len(args):
            node.replace(args[0].copy())


def _normalize_order_keys(select: exp.Select) -> None:
    """After qualify an unqualified name in ORDER BY refers to a select item: either the author wrote that name or a
    position, or qualify swapped the name in for an expression that equals a select item (it does that in grouped
    queries). Keep a name only where the author wrote one AND it is an alias the author wrote; otherwise the key is
    the item's expression. So a key never names an alias that is dropped, and an expression stays an expression.
    When two items carry the name, the author's expression (kept from before qualify) says which item was meant, and
    a name the author wrote is ambiguous, as it is for Postgres."""
    for ordered in _order_keys(select):
        key = ordered.this
        if not isinstance(key, exp.Column) or key.table:
            continue
        found = _the_item(select, key, ordered.meta.get(BARE_KEY), ordered.meta.get(ORIG_KEY), "ORDER BY")
        if found is None:
            continue
        index, item = found
        written_as_name = (ordered.meta.get(BARE_KEY) and alias_status(item, index) == "explicit"
                           and _wrote_this_name(ordered.meta.get(ORIG_KEY), key))
        if not written_as_name and isinstance(item, exp.Alias) and not contains_query(item.this):
            key.replace(item.this.copy())


def _distinct_marks(select: exp.Select) -> list[tuple[bool, exp.Expr]]:
    """Per DISTINCT ON key, before qualify: (the author wrote a bare name or a position, a copy of the key). Kept by
    the position of the SELECT in the tree, because qualify rebuilds the Select and Distinct nodes and their meta."""
    return [((isinstance(key, exp.Column) and not key.table) or (isinstance(key, exp.Literal) and key.is_int), key.copy())
            for key in _distinct_on_keys(select)]


def _normalize_distinct_keys(select: exp.Select, marks: list[tuple[bool, exp.Expr]] | None) -> None:
    """The keys of DISTINCT ON, which Postgres resolves like ORDER BY keys (an output name first, a number is a
    position), normalised like `_normalize_order_keys`: qualify swaps an expression that equals a select item for the
    item's name in grouped queries and turns a position into the item's alias (a generated one is dropped again), so
    a name the author did not write becomes the item's expression, the author's copy deciding between two items of
    one name; a name the author wrote that two items carry is ambiguous, as for Postgres."""
    keys = _distinct_on_keys(select)
    if not keys:
        return
    if marks is None or len(marks) != len(keys):
        marks = [(False, None)] * len(keys)
    out = []
    for key, (bare, orig) in zip(keys, marks):
        if isinstance(key, exp.Column) and not key.table:
            found = _the_item(select, key, bare, orig, "DISTINCT ON")
            if found is not None:
                index, item = found
                written_as_name = bare and alias_status(item, index) == "explicit" and _wrote_this_name(orig, key)
                if not written_as_name and isinstance(item, exp.Alias) and not contains_query(item.this):
                    key = item.this.copy()
        out.append(key)
    _set_distinct_on_keys(select, out)


def _the_item(select: exp.Select, key: exp.Column, bare: bool | None, orig: exp.Expr | None, where: str
              ) -> tuple[int, exp.Expr] | None:
    """The select item a bare key names after qualify; None when none carries the name."""
    matches = [(i, item) for i, item in enumerate(select.selects) if item.alias_or_name == key.name]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if bare:
        expr = lambda item: item.this if isinstance(item, exp.Alias) else item  # noqa: E731
        if isinstance(orig, exp.Column) and not orig.table and _folded_key(orig) != key.name:
            # the author wrote an INPUT column's bare name (no output carries it) and qualify swapped it for the name
            # of a select item that equals it; the items that are that column say which
            same = [(i, item) for i, item in matches if isinstance(expr(item), exp.Column) and expr(item).name == _folded_key(orig)]
            if same and all(_same_text(expr(item), expr(same[0][1])) for _, item in same[1:]):
                return same[0]
            raise UnsupportedSQL("name resolution", f"{where} {orig.name} was rewritten to the name {key.name}, which "
                                 f"{len(matches)} outputs carry")
        # Postgres accepts the name when every item that carries it is the same expression (`x AS k, x AS k ... ORDER
        # BY k`) and rejects it as ambiguous otherwise
        first = matches[0][1]
        if all(_same_text(expr(item), expr(first)) for _, item in matches[1:]):
            return matches[0]
        raise UnsupportedSQL("name resolution", f"{where} {key.name} is ambiguous: {len(matches)} outputs carry the name")
    for i, item in matches:
        if orig is not None and _same_text(item.this if isinstance(item, exp.Alias) else item, orig):
            return i, item
    raise UnsupportedSQL("name resolution", f"{where} {orig.sql() if orig is not None else key.name} was rewritten to the "
                         f"name {key.name}, which {len(matches)} outputs carry")


def _wrote_this_name(orig: exp.Expr | None, key: exp.Column) -> bool:
    """The author wrote the bare name the key carries after qualify (not an input column that qualify swapped for
    the name of an item it equals: such a key is written as the item's expression, the name could be ambiguous)."""
    return orig is None or not isinstance(orig, exp.Column) or bool(orig.table) or _folded_key(orig) == key.name


def _same_text(a: exp.Expr, b: exp.Expr) -> bool:
    """The two expressions are the same text once identifiers are compared without quotes and case (qualify quotes
    every identifier; the author's copy is as written)."""

    def plain(e: exp.Expr) -> str:
        e = e.copy()
        for ident in e.find_all(exp.Identifier):
            ident.set("this", ident.name.lower())
            ident.set("quoted", False)
        return e.sql()

    return plain(a) == plain(b)


def _first_select(query: exp.Expr) -> exp.Expr:
    """The SELECT that names the output columns of a query: the first branch of a set operation."""
    while isinstance(query, (exp.SetOperation, exp.Subquery, exp.Paren)):
        query = query.this
    return query


def _referenced_outputs(tree: exp.Expr) -> set[tuple[int, str]]:
    """(id of a SELECT, output name) for every output column that the query refers to BY NAME: `x.name` where x is a
    use of a CTE or derived table, and unqualified ORDER BY keys (of the SELECT itself or of a set operation over it)."""
    found: set[tuple[int, str]] = set()
    for scope in traverse_scope(tree):
        for column in scope.expression.find_all(exp.Column):
            if column.table:
                s: Scope | None = scope
                while s is not None and column.table not in s.sources:
                    s = s.parent
                source = s.sources[column.table] if s is not None else None
                if isinstance(source, Scope):
                    found.add((id(_first_select(source.expression)), column.name))
        order = scope.expression.args.get("order")
        for ordered in order.expressions if order else []:
            if isinstance(ordered.this, exp.Column) and not ordered.this.table:
                found.add((id(_first_select(scope.expression)), ordered.this.name))
        if isinstance(scope.expression, exp.Select):
            for key in _distinct_on_keys(scope.expression):
                if isinstance(key, exp.Column) and not key.table:
                    found.add((id(scope.expression), key.name))
    return found


def _drop_added_aliases(select: exp.Select, referenced: set[tuple[int, str]], dialect: str) -> None:
    """Take the aliases that qualify added back out of the tree (see `alias_status`), so that the SQL printed for a
    subquery or a derived table shows what the author wrote. The output name is left on the item as a mark."""
    for index, item in enumerate(list(select.selects)):
        status = alias_status(item, index)
        keep = status == "explicit" or (status == "generated" and (id(select), item.alias) in referenced)
        if isinstance(item, exp.Alias) and not keep:
            inner = item.this
            inner.meta[OUTPUT_NAME] = item.alias if status == "natural" else pg_default_name(inner, dialect)
            item.replace(inner)
