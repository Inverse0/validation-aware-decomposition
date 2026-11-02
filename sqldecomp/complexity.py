"""Component 3: the HUMAN-facing complexity model F, and the nineteen block features it reads.

    Features(memo, result, blocks)          the feature extractor of ONE query (cached per block class)
    HumanVerifier(...)                      theta: one optional threshold per feature, one beta, the excess mode, ...
    human_scoring(theta, features)          a `Scoring` for `sqldecomp.optimizer.optimize`, F = the model below
    risk(ctx, theta, features)              -log F with a per-feature breakdown (value, threshold, excess, contribution)
    python -m sqldecomp.complexity          the feature vectors of a query's groups / of one block, for inspection

THE MODEL. For a block b with feature
values v_x(b):

    F(theta, b) = prod_x p_x(v_x(b)) * s(b) ** gamma
    p_x(v)      = exp(-beta * excess_x(v))                       (the hinge link, default)
    excess_x(v) = max(0, v - tau_x) / max(tau_x, 1)              ("relative", default; tau_x = 0 degrades to absolute)
                = max(0, v - tau_x)                               ("absolute", an option)

so -log F = beta * sum_x excess_x(v_x(b)) - gamma * log s(b) >= 0, and it is EXACTLY 0 for a block within every
threshold (with gamma = 0): among decompositions whose blocks are all within the thresholds the optimizer decides by
Omega alone. theta = {tau_x: a number or None per feature (None: the feature does not participate, p_x = 1; every
feature is optional in this sense, there is no other tier), beta (one shared decay, default ln 2: at twice the limit
the reliability is halved), the excess mode, the link, gamma (the exponent of the semantic score of component 4,
the LLM judge of `sqldecomp.semantic_llm`: default 0; 0.25 when the score is switched on with `--semantic judge`), mu (Omega(k) = -mu * k
over the number of unique blocks, default 0.001)}. No hard caps, no per-block
reliability floor rho. Interpretation: a conjunctive (non-compensatory) multidimensional item-response model, one
dimension per feature, with a hinge link; `link="logistic"`, p_x = sigmoid(a * (c - excess_x)), is a documented
option so that a later fit from human verification data needs no rewrite (with it a block within every threshold
costs -log sigmoid(a c) per participating feature, not 0). The optimizer never sees the difference: it calls
`Scoring.block_cost` once per class, `Scoring.omega` on the feasible range of unique counts, and
`Scoring.meaningfulness` once per output signature.

THE FEATURES: nineteen registered named functions of the block, integers >= 0, computed from
the block's FRAGMENT (its memo expressions, the representative's, `BlockContext.fragment`), from the SIDE-TABLE TEXTS
of the conjuncts and operators it applies (each text parsed once with SQLGlot and cached by entry id) and from the
boundary information (inputs, parameters, exposed columns), NEVER from a printed block. The counting rules, the
FROM-use rule and the deliberate overlaps are in the docstrings of the feature functions below. The three "both" constructs (a boolean NOT: negation + bool_mix; NOT IN: negation +
null_sensitive; COALESCE / NULLIF / IFNULL / NVL: null_sensitive + expr) and the three aspect overlaps (predicates:
pred counts, negation / null_sensitive / bool_mix classify; subqueries: uncorr / corr count, depth measures,
negation classifies; n_optypes and length overlap everything by design) are the ONLY double counting.

Feature docstrings explain the counting conventions. `length` is counted from
the side-table texts using the printer's keyword, identifier and parenthesis rules.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from sqlglot import exp
from sqlglot.dialects.postgres import Postgres

from sqldecomp.blocks import Blocks, Parameter
from sqldecomp.model import Block, GroupId, ImportResult, Memo, MExpr, Op1, Op2, SideEntry
from sqldecomp.printer import sqltext
from sqldecomp.printer.context import PrintContext
from sqldecomp.printer.errors import PrintError
from sqldecomp.scoring import BlockContext, GroupContext, Scoring

LN2 = math.log(2.0)
_TOKENIZER = Postgres()

# --------------------------------------------------------------------------------------------------------------
# the registry of features
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Feature:
    number: int  # the number in section 3.3 of the design document (1-19)
    name: str
    definition: str  # one line, the counting rule
    fn: Callable[["_Analysis"], int]


FEATURES: dict[str, Feature] = {}
FEATURE_NAMES: list[str] = []  # in the order of section 3.3


def feature(number: int, definition: str):
    """Registers a feature function `fn(analysis) -> int` under its name; the docstring carries the full rule."""

    def register(fn: Callable[["_Analysis"], int]) -> Callable[["_Analysis"], int]:
        f = Feature(number, fn.__name__, definition, fn)
        FEATURES[fn.__name__] = f
        FEATURE_NAMES.append(fn.__name__)
        return fn

    return register


# --------------------------------------------------------------------------------------------------------------
# statistics of one side-table text (parsed once per entry id)
# --------------------------------------------------------------------------------------------------------------

QUERY_NODES = (exp.Subquery, exp.Select, exp.SetOperation)
ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod, exp.IntDiv, exp.BitwiseAnd, exp.BitwiseOr, exp.BitwiseXor,
              exp.BitwiseLeftShift, exp.BitwiseRightShift)
NEGATABLE_PREDICATES = (exp.Like, exp.ILike, exp.SimilarTo, exp.Between)  # SQLGlot may print `x NOT ILIKE y`


@dataclass
class Call:
    """One subquery call found in a text, in written order."""

    kind: str  # exists, not_exists, in_subquery, not_in_subquery, scalar_subquery, quantified_subquery, negated_quantified_subquery


@dataclass
class TextStats:
    tokens: int = 0  # tokens of the text WITHOUT its subquery bodies (the parentheses around a body stay)
    existence_negation: int = 0  # NOT IN (list or subquery), NOT EXISTS, <> ALL, NOT ... ANY
    boolean_not: int = 0  # a boolean NOT on an ordinary predicate (NOT x, NOT LIKE, NOT BETWEEN): negation AND bool_mix
    not_in: int = 0  # NOT IN over a list or a subquery: negation AND null_sensitive
    null_tests: int = 0  # IS [NOT] NULL / UNKNOWN / TRUE / FALSE, IS [NOT] DISTINCT FROM
    null_functions: int = 0  # COALESCE, NULLIF (IFNULL / NVL parse to COALESCE): null_sensitive AND expr
    distinct_in_agg: int = 0  # DISTINCT inside an aggregate call
    ors: int = 0  # OR (and XOR) operators
    bool_depth: int = 0  # the deepest chain of AND / OR / NOT connectives
    expr: int = 0  # non-aggregate function calls, CASE branches, arithmetic, CAST, ||, window calls
    window_calls: int = 0
    calls: list[Call] = field(default_factory=list)
    columns: set[tuple[str, str]] = field(default_factory=set)  # qualified columns outside subqueries (alias, column)
    agg_arg_columns: list[set[tuple[str, str]]] = field(default_factory=list)  # per aggregate call (not windowed)
    unparsed: bool = False  # the text could not be parsed: only the tokens were counted

    def add(self, other: "TextStats") -> None:
        for name in ("tokens", "existence_negation", "boolean_not", "not_in", "null_tests", "null_functions",
                     "distinct_in_agg", "ors", "expr", "window_calls"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.bool_depth = max(self.bool_depth, other.bool_depth)
        self.calls += other.calls
        self.columns |= other.columns
        self.agg_arg_columns += other.agg_arg_columns
        self.unparsed = self.unparsed or other.unparsed


def tokens_of(text: str) -> int:
    """The number of SQL tokens of a text (SQLGlot's Postgres tokenizer; punctuation counts)."""
    if not text:
        return 0
    try:
        return len(_TOKENIZER.tokenize(text))
    except Exception:  # noqa: BLE001 - an unreadable text: whitespace-separated words
        return len(text.split())


def _strip_parens(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _is_query(node: exp.Expression | None) -> bool:
    return isinstance(node, QUERY_NODES)


def _quantifier(node: exp.Expression | None) -> tuple[str, exp.Expression | None] | None:
    """("ANY" | "ALL", the operand) for a quantified comparison's right side: an Any / All node, or the Anonymous
    function ANY / ALL / SOME that SQLGlot's own text `x <> ALL(ARRAY[1, 2])` re-parses to."""
    if isinstance(node, exp.Any):
        return "ANY", node.this
    if isinstance(node, exp.All):
        return "ALL", node.this
    if isinstance(node, exp.Anonymous) and str(node.this).upper() in ("ANY", "ALL", "SOME"):
        args = node.expressions
        return ("ALL" if str(node.this).upper() == "ALL" else "ANY"), (args[0] if len(args) == 1 else None)
    return None


class _TextWalker:
    """Walks one parsed text and counts the constructs. Subquery bodies are OPAQUE: a body is a scope of its own whose
    conjuncts and operators are counted through the block's fragment when the body lies inside the block (a call is
    recorded here, its kind decided by the node around it)."""

    def __init__(self) -> None:
        self.s = TextStats()
        self.skipped: list[exp.Expression] = []  # query nodes not descended into
        self._agg_columns: list[set[tuple[str, str]]] | None = None  # the column sets being collected for agg calls

    # -- the connective chain depth --------------------------------------------------------------------------

    def _bool(self, depth: int) -> int:
        depth += 1
        self.s.bool_depth = max(self.s.bool_depth, depth)
        return depth

    def _call(self, kind: str, body: exp.Expression | None) -> None:
        self.s.calls.append(Call(kind))
        if body is not None:
            self.skipped.append(body)

    def visit(self, node: exp.Expression | None, depth: int = 0) -> None:
        if node is None:
            return
        s = self.s
        if isinstance(node, exp.Not):
            inner = _strip_parens(node.this)
            parenthesised = isinstance(node.this, exp.Paren)
            if isinstance(inner, exp.Exists):
                s.existence_negation += 1
                self._call("not_exists", inner.this)
                return
            if isinstance(inner, exp.In):
                self._not_in(inner, depth)
                return
            q = _quantifier(inner.expression) if isinstance(inner, exp.Binary) else None
            if q is not None and q[0] == "ANY":
                s.existence_negation += 1  # NOT (x = ANY (...)): the "NOT ... ANY" form of NOT IN, an existence negation
                self._quantified(q[1], "negated_quantified_subquery", depth)
                self.visit(inner.this, depth)
                return
            if isinstance(inner, exp.Is) and not parenthesised and isinstance(inner.expression, exp.Boolean):
                self.visit(inner, depth)  # `NOT x IS TRUE` is SQLGlot's spelling of `x IS NOT TRUE`: one null test
                return
            # a boolean NOT on an ordinary predicate (NOT x, NOT LIKE, NOT BETWEEN, NOT (a AND b), NOT (x IS NULL),
            # NOT (x > ALL (...))); what it negates is walked as usual (a `<> ALL` inside adds its own negation)
            s.boolean_not += 1
            self.visit(inner, self._bool(depth))
            return
        if isinstance(node, exp.In):
            if node.args.get("negate"):
                self._not_in(node, depth)
                return
            self.visit(node.this, depth)
            query = node.args.get("query")
            if query is not None:
                self._call("in_subquery", query)
            for e in node.expressions:
                self.visit(e, depth)
            return
        if isinstance(node, exp.Exists):
            self._call("exists", node.this)
            return
        q = _quantifier(node.expression) if isinstance(node, exp.Binary) else None
        if q is not None:
            if isinstance(node, exp.NEQ) and q[0] == "ALL":
                s.existence_negation += 1  # x <> ALL (...) is NOT IN by another name (over a subquery or an array)
                self._quantified(q[1], "negated_quantified_subquery", depth)
            else:
                self._quantified(q[1], "quantified_subquery", depth)
            self.visit(node.this, depth)
            return
        q = _quantifier(node)
        if q is not None:  # an ANY / ALL that is not the right side of a comparison
            self._quantified(q[1], "quantified_subquery", depth)
            return
        if _is_query(node):
            self._call("scalar_subquery", node)
            return
        if isinstance(node, NEGATABLE_PREDICATES) and node.args.get("negate"):
            s.boolean_not += 1
            depth = self._bool(depth)
        if isinstance(node, exp.Is):
            s.null_tests += 1
        elif isinstance(node, (exp.NullSafeEQ, exp.NullSafeNEQ)):
            s.null_tests += 1
        elif isinstance(node, (exp.Coalesce, exp.Nullif)):
            s.null_functions += 1
            s.expr += 1
        elif isinstance(node, exp.Case):
            s.expr += len(node.args.get("ifs") or ()) + (1 if node.args.get("default") is not None else 0)
            self.visit(node.this, depth)
            for arm in node.args.get("ifs") or ():
                self.visit(arm.this, depth)
                self.visit(arm.args.get("true"), depth)
            self.visit(node.args.get("default"), depth)
            return
        elif isinstance(node, exp.Window):
            s.expr += 1
            s.window_calls += 1
            fn = node.this
            if isinstance(fn, exp.Filter):  # SUM(x) FILTER (WHERE ...) OVER (...): a windowed call all the same
                self.visit(fn.expression, depth)
                fn = fn.this
            if isinstance(fn, exp.Func):  # the windowed function's arguments, the call itself counted once above
                for child in fn.iter_expressions():
                    self.visit(child, depth)
            else:
                self.visit(fn, depth)
            for key in ("partition_by", "order", "spec"):
                value = node.args.get(key)
                if isinstance(value, list):
                    for v in value:
                        self.visit(v, depth)
                else:
                    self.visit(value, depth)
            return
        elif isinstance(node, exp.Filter):  # COUNT(*) FILTER (WHERE ...): the predicate is part of the call's argument
            self._aggregate_call([node.this, node.expression], depth)
            return
        elif isinstance(node, exp.WithinGroup):  # PERCENTILE_CONT(...) WITHIN GROUP (ORDER BY x): x is the argument
            self._aggregate_call([node.this, node.expression], depth)
            return
        elif isinstance(node, exp.AggFunc):  # an aggregate call: not counted (agg owns it); its argument is walked
            self._aggregate_call([node], depth)
            return
        elif isinstance(node, exp.Distinct):
            s.distinct_in_agg += 1
        elif isinstance(node, exp.Connector):  # And / Or / Xor (Func subclasses in SQLGlot: handled before Func)
            if isinstance(node, (exp.Or, exp.Xor)):
                s.ors += 1
            depth = self._bool(depth)
        elif isinstance(node, exp.Func):
            s.expr += 1  # a scalar function call, CAST, CONCAT, EXTRACT, CURRENT_DATE, a standalone IF, ...
        elif isinstance(node, ARITHMETIC) or isinstance(node, (exp.DPipe, exp.Neg)):
            s.expr += 1
        elif isinstance(node, exp.Column):
            table = node.args.get("table")
            if table is not None:
                key = (sqltext.ident_key(table), sqltext.ident_key(node.this) if isinstance(node.this, exp.Identifier) else "*")
                s.columns.add(key)
                if self._agg_columns is not None:
                    self._agg_columns[0].add(key)
        for child in node.iter_expressions():
            self.visit(child, depth)

    def _quantified(self, operand: exp.Expression | None, kind: str, depth: int) -> None:
        """The operand of a quantifier: a subquery is a call (opaque), an array or a list is walked."""
        inner = _strip_parens(operand) if operand is not None else None
        if _is_query(inner):
            self._call(kind, inner)
        else:
            self.visit(operand, depth)

    def _aggregate_call(self, parts: list[exp.Expression | None], depth: int) -> None:
        """One aggregate call (the call itself is the aggregate operator's, feature 14): the columns of everything
        under it (its arguments, a FILTER predicate, a WITHIN GROUP key) are recorded for the outer-join rule. A
        FILTER around a WITHIN GROUP (or the other way round) is still ONE call."""
        columns: set[tuple[str, str]] = set()
        outer = self._agg_columns
        self._agg_columns = [columns]
        todo = list(parts)
        while todo:
            part = todo.pop(0)
            if isinstance(part, (exp.Filter, exp.WithinGroup)):
                todo = [part.this, part.expression] + todo
            elif isinstance(part, exp.AggFunc):
                for child in part.iter_expressions():
                    self.visit(child, depth)
            else:
                self.visit(part, depth)
        self._agg_columns = outer
        if outer is not None:
            outer[0] |= columns
        self.s.agg_arg_columns.append(columns)

    def _not_in(self, node: exp.In, depth: int) -> None:
        s = self.s
        s.existence_negation += 1
        s.not_in += 1
        self.visit(node.this, depth)
        query = node.args.get("query")
        if query is not None:
            self._call("not_in_subquery", query)
        for e in node.expressions:
            self.visit(e, depth)


def text_stats(text: str, role: str) -> TextStats:
    """The statistics of one side-table text. `role`: predicate (a conjunct, an ON conjunct), item (a select item, a
    DISTINCT ON expression, a LIMIT / OFFSET), group (a GROUP BY item, which may be ROLLUP (...)), order (an ORDER
    BY key, possibly a bare output name)."""
    walker = _TextWalker()
    try:
        if role == "predicate":
            tree = sqltext.parse_predicate(text)
        elif role == "group":
            tree = sqltext.parse_group_item(text)
        elif role == "values":
            tree = sqltext.parse_from_item(text)
        else:
            tree = sqltext.parse_item(text)
    except PrintError:
        s = TextStats(tokens=tokens_of(text), unparsed=True)
        return s
    if isinstance(tree, exp.Group):
        for e in tree.expressions:
            walker.visit(e)
    elif role == "values":  # `(VALUES (1 + 2, COALESCE(NULL, 3))) AS v(x, y)`: the rows' expressions
        for row in tree.find_all(exp.Tuple):
            for e in row.expressions:
                walker.visit(e)
    else:
        walker.visit(tree)
    s = walker.s
    s.tokens = max(0, tokens_of(text) - sum(tokens_of(_bare_query(q).sql(dialect="postgres")) for q in walker.skipped))
    return s


def _bare_query(q: exp.Expression) -> exp.Expression:
    """A subquery body without the parentheses around it: `x > (SELECT ...)` keeps its two parentheses in the text's
    count, and the body is counted through its own groups (or as a call of `b_<id>` when it is an input)."""
    while isinstance(q, (exp.Subquery, exp.Paren)) and isinstance(q.this, exp.Expression):
        q = q.this
    return q


@dataclass
class EntryStats:
    """The statistics of one side entry (all of its texts), plus what the entry is."""

    kind: str
    stats: TextStats  # merged over the entry's texts
    texts: list[TextStats] = field(default_factory=list)  # per text, in `entry_texts` order
    conjuncts: int = 0  # 1 for a conjunct entry; the number of ON conjuncts of an outer join; 0 otherwise
    order_keys: int = 0
    select_items: int = 0
    aliased_items: int = 0
    group_items: int = 0
    subqueries: tuple[str, ...] = ()  # the scope ids the entry names, in written order
    has_offset: bool = False
    has_limit: bool = False
    distinct_on: int = 0
    has_window: bool = False
    grouping_sets: int = 0  # GROUP BY items that are ROLLUP / CUBE / GROUPING SETS (MySQL's `WITH ROLLUP` included)
    group_modifiers: int = 0  # `WITH ROLLUP` / `WITH CUBE` after the items: no item, no comma
    window_over_aggregate: bool = False  # a window projection over an aggregate, reduced to its window items


def entry_texts(entry: SideEntry) -> list[tuple[str, str]]:
    """(text, role) for every SQL text a side entry carries. A scan, a ref and a set operation carry none."""
    d = entry.data
    if entry.kind == "conjunct":
        return [(entry.sql, "predicate")]
    if entry.kind == "scan":
        return [(entry.sql, "values")] if d.get("columns") and entry.sql else []
    if entry.kind == "outer_join":
        return [(t, "predicate") for t in d.get("on") or ()]
    out: list[tuple[str, str]] = []
    if entry.kind in ("aggregate", "project"):
        out += [(i.get("sql", ""), "item") for i in d.get("select") or ()]
        out += [(g, "group") for g in d.get("group_by") or ()]
    elif entry.kind == "order":
        out += [(k.get("sql", ""), "order") for k in d.get("keys") or ()]
    elif entry.kind == "limit":
        out += [(t, "item") for t in (d.get("limit"), d.get("offset")) if t]
    elif entry.kind == "distinct":
        out += [(t, "item") for t in d.get("on") or ()]
    return out


def window_items_only(entry: SideEntry, es: EntryStats) -> EntryStats:
    """A window projection over an aggregate: the importer copies the aggregate's select items into it (only the
    window items are new), so the aggregate's items are counted through the aggregate entry and the projection
    contributes its window items alone."""
    items = entry.data.get("select") or ()
    keep = [k for k, ts in enumerate(es.texts) if k < len(items) and ts.window_calls > 0]
    merged = TextStats()
    for k in keep:
        merged.add(es.texts[k])
    # the entry's scopes are listed in written order, one per call: the calls of text k take the next len(calls) scopes
    scopes: list[str] = []
    offset = 0
    for k, ts in enumerate(es.texts):
        n = len(ts.calls)
        if k in keep:
            scopes += list(es.subqueries[offset:offset + n])
        offset += n
    out = EntryStats(es.kind, merged, [es.texts[k] for k in keep], subqueries=tuple(scopes),
                     select_items=len(keep), aliased_items=sum(1 for k in keep if items[k].get("alias")), has_window=True,
                     window_over_aggregate=True)
    return out


def entry_stats(entry: SideEntry) -> EntryStats:
    merged = TextStats()
    texts = []
    for text, role in entry_texts(entry):
        ts = text_stats(text, role)
        texts.append(ts)
        merged.add(ts)
    d = entry.data
    es = EntryStats(entry.kind, merged, texts, subqueries=tuple(str(s) for s in d.get("subqueries") or ()))
    if entry.kind == "conjunct":
        es.conjuncts = 1
    elif entry.kind == "outer_join":
        es.conjuncts = len(d.get("on") or ())
    elif entry.kind == "order":
        es.order_keys = len(d.get("keys") or ())
    elif entry.kind in ("aggregate", "project"):
        items = d.get("select") or ()
        es.select_items = len(items)
        es.aliased_items = sum(1 for i in items if i.get("alias"))
        groups = [str(g).upper() for g in d.get("group_by") or ()]
        es.group_items = len(groups)
        es.group_modifiers = sum(1 for g in groups if g.startswith("WITH "))  # `GROUP BY a, b WITH ROLLUP`
        es.has_window = bool(d.get("has_window"))
        es.grouping_sets = sum(1 for g in groups if g.startswith(("ROLLUP", "CUBE", "GROUPING SETS", "WITH ROLLUP", "WITH CUBE")))
    elif entry.kind == "limit":
        es.has_limit, es.has_offset = bool(d.get("limit")), bool(d.get("offset"))
    elif entry.kind == "distinct":
        es.distinct_on = len(d.get("on") or ())
    return es


# --------------------------------------------------------------------------------------------------------------
# the analysis of one block: everything the feature functions read
# --------------------------------------------------------------------------------------------------------------

OPTYPE_OF_SETOP = {"UNION": "union", "UNION ALL": "union_all", "INTERSECT": "intersect", "INTERSECT ALL": "intersect_all",
                   "EXCEPT": "except", "EXCEPT ALL": "except_all"}
OPTYPE_OF_JOIN = {"LEFT": "left_join", "RIGHT": "right_join", "FULL": "full_join", "SEMI": "semi_join", "ANTI": "anti_join"}
NULL_EXTENDING = {"LEFT", "RIGHT", "FULL"}

# The fixed alphabet of `n_optypes` (feature 16): plan operators plus subquery calls by kind. Scans and refs are not
# operators; expression-level constructs (CASE, functions) are not either.
OPTYPE_ALPHABET = (
    "join", "cross_join", "left_join", "right_join", "full_join", "semi_join", "anti_join", "filter", "having",
    "aggregate", "project", "window", "distinct", "distinct_on", "order", "limit", "offset",
    "union", "union_all", "intersect", "intersect_all", "except", "except_all",
    "exists", "not_exists", "in_subquery", "not_in_subquery", "scalar_subquery", "quantified_subquery",
    "negated_quantified_subquery", "subquery")


@dataclass
class _CallInfo:
    entry: str
    scope: str
    body: GroupId | None
    kind: str
    is_input: bool
    correlated: bool  # with the block's own scopes


class _Analysis:
    """One pass over a block's fragment; the feature functions read its fields."""

    def __init__(self, features: "Features", block: Block, fragment: dict[GroupId, MExpr],
                 parameters: Iterable[Parameter], exposed: Iterable[str] = (), star: bool = False,
                 nullable_inputs: Iterable[tuple[str, str]] = ()):
        self.features = features
        self.blocks = features.blocks
        self.side = features.side
        self.block = block
        self.inputs = frozenset(block.inputs)
        # (occurrence, column) pairs that an INPUT exposes from the NULL-extended side of an outer join inside it
        # (`ExposedColumn.nullable`, CPX-O2): a predicate or an aggregate argument of this block that reads one is
        # NULL-sensitive
        self.nullable_inputs = frozenset(nullable_inputs)
        self.fragment = fragment
        self.parameters = list(parameters)
        self.exposed = tuple(exposed)  # per column the block exposes at a boundary inside a FROM clause: its occurrence
        self.exposed_columns = len(self.exposed)
        self.star = star  # the boundary exposes every column (`SELECT *`): nothing is read by name
        self.notes: list[str] = []
        blocks = self.blocks
        # the expressions, by kind
        self.joins: list[tuple[GroupId, MExpr]] = []
        self.filters: list[tuple[GroupId, MExpr]] = []
        self.op1: list[tuple[GroupId, MExpr, SideEntry]] = []
        self.op2: list[tuple[GroupId, MExpr, SideEntry]] = []
        self.scans: list[tuple[GroupId, MExpr, SideEntry]] = []
        self.refs: list[tuple[GroupId, MExpr, SideEntry]] = []
        for g, e in fragment.items():
            if e.kind == "join":
                self.joins.append((g, e))
            elif e.kind == "filter":
                self.filters.append((g, e))
            elif e.kind == "scan":
                self.scans.append((g, e, self.side[e.arg]))
            elif e.kind == "op1":
                entry = self.side[e.arg]
                (self.refs if entry.kind == "ref" else self.op1).append((g, e, entry))
            else:
                self.op2.append((g, e, self.side[e.arg]))
        # every entry the fragment applies, with the group applying it; a VALUES scan counts its rows' expressions
        self.applied: list[tuple[GroupId, str, EntryStats]] = []
        seen: set[str] = set()
        aggregate_scopes = {blocks.owner_scope(e.arg) for g, e in fragment.items() if e.kind == "op1" and self.side[e.arg].kind == "aggregate"}
        for g, e in fragment.items():
            ids = blocks.applied(e) or ((e.arg,) if e.kind == "scan" and self.side[e.arg].data.get("columns") else ())
            for id_ in ids:
                if id_ in seen:
                    continue  # a CTE body under two refs is one group: already listed
                seen.add(id_)
                es = features.entry_stats(id_)
                if es.kind == "project" and es.has_window and blocks.owner_scope(id_) in aggregate_scopes:
                    es = features.window_only_stats(id_)
                self.applied.append((g, id_, es))
        # the subquery calls, with what they call
        self.calls: list[_CallInfo] = []
        for g, id_, es in self.applied:
            kinds = [c.kind for c in es.stats.calls]
            if len(kinds) != len(es.subqueries):
                self.notes.append(f"entry {id_}: {len(kinds)} calls found in the text, {len(es.subqueries)} scopes listed")
            for k, sid in enumerate(es.subqueries):
                body = features.scope_group(sid)
                kind = kinds[k] if k < len(kinds) else "subquery"
                is_input = body in self.inputs
                self.calls.append(_CallInfo(id_, sid, body, kind, is_input, self._correlated_with_block(body)))
        # the scope of each fragment group's expression (for the outer-join rule)
        self._below_cache: dict[GroupId, frozenset[GroupId]] = {}
        self.memo: dict[str, Any] = {}  # results of the helpers that several features share

    # -- helpers -----------------------------------------------------------------------------------------------

    def _correlated_with_block(self, body: GroupId | None) -> bool:
        """Does the body read a column of the block's own scopes (an occurrence inside the block, outside the body)?"""
        if body is None:
            return False
        own = self.blocks.occurrences_under(self.block.output) - self.blocks.occurrences_under(body)
        return any(p.occurrence in own for p in self.features.body_parameters(body))

    def scope_of_group(self, g: GroupId) -> str | None:
        return self.features.scope_of_group(g)

    def below(self, g: GroupId) -> frozenset[GroupId]:
        """Groups strictly below g in the fragment through memo children, not entering another scope (a ref's body)
        and not following subquery links."""
        if g in self._below_cache:
            return self._below_cache[g]
        self._below_cache[g] = frozenset()
        e = self.fragment.get(g)
        out: set[GroupId] = set()
        if e is not None and not (e.kind == "op1" and self.side[e.arg].kind == "ref"):
            for c in e.children:
                out.add(c)
                out |= self.below(c)
        self._below_cache[g] = frozenset(out)
        return self._below_cache[g]

    def under_inputs(self) -> frozenset[str]:
        """The occurrences inside the block's inputs and not in the block itself (a CTE body can be both: written in
        the block and used again below an input). A column of one of them is read under the name the input gives it
        (`b_<id>.<alias>__<column>` for an input cut inside a FROM clause)."""
        found = self.memo.get("under_inputs")
        if found is None:
            out: set[str] = set()
            for i in self.inputs:
                out |= self.blocks.occurrences_under(i)
            own = {entry.id for _, _, entry in self.scans} | {entry.id for _, _, entry in self.refs}
            found = self.memo["under_inputs"] = frozenset(out - own)
        return found

    def group_applying(self, entry_id: str) -> GroupId | None:
        for g, id_, _ in self.applied:
            if id_ == entry_id:
                return g
        return None

    def climb_to_scope(self, g: GroupId, target: str) -> GroupId | None:
        """From a fragment group of a nested scope, the fragment group in scope `target` under which the nested scope
        hangs (through the consumer of a subquery body or the ref of a CTE / derived table); None if not found."""
        seen: set[GroupId] = set()
        while g is not None and g not in seen:
            seen.add(g)
            sid = self.scope_of_group(g)
            if sid == target:
                return g
            if sid is None or sid not in self.blocks.scopes:
                return None
            scope = self.blocks.scopes[sid]
            top = self.features.scope_group(sid)
            if scope.kind == "subquery":  # the consumer: the expression applying the entry that names the scope
                found = None
                for gg, id_, es in self.applied:
                    if sid in es.subqueries:
                        found = gg
                        break
                g = found
            else:  # a CTE, derived table or set-operation branch: the ref / set operation above its top
                found = None
                for gg, e in self.fragment.items():
                    if top in e.children and gg != top:
                        found = gg
                        break
                g = found
        return None

    def null_extended(self, g: GroupId, e: MExpr, entry: SideEntry) -> frozenset[str]:
        """The occurrences (scans and refs of the join's own scope) that the outer join NULL-extends."""
        return self.features.null_extended(g, e, entry)


# --------------------------------------------------------------------------------------------------------------
# the nineteen features (section 3.3)
# --------------------------------------------------------------------------------------------------------------


@feature(1, "inner join nodes in the fragment (a region with m inputs has m - 1; a cross join the query wrote counts)")
def inner_join(a: _Analysis) -> int:
    """1. `inner_join`: the number of inner Join nodes in the fragment. Every Join is binary, so a region with m inputs
    has m - 1 of them. A cross join that the query itself contains (a Join without a condition) is an inner join
    too (INTERPRETATION: the document says "inner join nodes"; a cross product is one)."""
    return len(a.joins)


@feature(2, "Op2 outer-join nodes of kind LEFT / RIGHT / FULL")
def outer_join(a: _Analysis) -> int:
    """2. `outer_join`: outer-join nodes with join_type LEFT, RIGHT or FULL only. SEMI and ANTI joins (which the
    importer produces for the SEMI / ANTI JOIN syntax only, never for Postgres text) are not outer joins; an ANTI
    join counts under `negation` (3), a SEMI join under `n_optypes` and `pred` only."""
    return sum(1 for _, _, entry in a.op2 if entry.kind == "outer_join" and entry.data.get("join_type") in NULL_EXTENDING)


@feature(3, "NOT IN, NOT EXISTS, <> ALL / NOT ... ANY, ANTI joins, and boolean NOTs on ordinary predicates (both with 8)")
def negation(a: _Analysis) -> int:
    """3. `negation`: NOT IN (over a subquery or a list) ["both" with 4], NOT EXISTS, `<> ALL` and `NOT ... ANY`
    forms, ANTI-join nodes if the importer ever produces them, AND a boolean NOT applied to an ordinary predicate
    (NOT x, NOT LIKE, NOT BETWEEN, NOT (a AND b), NOT (x IS NULL), NOT (x > ALL (...))) ["both" with 8]; what a
    boolean NOT negates is counted as usual, so NOT (x <> ALL (...)) is two negations and NOT (NOT EXISTS ...) too.
    Positive EXISTS / IN / ANY / ALL are calls (18 / 19), not negation; EXCEPT is a set operation (7).
    INTERPRETATION: `IS NOT NULL`, `IS NOT TRUE` (which SQLGlot spells `NOT x IS TRUE`, without parentheses), `<>`
    and `!=` are single predicates, not negations (`IS NOT ...` counts under 4); an explicit `NOT (x IS NULL)` is a
    boolean NOT around a null test; a NOT anywhere in the block's texts counts, a CASE condition or a boolean select
    item included, so that the "both" rule with `bool_mix` holds everywhere."""
    n = sum(es.stats.existence_negation + es.stats.boolean_not for _, _, es in a.applied)
    n += sum(1 for _, _, entry in a.op2 if entry.kind == "outer_join" and entry.data.get("join_type") == "ANTI")
    return n


@feature(4, "IS [NOT] NULL, COALESCE / NULLIF / IFNULL / NVL (both with 9), NOT IN (both with 3), a NULL-extended column used above its outer join in a predicate or an aggregate argument")
def null_sensitive(a: _Analysis) -> int:
    """4. `null_sensitive`: IS NULL / IS NOT NULL (INTERPRETATION: IS [NOT] UNKNOWN / TRUE / FALSE and IS [NOT]
    DISTINCT FROM as well: three-valued tests whose value on NULL is the point), COALESCE, NULLIF, IFNULL / NVL
    (SQLGlot parses the last two to COALESCE) ["both" with 9], NOT IN over a subquery or a list ["both" with 3],
    the IS NULL test inside a CASE (the CASE's branches are 9), and a NULL-extended column of an outer join used
    ABOVE that join in a WHERE / HAVING / join predicate or in an aggregate call's argument (once per predicate,
    once per aggregate call; merely selecting the column does not count; a use through the output of a derived
    table or CTE is not traced). No schema nullability exists and none is used."""
    n = sum(es.stats.null_tests + es.stats.null_functions + es.stats.not_in for _, _, es in a.applied)
    return n + _outer_join_uses(a)


def _outer_join_uses(a: _Analysis) -> int:
    """Predicates and aggregate calls above an outer join that read a column it NULL-extends."""
    if "outer_join_uses" not in a.memo:
        a.memo["outer_join_uses"] = _outer_join_uses_compute(a)
    return a.memo["outer_join_uses"]


def _outer_join_uses_compute(a: _Analysis) -> int:
    joins = [(g, a.null_extended(g, e, entry)) for g, e, entry in a.op2 if entry.kind == "outer_join"]
    joins = [(g, n) for g, n in joins if n]
    if not joins and not a.nullable_inputs:
        return 0
    count = 0
    f = a.features
    for g, id_, es in a.applied:
        kind = es.kind
        if kind not in ("conjunct", "aggregate", "project", "outer_join", "order", "distinct"):
            continue
        variant = "w" if es.window_over_aggregate else ""
        if kind in ("conjunct", "outer_join"):  # per predicate: a conjunct, or each ON conjunct of a further outer join
            texts = [(f.resolved(id_, (variant, "t", k), ts.columns), ts.columns) for k, ts in enumerate(es.texts)]
        else:  # per aggregate call
            texts = [(f.resolved(id_, (variant, "a", k), columns), columns) for k, columns in enumerate(es.stats.agg_arg_columns)]
        above: dict[GroupId, bool] = {}
        for occs, columns in texts:
            if not occs:
                continue
            counted = False
            for oj, n in joins:  # the text reads a column the join NULL-extends, and is applied above that join
                if oj != g and occs & n:
                    if oj not in above:
                        above[oj] = _above(a, g, oj)
                    if above[oj]:
                        count += 1
                        counted = True
                        break
            if counted or not a.nullable_inputs:
                continue
            # the join lies inside an INPUT: the text reads a column that input exposes as nullable (CPX-O2)
            if f.nullable_read(id_, columns, a.nullable_inputs):
                count += 1
    return count


def _above(a: _Analysis, g: GroupId, oj: GroupId) -> bool:
    """Is the expression at fragment group g applied above the outer join at group oj? In the join's own scope: the
    join lies below g. In a nested scope (a subquery body, a derived table): the point where the nested scope hangs
    in the join's scope lies above the join, and is not the join itself (a body inside the join's ON list)."""
    target = a.scope_of_group(oj)
    if a.scope_of_group(g) == target:
        return oj in a.below(g)
    hook = a.climb_to_scope(g, target)
    return hook is not None and hook != oj and oj in a.below(hook)


@feature(5, "DISTINCT operators, DISTINCT ON, DISTINCT inside an aggregate call")
def distinct(a: _Analysis) -> int:
    """5. `distinct`: DISTINCT operators (a `distinct` node, DISTINCT ON included) and DISTINCT inside an aggregate
    call (COUNT(DISTINCT x)). UNION's implicit deduplication is 7 only."""
    return sum(1 for _, _, entry in a.op1 if entry.kind == "distinct") + sum(es.stats.distinct_in_agg for _, _, es in a.applied)


def _relation_uses(a: _Analysis) -> tuple[dict[tuple, int], set[str]]:
    """FROM uses per relation (the FROM-use rule of section 3.2) and the distinct base tables read anywhere inside
    the block. A relation is a base table (by name), a block input (by group), or an in-block CTE (by its declaring
    scope and name) or derived table (by its body scope: one FROM item, one use). A ref to a block input is a use of
    that input; an input reached as a region leaf or through a subquery call is one use."""
    if "relation_uses" not in a.memo:
        a.memo["relation_uses"] = _relation_uses_compute(a)
    return a.memo["relation_uses"]


def _relation_uses_compute(a: _Analysis) -> tuple[dict[tuple, int], set[str]]:
    uses: dict[tuple, int] = {}
    tables: set[str] = set()
    for _, _, entry in a.scans:
        table = entry.data.get("table")
        if table:  # `$dual` / `$values` are no tables
            name = str(table).lower()
            tables.add(name)
            uses[("table", name)] = uses.get(("table", name), 0) + 1
    referenced_inputs: set[GroupId] = set()
    for _, e, entry in a.refs:
        child = e.children[0]
        if child in a.inputs:
            key = ("input", child)
            referenced_inputs.add(child)
        elif entry.data.get("name"):
            key = ("cte", str(entry.data.get("declared_in")), str(entry.data.get("name")).lower())
        else:
            key = ("derived", str(entry.data.get("scope")))
        uses[key] = uses.get(key, 0) + 1
    for i in sorted(a.inputs - referenced_inputs):
        uses[("input", i)] = 1
    return uses, tables


@feature(6, "sum over relations (base table, block input, in-block CTE / derived table by name) of FROM uses - 1")
def table_reuse(a: _Analysis) -> int:
    """6. `table_reuse`: the sum over relations of (FROM uses - 1); a self-join of a base table, a block input read
    twice or an in-block CTE referenced twice gives >= 1 (a CTE body used twice inside the block is one memo group but
    two FROM uses of that name); a derived table is one FROM item and never reused."""
    uses, _ = _relation_uses(a)
    return sum(n - 1 for n in uses.values())


@feature(7, "UNION, UNION ALL, INTERSECT, EXCEPT nodes in the fragment (EXCEPT counts here only)")
def setop(a: _Analysis) -> int:
    """7. `setop`: set-operation nodes in the fragment. EXCEPT is not a negation and UNION's deduplication is no
    DISTINCT: both count here only."""
    return sum(1 for _, _, entry in a.op2 if entry.kind == "setop")


@feature(8, "OR operators plus boolean NOTs on ordinary predicates inside the block's predicates (both with 3)")
def bool_mix(a: _Analysis) -> int:
    """8. `bool_mix`: the boolean structure inside conjuncts (the importer already splits every top-level AND): the
    number of OR operators plus boolean NOTs on ordinary predicates ["both" with 3]. The maximum AND / OR / NOT
    nesting depth is reported in the breakdown (`bool_depth`). INTERPRETATION: counted over every text of the
    block where a predicate can stand (conjuncts, ON lists, and a CASE condition, FILTER clause or boolean-valued
    item inside a select list or key), the same scope as 3's boolean NOT, so that the "both" rule holds exactly."""
    return sum(es.stats.ors + es.stats.boolean_not for _, _, es in a.applied)


@feature(9, "non-aggregate function calls (COALESCE & co. both with 4), CASE branches, arithmetic, CAST / ::, ||, window calls; aggregate calls are not counted")
def expr(a: _Analysis) -> int:
    """9. `expr`: scalar expression complexity in select items, conjuncts, GROUP BY / ORDER BY keys, DISTINCT ON and
    LIMIT expressions, inside an aggregate's argument included: non-aggregate function calls INCLUDING COALESCE /
    NULLIF / IFNULL / NVL ["both" with 4], CASE branches (one per WHEN arm and one for an ELSE), arithmetic
    operators (+ - * / % unary minus, bit operators), CAST and `::`, string concatenation (each `||`, a CONCAT
    call), and a window function call (RANK() OVER (...), SUM(x) OVER (...): one per call; INTERPRETATION: the
    windowed function is no aggregate operator). Aggregate calls themselves are not counted (they belong to 14);
    comparisons, IN lists, BETWEEN, LIKE and literals are not expressions of this kind."""
    return sum(es.stats.expr for _, _, es in a.applied)


@feature(10, "ORDER BY and LIMIT / OFFSET operators in the fragment (key count in the breakdown)")
def order_limit(a: _Analysis) -> int:
    """10. `order_limit`: the number of ORDER BY and LIMIT / OFFSET operators (a LIMIT with an OFFSET is one
    operator); the number of ORDER BY keys is reported in the breakdown, key expressions count under 9."""
    return sum(1 for _, _, entry in a.op1 if entry.kind in ("order", "limit"))


@feature(11, "distinct base tables read anywhere inside the block + distinct block inputs referenced")
def n_relations(a: _Analysis) -> int:
    """11. `n_relations`: the distinct base tables read anywhere inside the block (inside in-block CTE / derived-table
    / subquery bodies included) plus the distinct block inputs referenced. In-block CTEs and derived tables are
    transparent (their base tables count, they do not); a CTE that is a block input counts ONCE however often it is
    referenced; `$dual` / `$values` pseudo scans are no tables."""
    _, tables = _relation_uses(a)
    return len(tables) + len(a.inputs)


@feature(12, "filter and join conjuncts applied in the fragment, HAVING and outer-join ON conjuncts included")
def pred(a: _Analysis) -> int:
    """12. `pred`: the conjuncts the fragment applies: WHERE filters, join conjuncts, HAVING filters, and the ON
    conjuncts of outer / semi / anti joins. The COUNT aspect of predicates; 3, 4 and 8 classify the same conjuncts."""
    n = len(a.filters) + sum(len(e.conjuncts) for _, e in a.joins)
    n += sum(len(entry.data.get("on") or ()) for _, _, entry in a.op2 if entry.kind == "outer_join")
    return n


def _nesting(a: _Analysis, derived_tables_nest: bool) -> int:
    """The subquery nesting depth of the fragment: a subquery call contributes 1 + its body's depth when the body
    lies inside the block, and exactly 1 when the body is a block input (the reader still shows a nested SELECT; the
    body's own nesting is the input's: design decision CPX-O13). A CTE body reached through a ref contributes
    its own depth and no level of its own; a derived table's body likewise, unless `derived_tables_nest` (the
    breakdown's `scope_nesting`)."""
    memo: dict[GroupId, int] = {}

    def depth(g: GroupId) -> int:
        if g in memo:
            return memo[g]
        memo[g] = 0  # a cyclic memo is not expected; this only stops the recursion
        e = a.fragment.get(g)
        if e is None:
            return 0
        d = 0
        for c in e.children:
            dc = depth(c)
            if derived_tables_nest and e.kind == "op1" and a.side[e.arg].kind == "ref" and not a.side[e.arg].data.get("name") \
                    and c not in a.inputs:
                dc += 1
            d = max(d, dc)
        for _, body in a.features.bodies_of(e):
            d = max(d, 1 + (0 if body in a.inputs else depth(body)))  # a call is a level; a body that is another block adds none (CPX-O13)
        memo[g] = d
        return d

    return depth(a.block.output)


@feature(13, "block-local subquery nesting depth: a call is 1 + its body's depth when the body lies inside the block, 1 when the body is another block")
def depth(a: _Analysis) -> int:
    """13. depth: block-local predicate/select-item subquery nesting.
    A subquery call contributes one plus its body's depth. If the body is an input
    block, its own nesting contributes zero, but the visible nested SELECT still
    contributes one. CTE and derived-table references inherit their bodies' depth
    without adding a level. The breakdown also reports scope_nesting, where a
    derived table adds one level and a CTE does not. Features 18/19 count subqueries;
    feature 3 classifies negation.
    """
    return _nesting(a, derived_tables_nest=False)


@feature(14, "aggregate operators (GROUP BY or a global aggregate); aggregate calls belong to it, HAVING is 12")
def agg(a: _Analysis) -> int:
    """14. `agg`: the number of aggregate operators (a GROUP BY or a global aggregate). The aggregate calls belong to
    the operator (never to 9); HAVING conjuncts belong to 12; a windowed aggregate is a window call (9), no operator."""
    return sum(1 for _, _, entry in a.op1 if entry.kind == "aggregate")


@feature(15, "distinct parameters of the block (columns of an enclosing query it depends on)")
def param(a: _Analysis) -> int:
    """15. `param`: the number of distinct parameters of the block (`Blocks.parameters`: the columns of an enclosing
    query that the block's result depends on, own or inherited from an input block): the block lies inside a
    correlated body. Distinct = distinct (occurrence, column) pairs, a parameter without a column counting once for
    its occurrence. Disjoint from 19 (which counts correlated subqueries the block contains)."""
    keys: set[tuple[str, str]] = set()
    for p in a.parameters:
        if p.columns:
            keys.update((p.occurrence, c) for c in p.columns)
        else:
            keys.add((p.occurrence, ""))
    return len(keys)


def _optypes(a: _Analysis) -> list[str]:
    if "optypes" not in a.memo:
        a.memo["optypes"] = _optypes_compute(a)
    return a.memo["optypes"]


def _optypes_compute(a: _Analysis) -> list[str]:
    found: set[str] = set()
    for _, e in a.joins:
        found.add("join" if e.conjuncts else "cross_join")
    for _, e in a.filters:
        found.add("having" if a.side[e.arg].data.get("source") == "having" else "filter")
    for g, e, entry in a.op1:
        if entry.kind == "aggregate":
            found.add("aggregate")
        elif entry.kind == "project":
            # a projection is an operator when its list computes something (the legality module's notion): a
            # `SELECT 1`, a plain list under DISTINCT or in a set-operation branch is a node but no operation
            if a.features.computes(e):
                found.add("project")
        elif entry.kind == "distinct":
            found.add("distinct_on" if entry.data.get("on") else "distinct")
        elif entry.kind == "order":
            found.add("order")
        elif entry.kind == "limit":
            if entry.data.get("limit"):
                found.add("limit")
            if entry.data.get("offset"):
                found.add("offset")
    if any(es.stats.window_calls for _, _, es in a.applied):
        found.add("window")  # a window function anywhere (a select list, an ORDER BY key): the type is added, never swapped
    for _, _, entry in a.op2:
        if entry.kind == "setop":
            found.add(OPTYPE_OF_SETOP.get(str(entry.data.get("op")).upper(), "union"))
        elif entry.kind == "outer_join":
            found.add(OPTYPE_OF_JOIN.get(str(entry.data.get("join_type")), "left_join"))
    for c in a.calls:
        found.add(c.kind)
    return sorted(found, key=OPTYPE_ALPHABET.index)


@feature(16, "distinct operator types in the fragment over the fixed alphabet OPTYPE_ALPHABET (scans and refs are none)")
def n_optypes(a: _Analysis) -> int:
    """16. `n_optypes`: the number of distinct operator types in the fragment over the fixed alphabet
    `OPTYPE_ALPHABET`: inner join / cross join, outer joins by kind, WHERE filter, HAVING filter, aggregate,
    projection (plus the window type when the select list holds a window function: a type is added, never
    swapped), DISTINCT / DISTINCT ON, ORDER BY, LIMIT, OFFSET, set operations by kind, and subquery calls by kind
    (EXISTS, NOT EXISTS, IN, NOT IN, scalar, quantified, negated quantified).
    Scans and refs are not operators, nor are expression-level constructs. The TYPE aspect: it overlaps every
    instance count by design."""
    return len(_optypes(a))


def _length_parts(a: _Analysis) -> dict[str, int]:
    if "length_parts" not in a.memo:
        a.memo["length_parts"] = _length_parts_compute(a)
    return a.memo["length_parts"]


def _cross_joins(a: _Analysis) -> int:
    """The joins the printer writes as CROSS JOIN. The printer does not follow the memo's join order: per FROM clause
    it takes the items in written order, preferring one that a condition connects to those already listed, and puts
    every join condition on the first item at which all of its inputs are present; an item that receives no
    condition is a CROSS JOIN. The same placement is computed here from the block's FROM items and join conditions
    (the memo's own order may hold a join without a condition where the printer has none, and the reverse for a
    condition over three items). Inside a side of an outer join, which has no WHERE of its own, the filters land in
    the ON clauses as well."""
    if not a.joins:
        return 0
    f = a.features
    if f.plain_join_conditions() and all(e.conjuncts for _, e in a.joins):
        return 0  # every join has a condition between two items: the items are connected, each JOIN gets its ON
    written = f.naming().written
    owners = f.result.facts.owners
    parent_join: dict[GroupId, GroupId] = {}
    items_of: dict[GroupId, list[GroupId]] = {}
    filters_of: dict[GroupId, list[str]] = {}
    for g, e in a.joins:
        items_of[g], filters_of[g] = [], []
        for c in e.children:
            while c not in a.inputs and c in a.fragment and a.fragment[c].kind == "filter":
                filters_of[g].append(a.fragment[c].arg)
                c = a.fragment[c].children[0]
            inner = None if c in a.inputs else a.fragment.get(c)
            if inner is not None and inner.kind == "join":
                parent_join[c] = g  # another join of the same FROM clause
            else:
                items_of[g].append(c)
    # a side of an outer join has no WHERE: the filters of a side with several items land in ON clauses too
    side_filters: dict[GroupId, list[str]] = {}
    for g, e, entry in a.op2:
        if entry.kind != "outer_join":
            continue
        for c in e.children:
            above: list[str] = []
            while c not in a.inputs and c in a.fragment and a.fragment[c].kind == "filter":
                above.append(a.fragment[c].arg)
                c = a.fragment[c].children[0]
            if c not in a.inputs and c in a.fragment and a.fragment[c].kind == "join":
                side_filters[c] = above
    clauses: dict[GroupId, tuple[list[GroupId], list[str], list[str]]] = {}
    for g, e in a.joins:
        root = g
        while root in parent_join:
            root = parent_join[root]
        items, conditions, filters = clauses.setdefault(root, ([], [], list(side_filters.get(root, ()))))
        items += items_of[g]
        conditions += list(e.conjuncts)
        filters += filters_of[g]
    cross = 0
    for root, (items, conditions, filters) in clauses.items():
        n = len(items)
        covers = [f.region_ids(i) if i in a.inputs or i not in a.fragment else frozenset({a.fragment[i].arg}) for i in items]
        position = [min((written.get(i, len(written)) for i in cover), default=len(written)) for cover in covers]
        rest = sorted(range(n), key=lambda i: (position[i], i))
        needs = [frozenset(f.conjunct_inputs(c)) for c in conditions]
        order = [rest.pop(0)]
        have = set(covers[order[0]])
        while rest:
            pick = next((i for i in rest if any(nd & covers[i] and nd <= have | covers[i] for nd in needs)), rest[0])
            rest.remove(pick)
            order.append(pick)
            have |= covers[pick]
        landed: set[int] = set()
        landing = list(zip(conditions, needs))
        if root in side_filters:
            landing += [(c, frozenset(f.conjunct_inputs(c))) for c in filters]
        for c, nd in landing:
            owner = owners.get(c)
            need = {owner} if owner is not None else set(nd)
            seen: set[str] = set()
            at = n - 1
            for k, i in enumerate(order):
                seen |= covers[i]
                if need <= seen:
                    at = min(max(k, 1), n - 1)
                    break
            landed.add(at)
        cross += (n - 1) - len(landed)
    return max(0, cross)


def _joins_inside(a: _Analysis, g: GroupId) -> bool:
    """Is the right side of an outer join more than one FROM item written in the block (inner joins, or another outer
    join)? The printer then puts it in parentheses."""
    while g not in a.inputs and g in a.fragment and a.fragment[g].kind == "filter":
        g = a.fragment[g].children[0]
    e = None if g in a.inputs else a.fragment.get(g)
    return e is not None and (e.kind == "join" or (e.kind == "op2" and a.side[e.arg].kind == "outer_join"))


def _filters_on_one_leaf(a: _Analysis, g: GroupId) -> bool:
    """Is the region at group g (a side of an outer join) one FROM item with filters on it? With several items the
    printer moves the filters into an ON clause of the side's inner joins; with one it has to wrap the item."""
    filters = 0
    while g not in a.inputs:
        e = a.fragment.get(g)
        if e is None or e.kind != "filter":
            break
        filters += 1
        g = e.children[0]
    e = None if g in a.inputs else a.fragment.get(g)
    return filters > 0 and (e is None or e.kind != "join")


def _setop_word(entry: SideEntry) -> str:
    """UNION, INTERSECT or EXCEPT (without ALL / DISTINCT)."""
    words = str(entry.data.get("op") or entry.sql or "UNION").upper().split()
    return words[0] if words else "UNION"


# keywords as the tokenizer of `tokens_of` counts them (SQLGlot reads `ORDER BY` and `GROUP BY` as one token each)
T_ORDER_BY, T_GROUP_BY = tokens_of("ORDER BY"), tokens_of("GROUP BY")


def _length_parts_compute(a: _Analysis) -> dict[str, int]:
    """The token count of the block's SQL (its reader form, inputs by name), by part, WITHOUT printing: the tokens
    of the applied entries' texts (without their subquery bodies, which are counted through their own groups; the
    parentheses around a body stay), the FROM items, and the keywords and punctuation each operator brings. It
    follows the printer's rules:

    - a conjunct brings its WHERE / AND / HAVING / ON; an inner join JOIN alone, and CROSS JOIN where no condition
      lands on a FROM item in the printer's own order of the items (`_cross_joins`; the memo's join order does not
      decide it); an outer join `LEFT JOIN ... ON` (`ON TRUE` without a condition); ORDER BY and GROUP BY are one
      token each for the tokenizer; `WITH ROLLUP` is no GROUP BY item;
    - a select list: SELECT, the items, the commas, `AS name` where the name is not the item's own; a column read
      through an input cut inside a FROM clause is `b_<id>.<alias>__<column>` and needs `AS <column>` to keep its
      output name; a set operation writes no list of its own;
    - FROM once per scope that has a FROM clause (its items are in the block or inputs); a table as `orders AS o`,
      or `orders` alone when the alias repeats the name; `name AS alias` for a CTE, `( ... ) AS alias` for a
      derived table, `name AS ( ... )` under WITH for a CTE written in the block;
    - an input is `b_<id>` (`AS alias` through a ref; with its arguments when it is parameterized); a subquery body
      or a set-operation branch that is an input is read by a statement of its own, by column name when the query
      spells every output name of that scope and by `SELECT *` otherwise; a branch written in the block is
      parenthesised when it has its own ORDER BY / LIMIT or is a nested set operation on the right or of another kind;
    - the block's own output inside a FROM clause: `SELECT alias.column AS alias__column, ... FROM`, a column that an
      input already exposes passing through as `b_<id>.alias__column`; `SELECT * FROM` when nothing is read by name.

    - a side of an outer join has no WHERE of its own: filters on its one FROM item print as the derived table
      `(SELECT * FROM <item> WHERE ...) AS <name>` (7 tokens more); a right side that is itself a join is
      parenthesised."""
    side, blocks = a.side, a.blocks
    parts = {"texts": 0, "operators": 0, "from_items": 0, "scopes": 0, "boundary": 0, "inputs": 0}
    under = a.under_inputs()  # a column of an occurrence inside an input is read as `b_<id>.<alias>__<column>`
    for g, id_, es in a.applied:
        parts["texts"] += es.stats.tokens
        k = es.kind
        if k == "conjunct":
            parts["operators"] += 1  # WHERE / AND / HAVING / ON
        elif k == "outer_join":
            parts["operators"] += 3 + (max(0, es.conjuncts - 1) if es.conjuncts else 1)  # LEFT JOIN ... ON ... AND ... / ON TRUE
            sides = a.fragment[g].children
            for side_group in sides:  # a side has no WHERE of its own: filters on its ONE input need
                if _filters_on_one_leaf(a, side_group):  # `(SELECT * FROM <input> WHERE ...) AS <name>`
                    parts["operators"] += 7
            if len(sides) == 2 and _joins_inside(a, sides[1]):  # `LEFT JOIN ( b JOIN c ON ... ) ON ...`
                parts["operators"] += 2
        elif k in ("aggregate", "project"):
            if es.window_over_aggregate:  # the window items join the aggregate's one select list: a comma each
                listed = any(o.kind == "aggregate" and o.select_items and blocks.owner_scope(i) == blocks.owner_scope(id_)
                             for _, i, o in a.applied)
                parts["operators"] += es.select_items - (0 if listed else 1) + 2 * es.aliased_items
            else:
                parts["operators"] += 1 + max(0, es.select_items - 1) + 2 * es.aliased_items  # SELECT, commas, AS alias
                if under:  # `c.c_id` read through an input: `b_<id>.c__c_id AS c_id`
                    parts["operators"] += 2 * sum(1 for occ in a.features.entry_bare_items(id_) if occ in under)
            if es.group_items:
                parts["operators"] += T_GROUP_BY + max(0, es.group_items - es.group_modifiers - 1)
        elif k == "order":
            parts["operators"] += T_ORDER_BY + max(0, es.order_keys - 1)
            parts["operators"] += sum(1 for key in (side[id_].data.get("keys") or ()) if key.get("desc"))
            parts["operators"] += sum(2 for key in (side[id_].data.get("keys") or ()) if key.get("nulls"))
        elif k == "limit":
            parts["operators"] += int(es.has_limit) + int(es.has_offset)
        elif k == "distinct":
            parts["operators"] += 4 + max(0, es.distinct_on - 1) if es.distinct_on else 1
        elif k == "setop":
            parts["operators"] += len(str(side[id_].data.get("op") or "UNION").split())
            e = a.fragment[g]
            for position, child in enumerate(e.children):  # a branch written here may need parentheses
                sid = None if child in a.inputs else blocks.scope_top_of(child)
                if sid is not None and a.features.branch_in_parentheses(sid, _setop_word(side[id_]), right=position == 1):
                    parts["operators"] += 2
    parts["operators"] += len(a.joins) + _cross_joins(a)  # JOIN (its conjuncts bring ON / AND above); CROSS JOIN
    # FROM items: scans as `table AS alias`, refs as `name AS alias` / `(...) AS alias`, inputs as `b_<id>`
    for _, _, entry in a.scans:
        parts["from_items"] += a.features.scan_tokens(entry.id)
    referenced_inputs: set[GroupId] = set()
    for _, e, entry in a.refs:
        child = e.children[0]
        alias, name = entry.data.get("alias"), entry.data.get("name")
        if child in a.inputs:
            referenced_inputs.add(child)
            parts["inputs"] += 1 + (2 if alias and alias != f"b_{child}" else 0)
        elif name:
            parts["from_items"] += 1 + (2 if alias and str(alias).lower() != str(name).lower() else 0)
        else:
            parts["from_items"] += 4  # ( ... ) AS alias
    for i in a.inputs - referenced_inputs:
        parts["inputs"] += 1  # a region leaf, a subquery body or a set-operation branch: `b_<id>`
    called = {c.body: c.scope for c in a.calls if c.is_input}
    for i in a.inputs:  # a parameterized input is called with its arguments: `b_<id>(c.c_id, c.c_name)`, one per distinct column
        args = {(q.occurrence, c) for q in a.features.body_parameters(i) for c in (q.columns or ("",))}
        if args:
            parts["inputs"] += 2 + 3 * len(args) + max(0, len(args) - 1)
        # a subquery body or a set-operation branch that is an input is read by a statement of its own: by column
        # name when the query itself spells every output name of that scope, otherwise by star
        sid = called.get(i)
        if sid is None:
            top = blocks.scope_top_of(i)
            sid = top if top is not None and blocks.scopes[top].kind == "setop_branch" else None
        if sid is not None:
            if a.features.spells_its_names(sid):
                n = len(blocks.scopes[sid].outputs)  # SELECT b.x, b.y FROM b_<id>
                parts["inputs"] += 2 + 3 * n + max(0, n - 1)
            else:
                parts["inputs"] += 3  # SELECT * FROM b_<id>
    # scopes inside the block: SELECT and the select list where no aggregate / project node carries it (those count
    # theirs above), FROM, and `name AS ( ... )` under WITH for an in-block CTE that is not the block's own output
    # (a CTE body printed as a block is a plain statement)
    scopes_inside: dict[str, GroupId] = {}
    for g in a.fragment:
        sid = blocks.scope_top_of(g)
        if sid is not None:
            scopes_inside.setdefault(sid, g)
    computed_lists = {blocks.owner_scope(id_) for _, id_, es in a.applied if es.kind in ("aggregate", "project")}
    ctes = 0
    for sid, top in scopes_inside.items():
        scope = blocks.scopes[sid]
        if scope.kind == "cte" and top != a.block.output:
            ctes += 1
            parts["scopes"] += 4  # name AS ( ... )
        if sid not in computed_lists and not a.features.is_setop_scope(sid):  # a set operation writes no list of its own
            parts["scopes"] += a.features.scope_list_tokens(sid)  # SELECT, the plain items, their names, the commas
            if under:
                parts["scopes"] += 2 * sum(1 for occ in a.features.scope_bare_items(sid) if occ in under)
        if a.features.scope_has_from(sid):
            parts["scopes"] += 1  # FROM (the items are in the block, or inputs)
    if ctes:
        parts["scopes"] += 1 + (ctes - 1)  # WITH, commas
    # the block's own output inside a region: the printer's boundary select list, `SELECT` + FROM; `SELECT *` when
    # nothing is read by name (star). A column of the block's own FROM items is `alias.column AS alias__column`; a
    # column that an input already exposes passes through under that name, `b_<id>.alias__column`
    if not blocks.is_scope_top(a.block.output) and a.block.output != blocks.memo.root:
        if a.star:
            parts["boundary"] += 3
        else:
            n = a.exposed_columns
            passed = sum(1 for occ in a.exposed if occ in under)
            parts["boundary"] += 2 + 5 * (n - passed) + 3 * passed + max(0, n - 1)
    return parts


@feature(17, "SQL length in tokens, counted from the side-table texts, FROM items and operator keywords by the printer's rules (never printed)")
def length(a: _Analysis) -> int:
    """17. `length`: the length of the block's SQL in tokens (SQLGlot's Postgres tokenizer), counted WITHOUT printing
    from the side-table texts of its conjuncts and operators (a subquery body inside the block counted through its
    own groups, a body that is an input as a call of `b_<id>`), its FROM items and the keywords / punctuation each
    operator adds, by the printer's own rules (`_length_parts_compute`); validated against printed samples in the
    report, where it equals the printed count on every sampled block. A proxy that overlaps everything by design."""
    return sum(_length_parts(a).values())


@feature(18, "uncorrelated subquery calls inside the block (the body reads no column of the block's scopes)")
def uncorr(a: _Analysis) -> int:
    """18. `uncorr`: UNCORRELATED subquery calls inside the block: predicate and select-item subqueries whose body reads
    no column of the block's own scopes; a call whose body is another block (an input) still counts as a call. The
    old `subqcall` counted ALL calls; 18 + 19 = all calls, no call is counted twice."""
    return sum(1 for c in a.calls if not c.correlated)


@feature(19, "correlated subquery calls inside the block (the body reads a column of the block's own scopes)")
def corr(a: _Analysis) -> int:
    """19. `corr`: CORRELATED subquery calls inside the block: the body reads a column of an occurrence of the block
    (its own scopes, an input's FROM item included), through any depth of nesting. A body correlated only with a
    query OUTSIDE the block is an uncorrelated call here (the block's own dependence on that column is 15)."""
    return sum(1 for c in a.calls if c.correlated)


assert [FEATURES[n].number for n in FEATURE_NAMES] == list(range(1, 20)), "the registry must hold features 1-19 in order"


# --------------------------------------------------------------------------------------------------------------
# the feature vector and the extractor
# --------------------------------------------------------------------------------------------------------------


@dataclass
class FeatureVector:
    values: dict[str, int]  # the nineteen, in the order of section 3.3
    breakdown: dict[str, Any]  # what the definitions ask to report beside the values, and the parts of the estimates
    notes: list[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> int:
        return self.values[name]

    def as_dict(self) -> dict[str, Any]:
        return {"values": dict(self.values), "breakdown": dict(self.breakdown), "notes": list(self.notes)}

    def __str__(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.values.items())


class Features:
    """The feature extractor of ONE query. `of(ctx)` computes the nineteen features of a block class from its
    context (cached per class id), `of_block(block)` from a block alone (for the command line and the tests)."""

    def __init__(self, memo: Memo, result: ImportResult, blocks: Blocks, boundary: Any = None):
        self.memo, self.result, self.blocks = memo, result, blocks
        self.boundary = boundary  # a `sqldecomp.boundary.Boundary`, for the exposed columns of a block printed alone
        self.side = result.side
        self._entry_stats: dict[str, EntryStats] = {}
        self._body_params: dict[GroupId, list[Parameter]] = {}
        self._by_class: dict[str, FeatureVector] = {}
        self._computing: dict[str, bool] = {}
        self._scope_tokens: dict[str, int] = {}
        self._scan_tokens: dict[str, int] = {}
        self._scope_group: dict[str, GroupId | None] = {}
        self._bodies: dict[MExpr, tuple] = {}
        self._scope_of: dict[GroupId, str | None] = {}
        self._null_extended: dict[tuple[GroupId, MExpr], frozenset[str]] = {}
        self._resolved: dict[tuple, frozenset[str]] = {}
        self._scope_bare: dict[str, tuple[str, ...]] = {}
        self._entry_bare: dict[str, tuple[str, ...]] = {}
        self._from_scopes: set[str | None] | None = None
        self._region_ids: dict[GroupId, frozenset[str]] = {}
        self._edge_inputs: dict[str, tuple[str, ...]] | None = None
        self._plain_conditions: bool | None = None
        self._setop_scope: dict[str, str | None] = {}
        self._ordered_scopes: set[str | None] | None = None
        self._spells_names: dict[str, bool] = {}
        self._naming: PrintContext | None = None  # the printer's rules about names (nothing is printed through it)
        self.notes: list[str] = []

    def entry_stats(self, entry_id: str) -> EntryStats:
        if entry_id not in self._entry_stats:
            self._entry_stats[entry_id] = entry_stats(self.side[entry_id])
        return self._entry_stats[entry_id]

    def window_only_stats(self, entry_id: str) -> EntryStats:
        key = entry_id + "#window"
        if key not in self._entry_stats:
            self._entry_stats[key] = window_items_only(self.side[entry_id], self.entry_stats(entry_id))
        return self._entry_stats[key]

    # Everything below is a function of ONE side entry or ONE scope and is asked for by every block class that holds
    # it: computed once per query, never per class (a query can have over a hundred thousand classes; parsing a
    # select list per class made the raw BEAVER set's largest queries take minutes instead of seconds).

    def computes(self, e: MExpr) -> bool:
        """Is this projection node an operation (its list computes something)? `Blocks._is_operator`, cached per entry."""
        found = self._computing.get(e.arg)
        if found is None:
            found = self._computing[e.arg] = bool(self.blocks._is_operator(e))
        return found

    def scope_list_tokens(self, sid: str) -> int:
        """The tokens of a scope's plain select list as it prints: SELECT, the items, `AS name` where the name is not
        the item's own, the commas."""
        if sid not in self._scope_tokens:
            self._scope_list(sid)
        return self._scope_tokens[sid]

    def scope_bare_items(self, sid: str) -> tuple[str, ...]:
        """Per item of a scope's plain select list that is a column under its own name (`c.c_id`): the occurrence it
        reads. Such an item prints as `b_<id>.c__c_id AS c_id` when the occurrence lies inside an input of the block."""
        if sid not in self._scope_bare:
            self._scope_list(sid)
        return self._scope_bare[sid]

    def _scope_list(self, sid: str) -> None:
        outputs = self.blocks.scopes[sid].outputs
        tokens, bare = 1 + max(0, len(outputs) - 1), []
        for o in outputs:
            tokens += tokens_of(o.sql)
            try:
                item = sqltext.parse_item(o.sql)
                natural = sqltext.natural_name(item)
            except PrintError:
                item = natural = None
            if natural != (o.name or "").lower():
                tokens += 2
            elif isinstance(item, exp.Column) and item.table:
                occ = self.blocks.resolve_alias(sid, item.table)
                if occ is not None:
                    bare.append(occ)
        self._scope_tokens[sid], self._scope_bare[sid] = tokens, tuple(bare)

    def entry_bare_items(self, entry_id: str) -> tuple[str, ...]:
        """The same for the select list of an aggregate / projection entry: per item that is a column without an
        alias, the occurrence it reads."""
        found = self._entry_bare.get(entry_id)
        if found is None:
            sid, bare = self.blocks.owner_scope(entry_id), []
            for i in self.side[entry_id].data.get("select") or ():
                if i.get("alias"):
                    continue
                try:
                    item = sqltext.parse_item(i.get("sql", ""))
                except PrintError:
                    continue
                if isinstance(item, exp.Column) and item.table:
                    occ = self.blocks.resolve_alias(sid, item.table)
                    if occ is not None:
                        bare.append(occ)
            found = self._entry_bare[entry_id] = tuple(bare)
        return found

    def region_ids(self, g: GroupId) -> frozenset[str]:
        """The ids that join conditions name inside a group read as one FROM item (an input): the occurrences of its
        scans, the ids of its operators, through its joins and filters."""
        found = self._region_ids.get(g)
        if found is None:
            self._region_ids[g] = frozenset()  # stops a cycle
            e = self.memo.groups[g][0]
            if e.kind in ("join", "filter"):
                found = frozenset().union(*(self.region_ids(c) for c in e.children))
            else:
                found = frozenset({e.arg})
            self._region_ids[g] = found
        return found

    def plain_join_conditions(self) -> bool:
        """Does every join condition of the query reference at most two FROM items, with no condition tied to an
        operator? Then a join order in which every join has a condition has no CROSS JOIN in print either."""
        if self._plain_conditions is None:
            facts = self.result.facts
            self._plain_conditions = not facts.owners and all(len(e.inputs) <= 2 for e in facts.edges)
        return self._plain_conditions

    def conjunct_inputs(self, conjunct: str) -> tuple[str, ...]:
        """The FROM items a join condition references (the importer's edges, else the `refs` of its side entry)."""
        if self._edge_inputs is None:
            self._edge_inputs = {e.conjunct: tuple(e.inputs) for e in self.result.facts.edges}
        found = self._edge_inputs.get(conjunct)
        return found if found is not None else tuple(self.side[conjunct].data.get("refs") or ())

    def scope_has_from(self, sid: str) -> bool:
        """The scope writes a FROM clause: it has a table, a VALUES list or a nested scope as a FROM item (whether the
        block holds the item itself or reads it as an input `b_<id>`)."""
        if self._from_scopes is None:
            self._from_scopes = {self.blocks.owner_scope(id_) for id_, e in self.side.items()
                                 if e.kind == "ref" or (e.kind == "scan" and (e.data.get("table") or e.data.get("columns")))}
        return sid in self._from_scopes

    def setop_of_scope(self, sid: str) -> str | None:
        """The set operation (UNION, INTERSECT, EXCEPT) that the scope's own statement is, or None. ORDER BY / LIMIT
        may follow it. Such a scope has output columns but writes no select list: `A UNION B` is its branches' lists
        and the operator."""
        if sid not in self._setop_scope:
            plan = self.blocks.scopes[sid].plan
            while isinstance(plan, Op1) and self.side[plan.op].kind in ("limit", "order", "distinct"):
                plan = plan.input
            word = None
            if isinstance(plan, Op2) and self.side[plan.op].kind == "setop":
                word = _setop_word(self.side[plan.op])
            self._setop_scope[sid] = word
        return self._setop_scope[sid]

    def is_setop_scope(self, sid: str) -> bool:
        return self.setop_of_scope(sid) is not None

    def branch_in_parentheses(self, sid: str, parent: str, right: bool) -> bool:
        """The printer's rule for a branch written inside the block: a branch with its own ORDER BY / LIMIT is
        parenthesised, and so is a nested set operation on the right side or of another kind than its parent."""
        if self._ordered_scopes is None:
            self._ordered_scopes = {self.blocks.owner_scope(id_) for id_, e in self.side.items() if e.kind in ("order", "limit")}
        trailing = sid in self._ordered_scopes
        word = self.setop_of_scope(sid)
        return trailing if word is None else (right or trailing or word != parent)

    def naming(self) -> PrintContext:
        """The printer's rules about names (what a FROM item or an output column is called). Nothing is printed."""
        if self._naming is None:
            self._naming = PrintContext(self.result)
        return self._naming

    def spells_its_names(self, sid: str) -> bool:
        """The printer's rule for reading a subquery body or a set-operation branch that is another block: by column
        name only when the query itself spells every output name of that scope, otherwise `SELECT * FROM b_<id>`
        (never a name that Postgres invents). Asked once per scope."""
        found = self._spells_names.get(sid)
        if found is None:
            try:
                found = bool(self.naming().explicit_output_names(self.blocks.scopes[sid]))
            except PrintError:
                found = False
            self._spells_names[sid] = found
        return found

    def scan_tokens(self, occ: str) -> int:
        """The tokens of a base table as a FROM item: `orders AS o`, or `orders` alone when the alias only repeats the
        table's name (the printer's rule). A VALUES list is counted with the texts."""
        found = self._scan_tokens.get(occ)
        if found is None:
            entry = self.side[occ]
            if not entry.data.get("table"):
                found = 0
            else:
                try:
                    found = tokens_of(sqltext.to_sql(self.naming().scan_item(self.naming().scans[occ])))
                except (PrintError, KeyError):
                    found = tokens_of(entry.sql)
            self._scan_tokens[occ] = found
        return found

    def scope_group(self, sid: str) -> GroupId | None:
        """The memo group of a scope's whole plan (None for an unknown scope), once per scope."""
        if sid not in self._scope_group:
            scope = self.blocks.scopes.get(sid)
            self._scope_group[sid] = None if scope is None else self.memo.group_of(scope.plan)
        return self._scope_group[sid]

    def bodies_of(self, e: MExpr) -> tuple[tuple[str, GroupId], ...]:
        """`Blocks.subquery_bodies(e)`: (scope id, root group) of the subqueries an expression holds, once per expression."""
        found = self._bodies.get(e)
        if found is None:
            found = self._bodies[e] = self.blocks.subquery_bodies(e)
        return found

    def scope_of_group(self, g: GroupId) -> str | None:
        if g not in self._scope_of:
            self._scope_of[g] = self.blocks.scope_of(g)
        return self._scope_of[g]

    def null_extended(self, g: GroupId, e: MExpr, entry: SideEntry) -> frozenset[str]:
        """The occurrences (scans and refs of the join's own scope) that the outer join at group g NULL-extends: a
        fact of the join expression, the same in every block that holds it."""
        found = self._null_extended.get((g, e))
        if found is None:
            kind = str(entry.data.get("join_type") or "")
            out: set[str] = set()
            if kind in NULL_EXTENDING and len(e.children) == 2:
                left, right = e.children
                scope = self.scope_of_group(g)
                for side in ([right] if kind == "LEFT" else [left] if kind == "RIGHT" else [left, right]):
                    out.update(occ for occ in self.blocks.occurrences_under(side) if self.blocks.owner_scope(occ) == scope)
            found = self._null_extended[(g, e)] = frozenset(out)
        return found

    def resolved(self, entry_id: str, key: tuple, columns: Iterable[tuple[str, str]]) -> frozenset[str]:
        """The occurrences that the qualified columns of one text (or one aggregate call) of an entry refer to: the
        entry's own scope first, then the scopes around it. `key` names the text within the entry."""
        k = (entry_id, key)
        found = self._resolved.get(k)
        if found is None:
            sid = self.blocks.owner_scope(entry_id)
            out = set()
            for alias, _ in columns:
                if alias:
                    occ = self.blocks.resolve_alias(sid, alias)
                    if occ is not None:
                        out.add(occ)
            found = self._resolved[k] = frozenset(out)
        return found

    def body_parameters(self, body: GroupId) -> list[Parameter]:
        if body not in self._body_params:
            self._body_params[body] = self.blocks.parameters(Block(body))
        return self._body_params[body]

    def nullable_read(self, entry_id: str, columns: Iterable[tuple[str, str]], nullable_inputs: frozenset[tuple[str, str]]) -> bool:
        """Does one text of an entry read a column that an input of the block exposes as nullable (CPX-O2)? The alias is
        resolved in the entry's scope, the pair (occurrence, column) looked up among the inputs' nullable columns."""
        sid = self.blocks.owner_scope(entry_id)
        for alias, column in columns:
            if alias and column:
                occ = self.blocks.resolve_alias(sid, alias)
                if occ is not None and (occ, column) in nullable_inputs:
                    return True
        return False

    def nullable_inputs_of(self, inputs: Iterable[GroupId]) -> frozenset[tuple[str, str]]:
        """The (occurrence, column) pairs that the block's inputs expose as nullable, from the boundary."""
        if self.boundary is None:
            return frozenset()
        out: set[tuple[str, str]] = set()
        for g in inputs:
            try:
                columns, _star = self.boundary.columns(g)
            except Exception:  # noqa: BLE001
                continue
            out.update((c.occurrence, c.column) for c in columns if c.nullable and c.occurrence and c.column)
        return frozenset(out)

    def of(self, ctx: BlockContext) -> FeatureVector:
        """The features of a block class, from the context the optimizer built (the representative's fragment, the
        parameters, the union of exposed columns), cached per class id."""
        found = self._by_class.get(ctx.class_id)
        if found is None:
            nullable = frozenset((c.occurrence, c.column) for i in ctx.inputs for c in i.columns if c.nullable and c.occurrence and c.column)
            found = self.compute(ctx.block, ctx.fragment, ctx.parameters, [c.occurrence for c in ctx.exposed], ctx.star, nullable)
            self._by_class[ctx.class_id] = found
        return found

    def of_block(self, block: Block) -> FeatureVector:
        """The features of one block (output, inputs); ValueError when the pair is not a block."""
        fragment = self.blocks.fragment(block)
        exposed: list[str] = []
        star = False
        if self.boundary is not None and not self.blocks.is_scope_top(block.output) and block.output != self.memo.root:
            try:
                columns, star = self.boundary.columns(block.output)
                exposed = [c.occurrence for c in columns]
            except Exception:  # noqa: BLE001 - the estimate of `length` only
                exposed, star = [], False
        return self.compute(block, fragment, self.blocks.parameters(block, fragment), exposed, star, self.nullable_inputs_of(block.inputs))

    def compute(self, block: Block, fragment: dict[GroupId, MExpr], parameters: Iterable[Parameter],
                exposed: Iterable[str] = (), star: bool = False, nullable_inputs: Iterable[tuple[str, str]] = ()) -> FeatureVector:
        """`exposed`: per column the block exposes at a boundary inside a FROM clause, the occurrence it comes from;
        `nullable_inputs`: the (occurrence, column) pairs the block's inputs expose as nullable (CPX-O2)."""
        a = _Analysis(self, block, fragment, parameters, exposed, star, nullable_inputs)
        values = {name: int(FEATURES[name].fn(a)) for name in FEATURE_NAMES}
        for name, v in values.items():
            if v < 0:
                raise AssertionError(f"feature {name} is negative ({v}) on block {block}")
        uses, tables = _relation_uses(a)
        breakdown = {
            "bool_depth": max((es.stats.bool_depth for _, _, es in a.applied), default=0),
            "order_keys": sum(es.order_keys for _, _, es in a.applied),
            "scope_nesting": _nesting(a, derived_tables_nest=True),
            "optypes": _optypes(a),
            "length_parts": _length_parts(a),
            "calls": [{"entry": c.entry, "scope": c.scope, "body": c.body, "kind": c.kind, "input": c.is_input,
                       "correlated": c.correlated} for c in a.calls],
            "relations": {"tables": sorted(tables), "uses": {" ".join(map(str, k)): n for k, n in uses.items()},
                          "inputs": sorted(a.inputs)},
            "negation_parts": {"existence": sum(es.stats.existence_negation for _, _, es in a.applied),
                               "boolean_not": sum(es.stats.boolean_not for _, _, es in a.applied),
                               "anti_joins": sum(1 for _, _, entry in a.op2 if entry.kind == "outer_join" and entry.data.get("join_type") == "ANTI")},
            "null_sensitive_parts": {"tests": sum(es.stats.null_tests for _, _, es in a.applied),
                                     "functions": sum(es.stats.null_functions for _, _, es in a.applied),
                                     "not_in": sum(es.stats.not_in for _, _, es in a.applied),
                                     "outer_join_uses": _outer_join_uses(a)},
            "window_calls": sum(es.stats.window_calls for _, _, es in a.applied),
            "grouping_sets": sum(es.grouping_sets for _, _, es in a.applied),
            "unparsed_entries": [id_ for _, id_, es in a.applied if es.stats.unparsed],
            "entries": [id_ for _, id_, _ in a.applied],
        }
        vector = FeatureVector(values, breakdown, list(a.notes))
        self.notes += a.notes
        return vector


# --------------------------------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------------------------------



@dataclass
class RiskRow:
    feature: str
    value: int
    threshold: float | None
    excess: float  # 0 when the feature does not participate
    contribution: float  # -log p_x


@dataclass
class Risk:
    total: float  # -log F
    rows: list[RiskRow]
    meaningfulness: float
    semantic_contribution: float  # -gamma * log s

    @property
    def F(self) -> float:
        return math.exp(-self.total)

    def over_threshold(self) -> list[RiskRow]:
        return [r for r in self.rows if r.excess > 0]


@dataclass
class HumanVerifier:
    """theta of the human-facing model: one optional threshold per feature, one shared beta, the excess mode, the
    link, gamma and mu. `tau` holds every feature name; None means the feature does not participate."""

    tau: dict[str, float | None] = field(default_factory=lambda: {n: None for n in FEATURE_NAMES})
    beta: float = LN2
    excess: str = "relative"  # or "absolute"
    link: str = "hinge"  # or "logistic"
    gamma: float = 0.0
    mu: float = 0.001
    logistic_a: float = 4.0  # p_x = sigmoid(a * (c - excess_x)); c = 1: at twice the limit the reliability is 1/2
    logistic_c: float = 1.0
    notes: dict[str, Any] = field(default_factory=dict)  # what a loader recorded (never a secret)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.tau) - set(FEATURE_NAMES))
        if unknown:
            raise ValueError(f"unknown features in tau: {unknown}")
        self.tau = {n: (None if self.tau.get(n) is None else float(self.tau[n])) for n in FEATURE_NAMES}
        if any(t is not None and (t < 0 or math.isnan(t) or math.isinf(t)) for t in self.tau.values()):
            raise ValueError("a threshold must be a finite number >= 0, or None")
        if self.excess not in ("relative", "absolute"):
            raise ValueError("excess must be 'relative' or 'absolute'")
        if self.link not in ("hinge", "logistic"):
            raise ValueError("link must be 'hinge' or 'logistic'")
        if not (self.beta >= 0) or math.isinf(self.beta):
            raise ValueError("beta must be a finite number >= 0")
        if not (self.gamma >= 0) or math.isinf(self.gamma):
            raise ValueError("gamma must be a finite number >= 0")
        if math.isnan(self.mu) or math.isinf(self.mu):
            raise ValueError("mu must be finite")

    # -- the per-feature terms ----------------------------------------------------------------------------------

    def excess_of(self, name: str, value: float) -> float:
        tau = self.tau[name]
        if tau is None:
            return 0.0
        over = max(0.0, float(value) - tau)
        return over / max(tau, 1.0) if self.excess == "relative" else over

    def p(self, name: str, value: float) -> float:
        if self.tau[name] is None:
            return 1.0
        e = self.excess_of(name, value)
        if self.link == "hinge":
            return math.exp(-self.beta * e)
        return 1.0 / (1.0 + math.exp(-self.logistic_a * (self.logistic_c - e)))

    def risk(self, features: FeatureVector, meaningfulness: float = 1.0) -> Risk:
        """-log F with the per-feature breakdown (value, threshold, excess, contribution)."""
        rows = []
        total = 0.0
        for name in FEATURE_NAMES:
            v = features.values[name]
            e = self.excess_of(name, v)
            c = -math.log(self.p(name, v)) if self.tau[name] is not None else 0.0
            rows.append(RiskRow(name, v, self.tau[name], e, c))
            total += c
        if not (meaningfulness > 0.0) or math.isinf(meaningfulness) or math.isnan(meaningfulness):
            raise ValueError(f"the semantic score must be a positive finite number, got {meaningfulness!r}")
        semantic = -self.gamma * math.log(meaningfulness)
        return Risk(total + semantic, rows, meaningfulness, semantic)

    def F(self, features: FeatureVector, meaningfulness: float = 1.0) -> float:
        return math.exp(-self.risk(features, meaningfulness).total)

    # -- serialisation (never a secret: only the fields above are written) ---------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {"model": "human-verifier", "tau": {n: self.tau[n] for n in FEATURE_NAMES}, "beta": self.beta,
                "excess": self.excess, "link": self.link, "gamma": self.gamma, "mu": self.mu,
                "logistic_a": self.logistic_a, "logistic_c": self.logistic_c, "notes": dict(self.notes)}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "HumanVerifier":
        allowed = {"tau", "beta", "excess", "link", "gamma", "mu", "logistic_a", "logistic_c", "notes"}
        kw = {k: v for k, v in d.items() if k in allowed}
        return cls(**kw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "HumanVerifier":
        return cls.from_json(json.loads(Path(path).read_text()))

    def summary(self) -> str:
        taus = ",".join(f"{n}={self.tau[n]:g}" for n in FEATURE_NAMES if self.tau[n] is not None) or "no thresholds"
        return f"tau[{taus}] beta={self.beta:g} excess={self.excess} link={self.link} gamma={self.gamma:g} mu={self.mu:g}"

    def with_tau(self, **tau: float | None) -> "HumanVerifier":
        """A copy with some thresholds changed."""
        d = self.to_json()
        d["tau"].update(tau)
        return HumanVerifier.from_json(d)

# The threshold preset: user-study sets all nineteen features from USER_STUDY_TAU.
USER_STUDY_TAU = {"inner_join": 2.0, "outer_join": 1.0, "negation": 1.0, "null_sensitive": 1.0, "distinct": 1.0, "table_reuse": 1.0,
                  "setop": 1.0, "bool_mix": 2.0, "expr": 4.0, "order_limit": 2.0, "n_relations": 4.0, "pred": 3.0, "depth": 1.0,
                  "agg": 1.0, "param": 1.0, "n_optypes": 5.0, "length": 150.0, "uncorr": 1.0, "corr": 1.0}
TAU_PRESETS = {"user-study": USER_STUDY_TAU}
DEFAULT_TAU = "user-study"


def human_preset(name: str) -> HumanVerifier:
    """The theta of the named thresholds (`TAU_PRESETS`) with every other field at its default: beta ln 2, the
    relative excess, the hinge link, gamma 0, mu 0.001. A fresh object on every call."""
    if name not in TAU_PRESETS:
        raise ValueError(f"unknown thresholds {name!r}; the named ones are {', '.join(TAU_PRESETS)}")
    return HumanVerifier(tau=dict(TAU_PRESETS[name]))


def human_default() -> HumanVerifier:
    """Return a fresh default human-facing scoring model.
    All nineteen thresholds use USER_STUDY_TAU; beta is ln(2), excess is relative,
    the link is hinge, gamma is zero and mu is 0.001. Thresholds are configurable
    user parameters.
    """
    return human_preset(DEFAULT_TAU)


def preset_of(tau: dict[str, Any]) -> str | None:
    """The name of a theta's thresholds (`TAU_PRESETS`: every feature's threshold equal, an unset one None), else None."""
    given = {n: (None if tau.get(n) is None else float(tau[n])) for n in FEATURE_NAMES}
    return next((name for name, t in TAU_PRESETS.items() if given == {n: t.get(n) for n in FEATURE_NAMES}), None)


def scaled(theta: HumanVerifier, alpha: float) -> HumanVerifier:
    """`theta` with every threshold times `alpha`, not rounded (outer_join 1 becomes 0.5 at alpha 0.5): the paper's
    granularity. The notes keep `alpha` and the thresholds before scaling (`tau0`), so the scaled theta names its
    origin (`scoring_args.scoring_label`). `alpha` 1 is recorded too: it names the setting, the thresholds stay."""
    if not (alpha > 0) or math.isinf(alpha):
        raise ValueError("alpha must be a finite number > 0")
    if "alpha" in theta.notes:
        raise ValueError("this theta is scaled already (alpha in its notes)")
    d = theta.to_json()
    d["notes"] = {**d["notes"], "alpha": float(alpha), "tau0": {n: t for n, t in d["tau"].items() if t is not None}}
    d["tau"] = {n: (None if t is None else t * alpha) for n, t in d["tau"].items()}
    return HumanVerifier.from_json(d)


def constant_meaningfulness(group: GroupContext) -> float:
    """The semantic score when it is switched off (`--semantic none`, the default): 1.0 for every group, so that
    s ** gamma = 1 whatever gamma. Component 4 itself is the LLM judge of `sqldecomp.semantic_llm`."""
    return 1.0


class HumanScoring(Scoring):
    """A `Scoring` whose F is the human-facing model on the features of ONE query. The optimizer calls it like any
    other Scoring; `features` and `verifier` are there for the reports (the breakdown of every class it scored)."""

    def __init__(self, verifier: HumanVerifier, features: Features,
                 meaningfulness: Callable[[GroupContext], float] | None = None):
        semantic = meaningfulness or constant_meaningfulness
        super().__init__(F=lambda theta, ctx: theta.F(features.of(ctx), ctx.meaningfulness),
                         Omega=lambda k: -verifier.mu * k, meaningfulness=semantic, theta=verifier,
                         name=f"human {verifier.summary()}")
        self.verifier = verifier
        self.features = features

    def risk(self, ctx: BlockContext) -> Risk:
        return self.verifier.risk(self.features.of(ctx), ctx.meaningfulness)


def human_scoring(theta: HumanVerifier, features: Features,
                  meaningfulness: Callable[[GroupContext], float] | None = None) -> HumanScoring:
    """The Scoring of the human-facing model for one query: F(theta, ctx) = prod p_x(v_x) * s ** gamma on the
    features of the class's representative, Omega(k) = -mu k, the semantic hook of component 4 (constant 1.0 unless
    given: the LLM judge's, `sqldecomp.semantic_llm.judge_hook`). It
    plugs into `sqldecomp.optimizer.optimize` unchanged; the query binding is what lets F read the side-table texts
    through the context's fragment (the context carries the fragment, the texts live in the query's side table)."""
    return HumanScoring(theta, features, meaningfulness)


def risk(ctx: BlockContext, theta: HumanVerifier, features: Features) -> Risk:
    """-log F of a block class with the per-feature breakdown (value, threshold, excess, contribution)."""
    return theta.risk(features.of(ctx), ctx.meaningfulness)


def parse_tau(text: str) -> dict[str, float | None]:
    """`inner_join=2,pred=3,corr=none` -> thresholds; unknown names are an error; `none` / `off` unsets one. A part
    without `=` is a named set of thresholds (`TAU_PRESETS`: `user-study`), which later pairs change
    (`user-study,length=150`)."""
    out: dict[str, float | None] = {}
    for part in (p.strip() for p in text.split(",") if p.strip()):
        if "=" not in part:
            if part not in TAU_PRESETS:
                raise ValueError(f"--tau expects name=value pairs or a named set of thresholds ({', '.join(TAU_PRESETS)}), got {part!r}")
            out.update(TAU_PRESETS[part])
            continue
        name, value = (x.strip() for x in part.split("=", 1))
        if name not in FEATURES:
            raise ValueError(f"unknown feature {name!r}; the features are {', '.join(FEATURE_NAMES)}")
        out[name] = None if value.lower() in ("none", "off", "null", "") else float(value)
    return out


# --------------------------------------------------------------------------------------------------------------
# command line: the feature vectors of a query's groups / of one block, for inspection
# --------------------------------------------------------------------------------------------------------------


def block_report(features: Features, block: Block, verifier: HumanVerifier | None = None, printer: Any = None) -> list[str]:
    v = features.of_block(block)
    head = f"### block {block.output} <- {sorted(block.inputs)}"
    lines = [head, "features: " + str(v),
             f"breakdown: bool_depth={v.breakdown['bool_depth']} order_keys={v.breakdown['order_keys']} "
             f"scope_nesting={v.breakdown['scope_nesting']} optypes={v.breakdown['optypes']} "
             f"length_parts={v.breakdown['length_parts']} calls={v.breakdown['calls']} relations={v.breakdown['relations']}"]
    if v.notes:
        lines.append("notes: " + "; ".join(v.notes))
    if verifier is not None:
        r = verifier.risk(v)
        lines.append(f"-log F = {r.total:.6f}  (F = {r.F:.6f})  " + " ".join(
            f"{row.feature}:{row.value}>{row.threshold:g}(+{row.contribution:.4f})" for row in r.over_threshold()))
    if printer is not None:
        try:
            lines.append(printer.block_sql(block, inputs="name"))
        except Exception as e:  # noqa: BLE001
            lines.append(f"-- {type(e).__name__}: {e}")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sqldecomp.complexity", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sql", help="one query (text, or @file)")
    ap.add_argument("--schema", help="schema file: contract JSON or the BEAVER format (required with --sql)")
    ap.add_argument("--block", action="append", default=[],
                    help="a block OUTPUT:INPUT1,INPUT2 (repeatable); default: every cut point and the root from base tables")
    ap.add_argument("--theta-json", help="a theta JSON (HumanVerifier.save) to print -log F under")
    ap.add_argument("--tau", help="thresholds name=value,... (with --beta / --excess / --link / --gamma / --mu)")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--excess", choices=("relative", "absolute"))
    ap.add_argument("--link", choices=("hinge", "logistic"))
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--mu", type=float)
    ap.add_argument("--json", action="store_true", help="print the feature vectors as JSON")
    ap.add_argument("--print", dest="print_sql", action="store_true", help="also print each block's SQL (reader form)")
    ap.add_argument("--definitions", action="store_true", help="print the nineteen definitions and exit")
    args = ap.parse_args(argv)
    if args.definitions:
        for name in FEATURE_NAMES:
            f = FEATURES[name]
            print(f"{f.number:2}. {name}: {f.definition}")
        return 0
    if not args.sql or not args.schema:
        ap.error("--sql and --schema are required (or --definitions)")
    from sqldecomp.boundary import Boundary
    from sqldecomp.optimize import prepare
    from sqldecomp.pipeline import load_any_schema
    from sqldecomp.importer import UnsupportedSQL
    from sqldecomp.optimize import IncompleteMemo
    try:  # an unreadable schema file or query file is a finding (a KeyError here is the file's shape)
        sql = Path(args.sql[1:]).read_text() if args.sql.startswith("@") else args.sql
        schema = load_any_schema(args.schema)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(json.dumps({"finding": f"cannot read the inputs: {type(e).__name__}: {e}"}, indent=1))
        return 1
    try:  # a refusal of the query is a finding, never a traceback; an internal error stays a traceback
        result, memo, blocks, printer = prepare(sql, schema)
        verifier = verifier_from_args(args) if human_options_given(args) else None
    except (UnsupportedSQL, IncompleteMemo, ValueError) as e:
        print(json.dumps({"finding": f"{type(e).__name__}: {e}"}, indent=1))
        return 1
    boundary = Boundary(memo, result, blocks, printer, None, schema)
    features = Features(memo, result, blocks, boundary)
    if args.block:
        wanted = []
        for spec in args.block:
            output, _, inputs = spec.partition(":")
            b = Block(output.strip(), frozenset(i.strip() for i in inputs.split(",") if i.strip()))
            why = blocks.why_not_a_block(b) if b.output in memo.groups else f"memo has no group {b.output!r}"
            if why:
                print(json.dumps({"finding": f"not a block: {why}", "groups": sorted(memo.groups)}, indent=1))
                return 1
            wanted.append(b)
    else:
        wanted = [Block(memo.root)] + [Block(g) for g in blocks.cut_points()]
    if args.json:
        out = {}
        for b in wanted:
            v = features.of_block(b)
            rec: dict[str, Any] = v.as_dict()
            if verifier is not None:
                r = verifier.risk(v)
                rec["risk"] = {"total": r.total, "F": r.F, "rows": [row.__dict__ for row in r.rows]}
            out[f"{b.output}<-{','.join(sorted(b.inputs))}"] = rec
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    lines = [f"# {len(wanted)} blocks; features " + ", ".join(f"{FEATURES[n].number}:{n}" for n in FEATURE_NAMES), ""]
    for b in wanted:
        lines += block_report(features, b, verifier, printer if args.print_sql else None)
    print("\n".join(lines))
    return 0


HUMAN_OPTIONS = ("theta_json", "tau", "beta", "excess", "link", "gamma", "mu")


def human_options_given(args) -> bool:
    """Whether a command line names any option of the human model's theta."""
    return any(getattr(args, k, None) is not None for k in HUMAN_OPTIONS)


def verifier_from_args(args) -> HumanVerifier:
    """theta from `--theta-json` (a JSON file) and / or the `--tau`, `--beta`, ... options. The thresholds are the
    file's (updated by `--tau`), else exactly the features `--tau` names, else the default ones (`human_default()`:
    every feature at `USER_STUDY_TAU`); every other field is the file's or the default, overridden by its option."""
    if getattr(args, "theta_json", None):
        base = HumanVerifier.load(args.theta_json)
    elif getattr(args, "tau", None):
        base = HumanVerifier()  # exactly the thresholds named
    else:
        base = human_default()
    d = base.to_json()
    if getattr(args, "tau", None):
        d["tau"].update(parse_tau(args.tau))
    for k in ("beta", "excess", "link", "gamma", "mu"):
        if getattr(args, k, None) is not None:
            d[k] = getattr(args, k)
    return HumanVerifier.from_json(d)


if __name__ == "__main__":
    sys.exit(main())
