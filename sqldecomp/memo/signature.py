"""Component 5, first half: the alias-free signature of an occurrence, same_task and same_task_across.

A memo group is one OCCURRENCE: it keeps the aliases of the place it was written at. The signature names the GROUP in
the design document's sense: what the occurrence computes, up to renaming of aliases and up to operator order.

DEFINITION. The signature of group g is a JSON text built from the order-free content of g:

    scan      ["scan", <placeholder>, <table>]
    region    ["region", sorted inputs, sorted conjuncts]; a conjunct is "F:<sql>" (a filter under g) or "J:<sql>"
              (a join condition under g). Operator order inside the region does not exist in this form.
    operator  ["op", <kind>, <sql>, <data>, [children in order]]
    ref       ["ref", <placeholder>, <column alias list or null>, child]; the CTE name and the text of the nested
              query are dropped, because the child says what the nested scope computes
    a child that is the plan of a nested scope is wrapped as ["scope", [[output name, <sql>], ...], child], because
    the scope's output list (which the skeleton does not hold) decides what `alias.column` means above it.

Every <sql> is the side table's canonical text, parsed with SQLGlot, with each column qualifier replaced: an
occurrence under g (a scan or a `ref` use) by its placeholder t0, t1, ...; an occurrence outside g (a correlated
reference) by "free:<query fingerprint>:<occurrence id>", which no other occurrence can share. A key of ORDER BY or
DISTINCT ON that stands alone as a bare name refers to an OUTPUT column of its scope (`ORDER BY total`); it is written
as that output's expression, because a select list that only renames columns is no plan node and the name alone would
not say what is sorted (two outputs of one name: unique; the output of a set operation is its own name, defined by the
branches' output lists). A VALUES list is named by its rows and column names.
Placeholders are chosen canonically: occurrences are ordered by (table, how they are used), where "how they are used"
is refined until stable from the conjuncts and operators that mention them; occurrences that remain tied are tried in
every order and the smallest text wins.

SAFETY. Equal signatures must imply equal up to renaming, so everything unclear makes the signature UNIQUE to the
occurrence (["unique", fingerprint, group, reason]): a fragment that contains a nested query, a qualifier that does not
resolve to exactly one occurrence, a bare name that could be a whole-row reference to an alias, a fragment that cannot
be read. A missed match only loses sharing; a wrong match would be a bug.

ACROSS IMPORTS. The signature never mentions an id of its own import except inside a "free:" or "unique" token, which
carry the import's fingerprint, so signatures of two imports over one schema are comparable: `same_task_across` is
component 5 for two occurrences of two different queries, up to renaming of aliases. It is the alias-FREE counterpart
of `sqldecomp.memo.align`, whose keys keep the aliases (a re-import of a print of the same query).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from math import factorial

import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

from sqldecomp.model import GroupId, ImportResult, Memo, Op1, Op2, Plan, Scan, plan_children

PERMUTATION_CAP = 5040  # tied occurrences are tried in every order up to this many orders (7 tied ones)
_MARK = re.compile(r"__occ_(\d+)__")
# Id lists, and facts the importer derives from the structure: the structure already says what they say. `exposes` of
# an outer join follows from its join_type and its children; `key` of a distinct from its ON list and the select list
# below it. Rendered as SQL they made every outer join and every DISTINCT ON unique ("bare name c could be a whole-row
# reference"): 2,704 of the 6,604 unique groups of the whole BEAVER dw set.
_DROP_KEYS = {"refs", "outer_refs", "subqueries", "exposes", "key"}
_VERBATIM_KEYS = {"alias", "name", "op", "join_type", "nulls", "source", "kind"}  # never SQL over FROM aliases
_NAME_TOKENS = (TokenType.VAR, TokenType.IDENTIFIER)
_SYM = "\x1f"  # separates the two operands of a top-level `=` / `<>` until the placeholders are known
_FREE_TOKEN = '"free:'  # how `_free` starts a correlated reference; its presence makes a signature import-bound
_KEY_KINDS = ("order", "distinct")  # operators whose keys may be bare names of OUTPUT columns of their scope
_FLIP = {exp.GT: exp.LT, exp.GTE: exp.LTE}  # x > y is written y < x


class _Unclear(Exception):
    """Something the signature cannot name safely; the occurrence gets a unique signature instead."""


def signature(memo: Memo, result: ImportResult, group: GroupId) -> str:
    """The alias-free name of what an occurrence computes: tables, every conjunct applied under it (filters and join
    conditions) and the opaque operators under it, with aliases replaced by canonical placeholders. Two occurrences
    with the same signature are the same GROUP in the design document's sense (equal up to renaming). Equal
    signatures must imply that; when unsure, make signatures differ."""
    if group not in memo.groups:
        raise KeyError(group)
    return _context(memo, result).signature(group)


def same_task(memo: Memo, result: ImportResult, g1: GroupId, g2: GroupId) -> bool:
    """True when g1 and g2 are the same occurrence or have the same signature."""
    if g1 not in memo.groups or g2 not in memo.groups:
        raise KeyError(g1 if g1 not in memo.groups else g2)
    return g1 == g2 or signature(memo, result, g1) == signature(memo, result, g2)


def same_task_across(memo_a: Memo, a: ImportResult, ga: GroupId, memo_b: Memo, b: ImportResult, gb: GroupId) -> bool:
    """Component 5 across two imports, alias-free: True when the occurrence `ga` of `a` and the occurrence `gb` of `b`
    are the same task, that is equivalent under R1-R4 up to alias renaming (the design document's alpha-equivalence).
    Decided by the signatures alone: True iff the two signatures are equal and neither is unique.

    CONSERVATIVE: a False can be a missed match, a True is never wrong. A True rests on the invariant of `signature`
    (equal signatures imply equal up to renaming) and on nothing else; in particular the fingerprint inside a "free:"
    token is never compared. Never matched, whatever the two groups compute:
    - a unique signature (a group with a predicate subquery, an unreadable fragment, ...), not even against the same
      query imported again; the same occurrence of ONE memo is the one exception, as in `same_task`;
    - a correlated body of another TEXT: its outer reference is a "free:" token bound to one occurrence of its own
      import, so it matches only the same text (same dialect, same side table, compared exactly) imported again;
    - two imports of different dialects (equal Postgres-canonical texts can mean other rows under MySQL's collations);
      ties beyond PERMUTATION_CAP (sound but not canonical).
    Both imports must be over ONE schema: a base table is named by its name only, and nothing here checks it.

    What is compared is the RELATION the group computes, as `same_task` does: the root groups of `SELECT c.c_id FROM
    customers c` and of `SELECT c.c_name FROM customers c` are one task (a rename-or-drop projection is not a node), so
    a caller that certifies a whole query compares the output lists as well, THROUGH the renaming (`placeholders`)
    and never by text: `SELECT a.o_id FROM orders a JOIN orders b ON a.o_cust = b.o_cust WHERE a.o_total > 5 AND
    b.o_total < 3` and the same statement with its two filters swapped are one relation up to the renaming a <-> b,
    both select `a.o_id`, and they return different rows; through the placeholders the outputs read `t0.o_id` and
    `t1.o_id` (`test_memo_same_task_across.py` shows the composition). Unlike `equivalent` this answer is
    symmetric, does not depend on aliases and is True both ways around a removable cross join; unlike `equivalent` it
    does not look into the other memo, so it finds exactly what the order-free signature names (every join order,
    every filter placement inside a region). KeyError when a group is not in its memo."""
    if ga not in memo_a.groups:
        raise KeyError(ga)
    if gb not in memo_b.groups:
        raise KeyError(gb)
    if memo_a is memo_b and a is b and ga == gb:
        return True
    if a.dialect != b.dialect:
        return False
    sa, sb = signature(memo_a, a, ga), signature(memo_b, b, gb)
    if sa != sb or is_unique(sa):
        return False
    if _FREE_TOKEN in sa and not _same_import(a, b):
        return False
    return True


def _same_import(a: ImportResult, b: ImportResult) -> bool:
    """The two results are one import: the same text, dialect and side table (what the fingerprint hashes, compared
    exactly). The importer gives the same text the same ids, so a "free:" token then names the same occurrence."""
    if a is b:
        return True
    return (a.sql == b.sql and a.dialect == b.dialect
            and sorted((e.id, e.kind, e.sql) for e in a.side.values()) == sorted((e.id, e.kind, e.sql) for e in b.side.values()))


def placeholders(memo: Memo, result: ImportResult, group: GroupId) -> dict[str, str] | None:
    """The renaming behind a signature: occurrence id (a scan's occ or a `ref` operator id) -> placeholder. Two
    occurrences with the same signature compute the same thing THROUGH these renamings, so whoever carries anything
    alias-specific from one occurrence to the other (a printed block, a sub-question) composes the two dicts.
    None when the signature is unique. With tied occurrences several renamings give the same text; this is one."""
    if group not in memo.groups:
        raise KeyError(group)
    ctx = _context(memo, result)
    ctx.signature(group)
    return ctx._placeholders.get(group)


def is_unique(sig: str) -> bool:
    """True when a signature is the 'unclear, never shared' kind."""
    return sig.startswith('["unique"')


def _context(memo: Memo, result: ImportResult) -> "_Context":
    held = getattr(memo, "_sig_ctx", None)
    if held is None or held[0] is not result:
        held = (result, _Context(memo, result))
        memo._sig_ctx = held
    return held[1]


class _Context:
    def __init__(self, memo: Memo, result: ImportResult):
        self.memo, self.result, self.side = memo, result, result.side
        # names this query in "free:" and "unique" tokens, so that signatures of two queries can share a cache safely
        seed = json.dumps([result.sql, sorted((e.id, e.kind, e.sql) for e in result.side.values())])
        self.fingerprint = hashlib.sha1(seed.encode()).hexdigest()[:12]
        self.scopes = {s.id: s for s in result.scopes}
        by_plan: dict[Plan, list[str]] = {}
        for s in result.scopes:
            by_plan.setdefault(s.plan, []).append(s.id)
        # home scope of every id, and the FROM aliases of every scope: walk a scope's plan, stop at nested scopes
        self.home: dict[str, str] = {}
        self.aliases: dict[str, dict[str, list[str]]] = {s.id: {} for s in result.scopes}
        self.tables: dict[str, str] = {}
        for s in result.scopes:
            todo = [s.plan]
            while todo:
                n = todo.pop()
                if n is not s.plan and any(x != s.id for x in by_plan.get(n, ())):
                    continue  # the plan of a nested scope: its ids and aliases belong to that scope
                entry = None
                if isinstance(n, Scan):
                    self.tables[n.occ] = self._relation(n)
                    self.home.setdefault(n.occ, s.id)
                    entry = (n.occ, self.side.get(n.occ))
                elif isinstance(n, (Op1, Op2)):
                    self.home.setdefault(n.op, s.id)
                    if n.op in self.side and self.side[n.op].kind == "ref":
                        entry = (n.op, self.side[n.op])
                elif hasattr(n, "conjunct"):
                    self.home.setdefault(n.conjunct, s.id)
                for c in getattr(n, "conjuncts", ()):
                    self.home.setdefault(c, s.id)
                if entry is not None and entry[1] is not None:
                    # a FROM item without an alias is qualified by its own name (`orders.o_total`)
                    name = entry[1].data.get("alias") or (entry[1].data.get("table") if isinstance(n, Scan) else None)
                    if name:
                        self.aliases[s.id].setdefault(name, []).append(entry[0])
                todo.extend(plan_children(n))
        self.occ_ids: list[str] = []  # marker number -> occurrence id
        self.occ_no: dict[str, int] = {}
        self.group_scopes: dict[GroupId, list[str]] = {}
        for s in result.scopes:
            if s.kind != "root":
                g = memo.group_of(s.plan)
                if g is not None:
                    self.group_scopes.setdefault(g, []).append(s.id)
        self._templates: dict[tuple[str, str], str] = {}
        self._content: dict[GroupId, tuple] = {}
        self._signatures: dict[GroupId, str] = {}
        self._placeholders: dict[GroupId, dict[str, str]] = {}

    # ---- signature of one group ------------------------------------------------------------------------------
    def signature(self, g: GroupId) -> str:
        if g not in self._signatures:
            try:
                self._signatures[g] = self._canonical(g)
            except _Unclear as why:
                self._signatures[g] = json.dumps(["unique", self.fingerprint, g, str(why)])
        return self._signatures[g]

    def _canonical(self, g: GroupId) -> str:
        tree = self._tree(g)  # nested lists whose strings carry __occ_N__ markers
        bound = sorted(_occurrences(tree))  # marker numbers of the occurrences under g
        mentions = _fragments(tree)
        classes = self._refine(bound, mentions)
        orders = 1
        for c in classes:
            orders *= factorial(len(c))
        if orders > PERMUTATION_CAP:  # sound but no longer canonical: ties keep the order of their ids
            classes = [[o] for c in classes for o in sorted(c, key=lambda n: self.occ_ids[n])]
        best = None
        for combo in itertools.product(*[itertools.permutations(c) for c in classes]):
            names = {o: f"t{k}" for k, o in enumerate(o for c in combo for o in c)}
            text = json.dumps(self._render(tree, names), ensure_ascii=False)
            if best is None or text < best:
                best = text
                self._placeholders[g] = {self.occ_ids[o]: name for o, name in names.items()}
        return best

    def _refine(self, bound: list[int], mentions: list[str]) -> list[list[int]]:
        """Partition the occurrences under g into ordered classes, by table and then by how they are used, without
        looking at aliases: colour refinement over the fragments that mention them."""
        label = {o: self._label(o) for o in bound}
        colour = {o: label[o] for o in bound}
        uses = {o: [m for m in mentions if f"__occ_{o}__" in m] for o in bound}
        while True:
            def view(o: int) -> tuple:
                def sub(m: re.Match) -> str:
                    n = int(m.group(1))
                    return "@" if n == o else f"<{colour[n]}>" if n in colour else self._free(n)
                return (label[o], tuple(sorted(_symmetric(_MARK.sub(sub, u)) for u in uses[o])))
            views = {o: view(o) for o in bound}
            ranks = {v: k for k, v in enumerate(sorted(set(views.values())))}
            new = {o: f"{label[o]}#{ranks[views[o]]}" for o in bound}
            if len(set(new.values())) == len(set(colour.values())):
                break
            colour = new
        classes: dict[tuple, list[int]] = {}
        for o in bound:
            classes.setdefault(views[o], []).append(o)
        return [classes[v] for v in sorted(classes)]

    def _label(self, o: int) -> str:
        occ = self.occ_ids[o]
        return f"scan:{self.tables[occ]}" if occ in self.tables else "ref"

    def _free(self, n: int) -> str:
        return f'"free:{self.fingerprint}:{self.occ_ids[n]}"'

    def _render(self, tree, names: dict[int, str]):
        def text(s: str) -> str:
            return _symmetric(_MARK.sub(lambda m: names.get(int(m.group(1))) or self._free(int(m.group(1))), s))

        def walk(x):
            if isinstance(x, str):
                return text(x)
            if isinstance(x, (list, tuple)):
                out = [walk(y) for y in x]
                if isinstance(x, tuple) and x[0] == "region":  # the one place where order must not exist
                    out[1] = sorted(out[1], key=lambda i: json.dumps(i, ensure_ascii=False))
                    out[2] = sorted(out[2])
                return out
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            return x
        return walk(tree)

    def _relation(self, scan: Scan) -> str:
        """What a scan reads. For a base table its name. A VALUES list in FROM is the pseudo table `$values` for every
        such list, so its CONTENT and its column names have to be part of the name, or `(VALUES (1)) v(a)` and
        `(VALUES (2)) w(b)` would be one task."""
        entry = self.side.get(scan.occ)
        if entry is None or entry.data.get("table") is not None or not entry.sql:
            return scan.table
        try:
            item = sqlglot.parse_one("SELECT * FROM " + entry.sql, read=self.result.dialect or "postgres").args["from_"].this
            alias = item.args.get("alias")
            columns = [c.name for c in (alias.args.get("columns") or [])] if alias is not None else []
            if alias is not None:
                item.set("alias", None)
            return f"{scan.table} {item.sql(dialect=self.result.dialect or 'postgres')} ({', '.join(columns)})"
        except Exception:  # unreadable: the occurrence id makes it a relation of its own, so it is never shared
            return f"{scan.table} {scan.occ}"

    # ---- order-free content of a group, as a marker-carrying tree ----------------------------------------------
    # Structure nodes are TUPLES ("scan" | "ref" | "op" | "region" | "scope", ...); side-table data stays lists and
    # dicts, so a GROUP BY column that happens to be called `region` can never be taken for structure.
    def _tree(self, g: GroupId):
        e = self.memo.groups[g][0]
        if e.kind == "scan":
            return ("scan", self._mark(e.arg), self.tables.get(e.arg) or self._fail(f"scan {e.arg} is in no scope plan"))
        if e.kind in ("op1", "op2"):
            entry = self.side.get(e.arg) or self._fail(f"operator {e.arg} has no side entry")
            kids = [self._child(c) for c in e.children]
            if entry.kind == "ref":
                nested = entry.data.get("scope")
                if self.group_scopes.get(e.children[0]) != [nested]:
                    self._fail(f"ref {e.arg} does not sit on the plan of its scope")
                return ("ref", self._mark(e.arg), _alias_columns(entry, self.result.dialect), kids[0])
            scope = self.home.get(e.arg) or self._fail(f"operator {e.arg} is in no scope plan")
            sql, data = self._resolve_keys(scope, entry) if entry.kind in _KEY_KINDS else (entry.sql, entry.data)
            return ("op", entry.kind, self._sql(scope, sql), self._data(scope, data), kids)
        inputs, filters, joins, todo, seen = [], set(), set(), [g], set()
        while todo:  # the region under g, read from one expression per group (every expression agrees, by M1)
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            ex = self.memo.groups[x][0]
            if ex.kind == "filter":
                filters.add(ex.arg)
            elif ex.kind == "join":
                joins.update(ex.conjuncts)
            else:
                inputs.append(self._tree(x))
                continue
            todo.extend(ex.children)
        conjuncts = [f"{tag}:{self._conjunct(c)}" for tag, cs in (("F", filters), ("J", joins)) for c in sorted(cs)]
        return ("region", inputs, conjuncts)

    def _child(self, g: GroupId):
        scopes = self.group_scopes.get(g)
        if not scopes:
            return self._tree(g)
        if len(scopes) > 1:
            self._fail(f"group {g} is the plan of several scopes")
        s = self.scopes[scopes[0]]
        return ("scope", [[o.name, self._sql(s.id, o.sql)] for o in s.outputs], self._tree(g))

    def _resolve_keys(self, scope: str, entry) -> tuple[str, dict]:
        """`ORDER BY total`, `DISTINCT ON (k)`: a key that stands alone as a bare name is a reference to an OUTPUT column
        of the operator's scope (Postgres resolves such a name against the output columns first; the importer leaves
        it unqualified). When the select list only renames columns it is no plan node, so the name alone would not say
        what is sorted, and `c.c_name AS k ... ORDER BY k LIMIT 1` would be the task of `c.c_region AS k ... ORDER BY
        k LIMIT 1`. The key is therefore written as the output's expression, in the operator's text and in its data.
        A name that several outputs carry is unclear (unique). The output of a set operation is its own bare name and
        is defined by the branches' output lists, so it stays."""
        outputs: dict[str, list[str]] = {}
        for o in (self.scopes[scope].outputs if scope in self.scopes else ()):
            outputs.setdefault(o.name, []).append(o.sql)

        def expression(name: str | None) -> str | None:
            if name is None or name not in outputs:
                return None
            if len(outputs[name]) != 1:
                self._fail(f"key {name} names {len(outputs[name])} outputs of scope {scope}")
            expr = outputs[name][0]
            return None if _bare_name(expr) is not None else expr

        data = dict(entry.data)
        if entry.kind == "order":
            data["keys"] = [{**k, "sql": expression(_bare_name(k["sql"])) or k["sql"]} for k in data.get("keys") or []]
        elif data.get("on"):
            data["on"] = [expression(_bare_name(k)) or k for k in data["on"]]
        try:
            tokens = sqlglot.tokenize(entry.sql, dialect=self.result.dialect)
        except Exception:
            self._fail("operator text cannot be tokenized")
        text, shift = entry.sql, 0
        for i, tok in enumerate(tokens):  # a bare key is always a whole key: the importer refuses a name inside an expression
            prev = tokens[i - 1].text.upper() if i else "ORDER BY"
            nxt = tokens[i + 1].text.upper() if i + 1 < len(tokens) else ","
            if not _word(tok) or prev not in ("ORDER BY", "BY", ",", "(") or nxt not in (",", ")", "DESC", "ASC", "NULLS"):
                continue
            expr = expression(tok.text) or expression(tok.text.lower())
            if expr is not None:
                text = text[:tok.start + shift] + expr + text[tok.end + 1 + shift:]
                shift += len(expr) - (tok.end + 1 - tok.start)
        return text, data

    def _conjunct(self, c: str) -> str:
        entry = self.side.get(c)
        if entry is None or entry.kind != "conjunct":
            self._fail(f"conjunct {c} has no side entry")
        if entry.data.get("subqueries"):
            self._fail(f"conjunct {c} contains a subquery")
        return self._sql(self.home.get(c) or self._fail(f"conjunct {c} is in no scope plan"), entry.sql)

    def _data(self, scope: str, value, key: str | None = None):
        if isinstance(value, dict):
            return {k: self._data(scope, v, k) for k, v in sorted(value.items()) if k not in _DROP_KEYS}
        if isinstance(value, (list, tuple)):
            return [self._data(scope, v, key) for v in value]
        if isinstance(value, str) and key not in _VERBATIM_KEYS:
            return self._sql(scope, value)
        return value

    def _mark(self, occ: str) -> str:
        if occ not in self.occ_no:
            self.occ_no[occ] = len(self.occ_ids)
            self.occ_ids.append(occ)
        return f"__occ_{self.occ_no[occ]}__"

    @staticmethod
    def _fail(why: str):
        raise _Unclear(why)

    # ---- one SQL fragment: qualifiers -> markers ---------------------------------------------------------------
    def _sql(self, scope: str, sql: str) -> str:
        key = (scope, sql)
        if key not in self._templates:
            if "__occ_" in sql or _SYM in sql:
                self._fail("fragment uses the marker prefix")
            try:
                parsed = sqlglot.parse_one(sql, dialect=self.result.dialect) if sql.strip() else None
            except Exception:  # not an expression on its own ("UNION ALL", "LEFT JOIN ...", an elided SELECT ...)
                parsed = None
            # an operator's text may be a whole SELECT with its own FROM clause: that defines the scope's aliases
            # rather than using them, which only the token reading handles (`AS alias` is renamed like a use)
            whole = parsed is None or isinstance(parsed, exp.Query)
            self._templates[key] = self._from_tokens(scope, sql) if whole else self._from_ast(scope, parsed)
        return self._templates[key]

    def _chain(self, scope: str):
        """The scopes whose FROM aliases are visible inside `scope`, innermost first. A CTE body sees only itself."""
        seen: set[str] = set()
        s = self.scopes.get(scope)
        while s is not None and s.id not in seen:
            seen.add(s.id)
            yield s
            s = self.scopes.get(s.parent) if s.kind != "cte" and s.parent is not None else None

    def _resolve(self, scope: str, alias: str) -> str:
        """The occurrence a FROM alias means inside `scope`: innermost scope first, then the scopes around it."""
        for s in self._chain(scope):
            found = self.aliases[s.id].get(alias)
            if found:
                if len(found) > 1:
                    self._fail(f"alias {alias} names several FROM items of scope {s.id}")
                return self._mark(found[0])
        self._fail(f"qualifier {alias} does not resolve in scope {scope}")

    def _visible(self, scope: str) -> set[str]:
        return {a for s in self._chain(scope) for a in self.aliases[s.id]}

    def _from_ast(self, scope: str, tree: exp.Expression) -> str:
        if any(isinstance(n, (exp.Query, exp.Table)) for n in tree.walk()):
            self._fail("fragment contains a nested query")
        visible = self._visible(scope)
        for col in list(tree.find_all(exp.Column)):
            if col.args.get("db") or col.args.get("catalog"):
                self._fail(f"column {col.sql()} has more than one qualifier")
            if col.table:
                col.set("table", exp.to_identifier(self._resolve(scope, col.table)))
            elif col.name in visible:
                self._fail(f"bare name {col.name} could be a whole-row reference to a FROM alias")
        # exact symmetries of a top-level comparison, so that `b.id = a.id` and `a.id = b.id` are one text:
        # x > y is written y < x; the operands of = and <> are ordered once the placeholders are known (_symmetric)
        if type(tree) in _FLIP:
            tree = _FLIP[type(tree)](this=tree.expression, expression=tree.this)
        if isinstance(tree, (exp.EQ, exp.NEQ)):
            op = "=" if isinstance(tree, exp.EQ) else "<>"
            return _SYM.join((tree.this.sql(dialect=self.result.dialect), op, tree.expression.sql(dialect=self.result.dialect)))
        return tree.sql(dialect=self.result.dialect)

    def _from_tokens(self, scope: str, sql: str) -> str:
        """For operator texts that are not an expression on their own. Only the shape `alias . name` is renamed, a
        nested SELECT is refused (its own aliases could shadow ours), and every dotted head must resolve."""
        try:
            tokens = sqlglot.tokenize(sql, dialect=self.result.dialect)
        except Exception:
            self._fail("fragment cannot be tokenized")
        visible = self._visible(scope)
        out: list[str] = []
        for i, t in enumerate(tokens):
            prev = tokens[i - 1].token_type if i else None
            nxt = tokens[i + 1].token_type if i + 1 < len(tokens) else None
            if t.token_type == TokenType.SELECT and i > 0:
                self._fail("fragment contains a nested query")
            if _word(t) and prev != TokenType.DOT:
                if nxt == TokenType.DOT and i + 2 < len(tokens) and (_word(tokens[i + 2]) or tokens[i + 2].token_type == TokenType.STAR):
                    after = tokens[i + 3].token_type if i + 3 < len(tokens) else None
                    if after == TokenType.L_PAREN or (after == TokenType.DOT and i + 4 < len(tokens) and _word(tokens[i + 4])):
                        self._fail(f"dotted name at {t.text} is not alias.column")
                    out.append(self._resolve(scope, t.text))
                    continue
                if t.text in visible:
                    if prev != TokenType.ALIAS:
                        self._fail(f"bare name {t.text} could be a whole-row reference to a FROM alias")
                    out.append(self._resolve(scope, t.text))
                    continue
            if t.token_type == TokenType.STRING:
                out.append("'" + t.text.replace("'", "''") + "'")
            elif t.token_type == TokenType.IDENTIFIER:
                out.append('"' + t.text.replace('"', '""') + '"')
            else:
                out.append(t.text)
        return " ".join(out)


def _bare_name(text: str) -> str | None:
    """The name when `text` is one bare identifier (`k`, `"K"`), else None."""
    try:
        tokens = sqlglot.tokenize(text, dialect="postgres")
    except Exception:
        return None
    return tokens[0].text if len(tokens) == 1 and _word(tokens[0]) else None


def _word(token) -> bool:
    """A token that can be a name: an identifier, or any word (SQL keywords are legal alias and column names)."""
    return token.token_type in _NAME_TOKENS or bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", token.text))


def _alias_columns(entry, dialect: str) -> list[str] | None:
    """The column alias list of a `ref` use (`big AS b(x, y)`), or None. Everything else in a ref's text (the CTE
    name, the nested query, the alias) is dropped on purpose: the child says what the nested scope computes. The text
    must look like `<name or (...)> [AS] alias [(columns)]`, optionally LATERAL; anything else is unclear."""
    alias = entry.data.get("alias")
    try:
        tokens = sqlglot.tokenize(entry.sql, dialect=dialect)
    except Exception:
        raise _Unclear("ref text cannot be tokenized") from None
    depth, at = 0, None
    for i, t in enumerate(tokens):
        if t.token_type == TokenType.L_PAREN:
            depth += 1
        elif t.token_type == TokenType.R_PAREN:
            depth -= 1
        elif depth == 0 and t.token_type in _NAME_TOKENS and t.text == alias:
            at = i
    if at is None:
        raise _Unclear("ref text does not show its alias")
    head = [t for t in tokens[:at] if t.token_type not in (TokenType.ALIAS, TokenType.LATERAL)]
    if head and head[0].token_type == TokenType.L_PAREN:  # one parenthesised body and nothing else
        depth, closes = 0, []
        for k, t in enumerate(head):
            depth += (t.token_type == TokenType.L_PAREN) - (t.token_type == TokenType.R_PAREN)
            if depth == 0:
                closes.append(k)
        ok = closes == [len(head) - 1]
    else:  # a (possibly dotted) name: a table function such as generate_series(1, 5) is NOT a plain ref
        ok = all(_word(t) or t.token_type == TokenType.DOT for t in head)
    if not ok:
        raise _Unclear("ref is not a table or a subquery")
    rest = tokens[at + 1:]
    if not rest:
        return None
    names = [t.text for t in rest[1:-1] if t.token_type in _NAME_TOKENS]
    shape_ok = (rest[0].token_type == TokenType.L_PAREN and rest[-1].token_type == TokenType.R_PAREN and names
                and all(t.token_type in _NAME_TOKENS if k % 2 == 0 else t.token_type == TokenType.COMMA
                        for k, t in enumerate(rest[1:-1])))
    if not shape_ok:
        raise _Unclear("ref text has something after its alias")
    return names


def _symmetric(text: str) -> str:
    """Order the two operands of a top-level `=` / `<>` (kept apart by _SYM) now that their final text is known."""
    if _SYM not in text:
        return text
    head, body = (text[:2], text[2:]) if text[:2] in ("F:", "J:") else ("", text)
    left, op, right = body.split(_SYM)
    left, right = sorted((left, right), key=lambda t: (not t[:1].isalpha(), t))  # `t0.x = 'EU'`, not `'EU' = t0.x`
    return f"{head}{left} {op} {right}"


def _occurrences(tree) -> set[int]:
    """Marker numbers of the occurrences a tree BINDS: its scans and its ref uses (position 1 of their node)."""
    out: set[int] = set()
    if isinstance(tree, (list, tuple)):
        if isinstance(tree, tuple) and tree[0] in ("scan", "ref"):
            out.add(int(_MARK.fullmatch(tree[1]).group(1)))
        for x in tree:
            out |= _occurrences(x)
    elif isinstance(tree, dict):
        for x in tree.values():
            out |= _occurrences(x)
    return out


def _fragments(tree) -> list[str]:
    """Every string of a tree that mentions an occurrence, except the binding positions themselves."""
    out: list[str] = []
    if isinstance(tree, str):
        if _MARK.search(tree):
            out.append(tree)
    elif isinstance(tree, (list, tuple)):
        skip = 1 if isinstance(tree, tuple) and tree[0] in ("scan", "ref") else None
        for k, x in enumerate(tree):
            if k != skip:
                out.extend(_fragments(x))
    elif isinstance(tree, dict):
        for x in tree.values():
            out.extend(_fragments(x))
    return out
