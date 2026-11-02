"""The e-graph behind the memo: egglog encoding of the skeleton plan, rules R1-R4, saturation, raw read-out.

This is the only module that touches egglog (pinned to 13.2.0), so the memo engine can be replaced behind it.

ENCODING. One sort `Rel` with five constructors that mirror the skeleton: scan(i), join(l, r), filter(x, c),
op1(x, o), op2(o, l, r). Every id is an i64 index into a Python table, so no id text ever has to survive egglog's
string syntax. Two facts drive the rules: covers(term, input) for the leaves of a region, propagated through join
and filter only, and owner(conjunct, input) for local conjuncts.

THE JOIN CONDITION IS DERIVED, NOT CARRIED. A join term is `join(l, r)` with no condition. The contract guarantees
that every region input is one distinct occurrence, so the condition of a join is a function of the inputs its two
sides cover: exactly the edges whose inputs all lie in cover(l) | cover(r) but neither in cover(l) alone nor in
cover(r) alone. Two join terms over the same two classes therefore always have the same condition, which is why
leaving it off the term cannot merge two different joins. The read-out (sqldecomp.memo.build) recomputes that set
for every join expression and stores it in MExpr.conjuncts. The same test, inside egglog, is the guard of R2: the
new inner join must have at least one spanning edge. A binary edge needs one input on each side. An edge over m > 2
inputs (a hyperedge) needs all m inputs covered with at least one on each side, which is one rule per split s | m-s
of its inputs over a relation that holds every permutation of the edge's inputs.

RULES (no rule mentions op1 / op2, so every operator, including `ref`, is a barrier):
  R1  join(a, b) = join(b, a)
  R2  join(join(a, b), c) = join(a, join(b, c))  when an edge spans (b, c); and the mirror image when one spans (a, b)
  R3  filter(filter(a, p), q) = filter(filter(a, q), p)
  R4  join(filter(a, p), b) -> filter(join(a, b), p)  always;  back down only onto the side that covers owner(p).
A conjunct with no owner has no owner fact, so it never moves down. Cross joins that the query contains stay,
because saturation starts from P0 and R2 only ever refuses to create a join without a condition.

ENGINE REGISTRATION. egglog-python's `register` turns every
sub-expression that occurs twice in one registered expression into a global `let` named `$__expr_<n>` (the covers
fact of a scan, `covers(scan(4), 4)`, holds the literal 4 twice). Inside egglog 13.2.0 a reference to a global is a
call of a zero-argument function, and every call in a command gets a temporary named callee name + a per-name counter,
with no separator and the count 0 left off (`SymbolGen::fresh`, src/util.rs; `to_core_actions`, src/core.rs, which
never checks that the name is free). So the temporaries of `$__expr_1` at its 14th use and of `$__expr_11` at its 4th
are both `$__expr_114`: one variable, and the value bound later wins for both arguments of `(owner $__expr_1
$__expr_11)`, stored as owner(4, 4). It needs two callee names of which one is the other plus digits, and the right
number of earlier uses; six commands reproduce it. Which sub-expressions
egglog-python let-binds depends on set iteration order. A name collision can
corrupt owner or cover facts, yielding an incomplete or unsound memo.

So `saturate` hands the plans and the facts to the engine through the low-level program as plain expression actions
with every literal inline, never a global let, and then reads the fact tables and the registered plans back from the
engine before any rule runs: a fact the engine does not hold as given, a plan it does not hold, or an e-node it holds
that was not registered is `FactsNotHeld`, never a memo. What keeps this path safe is that no command sent from here
holds two callee names of which one is the other plus digits (`_egg_names` checks it: scan / join / filter / op1 /
op2, covers, owner, edge<m>) and that no rule action repeats a ground term (egglog-python would let-bind it; the
rules bind variables only, and the test above asserts that no `let` is sent). The freeze read-out is therefore
required at registration time; the public `function_values` read-out stays the fallback of the final read-out only.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Hashable, Iterable

from egglog import EGraph, Expr, bindings, get_callable_args, get_literal_value, i64, i64Like, relation, rewrite, rule, var, vars_, eq

KINDS = ("scan", "join", "filter", "op1", "op2")


class Rel(Expr):
    @classmethod
    def scan(cls, inp: i64Like) -> Rel: ...

    @classmethod
    def join(cls, left: Rel, right: Rel) -> Rel: ...

    @classmethod
    def filter(cls, inp: Rel, conjunct: i64Like) -> Rel: ...

    @classmethod
    def op1(cls, inp: Rel, op: i64Like) -> Rel: ...

    @classmethod
    def op2(cls, op: i64Like, left: Rel, right: Rel) -> Rel: ...


covers = relation("covers", Rel, i64)
owner = relation("owner", i64, i64)
_edge_relations: dict[int, object] = {}
_SPAN = bindings.RustSpan("", 0, 0)


class FactsNotHeld(AssertionError):
    """Right after the plans and facts were registered, the engine's tables do not hold them as given (an engine
    defect, see the module header). Raised before any rule runs; never a memo."""

    def __init__(self, problems: list[str]):
        super().__init__("the e-graph does not hold the facts it was given: " + "; ".join(problems))
        self.problems = problems


def _edge_relation(arity: int):
    """edge_m(i1, ..., im): one relation per edge arity, holding every permutation of each edge's inputs."""
    if arity not in _edge_relations:
        _edge_relations[arity] = relation(f"edge{arity}", *([i64] * arity))
    return _edge_relations[arity]


# A term is a nested tuple; build.py produces these from Plan values:
#   ("scan", input_idx) | ("join", l, r) | ("filter", x, conj_idx) | ("op1", x, op_idx) | ("op2", op_idx, l, r)
Term = tuple


@dataclass
class RawRow:
    kind: str  # one of KINDS
    arg: int | None  # scan: input index, filter: conjunct index, op1 / op2: operator index, join: None
    children: tuple[Hashable, ...]  # class keys
    cls: Hashable  # class key of the e-class this e-node belongs to


@dataclass
class RawEGraph:
    rows: list[RawRow]
    complete: bool
    iterations: int
    seconds: float
    readout: str  # "freeze" (fast, raw tables) or "function_values" (public API)
    stats: dict[str, float] = field(default_factory=dict)

    def class_of(self, term: Term, _memo: dict | None = None) -> Hashable | None:
        """The class key of a term, resolved bottom-up against the rows; None if the e-graph does not hold it."""
        index = getattr(self, "_index", None)
        if index is None:
            index = {(r.kind, r.arg, r.children): r.cls for r in self.rows}
            self._index = index
        memo = {} if _memo is None else _memo
        if term in memo:
            return memo[term]
        kind = term[0]
        if kind == "scan":
            key = ("scan", term[1], ())
        elif kind == "join":
            kids = (self.class_of(term[1], memo), self.class_of(term[2], memo))
            key = None if None in kids else ("join", None, kids)
        elif kind in ("filter", "op1"):
            kid = self.class_of(term[1], memo)
            key = None if kid is None else (kind, term[2], (kid,))
        else:
            kids = (self.class_of(term[2], memo), self.class_of(term[3], memo))
            key = None if None in kids else ("op2", term[1], kids)
        memo[term] = None if key is None else index.get(key)
        return memo[term]


def saturate(plans: Iterable[Term], cover_facts: Iterable[tuple[Term, int]], edges: Iterable[tuple[int, ...]],
             owners: Iterable[tuple[int, int]], *, budget_s: float, readout: str = "auto") -> RawEGraph:
    """Register every plan in ONE e-graph, add the facts, run the rules to saturation or to the budget, read out.

    `edges` are tuples of input indices (two or more distinct inputs each); `owners` are (conjunct, input) pairs.
    The budget is checked between iterations, so one long iteration can overshoot it."""
    plans, cover_facts, edges, owners = list(plans), list(cover_facts), [tuple(e) for e in edges], list(owners)
    arities = sorted({len(e) for e in edges})
    egraph = EGraph()
    _register_rules(egraph, arities)

    # plans and facts as plain expression actions, every literal inline, no global let (module header)
    names = _egg_names(egraph, arities)
    cache: dict[Term, bindings.Call] = {}
    calls = [_egg_term(p, cache, names) for p in plans]
    calls += [bindings.Call(_SPAN, names["covers"], [_egg_term(t, cache, names), _egg_int(inp)]) for t, inp in cover_facts]
    for e in edges:
        for perm in sorted(set(itertools.permutations(e))):
            calls.append(bindings.Call(_SPAN, names[f"edge{len(e)}"], [_egg_int(x) for x in perm]))
    calls += [bindings.Call(_SPAN, names["owner"], [_egg_int(conj), _egg_int(inp)]) for conj, inp in owners]
    egraph._run_program(*[bindings.ActionCommand(bindings.Expr_(_SPAN, c)) for c in calls])
    _check_facts(egraph, names, plans, cover_facts, edges, owners)

    start = time.perf_counter()
    iterations, complete = 0, True
    while True:
        if time.perf_counter() - start >= budget_s:
            complete = False
            break
        report = egraph.run(1)
        iterations += 1
        if not report.updated:
            break
    seconds = time.perf_counter() - start

    t0 = time.perf_counter()
    if readout == "function_values":
        rows, used = _read_function_values(egraph), "function_values"
    else:
        try:
            rows, used = _read_freeze(egraph), "freeze"
        except Exception:  # the raw-table API is private to egglog; fall back to the public one
            if readout == "freeze":
                raise
            rows, used = _read_function_values(egraph), "function_values"
    return RawEGraph(rows, complete, iterations, seconds, used, {"readout_s": time.perf_counter() - t0})


def _egg_names(egraph: EGraph, arities: list[int]) -> dict[str, str]:
    """The engine's names of the constructors and relations (declaring each to the engine on first use). No name may
    be another plus digits: that is the shape the engine's temporary naming trips over (module header)."""
    names = {kind: egraph._callable_to_egg(ctor)[1] for kind, ctor in _CTORS.items()}
    names["covers"] = egraph._callable_to_egg(covers)[1]
    names["owner"] = egraph._callable_to_egg(owner)[1]
    for m in arities:
        names[f"edge{m}"] = egraph._callable_to_egg(_edge_relation(m))[1]
    check_callee_names(names.values())
    return names


def check_callee_names(names: Iterable[str]) -> None:
    """Refuse a set of engine function names of which one is another plus digits (the engine's temporaries of the two
    can collide, module header). The names used here never are; this guards a future rename."""
    seen = sorted(set(names))
    for a in seen:
        for b in seen:
            if a != b and b.startswith(a) and b[len(a):].isdigit():
                raise ValueError(f"engine names {a!r} and {b!r} differ by digits only: the engine's temporaries can collide")


def _egg_int(value: int) -> bindings.Lit:
    return bindings.Lit(_SPAN, bindings.Int(int(value)))


def _egg_term(term: Term, cache: dict[Term, bindings.Call], names: dict[str, str]) -> bindings.Call:
    """A term as a low-level engine expression with its literals inline (one Python object per distinct term)."""
    if term in cache:
        return cache[term]
    kind = term[0]
    if kind == "scan":
        out = bindings.Call(_SPAN, names["scan"], [_egg_int(term[1])])
    elif kind == "join":
        out = bindings.Call(_SPAN, names["join"], [_egg_term(term[1], cache, names), _egg_term(term[2], cache, names)])
    elif kind == "filter":
        out = bindings.Call(_SPAN, names["filter"], [_egg_term(term[1], cache, names), _egg_int(term[2])])
    elif kind == "op1":
        out = bindings.Call(_SPAN, names["op1"], [_egg_term(term[1], cache, names), _egg_int(term[2])])
    elif kind == "op2":
        out = bindings.Call(_SPAN, names["op2"], [_egg_int(term[1]), _egg_term(term[2], cache, names), _egg_term(term[3], cache, names)])
    else:
        raise ValueError(f"unknown term kind {kind!r}")
    cache[term] = out
    return out


def _check_facts(egraph: EGraph, names: dict[str, str], plans: list[Term], cover_facts: list[tuple[Term, int]],
                 edges: list[tuple[int, ...]], owners: list[tuple[int, int]]) -> None:
    """Read the owner, edge and covers tables and the registered plans back from the engine, before any rule runs:
    they must be exactly what was given (module header). One pass over the initial e-nodes."""
    problems: list[str] = []
    low = egraph._egraph
    frozen = low.freeze()

    def rows_of(name: str):
        t = frozen.functions.get(name)
        return [] if t is None else [r for r in t.rows if not r.subsumed]

    want_owner = sorted(set(owners))
    got_owner = sorted({(low.value_to_i64(r.inputs[0]), low.value_to_i64(r.inputs[1])) for r in rows_of(names["owner"])})
    if got_owner != want_owner:
        problems.append(f"owner facts given {want_owner}, held {got_owner}")
    for m in sorted({len(e) for e in edges}):
        want = sorted({perm for e in edges if len(e) == m for perm in itertools.permutations(e)})
        got = sorted({tuple(low.value_to_i64(v) for v in r.inputs) for r in rows_of(names[f"edge{m}"])})
        if got != want:
            problems.append(f"edge{m} facts given {want}, held {got}")
    raw = RawEGraph(_read_freeze(egraph), True, 0, 0.0, "freeze")
    memo: dict = {}
    for n, p in enumerate(plans):
        if raw.class_of(p, memo) is None:
            problems.append(f"plan {n} is not held")
    # the engine holds exactly the sub-terms of the registered plans and cover facts (an e-node it holds that was not
    # registered would be a term that arrived with a wrong literal)
    registered: set[Term] = set()
    todo = list(plans) + [t for t, _ in cover_facts]
    while todo:
        t = todo.pop()
        if t in registered:
            continue
        registered.add(t)
        todo.extend(x for x in t[1:] if isinstance(x, tuple))
    if len(raw.rows) != len(registered):
        problems.append(f"the engine holds {len(raw.rows)} e-nodes for {len(registered)} registered terms")
    want_covers = {(raw.class_of(t, memo), inp) for t, inp in cover_facts}  # a relation is a set: duplicates are one fact
    got_covers = {(r.inputs[0], low.value_to_i64(r.inputs[1])) for r in rows_of(names["covers"])}
    if got_covers != want_covers:
        problems.append(f"covers facts: {len(want_covers - got_covers)} given but not held, "
                        f"{len(got_covers - want_covers)} held but not given")
    if problems:
        raise FactsNotHeld(problems)


def _register_rules(egraph: EGraph, arities: list[int]) -> None:
    a, b, c, j, f = vars_("a b c j f", Rel)
    i, p, q = vars_("i p q", i64)
    rules = [
        # covers goes up through join and filter only: a barrier starts a new region whose leaf is its own term
        rule(eq(j).to(Rel.join(a, b)), covers(a, i)).then(covers(j, i)),
        rule(eq(j).to(Rel.join(a, b)), covers(b, i)).then(covers(j, i)),
        rule(eq(f).to(Rel.filter(a, p)), covers(a, i)).then(covers(f, i)),
        # R1
        rewrite(Rel.join(a, b)).to(Rel.join(b, a)),
        # R3
        rewrite(Rel.filter(Rel.filter(a, p), q)).to(Rel.filter(Rel.filter(a, q), p)),
        # R4 up (unconditional) and down (only onto the side that covers the conjunct's owner)
        rewrite(Rel.join(Rel.filter(a, p), b)).to(Rel.filter(Rel.join(a, b), p)),
        rewrite(Rel.filter(Rel.join(a, b), p)).to(Rel.join(Rel.filter(a, p), b), owner(p, i), covers(a, i)),
    ]
    # R2, both directions, guarded by "the new inner join has a spanning edge": one rule per arity and split
    for m in arities:
        xs = [var(f"x{m}_{k}", i64) for k in range(m)]
        edge = _edge_relation(m)(*xs)
        for s in range(1, m):
            left, right = xs[:s], xs[s:]
            rules.append(rewrite(Rel.join(Rel.join(a, b), c)).to(
                Rel.join(a, Rel.join(b, c)), *[covers(b, x) for x in left], *[covers(c, x) for x in right], edge))
            rules.append(rewrite(Rel.join(a, Rel.join(b, c))).to(
                Rel.join(Rel.join(a, b), c), *[covers(a, x) for x in left], *[covers(b, x) for x in right], edge))
    egraph.register(*rules)


_CTORS = {"scan": Rel.scan, "join": Rel.join, "filter": Rel.filter, "op1": Rel.op1, "op2": Rel.op2}
# positions of (literal argument, child arguments) in each constructor's argument list
_SHAPE = {"scan": (0, ()), "join": (None, (0, 1)), "filter": (1, (0,)), "op1": (1, (0,)), "op2": (0, (1, 2))}


def _read_freeze(egraph: EGraph) -> list[RawRow]:
    """Fast read-out: egglog's frozen function tables give (child class ids, class id) per e-node directly."""
    low = egraph._egraph
    frozen = low.freeze()
    rows: list[RawRow] = []
    for kind, ctor in _CTORS.items():
        _, name = egraph._callable_to_egg(ctor)
        table = frozen.functions.get(name)
        if table is None:
            continue
        lit, kids = _SHAPE[kind]
        for r in table.rows:
            if r.subsumed:
                continue
            arg = None if lit is None else low.value_to_i64(r.inputs[lit])
            rows.append(RawRow(kind, arg, tuple(r.inputs[k] for k in kids), r.output))
    return rows


def _read_function_values(egraph: EGraph) -> list[RawRow]:
    """Public-API read-out: a class is identified by the text of its canonical (extracted) term. Correct but slow
    (seconds from about 500 classes on), so it is only the fallback."""
    rows: list[RawRow] = []
    for kind, ctor in _CTORS.items():
        lit, kids = _SHAPE[kind]
        for call, canonical in egraph.function_values(ctor).items():
            args = get_callable_args(call)
            arg = None if lit is None else int(get_literal_value(args[lit]))
            rows.append(RawRow(kind, arg, tuple(str(args[k]) for k in kids), str(canonical)))
    return rows
