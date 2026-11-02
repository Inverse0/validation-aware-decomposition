"""Question-aware LLM semantic scoring, selected with --semantic judge (--semantic none switches it off). Before
the optimizer, GPT-6 Luna at high effort labels every cut point of the query on two scales, one call each:
COHERENCE (`semantic-coherence-v1`: does the relation make sense on its own? a name, what one row is, 0-3; the call is
shown nothing of the question) and RELEVANCE (`semantic-relevance-v3`: is it a concept the question mentions?
complete / narrower / split / technical / unrelated, the question's words it matches, what is missing); the optimizer's
`meaningfulness` then reads the score of an output group from its labels.
Templates are loaded from prompts/semantic_llm/.

THE SCORE (`JudgeSpec`). Each label scales the score by a factor set by one rule over the levels of its scale, counted
from the best (k = 1): the best level (coherence 3, complete) keeps it at 1; the second (coherence 2, narrower), which
still marks a sound unit, costs a nominal epsilon (0.05: 0.95); from the third on the factor is 2^-(k-2), the epsilon
not carried down (coherence 1: 1/2, 0: 1/4; technical 1/2, split 1/4, unrelated 1/8). s = f_coh * f_rel, in [1/32, 1].
The root group, which answers the question, scores 1, and so does a cut point the judge left unlabelled or that could
not be printed (`missing`, for each scale). The scoring raises s to gamma (`scoring_args.SEMANTIC_GAMMA`, 0.25): a block
costs -gamma ln s of its output.

THE CALLS. The cut points are the groups `Blocks.cut_points` lists, one per output signature (the first of each in that
order: the optimizer scores a signature once), printed from base tables (`Printer.group_sql`; a relation inside a
correlated subquery in closed form, which the prompts explain when they show one), shown in an order shuffled by a
seed taken from the query's key (a rerun builds the same prompt, so the cache answers it) and labelled C1..Cn after the
shuffle. The schema is that of the tables the shown relations read. The two calls run together when the query's model
calls may (`call_parallelism` > 1), on the decomposition clock.

    cut_points(memo, result, blocks, printer, signatures)   the relations to show
    rate(points, question, evidence, schema_json, key, ask, spec, parallelism)   the two calls -> `Rating`
    judge_hook(rating, spec, root_signature)                `meaningfulness` for `make_scoring(..., semantic_hook=)`
"""
from __future__ import annotations

from prompts import load_prompt

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

SPEC_MODEL = "semantic-llm-judge"  # the "model" of the spec's JSON, beside "semantic-distilled"
MODEL, EFFORT = "gpt-6-luna", "high"
COHERENCE = "semantic-coherence-v1"
RELEVANCE = "semantic-relevance-v3"
RETRY = "|retry"  # Retry an answer without a JSON object once under this template suffix.
LEVELS = (0, 1, 2, 3)
MATCHES = ("complete", "narrower", "split", "technical", "unrelated")
EPSILON = 0.05
COHERENCE_FACTORS = {3: 1.0, 2: 1.0 - EPSILON, 1: 0.5, 0: 0.25}
RELEVANCE_FACTORS = {"complete": 1.0, "narrower": 1.0 - EPSILON, "technical": 0.5, "split": 0.25, "unrelated": 0.125}

CLOSED_FORM = load_prompt('semantic_llm/closed_form.txt')

COHERENCE_PROMPT = load_prompt('semantic_llm/coherence_prompt.txt')

RELEVANCE_PROMPT = load_prompt('semantic_llm/relevance_prompt.txt')


# --------------------------------------------------------------------------------------------------------------
# the cut points and the prompts
# --------------------------------------------------------------------------------------------------------------


def signature_key(sig: Any) -> str:
    """One string per output signature (the optimizer's `GroupContext.signature`)."""
    return json.dumps(sig, sort_keys=True, default=str)


def cut_points(memo, result, blocks, printer, signatures: dict) -> list[dict[str, Any]]:
    """One record per output signature among the query's cut points (the first group of each in `Blocks.cut_points`
    order): its group, its signature, its SQL from base tables (None, with the error, when it cannot be printed: it is
    not shown and scores `missing`), whether it lies inside a correlated subquery, its share of the plan's groups."""
    from sqldecomp.model import Block
    from sqldecomp.printer import PrintError
    n = max(1, len(memo.groups))
    seen: set[str] = set()
    out = []
    for g in blocks.cut_points():
        sig = signature_key(signatures[g])
        if sig in seen:
            continue
        seen.add(sig)
        point: dict[str, Any] = {"group": g, "signature": sig, "parameterized": bool(blocks.parameters(Block(g))),
                                 "share_below": round(len(blocks.reachable(g)) / n, 4)}
        try:
            point["sql"] = printer.group_sql(g, bind_outer=True)
        except PrintError as e:
            point.update(sql=None, print_error=str(e)[:300])
        out.append(point)
    return out


def shuffled(points: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """The points in an order fixed by the query's key alone, labelled C1..Cn in that order."""
    order = list(points)
    random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)).shuffle(order)
    return [dict(p, label=f"C{i}") for i, p in enumerate(order, 1)]


def _relations(points: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{p['label']}]\n{p['sql'].strip()}" for p in points)


def _schema(schema_json: dict[str, Any], points: list[dict[str, Any]]) -> str:
    """The schema text of the tables the shown relations read, and of no other."""
    from sqldecomp.tasks import schema_text, tables_in
    tables: set[str] = set()
    for p in points:
        tables |= tables_in(p["sql"])
    return schema_text(schema_json, tables).strip()


def _closed_form(points: list[dict[str, Any]]) -> str:
    return CLOSED_FORM if any(p["parameterized"] for p in points) else ""


def coherence_prompt(schema_json: dict[str, Any], points: list[dict[str, Any]]) -> str:
    return COHERENCE_PROMPT.format(closed_form=_closed_form(points), schema_text=_schema(schema_json, points), relations=_relations(points))


def relevance_prompt(schema_json: dict[str, Any], points: list[dict[str, Any]], question: str, evidence: str = "") -> str:
    ev = f"\nADDITIONAL INFORMATION GIVEN WITH THE QUESTION\n{evidence.strip()}\n" if (evidence or "").strip() else ""
    return RELEVANCE_PROMPT.format(closed_form=_closed_form(points), question=question.strip(), evidence=ev,
                                   schema_text=_schema(schema_json, points), relations=_relations(points))


# --------------------------------------------------------------------------------------------------------------
# the answers
# --------------------------------------------------------------------------------------------------------------


def _json_object(answer: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", answer or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _level(v: Any) -> int | None:
    """0..3 from a number or a digit string, None for anything else."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str) and v.strip() in ("0", "1", "2", "3"):
        return int(v.strip())
    if isinstance(v, (int, float)) and v in LEVELS:
        return int(v)
    return None


def parse_coherence(answer: str, labels: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The well-formed coherence labels of an answer, and what is wrong with the others."""
    d = _json_object(answer)
    if d is None:
        return {}, ["no JSON object"]
    out, problems = {}, []
    for label in labels:
        v = d.get(label)
        level = _level(v.get("coherence")) if isinstance(v, dict) else None
        if level is None:
            problems.append(f"{label}: {'missing' if v is None else 'no rating 0-3'}")
            continue
        out[label] = {"name": str(v.get("name") or "")[:200], "row": str(v.get("row") or "")[:200], "coherence": level}
    return out, problems


def parse_relevance(answer: str, labels: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The well-formed relevance labels of an answer, and what is wrong with the others."""
    d = _json_object(answer)
    if d is None:
        return {}, ["no JSON object"]
    out, problems = {}, []
    for label in labels:
        v = d.get(label)
        match = str(v.get("match") or "").strip().lower() if isinstance(v, dict) else ""
        if match not in MATCHES:
            problems.append(f"{label}: {'missing' if v is None else 'no match of ' + '/'.join(MATCHES)}")
            continue
        out[label] = {"match": match, "phrase": str(v.get("phrase") or "")[:300], "missing": str(v.get("missing") or "")[:300]}
    return out, problems


# --------------------------------------------------------------------------------------------------------------
# the score and the hook
# --------------------------------------------------------------------------------------------------------------


@dataclass
class JudgeSpec:
    """The judge of a scoring spec (`spec["semantic"]`, JSON-able): the model and effort of its calls, the factors of
    every label, the factor of a label that is not there, the prompt templates."""

    model: str = MODEL
    effort: str = EFFORT
    coherence_factors: dict[int, float] = field(default_factory=lambda: dict(COHERENCE_FACTORS))
    relevance_factors: dict[str, float] = field(default_factory=lambda: dict(RELEVANCE_FACTORS))
    missing: float = 1.0
    templates: dict[str, str] = field(default_factory=lambda: {"coherence": COHERENCE, "relevance": RELEVANCE})

    def __post_init__(self) -> None:
        self.coherence_factors = {int(k): float(v) for k, v in self.coherence_factors.items()}
        self.relevance_factors = {str(k): float(v) for k, v in self.relevance_factors.items()}
        if set(self.coherence_factors) != set(LEVELS) or set(self.relevance_factors) != set(MATCHES):
            raise ValueError(f"the factors name every coherence level {LEVELS} and every relevance category {MATCHES}")
        if not all(0.0 < f <= 1.0 for f in [*self.coherence_factors.values(), *self.relevance_factors.values(), self.missing]):
            raise ValueError("every factor, and the one for a missing label, lies in (0, 1]")
        if self.templates != {"coherence": COHERENCE, "relevance": RELEVANCE}:
            raise ValueError(f"this version has the templates {COHERENCE} and {RELEVANCE}")

    def score(self, coherence: int | None, relevance: str | None) -> float:
        """s of one cut point from its labels (a missing label: `missing`)."""
        c = self.missing if coherence is None else self.coherence_factors[coherence]
        r = self.missing if relevance is None else self.relevance_factors[relevance]
        return c * r

    def to_json(self) -> dict[str, Any]:
        return {"model": SPEC_MODEL, "llm": {"model": self.model, "effort": self.effort},
                "coherence_factors": {str(k): v for k, v in sorted(self.coherence_factors.items())},
                "relevance_factors": {k: self.relevance_factors[k] for k in MATCHES}, "missing": self.missing,
                "templates": dict(self.templates)}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "JudgeSpec":
        if d.get("model") != SPEC_MODEL:
            raise ValueError(f"not a judge spec (model {d.get('model')!r})")
        llm = d.get("llm") or {}
        return cls(model=llm.get("model", MODEL), effort=llm.get("effort", EFFORT), coherence_factors=d.get("coherence_factors", COHERENCE_FACTORS),
                   relevance_factors=d.get("relevance_factors", RELEVANCE_FACTORS), missing=d.get("missing", 1.0),
                   templates=d.get("templates") or {"coherence": COHERENCE, "relevance": RELEVANCE})

    def has_default_factors(self) -> bool:
        return (self.coherence_factors, self.relevance_factors, self.missing) == (COHERENCE_FACTORS, RELEVANCE_FACTORS, 1.0)

    def summary(self) -> str:
        return f"judge {self.model} {self.effort}"


@dataclass
class Rating:
    """The judge's labels of one query: per shown cut point (by signature) its two labels, the calls, the problems."""

    points: list[dict[str, Any]]  # the shown cut points, shuffled and labelled
    labels: dict[str, dict[str, Any]]  # signature -> {"coherence": {...} | None, "relevance": {...} | None}
    calls: list[Any]  # the two `sqldecomp.llm.Call`s (none when nothing was shown)
    problems: list[str]
    unprinted: list[dict[str, Any]] = field(default_factory=list)

    def score(self, signature: str, spec: JudgeSpec) -> float:
        lab = self.labels.get(signature) or {}
        c, r = lab.get("coherence"), lab.get("relevance")
        return spec.score(c["coherence"] if c else None, r["match"] if r else None)

    def to_json(self, spec: JudgeSpec) -> dict[str, Any]:
        """What the run record keeps: every shown cut point with its labels and score, the problems, the unprinted."""
        points = []
        for p in self.points:
            lab = self.labels.get(p["signature"]) or {}
            c, r = lab.get("coherence"), lab.get("relevance")
            points.append({"group": p["group"], "label": p["label"], "share_below": p["share_below"],
                           "coherence": c["coherence"] if c else None, "name": c["name"] if c else None, "row": c["row"] if c else None,
                           "relevance": r["match"] if r else None, "phrase": r["phrase"] if r else None, "missing": r["missing"] if r else None,
                           "s": round(self.score(p["signature"], spec), 6)})
        return {"judge": spec.summary(), "cut_points": len(self.points), "points": points, "problems": self.problems[:40],
                "unprinted": [{"group": p["group"], "error": p.get("print_error")} for p in self.unprinted]}


def rate(points: list[dict[str, Any]], question: str, evidence: str, schema_json: dict[str, Any], key: str,
         ask: Callable[[str, str], Any], spec: JudgeSpec, parallelism: int = 1) -> Rating:
    """The judge's two calls over the query's printable cut points (`ask`: (prompt, template) -> `sqldecomp.llm.Call`),
    together when `parallelism` > 1. An answer without a JSON object is asked once more, the same prompt under the
    template with `RETRY` (both retries together when `parallelism` > 1), and the second answer
    counts. A failed call, an answer unreadable twice or an unreadable label leaves those labels missing (a problem).
    """
    from sqldecomp.llm import in_parallel
    shown = shuffled([p for p in points if p.get("sql") is not None], key)
    unprinted = [p for p in points if p.get("sql") is None]
    if not shown:
        return Rating([], {}, [], [], unprinted)
    asks = [(spec.templates["coherence"], coherence_prompt(schema_json, shown)),
            (spec.templates["relevance"], relevance_prompt(schema_json, shown, question, evidence))]
    calls = in_parallel([lambda t=t, p=p: ask(p, t) for t, p in asks], 2 if parallelism > 1 else 1)
    labels = [p["label"] for p in shown]
    parsers = (parse_coherence, parse_relevance)
    unreadable = [i for i, c in enumerate(calls) if (c.exit_code == 0 or (c.answer or "").strip()) and _json_object(c.answer) is None]
    retries = dict(zip(unreadable, in_parallel([lambda i=i: ask(asks[i][1], asks[i][0] + RETRY) for i in unreadable],
                                               2 if parallelism > 1 else 1))) if unreadable else {}
    parsed, problems = [], []
    for i, ((template, _), c, parse) in enumerate(zip(asks, calls, parsers)):
        if i in retries:
            c, template = retries[i], template + RETRY
        if c.exit_code != 0 and not (c.answer or "").strip():
            parsed.append({})
            problems.append(f"{template}: the call failed (exit code {c.exit_code})")
            continue
        got, bad = parse(c.answer, labels)
        parsed.append(got)
        problems += [f"{template}: {b}" for b in bad]
    coh, rel = parsed
    return Rating(shown, {p["signature"]: {"coherence": coh.get(p["label"]), "relevance": rel.get(p["label"])} for p in shown},
                  [*calls, *(retries[i] for i in sorted(retries))], problems, unprinted)


def judge_hook(rating: Rating, spec: JudgeSpec, root_signature: Any) -> Callable[[Any], float]:
    """`meaningfulness(GroupContext)`: 1 for the root group, else the score of the group's signature."""
    root = signature_key(root_signature)

    def meaningfulness(ctx: Any) -> float:
        sig = signature_key(ctx.signature)
        return 1.0 if sig == root else rating.score(sig, spec)

    return meaningfulness
