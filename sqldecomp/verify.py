"""The VERIFICATION STEP: the tasks of a decomposition (component 9) are judged by the verifier, the verdicts are parsed
and combined into one verdict for the query, with the rejecting block(s) named; and the DIRECT arm: the whole query
judged against the question with the same prompt family, which experiments compare with the decomposed arm.

    verify_tasks(tasks, records, question=, evidence=, schema_json=, ask=, rule="conjunction") -> Verification
    verify_direct(sql, question=, evidence=, schema_json=, ask=, rule="conjunction") -> Verification
    model_ask(config) -> Ask                    the verifier's calls through `sqldecomp.llm` (one `ModelConfig`)

THE PROMPT (`PROMPT`): one template for both arms. A BLOCK item is the block's TASK TEXT in the question's place, the
evidence given with the question, the schema text of the base tables the block's own text reads, the INPUT RELATIONS
(the contracts of its inputs, and its parameters) and the block's SQL in the reader form; the WHOLE item (the direct
arm) is the question, the evidence, the schema text of the base tables the query reads, no input section, and the
query as the pipeline prints it. The base template is `verifier-v1`; the template LABEL, which is part
of the model call's cache key, names the variant a prompt actually renders: `verifier-v1` for the base schema/interface format, `verifier-v2` when the schema text renders a UNIQUE key (and
its legend), `verifier-v3` when it marks a NOT NULL column (and its legend) or an input contract carries a weak key,
written `unique where non-null: (...)` (`template_for`). A schema that declares none of these uses the base format. One item per block CLASS: the first occurrence of a shared class stands for every
occurrence (the class is verified once, its task written for the readers of all of them).

THE VERDICT (`parse`): the first `VERDICT: yes` or `VERDICT: no` anywhere in the
answer (case-insensitive, `*` emphasis after the colon tolerated); without one the verdict is `undecided`, counted and
never dropped.

THE AGGREGATION (`conjunction`): the query is accepted iff every item is accepted; it is `undecided` when an item is
undecided (even beside a rejection: such a query has no decomposed verdict); otherwise it is rejected, and the record
names the rejecting block(s) in every case. The direct arm's verdict is the whole item's, under the same rule.

Every call goes through an `Ask` (prompt, template label) -> `sqldecomp.llm.Call`; `model_ask(config)` is the
pipeline's (the one `ModelConfig` of the task writer and the verifier, in the sandbox by default). The verifier's time
is never limited: verification is outside the decomposition timeout and is only measured.
"""
from __future__ import annotations

from prompts import load_prompt

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from sqldecomp.llm import Call, ModelConfig, Price, in_parallel

TEMPLATE = "verifier-v1"
TEMPLATE_UNIQUE = "verifier-v2"  # the same prompt with the UNIQUE legend, for an item whose schema text renders a UNIQUE key
TEMPLATE_NOT_NULL = "verifier-v3"  # the same prompt rendering NOT NULL columns (and their legend) or weak keys in the contracts
TEMPLATE_FAMILY = TEMPLATE_NOT_NULL  # the prompt family of this version: the newest label it renders (recorded in every run)
RULES = ("conjunction",)
VERDICTS = ("yes", "no", "undecided")

PROMPT = load_prompt('verify/prompt.txt')

_VERDICT = re.compile(r"VERDICT\s*:\s*\**\s*(yes|no)\b", re.I)
_REASON = re.compile(r"REASON\s*:\s*(.*)", re.I)

Ask = Callable[[str, str], Call]  # (prompt, template label) -> the call


# --------------------------------------------------------------------------------------------------------------
# items, the prompt, the parser
# --------------------------------------------------------------------------------------------------------------


def template_for(item: dict[str, Any], template: str = TEMPLATE) -> str:
    """The template label of an item's prompt: `verifier-v3` when its schema text marks a NOT NULL column or its input
    contracts carry a weak key (`unique where non-null`), else `verifier-v2` when the schema text carries a UNIQUE
    key, else `verifier-v1` (the prompt is then byte-identical to the one the published experiments used, and their
    cached verdicts apply)."""
    if template != TEMPLATE:
        return template
    from sqldecomp.tasks import NOT_NULL_MARK, WEAK_KEY_LEAD
    schema = item.get("schema_text") or ""
    if NOT_NULL_MARK in schema or WEAK_KEY_LEAD in (item.get("inputs_text") or ""):
        return TEMPLATE_NOT_NULL
    return TEMPLATE_UNIQUE if " UNIQUE(" in schema else TEMPLATE


def render(item: dict[str, Any]) -> str:
    from sqldecomp.tasks import schema_legend
    evidence = item.get("evidence") or ""
    evidence_block = f"\nADDITIONAL INFORMATION GIVEN WITH THE QUESTION\n{evidence}\n" if evidence.strip() else ""
    inputs = item.get("inputs_text") or ""
    inputs_block = f"\n{inputs}\n" if inputs.strip() else ""
    return PROMPT.format(question=item["question"].strip(), evidence=evidence_block, schema_text=item["schema_text"].strip(),
                         schema_legend=schema_legend(item["schema_text"]), inputs_text=inputs_block, sql=item["sql"].strip())


def parse(answer: str) -> tuple[str, str]:
    """('yes' | 'no' | 'undecided', reason)."""
    m = _VERDICT.search(answer or "")
    verdict = m.group(1).lower() if m else "undecided"
    r = _REASON.search(answer or "")
    reason = r.group(1).strip() if r else (answer or "").strip()[:300]
    return verdict, reason


def block_items(tasks: Iterable[Any], records: list[dict[str, Any]], *, evidence: str, schema_json: dict[str, Any]) -> list[dict[str, Any]]:
    """One item per block CLASS of the decomposition, in decomposition order (the first occurrence of a shared class
    stands for the class): the task text, the evidence, the schema text of the block's own base tables, the input
    contracts and parameters (`Task.inputs_text`), the reader-form SQL."""
    from sqldecomp.tasks import schema_text
    by_output = {r["output"]: r for r in records}
    seen: set[str] = set()
    out = []
    for t in tasks:
        if t.class_id and t.class_id in seen:
            continue
        if t.class_id:
            seen.add(t.class_id)
        b = by_output[t.output]
        out.append({"item": f"block:{t.output}", "output": t.output, "inputs": list(t.inputs), "class_id": t.class_id,
                    "shared_with": list(t.shared_with), "is_root": t.is_root,
                    "question": t.text, "evidence": evidence or "", "schema_text": schema_text(schema_json, set(b["tables_own"])),
                    "inputs_text": t.inputs_text(), "sql": b["sql"]})
    return out


def whole_item(sql: str, *, question: str, evidence: str, schema_json: dict[str, Any]) -> dict[str, Any]:
    """The direct arm's item: the question, the evidence, the schema text of the base tables the query reads, the
    query as given here (the pipeline passes its own print, `canonical_sql(print_plan)`, as the published experiments
    did)."""
    from sqldecomp.tasks import schema_text, tables_in
    return {"item": "whole", "output": None, "inputs": [], "class_id": None, "shared_with": [], "is_root": True,
            "question": question, "evidence": evidence or "", "schema_text": schema_text(schema_json, tables_in(sql)),
            "inputs_text": "", "sql": sql}


# --------------------------------------------------------------------------------------------------------------
# judging and the aggregation
# --------------------------------------------------------------------------------------------------------------


@dataclass
class ItemVerdict:
    item: str  # "block:<output>" or "whole"
    output: str | None
    class_id: str | None
    shared_with: list[str]
    is_root: bool
    verdict: str  # the item's verdict under the rule
    first: str  # the first judgement
    reason: str  # the reason of the judgement that decided the verdict
    template: str  # the template label of the call (a run tag is never part of it)
    key: str  # the cache key of the first call
    prompt_chars: int
    calls: list[dict[str, Any]] = field(default_factory=list)  # `Call.record(price)` of every call made for the item

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verification:
    rule: str
    final: str  # yes | no | undecided
    rejecting: list[str]  # the outputs of the rejecting items, each with the other occurrences of its class
    undecided: list[str]
    items: list[ItemVerdict]
    time_ms: float

    @property
    def calls(self) -> list[dict[str, Any]]:
        return [c for it in self.items for c in it.calls]

    def to_json(self) -> dict[str, Any]:
        return {"rule": self.rule, "final": self.final, "rejecting_blocks": self.rejecting, "undecided_blocks": self.undecided,
                "items": [it.to_json() for it in self.items], "calls": len(self.calls), "time_ms": round(self.time_ms, 1)}


def judge_item(item: dict[str, Any], ask: Ask, *, rule: str = "conjunction", price: Price | None = None) -> ItemVerdict:
    """One item, one call."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    prompt = render(item)
    label = template_for(item)
    first = ask(prompt, label)
    verdict, reason = parse(first.answer)
    out = ItemVerdict(item["item"], item.get("output"), item.get("class_id"), list(item.get("shared_with") or ()), bool(item.get("is_root")),
                      verdict, verdict, reason, label, first.key, len(prompt), calls=[first.record(price)])
    return out


def conjunction(verdicts: list[str]) -> str:
    """yes iff every verdict is yes; undecided when one is undecided; else no."""
    if not verdicts or any(v == "undecided" for v in verdicts):
        return "undecided"
    return "yes" if all(v == "yes" for v in verdicts) else "no"


def _occurrences(it: ItemVerdict) -> list[str]:
    return [o for o in [it.output, *it.shared_with] if o]


def combine(items: list[ItemVerdict], rule: str, time_ms: float) -> Verification:
    final = conjunction([it.verdict for it in items])
    rejecting = [o for it in items if it.verdict == "no" for o in (_occurrences(it) or [it.item])]
    undecided = [o for it in items if it.verdict == "undecided" for o in (_occurrences(it) or [it.item])]
    return Verification(rule, final, rejecting, undecided, items, time_ms)


def verify_tasks(tasks: list[Any], records: list[dict[str, Any]], *, question: str, evidence: str, schema_json: dict[str, Any], ask: Ask,
                 rule: str = "conjunction", price: Price | None = None, parallelism: int = 1) -> Verification:
    """The decomposed arm: every block class of the decomposition judged against its task, then the rule. `question`
    is N (the root's task text is N itself; it is not read here otherwise). `parallelism`: the classes judged at a time
    (each is judged whatever the others' verdicts are, so the verdicts are the same)."""
    t0 = time.perf_counter()
    items = in_parallel([lambda it=it: judge_item(it, ask, rule=rule, price=price)
                         for it in block_items(tasks, records, evidence=evidence, schema_json=schema_json)], parallelism)
    return combine(items, rule, (time.perf_counter() - t0) * 1000)


def verify_direct(sql: str, *, question: str, evidence: str, schema_json: dict[str, Any], ask: Ask, rule: str = "conjunction",
                  price: Price | None = None) -> Verification:
    """The direct arm: the whole query against the question, the same prompt, the same rule."""
    t0 = time.perf_counter()
    item = judge_item(whole_item(sql, question=question, evidence=evidence, schema_json=schema_json), ask, rule=rule, price=price)
    return combine([item], rule, (time.perf_counter() - t0) * 1000)


def model_ask(config: ModelConfig) -> Ask:
    """The verifier's calls through `sqldecomp.llm.call` under `config` (its `cache_dir` holds the verdict calls)."""
    from sqldecomp import llm
    return lambda prompt, template: llm.call(prompt, template, config)
