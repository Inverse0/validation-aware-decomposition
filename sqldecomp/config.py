"""Serializable configuration for a verification experiment.
ExperimentConfig.to_json() is recorded in every result; from_json() restores it.
Settings cover structural/semantic scoring, task assignment, model overrides,
aggregation, decomposition strategy, caches, concurrency and time budgets.

Library defaults use human structural scoring without a semantic term, td3-desc
task assignment, a GPT-6 Luna writer, conjunction aggregation and a 3,600-second
decomposition budget. The public experiment planner supplies the paper's model
and semantic-scoring settings. Verification is outside the decomposition budget.

replay serves task and semantic-judge answers from caches while calling the
selected validator. random-k uses k_by_key to match an Opt run's block counts.
Prices carry source/date metadata; an empty price table makes cost unavailable.
See the dataclass fields and README.md for the available settings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sqldecomp.clock import TASKS_TIMEOUT_S
from sqldecomp.llm import ModelConfig, Price

TASKS_METHOD = "td3-desc"  # Default task-assignment method.
WRITER_MODEL, WRITER_EFFORT = "gpt-6-luna", "max"  # Default requirement and input-description writer.
AGGREGATIONS = ("conjunction",)

# Dated price snapshot for the library configuration; see PRICES_DATE/SOURCE.
# The public experiment CLI uses --prices none unless a price file is supplied.
# Prices are USD per million tokens, with long-context multipliers.
PRICES = {"gpt-5.5": Price(input=5.0, cached_input=0.5, output=30.0, long_input_threshold=272_000, long_input_multiplier=2.0,
                           long_output_multiplier=1.5),
          "gpt-6-luna": Price(input=0.1, cached_input=0.01, output=0.5, long_input_threshold=272_000, long_input_multiplier=2.0,
                              long_output_multiplier=1.5)}
PRICES_DATE = "2026-10-07"
PRICES_SOURCE = ("gpt-5.5: https://developers.openai.com/api/docs/models/gpt-5.5 (standard tier, read 2026-10-06); gpt-6-luna: "
                 "platform.openai.com/docs/pricing and .../docs/models/gpt-6-luna (standard tier, read 2026-10-07); reasoning tokens "
                 "are billed as output tokens: https://developers.openai.com/api/docs/guides/reasoning")


def _default_scoring() -> dict[str, Any]:
    from sqldecomp.scoring_args import default_spec
    return default_spec()


def _verifier_family() -> str:
    from sqldecomp.verify import TEMPLATE_FAMILY
    return TEMPLATE_FAMILY


@dataclass
class ExperimentConfig:
    scoring: dict[str, Any] = field(default_factory=_default_scoring)
    tasks_method: str = TASKS_METHOD
    model: ModelConfig = field(default_factory=ModelConfig)
    writer_model: str | None = WRITER_MODEL  # None inherits the base model.
    writer_effort: str | None = WRITER_EFFORT
    verifier_model: str | None = None  # the verifier alone, on purpose (None: `model`)
    verifier_effort: str | None = None
    verifier_template: str = field(default_factory=_verifier_family)
    aggregation: str = "conjunction"
    decomposed_arm: bool = True  # write the tasks and judge the blocks
    direct_arm: bool = False  # judge the whole query against the question
    seed: int = 0
    task_cache: str | None = None
    verdict_cache: str | None = None
    semantic_cache: str | None = None  # the LLM judge's calls (`--semantic judge`): CACHE/semantic
    run_tag: str = ""
    prices: dict[str, Price] = field(default_factory=lambda: dict(PRICES))
    prices_date: str | None = PRICES_DATE
    prices_source: str | None = PRICES_SOURCE
    checks: bool = False
    db: str | None = None
    decomposition_timeout_s: float = TASKS_TIMEOUT_S
    call_parallelism: int = 1  # model calls of one query at a time (1: one after another)
    replay: bool = False  # the writer and the LLM judge answer from their caches only (never a new call)
    decomposition: str = "optimum"  # or a baseline of `sqldecomp.baselines`: syntactic, random-k
    decomposition_seed: int = 0  # random-k: the run's seed (each query's is hashed with its key)
    k_by_key: dict[str, int] | None = None  # random-k: the number of distinct blocks wanted, per query key

    def __post_init__(self) -> None:
        from sqldecomp.tasks import METHODS
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
        if self.tasks_method not in METHODS:
            raise ValueError(f"tasks_method must be one of {METHODS}")
        if self.verifier_template != _verifier_family():
            raise ValueError(f"this version has one verifier prompt family, {_verifier_family()!r}")
        if self.scoring.get("kind") != "human":
            raise ValueError("scoring must be a spec of sqldecomp.scoring_args (kind human)")
        if not (self.decomposition_timeout_s > 0):
            raise ValueError("the decomposition timeout is a number of seconds > 0")
        if not (isinstance(self.call_parallelism, int) and self.call_parallelism >= 1):
            raise ValueError("call_parallelism is a whole number >= 1")
        from sqldecomp.baselines import STRATEGIES
        if self.decomposition not in STRATEGIES:
            raise ValueError(f"decomposition must be one of {STRATEGIES}")
        if self.decomposition == "random-k" and not self.k_by_key:
            raise ValueError("random-k needs the k of every query (k_by_key)")

    # -- the two sides' model configurations ---------------------------------------------------------------------

    def writer_config(self) -> ModelConfig:
        return self.model.with_(model=self.writer_model or self.model.model, effort=self.writer_effort or self.model.effort,
                                cache_dir=str(Path(self.task_cache) / "llm") if self.task_cache else None)

    def verifier_config(self) -> ModelConfig:
        return self.model.with_(model=self.verifier_model or self.model.model, effort=self.verifier_effort or self.model.effort,
                                cache_dir=self.verdict_cache)

    def semantic_config(self, judge) -> ModelConfig:
        """The LLM judge's calls (`semantic_llm.JudgeSpec`): its own model and effort, the sandbox and the timeout of
        `model`, `semantic_cache`."""
        return self.model.with_(model=judge.model, effort=judge.effort, cache_dir=self.semantic_cache)

    def price(self, model: str) -> Price | None:
        return self.prices.get(model)

    # -- serialisation -------------------------------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {"scoring": self.scoring, "tasks_method": self.tasks_method, "model": self.model.to_json(),
                "writer_model": self.writer_model, "writer_effort": self.writer_effort, "verifier_model": self.verifier_model,
                "verifier_effort": self.verifier_effort, "verifier_template": self.verifier_template, "aggregation": self.aggregation,
                "decomposed_arm": self.decomposed_arm, "direct_arm": self.direct_arm, "seed": self.seed, "task_cache": self.task_cache, "verdict_cache": self.verdict_cache,
                "semantic_cache": self.semantic_cache,
                "run_tag": self.run_tag, "prices": {m: p.to_json() for m, p in sorted(self.prices.items())},
                "prices_date": self.prices_date, "prices_source": self.prices_source, "checks": self.checks, "db": self.db,
                "decomposition_timeout_s": self.decomposition_timeout_s, "call_parallelism": self.call_parallelism, "replay": self.replay,
                "decomposition": self.decomposition, "decomposition_seed": self.decomposition_seed, "k_by_key": self.k_by_key}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        d["model"] = ModelConfig.from_json(d.get("model") or {})
        d["prices"] = {m: Price.from_json(p) for m, p in (d.get("prices") or {}).items()}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def with_(self, **changes: Any) -> "ExperimentConfig":
        return replace(self, **changes)


def load_prices(path: str | Path) -> tuple[dict[str, Price], str | None, str | None]:
    """A price table from a JSON file `{"date": ..., "source": ..., "prices": {model: {input, cached_input, output,
    ...}}}`."""
    d = json.loads(Path(path).read_text())
    return {m: Price.from_json(p) for m, p in d.get("prices", {}).items()}, d.get("date"), d.get("source")


def add_experiment_arguments(ap) -> None:
    """The command-line flags of a verification run that map onto `ExperimentConfig` (the scoring's and the model's
    have their own groups: `scoring_args.add_scoring_arguments`, `llm.add_model_arguments`)."""
    g = ap.add_argument_group("experiment", "the parameters of a verification run (recorded in every output record)")
    g.add_argument("--aggregation", choices=AGGREGATIONS, default="conjunction", help="how the block verdicts combine (default conjunction)")
    g.add_argument("--writer-model", help=f"the task writer's model (and td3-desc's describer's; default {WRITER_MODEL})")
    g.add_argument("--writer-effort", help=f"the task writer's reasoning effort (default {WRITER_EFFORT})")
    g.add_argument("--verifier-model", help="the verifier's model, overriding --model on purpose")
    g.add_argument("--verifier-effort", help="the verifier's reasoning effort, overriding --effort on purpose")
    g.add_argument("--cache", help="the root of the persistent caches: CACHE/tasks (assignments, the writer's calls under tasks/llm), "
                                   "CACHE/verdicts (the verifier's calls) and CACHE/semantic (the LLM judge's calls, --semantic judge)")
    g.add_argument("--verdict-cache", help="the persistent cache of the verifier's calls (default CACHE/verdicts)")
    g.add_argument("--run-tag", default="", help="a label written into every record of the run")
    g.add_argument("--prices", help="a JSON price table (date, source, prices per model) replacing the packaged one; 'none' leaves the "
                                    "prices unset (the cost is then reported as unavailable)")
    g.add_argument("--checks", action="store_true", help="also run checks C1 - C3 on the optimum (C3 needs --db)")
    g.add_argument("--decomposition", choices=("optimum", "syntactic", "random-k"), default="optimum",
                   help="the decomposition judged: the optimizer's (default), or a baseline of sqldecomp.baselines: a block at every CTE, "
                        "derived table and subquery as written where the cut is legal (syntactic), or a seeded random legal "
                        "decomposition with the query's k distinct blocks (random-k, --k-file)")
    g.add_argument("--decomposition-seed", type=int, default=0, help="random-k: the run's seed (hashed with each query's key)")
    g.add_argument("--k-file", help="random-k: a JSON object of query key -> the number of distinct blocks wanted")
    g.add_argument("--replay", action="store_true",
                   help="the task writer and the LLM judge answer from their caches only: a prompt not in the cache is a failed call, "
                        "never asked (another validator judging a run's exact decompositions and tasks, --cache holding copies of that "
                        "run's tasks/ and semantic/)")
    g.add_argument("--call-parallelism", type=int, default=1,
                   help="model calls of one query at a time (default 1, one after another): the writer's calls of the blocks whose "
                        "readers' tasks are written, td3-desc's descriptions beside them, the verifier's block calls, the LLM judge's two "
                        "calls, and the direct arm beside the "
                        "decomposed arm; the tasks and the verdicts are the same, only the times change")


def config_from_args(args, scoring: dict[str, Any], model: ModelConfig) -> ExperimentConfig:
    task_cache = getattr(args, "tasks_cache", None) or (str(Path(args.cache) / "tasks") if getattr(args, "cache", None) else None)
    verdict_cache = getattr(args, "verdict_cache", None) or (str(Path(args.cache) / "verdicts") if getattr(args, "cache", None) else None)
    semantic_cache = str(Path(args.cache) / "semantic") if getattr(args, "cache", None) else None
    prices, date, source = dict(PRICES), PRICES_DATE, PRICES_SOURCE
    if getattr(args, "prices", None) == "none":
        prices, date, source = {}, None, None
    elif getattr(args, "prices", None):
        try:
            prices, date, source = load_prices(args.prices)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
            raise SystemExit(f"error: cannot read the price table {args.prices}: {e}") from None
    return ExperimentConfig(scoring=scoring, tasks_method=getattr(args, "tasks", None) or TASKS_METHOD, model=model,
                            writer_model=getattr(args, "writer_model", None) or WRITER_MODEL,
                            writer_effort=getattr(args, "writer_effort", None) or WRITER_EFFORT,
                            verifier_model=getattr(args, "verifier_model", None), verifier_effort=getattr(args, "verifier_effort", None),
                            aggregation=getattr(args, "aggregation", "conjunction"), decomposed_arm=bool(getattr(args, "verify", True)),
                            direct_arm=bool(getattr(args, "direct", False)),
                            seed=getattr(args, "seed", 0), task_cache=task_cache, verdict_cache=verdict_cache, semantic_cache=semantic_cache,
                            run_tag=getattr(args, "run_tag", "") or "", prices=prices, prices_date=date, prices_source=source,
                            checks=bool(getattr(args, "checks", False)), db=getattr(args, "db", None),
                            decomposition_timeout_s=TASKS_TIMEOUT_S if getattr(args, "timeout", None) is None else args.timeout,
                            call_parallelism=getattr(args, "call_parallelism", None) or 1, replay=bool(getattr(args, "replay", False)),
                            decomposition=getattr(args, "decomposition", None) or "optimum", decomposition_seed=getattr(args, "decomposition_seed", 0) or 0,
                            k_by_key=_k_file(getattr(args, "k_file", None)))


def _k_file(path: str | None) -> dict[str, int] | None:
    if not path:
        return None
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: cannot read the k file {path}: {e}") from None
    return {str(k): int(v) for k, v in d.items()}
