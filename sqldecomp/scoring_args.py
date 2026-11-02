"""The scoring settings of the command lines (`python -m sqldecomp.pipeline`, `python -m sqldecomp.optimize`, `python -m
sqldecomp.tasks`): which `Scoring` to hand the optimizer, with which parameters.

A setting is a plain JSON-able SPEC (so that corpus workers in other processes can rebuild it, and so that every
output can record it): `{"kind": "human", "theta": <HumanVerifier.to_json()>}` for the human-facing model of
component 3 (`sqldecomp.complexity`); it may carry `"semantic": <JudgeSpec.to_json()>`, the semantic score of
component 4 (the LLM judge of `sqldecomp.semantic_llm`) behind `meaningfulness`. `make_scoring(spec, memo, result,
blocks, boundary)` builds the Scoring for one query: the model's F reads the side-table texts of the query through the
block's fragment and the boundary (the columns a block exposes, the nullable columns of its inputs), so the Scoring is
bound to the query and NEEDS the query's `Boundary` (the optimizer itself is not told the difference: it calls
`block_cost`, `omega`, `meaningfulness`). `scoring_label(spec)` names a spec uniquely; `make_scoring` gives the
Scoring that name.

THE DEFAULTS. `--scoring human` with the default theta `sqldecomp.complexity.human_default()` (every one of the
nineteen features thresholded at `USER_STUDY_TAU`; beta ln 2, relative excess, hinge link, gamma 0, mu 0.001) and no
semantic score (`--semantic none`): `default_spec()`. The thresholds are user parameters: no validated values exist.
`--tau` takes them by name too (`user-study`, the default) and `--alpha` scales every threshold (the paper's
granularity, not rounded; recorded in the theta's notes and the label). `--semantic judge` switches the semantic score
on; its exponent gamma is then 0.25 (`SEMANTIC_GAMMA`) unless `--gamma` or a `--theta-json` gives another. `--gamma`
without a semantic score is refused (it would have no effect). The judge's labels need the question and two model
calls per query, so only the verification run (`sqldecomp.runner`, `pipeline --verify`) rates the cut points and hands
`make_scoring` the hook (`semantic_hook=`); a command line without the question refuses it; `--semantic-model` /
`--semantic-effort` change its model (GPT-6 Luna) and effort (high), a judge spec file (`JudgeSpec.to_json`) its
factors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from sqldecomp.blocks import Blocks
from sqldecomp.model import ImportResult, Memo
from sqldecomp.scoring import Scoring

SCORINGS = ("human",)
DEFAULT_SCORING = "human"
DEFAULT_SEMANTIC = "none"
SEMANTIC_GAMMA = 0.25  # the exponent of the semantic score when it is switched on


def add_scoring_arguments(ap: argparse.ArgumentParser, *, default: str = DEFAULT_SCORING) -> None:
    """`default`: the `--scoring` the command line takes when none is given."""
    assert default in SCORINGS
    g = ap.add_argument_group("scoring", "which F / Omega the optimizer uses")
    g.add_argument("--scoring", choices=SCORINGS, default=default,
                   help=f"'human': the human-facing model of component 3 (default theta: USER_STUDY_TAU thresholds) (default: {default})")
    g.add_argument("--theta-json", help="a theta JSON written by HumanVerifier.save (the options below override its fields)")
    g.add_argument("--tau", help="the thresholds name=value,... (exactly the features named participate; =none leaves one out), "
                                 "or a named set changed by later pairs: user-study (the default: USER_STUDY_TAU)")
    g.add_argument("--alpha", type=float, help="every threshold times ALPHA, not rounded (the paper's granularity; recorded in the "
                                               "theta's notes and in the scoring label)")
    g.add_argument("--beta", type=float, help="the shared decay (default ln 2)")
    g.add_argument("--excess", choices=("relative", "absolute"), help="the excess mode (default relative)")
    g.add_argument("--link", choices=("hinge", "logistic"), help="the link (default hinge)")
    g.add_argument("--gamma", type=float, help=f"the exponent of the semantic score, only with --semantic (default {SEMANTIC_GAMMA} "
                                                "when the semantic score is on)")
    g.add_argument("--mu", type=float, help="Omega(k) = -mu * k (default 0.001)")
    g.add_argument("--semantic", default=None,
                   help="the semantic score (component 4) behind `meaningfulness`: 'none' (constant 1, the default), 'judge' (the LLM "
                        "judge of sqldecomp.semantic_llm: the verification run only) or a JSON written by JudgeSpec.to_json")
    g.add_argument("--semantic-model", help="--semantic judge: the judge's model (default gpt-6-luna)")
    g.add_argument("--semantic-effort", help="--semantic judge: the judge's reasoning effort (default high)")


def scoring_spec_from_args(args) -> dict[str, Any]:
    """The spec of a parsed command line. `--gamma` without a semantic score and a parameter file that cannot be read
    are errors (`SystemExit` with the message), never silently ignored; a scoring without any threshold is warned
    about (every block is then within its thresholds and every query becomes one block)."""
    try:
        return _spec_from_args(args)
    except OSError as e:  # a file that does not exist (`--theta 0.1` is argparse's abbreviation of --theta-json)
        raise SystemExit(f"error: cannot read the scoring parameters: {type(e).__name__}: {e}") from e


def _spec_from_args(args) -> dict[str, Any]:
    kind = getattr(args, "scoring", None) or DEFAULT_SCORING
    if kind not in SCORINGS:
        raise SystemExit(f"error: unknown scoring {kind!r} (choose one of {', '.join(SCORINGS)})")
    semantic = _semantic_spec(args)
    if getattr(args, "gamma", None) is not None and not semantic:
        raise SystemExit("error: --gamma is the exponent of the semantic score and has no effect with --semantic none "
                         "(give --semantic judge)")
    from sqldecomp.complexity import scaled, verifier_from_args
    verifier = verifier_from_args(args)
    if getattr(args, "alpha", None) is not None:
        try:
            verifier = scaled(verifier, args.alpha)
        except ValueError as e:
            raise SystemExit(f"error: --alpha: {e}") from None
    d = verifier.to_json()
    if semantic and getattr(args, "gamma", None) is None and not getattr(args, "theta_json", None):
        d["gamma"] = SEMANTIC_GAMMA
    if all(v is None for v in d["tau"].values()):
        print("warning: --scoring human without any threshold: F is 1 for every block, so the optimum is the coarsest decomposition",
              file=sys.stderr)
    from sqldecomp.complexity import HumanVerifier
    return {"kind": "human", "theta": HumanVerifier.from_json(d).to_json(), **semantic}


def _semantic_spec(args) -> dict[str, Any]:
    """`{"semantic": <JudgeSpec JSON>}` for `--semantic judge` or a judge spec file, nothing for `none` (the default).
    `--semantic-model` and `--semantic-effort` belong to the judge alone."""
    from sqldecomp.semantic_llm import SPEC_MODEL, JudgeSpec
    choice = getattr(args, "semantic", None) or DEFAULT_SEMANTIC
    model, effort = getattr(args, "semantic_model", None), getattr(args, "semantic_effort", None)
    if choice == "judge":
        return {"semantic": JudgeSpec(**{k: v for k, v in (("model", model), ("effort", effort)) if v}).to_json()}
    d = json.loads(Path(choice).read_text()) if choice != "none" else None
    if d is not None and d.get("model") == SPEC_MODEL:
        spec = JudgeSpec.from_json(d)
        return {"semantic": JudgeSpec.from_json({**spec.to_json(), "llm": {"model": model or spec.model, "effort": effort or spec.effort}}).to_json()}
    if model or effort:
        raise SystemExit("error: --semantic-model and --semantic-effort belong to the LLM judge (--semantic judge or a judge spec file)")
    if choice == "none":
        return {}
    raise SystemExit(f"error: --semantic {choice}: not 'none', 'judge' or a judge spec file (JudgeSpec.to_json)")


def is_judge(spec: dict[str, Any] | None) -> bool:
    """Whether the spec's semantic score is the LLM judge (`sqldecomp.semantic_llm`), whose hook the verification run
    makes from the judge's labels."""
    from sqldecomp.semantic_llm import SPEC_MODEL
    return bool(spec) and (spec.get("semantic") or {}).get("model") == SPEC_MODEL


def default_spec() -> dict[str, Any]:
    """The default scoring: the human-facing model with its default theta (`human_default()`), no semantic score."""
    from sqldecomp.complexity import human_default
    return {"kind": "human", "theta": human_default().to_json()}


def human_spec(verifier, semantic=None) -> dict[str, Any]:
    spec = {"kind": "human", "theta": verifier.to_json()}
    if semantic is not None:
        spec["semantic"] = semantic.to_json()
    return spec


def make_scoring(spec: dict[str, Any], memo: Memo, result: ImportResult, blocks: Blocks, boundary: Any = None,
                 semantic_hook: Any = None) -> Scoring:
    """The Scoring of a spec for ONE query, named by `scoring_label(spec)`. The model needs the query's `Boundary`
    (`sqldecomp.boundary.Boundary(memo, result, blocks, printer, constraints, schema)`): the feature extractor reads
    the columns a block exposes and the nullable columns of its inputs from it; without it the values would differ
    from the pipeline's, so it is refused rather than approximated. `semantic_hook`: the `meaningfulness` of a spec
    whose semantic score is the LLM judge (`semantic_llm.judge_hook`, made by the verification run from the judge's
    labels); required for that spec and refused for another."""
    kind = spec.get("kind")
    if kind not in SCORINGS:
        raise ValueError(f"unknown scoring kind {kind!r}")
    if spec.get("semantic") and not is_judge(spec):
        raise ValueError("unknown semantic score: the semantic score is the LLM judge of sqldecomp.semantic_llm")
    if boundary is None:
        raise ValueError(f"make_scoring needs the query's Boundary for the {kind} scoring (the features through inputs read it)")
    if (semantic_hook is not None) != is_judge(spec):
        raise ValueError("a semantic hook is given exactly for the LLM judge's spec (the verification run makes it from the judge's labels)")
    from sqldecomp.complexity import Features, HumanVerifier, human_scoring
    features = Features(memo, result, blocks, boundary)
    scoring = human_scoring(HumanVerifier.from_json(spec["theta"]), features, semantic_hook)
    scoring.name = scoring_label(spec)
    return scoring


def spec_label(spec: dict[str, Any]) -> str:
    if spec.get("kind") == "human":
        from sqldecomp.complexity import HumanVerifier
        return "human " + HumanVerifier.from_json(spec["theta"]).summary()
    raise ValueError(f"unknown scoring kind {spec.get('kind')!r}")


def theta_name(theta: dict[str, Any]) -> str:
    """The name of a theta JSON in the label, beside its summary (which prints the thresholds, beta, the excess, the
    link, gamma and mu): its named thresholds (`user-study (default)`) when nothing the summary leaves out differs from
    the default (the logistic parameters, the notes), the same followed by `x alpha A` for a theta `--alpha` scaled
    (`complexity.scaled`), else 8 hex digits of the JSON."""
    from sqldecomp.complexity import DEFAULT_TAU, HumanVerifier, preset_of
    bare = HumanVerifier().to_json()
    hidden = {k: theta.get(k) for k in ("logistic_a", "logistic_c")} == {k: bare[k] for k in ("logistic_a", "logistic_c")}
    notes = theta.get("notes") or {}
    name = preset_of(theta["tau"]) if hidden and not notes else None
    if name is not None:
        return f"{name} (default)" if name == DEFAULT_TAU else name
    if hidden and set(notes) == {"alpha", "tau0"}:
        base = {**theta, "tau": {n: notes["tau0"].get(n) for n in theta["tau"]}, "notes": {}}
        return f"{theta_name(base)} x alpha {notes['alpha']:g}"
    return _digest(theta)


def _digest(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def scoring_label(spec: dict[str, Any] | None) -> str:
    """One line that names the spec actually used, uniquely (None stands for `default_spec()`): the model's summary plus
    `; theta: <name>` (`theta_name`: `user-study (default)`, possibly followed by `x alpha A`, or 8 hex digits of the
    theta JSON: two thetas that print the same summary, such as `--mu 0.001000001`, get different digests) plus
    `; semantic: none | judge <model> <effort>[ factors <digest>]` (at gamma 0 the semantic score is evaluated and
    recorded but has no effect on F, and the label says so; so does a theta's gamma without a semantic score, whose
    constant 1 makes it ineffective).
    """
    spec = spec or default_spec()
    kind = spec.get("kind")
    if kind not in SCORINGS:
        raise ValueError(f"unknown scoring kind {kind!r}")
    label = spec_label(spec) + f"; theta: {theta_name(spec['theta'])}"
    semantic = spec.get("semantic")
    if not semantic:
        label += "; semantic: none"
        if spec["theta"].get("gamma"):
            label += " (gamma has no effect)"
    else:
        if not is_judge(spec):
            raise ValueError("unknown semantic score: the semantic score is the LLM judge of sqldecomp.semantic_llm")
        from sqldecomp.semantic_llm import JudgeSpec
        judge = JudgeSpec.from_json(semantic)
        label += f"; semantic: {judge.summary()}" + ("" if judge.has_default_factors() else f" factors {_digest(semantic)}")
        if not spec["theta"].get("gamma"):
            label += " (no effect at gamma 0)"
    return label
