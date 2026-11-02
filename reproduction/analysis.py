"""Paper metrics: incorrect SQL is the positive class; undecided items stay in denominators."""
from __future__ import annotations


import collections

import datetime as dt

import json

import math

import sys

from pathlib import Path

from typing import Any

from reproduction.data import new_dir

STRATA = (("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), (">=7", 7, 10**9))

PANELS = ("pooled", "beaver", "spider2")

SEED = 20261012

def load_records(spec: str) -> dict[str, dict[str, Any]]:
    """Load result records by key from comma-separated run directories.
    Semicolons separate layers: a later layer replaces earlier records with the same
    key. Duplicate keys within one layer are rejected.
    """
    out: dict[str, dict[str, Any]] = {}
    for layer in spec.split(";"):
        got: dict[str, dict[str, Any]] = {}
        for d in layer.split(","):
            files = sorted(Path(d).glob("*/results.jsonl"))
            if not files:
                raise SystemExit(f"no results in {d} (a run that has not finished keeps results.partial.jsonl)")
            for f in files:
                for line in open(f):
                    r = json.loads(line)
                    if r["key"] in got:
                        raise SystemExit(f"{r['key']} occurs twice in {layer}")
                    got[r["key"]] = r
        out.update(got)
    return out

def judge_failed(record: dict[str, Any]) -> bool:
    """Whether the LLM judge's call failed for this record while the semantic score was in effect (gamma > 0)."""
    theta = ((record.get("config") or {}).get("scoring") or {}).get("theta") or {}
    failed = any(f.get("kind") == "model_call_failed" and f.get("stage") == "semantic" for f in record.get("findings") or [])
    return failed and (theta.get("gamma") or 0) > 0

def verdict_of(record: dict[str, Any] | None, side: str) -> str | None:
    """The arm's verdict of a record: `decomposed` (None for a judge failure, `judge_failed`) or `direct`; yes / no,
    else None (undecided, a timeout, a refusal, no record)."""
    if record is None:
        return None
    if side == "direct":
        v = (record.get("direct") or {}).get("final")
    else:
        v = None if judge_failed(record) else record.get("verdict")
    return v if v in ("yes", "no") else None

def decomposition_of(record: dict[str, Any] | None) -> frozenset | None:
    """The decomposition of a record's optimum: the set of its blocks (output, sorted inputs); None without one."""
    blocks = ((record or {}).get("optimum") or {}).get("blocks")
    return frozenset((b["output"], tuple(sorted(b["inputs"]))) for b in blocks) if blocks else None

def label_of(row: dict[str, Any]) -> bool:
    """The official benchmark label; True means correct SQL."""
    official = row.get("official_correct")
    if not isinstance(official, bool):
        raise ValueError(f"Candidate {row.get('id', '<unknown>')} requires a boolean official_correct label")
    return official

def measures(pairs: list[tuple[bool, str | None]]) -> dict[str, Any]:
    """(correct, verdict) pairs -> the counts and the measures, the incorrect SQL as the positive class; an undecided
    verdict (None) is neither a rejection nor an acceptance and stays in the denominators."""
    n_err = sum(1 for ok, _ in pairs if not ok)
    n_ok = len(pairs) - n_err
    tp = sum(1 for ok, v in pairs if not ok and v == "no")
    tn = sum(1 for ok, v in pairs if ok and v == "yes")
    fp = sum(1 for ok, v in pairs if ok and v == "no")
    und_err = sum(1 for ok, v in pairs if not ok and v is None)
    und_ok = sum(1 for ok, v in pairs if ok and v is None)
    recall = tp / n_err if n_err else None
    spec = tn / n_ok if n_ok else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * tp / (2 * tp + fp + (n_err - tp)) if (2 * tp + fp + (n_err - tp)) else None
    return {"n": len(pairs), "n_err": n_err, "n_ok": n_ok, "tp": tp, "tn": tn, "fp": fp, "fn": n_err - tp, "undecided_err": und_err,
            "undecided_ok": und_ok, "ba": (recall + spec) / 2 if recall is not None and spec is not None else None,
            "accuracy": (tp + tn) / len(pairs) if pairs else None, "precision": precision, "recall": recall, "specificity": spec, "f1": f1}

def _strata_of(records: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, str]:
    out = {}
    for k in keys:
        u = ((records.get(k) or {}).get("optimum") or {}).get("unique_blocks")
        out[k] = next((name for name, lo, hi in STRATA if u and lo <= u <= hi), "none")
    return out

def refused_keys(records: dict[str, dict[str, Any]]) -> set[str]:
    """The candidates the importer refused (a finding of stage import) in a run's records."""
    return {k for k, r in records.items() if any(f.get("stage") == "import" for f in r.get("findings") or [])}

def _ba_arrays(np, correct, verdicts):
    """Per candidate: is it incorrect, was it rejected (TP), accepted while correct (TN)."""
    err = np.array([not c for c in correct], dtype=bool)
    tp = np.array([(not c) and v == "no" for c, v in zip(correct, verdicts)], dtype=bool)
    tn = np.array([c and v == "yes" for c, v in zip(correct, verdicts)], dtype=bool)
    return err, tp, tn

def bootstrap(correct: list[bool], groups: list[str], arms: dict[str, list[str | None]], contrasts: list[tuple[str, ...]], *,
              resamples: int = 10_000, seed: int = SEED) -> dict[str, dict[str, Any]]:
    """The BA difference of each contrast, (A, B) for A - B or (A, B, C, D) for (A - B) - (C - D), with the percentile
    95% interval of a paired bootstrap over the candidates, resampled within each group (the source) with
    replacement, every arm on the same resample. A resample in which a class is empty is left out (and counted)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(correct)
    by_group = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    idx = np.concatenate([np.array(ix)[rng.integers(0, len(ix), size=(resamples, len(ix)))] for _, ix in sorted(by_group.items())], axis=1)
    assert idx.shape == (resamples, n)
    names = sorted({a for c in contrasts for a in c})
    ba, point = {}, {}
    for a in names:
        err, tp, tn = _ba_arrays(np, correct, arms[a])
        e, t, s = err[idx].sum(axis=1), tp[idx].sum(axis=1), tn[idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ba[a] = (t / e + s / (n - e)) / 2
        point[a] = measures(list(zip(correct, arms[a])))["ba"]
    out = {}
    for c in contrasts:
        diff = ba[c[0]] - ba[c[1]] if len(c) == 2 else (ba[c[0]] - ba[c[1]]) - (ba[c[2]] - ba[c[3]])
        if any(point[a] is None for a in c):  # a class is empty: no BA, no difference
            out[contrast_name(c)] = {"estimate": None, "ci95": [None, None], "significant": False, "resamples": 0, "resamples_left_out": resamples, "n": n}
            continue
        est = point[c[0]] - point[c[1]] if len(c) == 2 else (point[c[0]] - point[c[1]]) - (point[c[2]] - point[c[3]])
        ok = np.isfinite(diff)
        lo, hi = (float(np.percentile(diff[ok], 2.5)), float(np.percentile(diff[ok], 97.5))) if ok.any() else (None, None)
        out[contrast_name(c)] = {"estimate": est, "ci95": [lo, hi], "significant": lo is not None and (lo > 0 or hi < 0),
                                 "resamples": int(ok.sum()), "resamples_left_out": int((~ok).sum()), "n": n}
    return out

def contrast_name(c: tuple[str, ...]) -> str:
    return f"{c[0]} - {c[1]}" if len(c) == 2 else f"({c[0]} - {c[1]}) - ({c[2]} - {c[3]})"

def parse_contrast(text: str) -> tuple[str, ...]:
    """`A-B` or `A-B~C-D` (arm names never hold `-` around these separators: `A - B` with spaces is accepted too)."""
    parts = [p.strip() for p in text.split("~")]
    out: list[str] = []
    for p in parts:
        a, sep, b = p.partition(" - ") if " - " in p else p.partition("-")
        if not sep or not a.strip() or not b.strip():
            raise SystemExit(f"a contrast is A-B or A-B~C-D: {text!r}")
        out += [a.strip(), b.strip()]
    if len(out) not in (2, 4):
        raise SystemExit(f"a contrast is A-B or A-B~C-D: {text!r}")
    return tuple(out)

R_BINS = (("0", 0.0, 0.0), ("(0, 0.5]", 1e-12, 0.5), ("(0.5, 1]", 0.5 + 1e-12, 1.0), ("(1, 2]", 1.0 + 1e-12, 2.0), ("> 2", 2.0 + 1e-12, math.inf))

def difficulty(features: dict[str, float]) -> float:
    """R(b): the human-facing hinge model's -log F of a block's features at the user-study thresholds tau^0 (beta ln 2,
    the relative excess), no semantic term.
    """
    from sqldecomp.complexity import human_default
    theta = human_default()
    return sum(-math.log(theta.p(name, features.get(name, 0))) for name in theta.tau if theta.tau[name] is not None) if features else 0.0

def block_acceptance(records: dict[str, dict[str, Any]], correct_keys: set[str]) -> dict[str, Any]:
    """On the correct candidates: every judged block class (the verdict item) with its R(b); accepted / judged per R bin."""
    rows = []
    for k in sorted(correct_keys):
        r = records.get(k) or {}
        blocks = {b["output"]: b for b in (r.get("optimum") or {}).get("blocks", [])}
        for it in (r.get("verdicts") or {}).get("items", []):
            b = blocks.get(it.get("output"))
            if b is None or b.get("features") is None:
                continue
            rows.append((difficulty(b["features"]), it.get("verdict")))
    out = {}
    for name, lo, hi in R_BINS:
        sel = [v for rr, v in rows if lo <= rr <= hi]
        out[name] = {"blocks": len(sel), "accepted": sum(1 for v in sel if v == "yes"), "rejected": sum(1 for v in sel if v == "no"),
                     "acceptance": (sum(1 for v in sel if v == "yes") / len(sel)) if sel else None}
    return out

def _f(x: float | None, digits: int = 3, sign: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"

def parse_arm(text: str) -> tuple[str, str, str]:
    """`NAME=RUN[,RUN...][@direct]` -> (name, runs, side)."""
    name, sep, rest = text.partition("=")
    if not sep or not name or not rest:
        raise SystemExit(f"--arm NAME=RUN[@direct]: {text!r}")
    runs, _, side = rest.partition("@")
    if side not in ("", "direct"):
        raise SystemExit(f"--arm ...@direct is the only side suffix: {text!r}")
    return name, runs, side or "decomposed"

def tables(dataset: dict[str, dict[str, Any]], arms: list[tuple[str, str, str]], strata_run: str, *, contrasts: list[tuple[str, ...]] = (),
           differs: list[tuple[str, str]] = (), blocks: list[str] = (), denominator: str = "all",
           resamples: int = 10_000, seed: int = SEED, multiblock: str | None = None, only: set[str] | None = None) -> dict[str, Any]:
    """`multiblock`: a run (spec as an arm's); only the candidates whose optimum there has more than one distinct block
    (the others' block prompt is the direct prompt: their verdicts equal Direct's by construction). `only`: only these
    candidate IDs."""
    strata_records = load_records(strata_run)
    refused = refused_keys(strata_records)
    keys = sorted(k for k in dataset if denominator == "all" or k not in refused)
    if only is not None:
        unknown = sorted(only - set(dataset))
        if unknown:
            raise SystemExit(f"--keys not in the dataset: {unknown[:5]}")
        keys = [k for k in keys if k in only]
    if multiblock:
        several = {k for k, r in load_records(multiblock).items() if (((r.get("optimum") or {}).get("unique_blocks")) or 0) > 1}
        keys = [k for k in keys if k in several]
    stratum = _strata_of(strata_records, keys)
    records = {name: load_records(runs) for name, runs, _ in arms}
    side = {name: s for name, _, s in arms}
    verdicts = {name: [verdict_of(records[name].get(k), side[name]) for k in keys] for name in records}
    verdicts["always reject"] = ["no"] * len(keys)
    correct = [label_of(dataset[k]) for k in keys]
    source = [dataset[k]["benchmark"] for k in keys]
    out: dict[str, Any] = {"label": "official", "denominator": denominator, "candidates": len(keys), "refused": sorted(refused),
                           "multiblock": str(Path(multiblock.split(",")[0]).resolve()) if multiblock else None,
                           "only": len(only) if only is not None else None,
                           "strata_run": str(Path(strata_run.split(",")[0]).resolve()), "resamples": resamples, "seed": seed,
                           "arms": {name: {"runs": [str(Path(d).resolve()) for d in runs.split(",")], "side": s} for name, runs, s in arms},
                           "missing_records": {name: sum(1 for k in keys if k not in records[name]) for name in records},
                           "judge_failed": {name: sorted(k for k in keys if k in records[name] and judge_failed(records[name][k]))
                                            for name in records if side[name] == "decomposed"},
                           "mean_unique_blocks": {name: _mean([((records[name].get(k) or {}).get("optimum") or {}).get("unique_blocks") for k in keys])
                                                  for name in records if side[name] == "decomposed"},
                           "panels": {}, "strata": {}, "contrasts": {}}
    for panel in PANELS:
        sel = [i for i, k in enumerate(keys) if panel == "pooled" or dataset[k]["benchmark"] == panel]
        out["panels"][panel] = {name: measures([(correct[i], v[i]) for i in sel]) for name, v in verdicts.items()}
    for name, _, _ in (*STRATA, ("none", 0, 0)):
        sel = [i for i, k in enumerate(keys) if stratum[k] == name]
        out["strata"][name] = {arm: measures([(correct[i], v[i]) for i in sel]) for arm, v in verdicts.items()}
    if contrasts:
        out["contrasts"]["all"] = bootstrap(correct, source, verdicts, list(contrasts), resamples=resamples, seed=seed)
    out["differs"] = {}
    for a, b in differs:
        sel = [i for i, k in enumerate(keys) if decomposition_of(records[a].get(k)) != decomposition_of(records[b].get(k))]
        sub = {name: [v[i] for i in sel] for name, v in verdicts.items()}
        out["differs"][f"{a} - {b}"] = {"candidates": len(sel), "keys": [keys[i] for i in sel],
                                        "measures": {name: measures([(correct[i], verdicts[name][i]) for i in sel]) for name in (a, b)},
                                        **(bootstrap([correct[i] for i in sel], [source[i] for i in sel], sub, [(a, b)], resamples=resamples,
                                                     seed=seed)[f"{a} - {b}"] if sel else {})}
    out["blocks"] = {name: block_acceptance(records[name], {k for k, c in zip(keys, correct) if c}) for name in blocks}
    return out

def _mean(values: list[Any]) -> float | None:
    v = [x for x in values if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None

def markdown(t: dict[str, Any]) -> str:
    lines = [f"Label: {t['label']}; denominator: {t['denominator']} ({t['candidates']} candidates; refused by the importer: "
             f"{len(t['refused'])}); positive class: incorrect SQL; undecided, timeouts and refusals stay in the denominators.", ""]
    if t.get("multiblock"):
        lines[0:1] = [lines[0][:-1] + f". Only the candidates whose optimum in {Path(t['multiblock']).name} has more than one distinct block."]
    if t.get("only") is not None:
        lines[0:1] = [lines[0][:-1] + f". Only the {t['only']} candidates given (--keys)."]
    head = ("| arm | BA (0-1) | accuracy (0-1) | precision (0-1) | recall (0-1) | F1 (0-1) | TP | TN | FP | FN | undecided err / ok |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for panel, by_arm in t["panels"].items():
        any_m = next(iter(by_arm.values()))
        lines += [f"**{panel}** ({any_m['n']} candidates: {any_m['n_err']} incorrect, {any_m['n_ok']} correct; counts in candidates)", "", head]
        for name, m in by_arm.items():
            lines.append(f"| {name} | {_f(m['ba'])} | {_f(m['accuracy'])} | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | "
                         f"{m['tp']} | {m['tn']} | {m['fp']} | {m['fn']} | {m['undecided_err']} / {m['undecided_ok']} |")
        lines.append("")
    lines += ["**Strata of the strata run's distinct blocks** (BA (0-1); counts in candidates)", "",
              "| stratum | candidates (incorrect / correct) | " + " | ".join(next(iter(t["strata"].values()))) + " |",
              "| --- | --- |" + " --- |" * len(next(iter(t["strata"].values())))]
    for name, by_arm in t["strata"].items():
        m0 = next(iter(by_arm.values()))
        lines.append(f"| {name} | {m0['n']} ({m0['n_err']} / {m0['n_ok']}) | " + " | ".join(_f(m["ba"]) for m in by_arm.values()) + " |")
    lines.append("")
    if t["contrasts"].get("all"):
        lines += ["**Contrasts** (BA difference in BA points; 95% interval of the paired bootstrap stratified by source)", "",
                  "| contrast | estimate | 95% interval | excludes 0 | resamples |", "| --- | --- | --- | --- | --- |"]
        for name, c in t["contrasts"]["all"].items():
            lines.append(f"| {name} | {_f(c['estimate'], sign=True)} | [{_f(c['ci95'][0], sign=True)}, {_f(c['ci95'][1], sign=True)}] | "
                         f"{'yes' if c['significant'] else 'no'} | {c['resamples']} |")
        lines.append("")
    for name, d in t.get("differs", {}).items():
        lines.append(f"Where the decompositions differ, {name}: {d['candidates']} candidates; "
                     + (f"{_f(d['estimate'], sign=True)} BA points [{_f(d['ci95'][0], sign=True)}, {_f(d['ci95'][1], sign=True)}]" if d["candidates"] else "none"))
    if t.get("mean_unique_blocks"):
        lines += ["", "Mean distinct blocks per candidate with an optimum (blocks): " + "; ".join(f"{n} {_f(v, 2)}" for n, v in t["mean_unique_blocks"].items())]
    if any(t["judge_failed"].values()):
        lines.append("Judge failures counted as undecided (candidates): " + "; ".join(f"{n} {len(v)}" for n, v in t["judge_failed"].items()))
    for name, bins in t.get("blocks", {}).items():
        lines += ["", f"**Block acceptance on the correct candidates against R(b) at tau^0**, {name} (blocks; acceptance 0-1)", "",
                  "| R(b) | blocks | accepted | acceptance |", "| --- | --- | --- | --- |"]
        lines += [f"| {b} | {v['blocks']} | {v['accepted']} | {_f(v['acceptance'])} |" for b, v in bins.items()]
    return "\n".join(lines) + "\n"

def cmd_tables(args) -> int:
    dataset = {r["id"]: r for r in map(json.loads, open(args.dataset))}
    arms = [parse_arm(a) for a in args.arm]
    names = [a[0] for a in arms]
    if len(set(names)) != len(names) or "always reject" in names:
        raise SystemExit("arm names must be distinct (and not 'always reject')")
    contrasts = [parse_contrast(c) for c in args.compare]
    differs = [parse_contrast(c) for c in args.differs]
    for c in [*contrasts, *differs]:
        unknown = [a for a in c if a not in names and a != "always reject"]
        if unknown:
            raise SystemExit(f"unknown arms in a contrast: {unknown}")
    if any(len(c) != 2 for c in differs) or any(a == "always reject" for c in differs for a in c):
        raise SystemExit("--differs takes A-B of two decomposed arms")
    out = new_dir(args.out)
    t = tables(dataset, arms, args.strata, contrasts=contrasts, differs=differs, blocks=args.blocks, multiblock=args.multiblock,
               denominator=args.denominator, resamples=args.resamples, seed=args.seed, only=_keys(args.keys))
    (out / "tables.json").write_text(json.dumps(t, indent=1, sort_keys=True) + "\n")
    (out / "tables.md").write_text(markdown(t))
    print(markdown(t), file=sys.stderr)
    return 0

def _pct(values: list[float], q: float) -> float | None:
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)

def run_summary(run: str) -> dict[str, Any]:
    """A run's records summed: the wall time (first start to last finish), the code, the verifier's calls by model and
    effort (asked or from a cache), the calls by stage (asked = calls - cached), the paid USD (a call of a model
    without a price counts as unpriced), the findings by kind, a candidate's time."""
    recs = list(load_records(run).values())
    starts = [dt.datetime.fromisoformat(r["run"]["started_at"]) for r in recs if (r.get("run") or {}).get("started_at")]
    ends = [dt.datetime.fromisoformat(r["run"]["finished_at"]) for r in recs if (r.get("run") or {}).get("finished_at")]
    stages: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in recs:
        for stage, c in (r.get("tokens_and_cost") or {}).items():
            for field in ("calls", "cached_calls", "failed_calls", "unpriced_calls", "paid_usd"):
                stages[stage][field] += c.get(field) or 0
    verifier = collections.Counter(f"{c.get('model')} {c.get('effort')}" for r in recs for side in ("verdicts", "direct")
                                   for it in ((r.get(side) or {}).get("items") or []) for c in it.get("calls") or [])
    totals = [r["time_ms"]["total"] / 1000 for r in recs if (r.get("time_ms") or {}).get("total") is not None]
    return {"run": str(Path(run).resolve()), "records": len(recs), "start": min(starts).isoformat(timespec="seconds") if starts else None,
            "end": max(ends).isoformat(timespec="seconds") if ends else None,
            "wall_s": round((max(ends) - min(starts)).total_seconds()) if starts and ends else None,
            "commits": sorted({((r.get("run") or {}).get("code") or {}).get("commit") for r in recs} - {None}),
            "dirty": sorted({((r.get("run") or {}).get("code") or {}).get("dirty") for r in recs} - {None}),
            "verifier_calls": dict(sorted(verifier.items())), "stages": {s: {k: round(v, 6) for k, v in c.items()} for s, c in sorted(stages.items())},
            "asked": int(sum(c["calls"] - c["cached_calls"] for c in stages.values())), "cached": int(sum(c["cached_calls"] for c in stages.values())),
            "failed": int(sum(c["failed_calls"] for c in stages.values())), "unpriced": int(sum(c["unpriced_calls"] for c in stages.values())),
            "paid_usd": round(sum(c["paid_usd"] for c in stages.values()), 2),
            "findings": dict(sorted(collections.Counter(f.get("kind") for r in recs for f in r.get("findings") or []).items())),
            "candidate_s_median": _pct(totals, 50), "candidate_s_max": max(totals) if totals else None}

def runs_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| run | records | start - end (local) | wall (h) | code | verifier calls (model effort: calls) | asked: writer and describer / judge / "
             "verifier (calls) | from a cache (calls) | failed (calls) | paid (USD) | a candidate's time, median / max (s) | findings (records) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        asked = {s: int(c["calls"] - c["cached_calls"]) for s, c in r["stages"].items()}
        paid = f"{r['paid_usd']:.2f}" + (f" + unpriced ({r['unpriced']} calls, asked or cached)" if r["unpriced"] else "")
        lines.append(f"| {Path(r['run']).name} | {r['records']} | {(r['start'] or '')[5:16].replace('T', ' ')} - {(r['end'] or '')[5:16].replace('T', ' ')} | "
                     f"{_f(r['wall_s'] / 3600 if r['wall_s'] is not None else None, 2)} | {', '.join(c[:8] for c in r['commits'])} | "
                     f"{', '.join(f'{m}: {n}' for m, n in r['verifier_calls'].items())} | {asked.get('decomposition', 0)} / {asked.get('semantic', 0)} / "
                     f"{asked.get('verification', 0) + asked.get('direct', 0)} | {r['cached']} | {r['failed']} | {paid} | "
                     f"{_f(r['candidate_s_median'], 0)} / {_f(r['candidate_s_max'], 0)} | {', '.join(f'{k} {v}' for k, v in r['findings'].items()) or 'none'} |")
    return "\n".join(lines) + "\n"

def cmd_runs(args) -> int:
    out = new_dir(args.out)
    rows = [run_summary(r) for r in args.run]
    (out / "runs.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / "runs.md").write_text(runs_markdown(rows))
    print(runs_markdown(rows), file=sys.stderr)
    return 0

def _keys(text: str | None) -> set[str] | None:
    """--keys: comma-separated, or @FILE with one per line; None when not given."""
    if not text:
        return None
    return {line.strip() for line in open(text[1:]) if line.strip()} if text.startswith("@") else set(text.split(","))

def structure(records: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    """A run's decompositions on `keys`: the candidates with an optimum, their distinct blocks (mean, single-block
    ones) and the optimizer's forced cuts (a CTE read twice is its own block: `optimum.search.forced_cuts`; mean per
    candidate, candidates with one or more, in all), the findings by kind."""
    opt = [((records.get(k) or {}).get("optimum") or {}) for k in keys]
    with_opt = [o for o in opt if o.get("unique_blocks")]
    forced = [(o.get("search") or {}).get("forced_cuts") or 0 for o in with_opt]
    risks = []
    for optimum in with_opt:
        seen = set()
        for block in optimum.get("blocks", []):
            identity = block.get("class_id") or block["output"]
            if identity not in seen and block.get("features") is not None:
                risks.append(difficulty(block["features"]))
                seen.add(identity)
    return {"candidates": len(keys), "records": sum(1 for k in keys if k in records), "with_optimum": len(with_opt),
            "mean_unique_blocks": _mean([o["unique_blocks"] for o in with_opt]), "single_block": sum(1 for o in with_opt if o["unique_blocks"] == 1),
            "mean_forced_cuts": _mean(forced), "with_forced_cuts": sum(1 for f in forced if f), "forced_cuts": sum(forced),
            "distinct_blocks_scored": len(risks), "mean_block_risk": _mean(risks),
            "block_out_of_range_fraction": sum(r > 0 for r in risks) / len(risks) if risks else None,
            "findings": dict(sorted(collections.Counter(f.get("kind") for k in keys for f in (records.get(k) or {}).get("findings") or []).items()))}

def cmd_structure(args) -> int:
    arms = [parse_arm(a) for a in args.run]
    if any(side != "decomposed" for _, _, side in arms):
        raise SystemExit("--run NAME=RUN: a decomposed arm (no @direct)")
    records = {name: load_records(runs) for name, runs, _ in arms}
    only = _keys(args.keys)
    keys = sorted(only if only is not None else set().union(*records.values()))
    out = new_dir(args.out)
    result = {"keys": len(keys), "runs": {name: {"run": runs, **structure(records[name], keys)} for name, runs, _ in arms}}
    (out / "structure.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = [f"{len(keys)} candidates. Distinct blocks and forced cuts over the candidates with an optimum.", "",
             "| run | records | with an optimum (candidates) | distinct blocks, mean (blocks) | single-block (candidates) | forced cuts, mean "
             "(cuts per candidate) | with a forced cut (candidates) | forced cuts in all (cuts) | findings (records) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in result["runs"].items():
        lines.append(f"| {name} | {s['records']} | {s['with_optimum']} | {_f(s['mean_unique_blocks'], 2)} | {s['single_block']} | "
                     f"{_f(s['mean_forced_cuts'], 2)} | {s['with_forced_cuts']} | {s['forced_cuts']} | "
                     f"{', '.join(f'{k} {v}' for k, v in s['findings'].items()) or 'none'} |")
    lines += ["", "Structural risk at the paper's unscaled thresholds (one score per distinct block class):", "",
              "| run | blocks scored | mean R | out of range (fraction) |", "| --- | --- | --- | --- |"]
    for name, s in result["runs"].items():
        lines.append(f"| {name} | {s['distinct_blocks_scored']} | {_f(s['mean_block_risk'])} | {_f(s['block_out_of_range_fraction'])} |")
    (out / "structure.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), file=sys.stderr)
    return 0
