"""Offline whole-query risk, risk-quartile comparisons, and combined verdicts."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from reproduction import analysis


def whole_risk(item):
    key, sql, schema_path = item
    from sqldecomp.clock import Deadline, DecompositionTimeout
    from sqldecomp.complexity import Features, human_default, human_scoring
    from sqldecomp.constraints import derive_constraints
    from sqldecomp.importer import UnsupportedSQL
    from sqldecomp.memo import signature
    from sqldecomp.model import Block
    from sqldecomp.optimize import prepare
    from sqldecomp.optimizer import Optimizer
    from sqldecomp.pipeline import load_any_schema
    deadline = Deadline(600)
    try:
        schema = load_any_schema(schema_path)
        result, memo, blocks, printer = prepare(sql, schema, deadline)
        features = Features(memo, result, blocks)
        theta = human_default()
        opt = Optimizer(memo, result, blocks, human_scoring(theta, features), schema=schema,
                        constraints=derive_constraints(memo, result, schema),
                        signatures={g: signature(memo, result, g) for g in memo.groups}, printer=printer, deadline=deadline)
        features.boundary = opt.boundary
        values = dict(features.of_block(Block(memo.root)).values)
        return key, {"risk": analysis.difficulty(values), "features": values}
    except (UnsupportedSQL, DecompositionTimeout) as error:
        return key, {"risk": None, "reason": type(error).__name__ + ": " + str(error)}


def conjunction(values):
    # Combination of whole-query methods used in the tradeoff figure.
    return "no" if "no" in values else "yes" if all(v == "yes" for v in values) else None


def run(args):
    import numpy as np
    from reproduction.cli import selected_keys
    rows, keys = selected_keys(args.dataset, args.keys)
    data = {r["id"]: r for r in rows if r["id"] in keys}
    schemas = Path(args.schemas) if args.schemas else Path(args.dataset).parent / "schemas"
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    items = [(k, data[k]["generated_sql"], str(schemas / Path(data[k]["schema_file"]).name)) for k in keys]
    if args.jobs == 1:
        risk = dict(map(whole_risk, items))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            risk = dict(pool.map(whole_risk, items))
    records = analysis.load_records(args.run)
    verdicts = {"opt": {k: analysis.verdict_of(records.get(k), "decomposed") for k in keys},
                "whole": {k: analysis.verdict_of(records.get(k), "direct") for k in keys}}
    for spec in args.arm:
        name, directory, side = analysis.parse_arm(spec)
        if name in verdicts:
            raise ValueError(f"Duplicate arm name: {name}")
        other = analysis.load_records(directory)
        verdicts[name] = {k: analysis.verdict_of(other.get(k), side) for k in keys}
    for name in list(verdicts):
        if name != "opt":
            verdicts["opt_and_" + name] = {k: conjunction([verdicts["opt"][k], verdicts[name][k]]) for k in keys}
    metrics = {name: analysis.measures([(data[k]["official_correct"], values[k]) for k in keys]) for name, values in verdicts.items()}
    supported = [k for k in keys if risk[k]["risk"] is not None]
    quantiles = np.quantile([risk[k]["risk"] for k in supported], [0.25, 0.5, 0.75]).tolist() if supported else []
    quartiles = []
    for i in range(4):
        members = [k for k in supported if int(np.digitize(risk[k]["risk"], quantiles, right=True)) == i]
        wrong = [k for k in members if not data[k]["official_correct"]]
        correct = [k for k in members if data[k]["official_correct"]]
        quartiles.append(dict(quartile=i + 1, keys=members, incorrect=len(wrong), correct=len(correct),
            opt_only_rejects=sum(verdicts["opt"][k] == "no" and verdicts["whole"][k] != "no" for k in wrong),
            whole_only_rejects=sum(verdicts["whole"][k] == "no" and verdicts["opt"][k] != "no" for k in wrong),
            opt_only_accepts=sum(verdicts["opt"][k] == "yes" and verdicts["whole"][k] != "yes" for k in correct),
            whole_only_accepts=sum(verdicts["whole"][k] == "yes" and verdicts["opt"][k] != "yes" for k in correct)))
    whole_values = [risk[k]["risk"] for k in supported]
    result = dict(candidates=len(keys), candidates_with_risk=len(supported), quantiles=quantiles, quartiles=quartiles,
                  risk=risk, metrics=metrics, whole_mean_risk=analysis._mean(whole_values),
                  whole_out_of_range_fraction=sum(r > 0 for r in whole_values) / len(whole_values) if whole_values else None)
    out.mkdir(parents=True)
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Offline diagnostics: {out / 'diagnostics.json'}")
    return 0
