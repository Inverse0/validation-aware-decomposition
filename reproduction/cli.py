"""Portable experiment plans and execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks/validation/dataset.jsonl"
METHODS = ("opt", "direct", "direct-k", "structural", "syntactic", "random-k", "rt-se", "opt-rt-se")
MODEL_IDS = {"v0": "gpt-5.5", "v1": "gpt-5.6-sol", "v2": "gpt-6-astra"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_keys(dataset, text):
    from reproduction.experiment import read_keys
    rows = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("The dataset must contain unique, nonempty candidate IDs")
    keys = read_keys(text)
    if keys is not None and (not keys or keys - set(ids)):
        raise ValueError("--keys must select at least one known candidate")
    return rows, sorted(keys) if keys is not None else ids


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def build_plan(args):
    """A dependency-ordered plan; planning does not create files or call a model."""
    root = Path(args.out).resolve()
    dataset = str(Path(args.dataset).resolve())
    rows, keys = selected_keys(dataset, args.keys)
    schemas = str(Path(args.schemas).resolve()) if args.schemas else str(Path(dataset).parent / "schemas")
    for r in rows:
        if r["id"] in keys and not (Path(schemas) / Path(r["schema_file"]).name).is_file():
            raise ValueError(f"Missing schema for {r['id']}")
    selected = ",".join(keys) if args.keys else None
    steps = []
    def add(name, action, **kw):
        steps.append({"name": name, "action": action, "params": kw})
    inputs = str(root / "inputs")
    add("inputs", "experiment.cmd_inputs", dataset=dataset, schemas=schemas, keys=selected, out=inputs)
    models = [MODEL_IDS.get(m, m) for m in args.models]
    if len(set(models)) != len(models):
        raise ValueError("--models contains duplicate models")
    alphas = args.alphas if args.suite in {"rq3", "all"} else [args.alpha]
    methods = ({"rq1": ["opt", "direct-k", "rt-se", "opt-rt-se"],
                "rq2": ["opt", "structural", "syntactic", "random-k"],
                "rq3": ["opt"], "all": ["opt", "direct-k", "rt-se", "opt-rt-se", "structural", "syntactic", "random-k"],
                "custom": [args.method]})[args.suite]
    if any(m in methods for m in ("direct-k", "random-k", "rt-se", "opt-rt-se")) and "opt" not in methods:
        methods = ["opt", *methods]
    all_arms = []
    first_opt = None
    first_cache = {}
    all_runs = []
    for model in models:
        for alpha in alphas:
            tag = f"{slug(model)}_a{alpha:g}"
            run_root = root / "runs" / tag
            cache_root = root / "cache" / tag
            opt = str(run_root / "opt")
            opt_cache = str(cache_root / "opt")
            local_arms = []
            for method in methods:
                # RQ1/RQ2 use alpha=1 within the combined all suite.
                if args.suite == "all" and alpha != 1 and method != "opt":
                    continue
                out = str(run_root / method)
                cache = str(cache_root / method)
                if method in {"opt", "direct", "structural", "syntactic", "random-k"}:
                    replay = first_cache.get(alpha) if method == "opt" else None
                    kw = dict(inputs=inputs, cache=cache, out=out, jobs=args.jobs, files=1, pool=True, run_tag=tag + "/" + method,
                              arm="direct" if method == "direct" else "default", alpha=None if method == "direct" else alpha,
                              gamma=0.0 if method == "structural" else None, verifier_model=model, verifier_effort=args.effort,
                              writer_model=None if method == "direct" else args.writer_model,
                              writer_effort=None if method == "direct" else args.writer_effort,
                              semantic_model=None if method == "direct" else args.semantic_model,
                              semantic_effort=None if method == "direct" else args.semantic_effort,
                              decomposed_only=method not in {"opt", "direct"}, cache_from=[], verdicts_from=[], parts_from=[],
                              tasks_from=replay, decomposition=method if method in {"syntactic", "random-k"} else "optimum",
                              k_from=opt if method == "random-k" else None, decomposition_seed=args.seed,
                              call_parallelism=args.call_parallelism, resume=False,
                              prices=str(Path(args.prices).resolve()) if args.prices != "none" else "none")
                    add(f"{tag}/{method}", "experiment.cmd_run", **kw)
                    all_runs.append(out)
                    arm_name = method.replace("-", "_")
                    local_arms.append(f"{arm_name}={out}" + ("@direct" if method == "direct" else ""))
                    if method == "opt":
                        first_cache.setdefault(alpha, opt_cache)
                        first_opt = first_opt or opt
                        local_arms.append(f"whole={out}@direct")
                elif method == "direct-k":
                    add(f"{tag}/direct-k", "direct_k.run", base=opt, k_from=opt, cache=cache, out=out,
                        jobs=args.jobs, call_parallelism=args.call_parallelism, attempts=3, retry_wait=60.0,
                        keys=selected, reuse_cache=False)
                    local_arms.append(f"whole_k={out}/conjunction@direct")
                elif method == "rt-se":
                    baseline_inputs = str(root / "inputs-baselines" / tag)
                    add(f"{tag}/baseline-inputs", "baselines.cmd_inputs", dataset=dataset, direct=opt, schemas=schemas, out=baseline_inputs)
                    add(f"{tag}/rt-se", "baselines.cmd_run", inputs=baseline_inputs, cache=cache, out=out,
                        jobs=args.jobs, keys=selected, reuse_cache=False, serial=True, model=model, effort=args.effort)
                    arms = str(run_root / "rt-se-arms")
                    add(f"{tag}/rt-se-arms", "baselines.cmd_arms", run=[out], out=arms)
                    local_arms += [f"{a}={arms}/{a}@direct" for a in ("rt", "se")]
                else:
                    block_inputs = str(root / "inputs-blocks" / tag)
                    add(f"{tag}/block-inputs", "block_baselines.cmd_inputs", dataset=dataset, opt=opt,
                        verifier_cache=opt_cache, schemas=schemas, out=block_inputs)
                    for approach in ("rt", "se"):
                        add(f"{tag}/opt-{approach}", "block_baselines.cmd_run", approach=approach, inputs=block_inputs,
                            cache=str(cache_root / f"opt-{approach}"), out=str(run_root / f"opt-{approach}"),
                            keys=None, limit=None, reuse_cache=False, seed_cache=None, dry_run=False, model=model, effort=args.effort)
                    arms = str(run_root / "opt-rt-se-arms")
                    add(f"{tag}/block-arms", "block_baselines.cmd_arms", inputs=block_inputs,
                        rt=[str(run_root / "opt-rt")], se=[str(run_root / "opt-se")], out=arms)
                    # RT/SE replace the block validator
                    local_arms += [f"opt_rt={arms}/rt_only@direct", f"opt_se={arms}/se_only@direct"]
            strata = opt if "opt" in methods else str(run_root / methods[0])
            add(f"{tag}/tables", "analysis.cmd_tables", dataset=dataset, arm=local_arms, strata=strata,
                out=str(root / "tables" / tag), compare=[], differs=[], blocks=["opt"] if "opt" in methods else [],
                denominator="all", multiblock=None, keys=selected, resamples=args.resamples, seed=args.seed)
            all_arms += [f"{slug(tag)}_{a}" for a in local_arms if a.startswith(("opt=", "whole="))]
    if len(all_arms) > 2:
        add("granularity-tables", "analysis.cmd_tables", dataset=dataset, arm=all_arms, strata=first_opt,
            out=str(root / "tables" / "granularity"), compare=[], differs=[], blocks=[],
            denominator="all", multiblock=None, keys=selected, resamples=args.resamples, seed=args.seed)
    if all_runs:
        add("cost-and-runtime", "analysis.cmd_runs", run=all_runs, out=str(root / "tables" / "cost-and-runtime"))
        add("structure", "analysis.cmd_structure", run=[f"run{i}={p}" for i, p in enumerate(all_runs)], keys=selected,
            out=str(root / "tables" / "structure"))
    return steps


def doctor(args):
    errors = []
    rows, keys = selected_keys(args.dataset, None)
    for module in ["sqlglot", "egglog"] + (["duckdb", "pandas", "google.genai", "pydantic_settings", "pandasql"] if args.baselines else []):
        try:
            importlib.import_module(module)
        except ImportError:
            errors.append(f"Missing dependency: {module}; run bash scripts/setup.sh" + (" --baselines" if args.baselines else ""))
    for r in rows:
        if not (Path(args.dataset).parent / "schemas" / Path(r["schema_file"]).name).is_file():
            errors.append(f"Missing schema: {r['schema_file']}")
    if not args.offline:
        from sqldecomp import llm
        if sys.platform != "linux" or not llm.sandbox_available():
            errors.append("Model experiments require Linux and bubblewrap (bwrap)")
        if not llm.codex_binary().is_file():
            errors.append("Set SQLDECOMP_CODEX to the installed Codex CLI executable")
        if not llm.master_auth().is_file():
            errors.append("Log in with codex login; SQLDECOMP_CODEX_AUTH can select the credential file")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"{len(keys)} benchmark candidates; {sum(r['official_correct'] is True for r in rows)} labeled correct by the official evaluation.")
    print("Environment checks passed." if not errors else "Environment checks failed.")
    return 1 if errors else 0


def execute(args):
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    use_baselines = any(s["action"].startswith(("baselines.", "block_baselines.")) for s in plan)
    if doctor(argparse.Namespace(dataset=args.dataset, offline=False, baselines=use_baselines)):
        return 1
    root = Path(args.out).resolve()
    # Bind resume to the complete plan, input bytes, schemas and prompt bytes.
    files = [Path(args.dataset), *(Path(args.schemas) if args.schemas else Path(args.dataset).parent / "schemas").glob("*.json"),
             *(p for p in (ROOT / "prompts").rglob("*") if p.is_file() and p.suffix in {".py", ".txt"})]
    files += [p for package in ("sqldecomp", "reproduction", "nl2sql_verification")
              for p in (ROOT / package).rglob("*") if p.is_file() and p.suffix in {".py", ".json"}]
    if args.prices != "none":
        files.append(Path(args.prices))
    identity = {"plan": plan, "inputs": {str(p.resolve()): digest(p) for p in files}}
    state = root / "experiment-plan.json"
    if root.exists():
        if not args.resume or not state.is_file() or json.loads(state.read_text()) != identity:
            raise ValueError("Output already exists. Use --resume with the identical plan, or choose a new --out.")
    else:
        root.mkdir(parents=True)
        state.write_text(json.dumps(identity, indent=2) + "\n")
    done = root / ".completed"
    done.mkdir(exist_ok=True)
    for step in plan:
        marker = done / (slug(step["name"]) + ".json")
        params = dict(step["params"])
        out = Path(params["out"])
        if marker.exists():
            recorded = json.loads(marker.read_text())
            if not recorded or any(not Path(p).is_file() or digest(p) != sha for p, sha in recorded.items()):
                raise ValueError(f"Completed output changed or is missing: {step['name']}")
            print(f"Already complete: {step['name']}", flush=True)
            continue
        if out.exists():
            if step["action"] == "experiment.cmd_run":
                params["resume"] = True
            else:
                # Preserve partial outputs from this owned plan before retrying.
                failed = root / "interrupted" / slug(step["name"])
                failed.parent.mkdir(exist_ok=True)
                i = 1
                while failed.with_name(failed.name + f"-{i}").exists():
                    i += 1
                out.rename(failed.with_name(failed.name + f"-{i}"))
                if "reuse_cache" in params:
                    params["reuse_cache"] = True
                if step["action"] == "block_baselines.cmd_run":
                    params["seed_cache"] = params["cache"]
        print(f"Running: {step['name']}", flush=True)
        module, function = step["action"].split(".")
        fn = getattr(importlib.import_module("reproduction." + module), function)
        code = fn(argparse.Namespace(**params))
        if code:
            return code
        outputs = {str(p): digest(p) for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".md"}}
        if not outputs:
            raise RuntimeError(f"Step produced no outputs: {step['name']}")
        marker.write_text(json.dumps(outputs, indent=2) + "\n")
    print(f"Finished. Tables: {root / 'tables'}")
    return 0


def smoke(args):
    """Exercise parsing, memo, legality, optimization and C1/C2 without any model."""
    from sqldecomp import runner
    from sqldecomp.config import ExperimentConfig
    from sqldecomp.pipeline import load_any_schema
    from sqldecomp.schema import schema_json_of
    schema_path = ROOT / "benchmarks/toy/schema.json"
    schema = load_any_schema(str(schema_path))
    sql = (ROOT / "benchmarks/toy/query.sql").read_text()
    # Disable judge/task/validator calls; retain mechanical checks and human scoring.
    config = ExperimentConfig(decomposed_arm=False, direct_arm=False, checks=True)
    record = runner.run_one("smoke", sql, "Which customers have more than two orders?", "", schema, config,
                            schema_json=schema_json_of(schema_path))
    failures = [f for f in record.get("findings", []) if f.get("kind") not in {"info"}]
    checks = record.get("checks", {})
    if not record.get("optimum") or failures or checks.get("c1", {}).get("failed") or not checks.get("c2", {}).get("ok"):
        print(json.dumps(record, indent=2, default=str))
        return 1
    print(f"Offline smoke passed: {record['optimum']['unique_blocks']} distinct blocks, C1/C2 enabled; no model calls.")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="check dependencies, schemas and model-call environment")
    d.add_argument("--offline", action="store_true")
    d.add_argument("--baselines", action="store_true")
    d.add_argument("--dataset", default=str(DATASET))
    d.set_defaults(fn=doctor)
    s = sub.add_parser("smoke", help="offline mechanical-check example; no credentials needed")
    s.set_defaults(fn=smoke)
    r = sub.add_parser("run", help="execute or preview a paper experiment suite")
    r.add_argument("--suite", choices=("rq1", "rq2", "rq3", "all", "custom"), default="rq2")
    r.add_argument("--method", choices=METHODS, default="opt", help="used with --suite custom")
    r.add_argument("--dataset", default=str(DATASET))
    r.add_argument("--schemas")
    r.add_argument("--keys", help="comma-separated IDs, or @FILE with one ID per line")
    r.add_argument("--out", required=True)
    r.add_argument("--models", nargs="+", default=["v1"], help="v0/v1/v2 or provider model IDs")
    r.add_argument("--effort", default="medium")
    r.add_argument("--writer-model", default="gpt-6-luna")
    r.add_argument("--writer-effort", default="max")
    r.add_argument("--semantic-model", default="gpt-6-luna")
    r.add_argument("--semantic-effort", default="high")
    r.add_argument("--alpha", type=float, default=1.0, help="RQ1/RQ2/custom threshold scale")
    r.add_argument("--alphas", type=float, nargs="+", default=[4.0, 2.0, 1.0, 0.5], help="RQ3/all granularity grid")
    r.add_argument("--jobs", type=int, default=1, help="concurrent candidate queries; paper used up to 8")
    r.add_argument("--call-parallelism", type=int, default=4)
    r.add_argument("--seed", type=int, default=20261012)
    r.add_argument("--resamples", type=int, default=10000)
    r.add_argument("--prices", default="none", help="explicit price-table JSON; default records tokens without estimating dollars")
    r.add_argument("--dry-run", action="store_true", help="print dependency plan; no writes or model calls")
    r.add_argument("--resume", action="store_true")
    r.set_defaults(fn=execute)
    e = sub.add_parser("evaluate", help="score existing runs with the official labels")
    e.add_argument("--dataset", default=str(DATASET))
    e.add_argument("--arm", action="append", required=True, help="NAME=RUN or NAME=RUN@direct")
    e.add_argument("--strata", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--keys")
    e.add_argument("--compare", action="append", default=[])
    e.add_argument("--resamples", type=int, default=10000)
    e.add_argument("--seed", type=int, default=20261012)
    e.set_defaults(fn=lambda a: importlib.import_module("reproduction.analysis").cmd_tables(a),
                   differs=[], blocks=[], denominator="all", multiblock=None)
    w = sub.add_parser("diagnostics", help="offline whole-query risk, risk quartiles and method combinations")
    w.add_argument("--dataset", default=str(DATASET))
    w.add_argument("--schemas")
    w.add_argument("--run", required=True, help="Opt run, containing both Opt and Whole verdicts")
    w.add_argument("--arm", action="append", default=[], help="additional NAME=RUN@direct, for example RT and SE")
    w.add_argument("--keys")
    w.add_argument("--jobs", type=int, default=1)
    w.add_argument("--out", required=True)
    w.set_defaults(fn=lambda a: importlib.import_module("reproduction.diagnostics").run(a))
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "diagnostics" and args.jobs < 1:
            raise ValueError("--jobs must be positive")
        if args.command == "run":
            if args.jobs < 1 or args.call_parallelism < 1 or args.resamples < 0:
                raise ValueError("Concurrency must be positive; resamples must be nonnegative")
            if any(not math.isfinite(x) or x <= 0 for x in [args.alpha, *args.alphas]):
                raise ValueError("All alpha values must be finite and positive")
            if len(set(args.alphas)) != len(args.alphas):
                raise ValueError("--alphas contains duplicates")
            if args.suite == "all" and 1.0 not in args.alphas:
                raise ValueError("--suite all requires alpha=1 in --alphas for RQ1 and RQ2")
            if args.prices != "none" and not Path(args.prices).is_file():
                raise ValueError("--prices must be none or an existing price-table JSON file")
        return args.fn(args)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
