"""Whole-query validation repeated once per distinct block in the optimal decomposition."""
from __future__ import annotations


import datetime as dt

import json

import subprocess

import sys

import threading

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from typing import Any

from reproduction.data import new_dir

from reproduction import analysis as paper

RULES = ("conjunction",)

SAMPLE_TAG = "sample"

def combine(verdicts: list[str]) -> dict[str, str]:
    """The verdict of one candidate's samples: it rejects if any sample rejects."""
    from sqldecomp.verify import conjunction
    return {"conjunction": conjunction(verdicts)}

def sample_label(template: str, i: int) -> str:
    """The template label of sample i (2..k): the base item's label plus `|sample-i`."""
    return f"{template}|{SAMPLE_TAG}-{i}"

def plan(base: dict[str, dict[str, Any]], only: set[str] | None = None, k_from: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Per candidate of the base run (by key): k (from `k_from`'s optimum when given, else the base run's), sample 1 and
    the prompt to ask again; a candidate the importer refused with k None. Refuses a candidate without an optimum (and
    not refused), whose Direct is not one judgement of the whole query, whose cached call is missing, does not hash to
    the record's key, was asked outside the sandbox, or parses to another verdict than the record's; and a `k_from` that
    lacks a candidate of the base run or refuses others."""
    from sqldecomp import llm
    from sqldecomp.verify import parse
    refused = paper.refused_keys(base)
    unknown = sorted((only or set()) - set(base))
    if unknown:
        raise SystemExit(f"--keys not in the base run: {unknown[:5]}")
    if k_from is not None:
        missing = sorted(set(base) - set(k_from))
        if missing or paper.refused_keys(k_from) != refused:
            raise SystemExit(f"--k-from lacks candidates of the base run ({missing[:5]}) or refuses others")
    sizes = k_from if k_from is not None else base
    out = []
    for key in sorted(base):
        if only is not None and key not in only:
            continue
        r = base[key]
        if key in refused:
            out.append({"key": key, "refused": True, "k": None})
            continue
        k = (sizes[key].get("optimum") or {}).get("unique_blocks")
        if not k:
            raise SystemExit(f"{key}: no optimum to take k from, and not refused")
        direct = r.get("direct") or {}
        items = direct.get("items") or []
        if direct.get("rule") != "conjunction" or len(items) != 1 or items[0].get("item") != "whole":
            raise SystemExit(f"{key}: the base run's Direct is not one judgement of the whole query")
        it = items[0]
        path = Path(r["config"]["verdict_cache"]) / f"{it['key']}.json"
        if not path.exists():
            raise SystemExit(f"{key}: the base call is not in its cache ({path})")
        entry = json.loads(path.read_text())
        if not entry.get("sandbox", True):
            raise SystemExit(f"{key}: the base call was asked outside the sandbox")
        if entry.get("template") != it["template"] or llm.cache_key(entry["prompt"], entry["model"], entry["effort"], entry["template"]) != it["key"]:
            raise SystemExit(f"{key}: the cached prompt does not hash to the record's key")
        if parse(entry["answer"])[0] != it["verdict"] or it["verdict"] != direct.get("final"):
            raise SystemExit(f"{key}: the cached answer does not parse to the record's verdict")
        price = (r.get("config") or {}).get("prices", {}).get(entry["model"])
        out.append({"key": key, "refused": False, "k": int(k), "template": it["template"], "model": entry["model"], "effort": entry["effort"],
                    "prompt": entry["prompt"], "price": price, "base": {"key": it["key"], "verdict": it["verdict"], "reason": it.get("reason"),
                                                                          "run": (r.get("run") or {}).get("tag"), "cache": str(path.parent),
                                                                          "time": entry.get("time")}})
    return out

def ask_sample(p: dict[str, Any], i: int, config, price, attempts: int, wait_s: float) -> dict[str, Any]:
    """Sample i of a candidate: its call under `sample_label`, asked again (same label) while it exits non-zero or
    answers nothing, at most `attempts` times."""
    from sqldecomp import llm
    from sqldecomp.verify import parse
    label = sample_label(p["template"], i)
    calls = []
    for attempt in range(1, attempts + 1):
        c = llm.call(p["prompt"], label, config)
        calls.append(c.record(price))
        if c.exit_code == 0 and c.answer.strip():
            verdict, reason = parse(c.answer)
            return {"i": i, "label": label, "key": c.key, "verdict": verdict, "reason": reason, "cached": c.cached, "attempts": attempt,
                    "failed": False, "calls": calls}
        if attempt < attempts:
            time.sleep(wait_s)
    return {"i": i, "label": label, "key": calls[-1]["key"], "verdict": "undecided", "reason": None, "cached": False, "attempts": attempts,
            "failed": True, "calls": calls}

def run_candidate(p: dict[str, Any], config, price, *, call_parallelism: int, attempts: int, wait_s: float) -> dict[str, Any]:
    """Every sample of one candidate and their verdict."""
    from sqldecomp.llm import in_parallel
    t0 = time.perf_counter()
    if p["refused"]:
        return {"key": p["key"], "refused": True, "k": None, "samples": [], "verdicts": [], "conjunction": None, "seconds": 0.0}
    first = {"i": 1, "label": p["template"], "key": p["base"]["key"], "verdict": p["base"]["verdict"], "reason": p["base"]["reason"],
             "source": "base", "run": p["base"]["run"], "cache": p["base"]["cache"], "asked": p["base"]["time"]}
    fresh = in_parallel([lambda i=i: ask_sample(p, i, config, price, attempts, wait_s) for i in range(2, p["k"] + 1)], call_parallelism)
    samples = [first, *fresh]
    verdicts = [s["verdict"] for s in samples]
    return {"key": p["key"], "refused": False, "k": p["k"], "template": p["template"], "model": p["model"], "effort": p["effort"],
            "samples": samples, "verdicts": verdicts, **combine(verdicts),
            "counts": {v: verdicts.count(v) for v in ("yes", "no", "undecided")}, "failed_samples": [s["i"] for s in fresh if s["failed"]],
            "seconds": round(time.perf_counter() - t0, 1)}

def rule_record(row: dict[str, Any], rule: str) -> dict[str, Any]:
    """The record `paper.py tables` reads for one rule: the rule's verdict as the direct arm's `final`."""
    final = row.get(rule)
    findings = [{"stage": "direct-k", "kind": "model_call_failed", "sample": i} for i in row.get("failed_samples") or []]
    if row.get("error"):
        findings.append({"stage": "direct-k", "kind": "error", "detail": row["error"]})
    return {"key": row["key"], "refused": row.get("refused", False), "k": row.get("k"), "findings": findings,
            "direct": {"final": final if final in ("yes", "no", "undecided") else None, "rule": rule, "k": row.get("k"),
                       "verdicts": row.get("verdicts", [])}}

def codex_version() -> str:
    from sqldecomp import llm
    try:
        return subprocess.run([str(llm.codex_binary()), "--version"], capture_output=True, text=True, timeout=60).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"

def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")

def run(args) -> int:
    from sqldecomp import llm
    from sqldecomp.runner import code_version
    base = paper.load_records(args.base)
    items = plan(base, paper._keys(args.keys), paper.load_records(args.k_from) if args.k_from else None)
    asked = [p for p in items if not p["refused"]]
    models = {(p["model"], p["effort"]) for p in asked}
    prices = {json.dumps(p["price"], sort_keys=True) for p in asked}
    if len(models) != 1 or len(prices) != 1:
        raise SystemExit(f"the base calls differ in model, effort or price: {sorted(models)}")
    (model, effort), price_json = models.pop(), prices.pop()
    price = llm.Price.from_json(json.loads(price_json)) if price_json != "null" else None
    cache = Path(args.cache)
    if cache.exists() and not args.reuse_cache:
        raise SystemExit(f"--cache must be new (or --reuse-cache after a crash): {cache}")
    out = new_dir(args.out)
    cache.mkdir(parents=True, exist_ok=True)
    config = llm.ModelConfig(model=model, effort=effort, cache_dir=str(cache), sandbox=True)
    llm.require_sandbox(config)
    meta = {"argv": list(sys.argv), "base": args.base, "k_from": args.k_from, "cache": str(cache.resolve()), "out": str(out.resolve()), "model": model, "effort": effort,
            "sandbox": True, "timeout_s": config.timeout_s, "price": json.loads(price_json), "jobs": args.jobs, "call_parallelism": args.call_parallelism,
            "attempts": args.attempts, "retry_wait_s": args.retry_wait, "reuse_cache": args.reuse_cache, "keys": args.keys,
            "code": code_version(), "codex": codex_version(), "started": _now(), "candidates": len(items),
            "refused": sum(1 for p in items if p["refused"]), "k1": sum(1 for p in asked if p["k"] == 1),
            "samples_to_ask": sum(p["k"] - 1 for p in asked)}
    (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    order = sorted(items, key=lambda p: (-(p["k"] or 0), p["key"]))
    partial = out / "samples.partial.jsonl"
    lock = threading.Lock()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def one(p: dict[str, Any]) -> dict[str, Any]:
        try:
            return run_candidate(p, config, price, call_parallelism=args.call_parallelism, attempts=args.attempts, wait_s=args.retry_wait)
        except Exception as e:  # recorded, the others go on; the exit code says so
            return {"key": p["key"], "refused": p["refused"], "k": p["k"], "samples": [], "verdicts": [], "conjunction": None,
                    "error": repr(e)[:500]}

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(one, p) for p in order]
        for f in as_completed(futures):
            row = f.result()
            with lock:
                rows.append(row)
                with partial.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"[{len(rows)}/{len(order)}] {time.perf_counter() - t0:7.0f} s {row['key']} k={row['k']} "
                      f"{''.join(v[0] for v in row['verdicts'])} conj={row['conjunction']}"
                      + (f" ERROR {row['error']}" if row.get("error") else ""), file=sys.stderr, flush=True)
    rows.sort(key=lambda r: r["key"])
    (out / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for rule in RULES:
        d = out / rule / "pool"
        d.mkdir(parents=True)
        (d / "results.jsonl").write_text("".join(json.dumps(rule_record(r, rule)) + "\n" for r in rows))
    partial.unlink()
    calls = [c for r in rows for s in r["samples"] for c in s.get("calls", [])]
    fresh = [s for r in rows for s in r["samples"] if s.get("source") != "base"]
    meta.update({"finished": _now(), "seconds": round(time.perf_counter() - t0, 1),
                 "calls": {"asked": sum(1 for c in calls if not c["cached"]), "cache_hits": sum(1 for c in calls if c["cached"]),
                           "answered": sum(1 for c in calls if c["exit_code"] == 0 and not c["cached"]),
                           "failed": sum(1 for c in calls if c["exit_code"] != 0), "samples": len(fresh),
                           "samples_failed": sum(1 for s in fresh if s["failed"]), "samples_retried": sum(1 for s in fresh if s["attempts"] > 1)},
                 "usd": round(sum(c["cost_usd"] or 0 for c in calls if not c["cached"]), 4),
                 "unpriced": sum(1 for c in calls if not c["cached"] and c["cost_usd"] is None),
                 "latency_s": _latency(calls), "errors": sorted(r["key"] for r in rows if r.get("error")),
                 "verdicts": {rule: {v: sum(1 for r in rows if r[rule] == v) for v in ("yes", "no", "undecided", None)} for rule in RULES},
                 "sample_verdicts": {v: sum(1 for s in fresh if s["verdict"] == v) for v in ("yes", "no", "undecided")}})
    meta["verdicts"] = {rule: {str(v): n for v, n in by.items()} for rule, by in meta["verdicts"].items()}
    (out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("seconds", "calls", "usd", "verdicts", "errors")}), file=sys.stderr)
    return 1 if meta["errors"] else 0

def _latency(calls: list[dict[str, Any]]) -> dict[str, Any]:
    v = sorted(c["latency_ms"] / 1000 for c in calls if not c["cached"] and c["exit_code"] == 0)
    if not v:
        return {}
    return {"answered": len(v), "median": round(v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2, 1),
            "p90": round(v[min(len(v) - 1, int(0.9 * len(v)))], 1), "max": round(v[-1], 1)}
