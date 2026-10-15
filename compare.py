#!/usr/bin/env python3
"""Compare two ML-Bench run files (run_*.json) on each test's headline metric.

  python compare.py BASELINE.json CURRENT.json [--threshold 5]

A change counts as a regression when the headline metric moves the wrong way
(respecting higher/lower-is-better) by more than --threshold percent. Exits 1
if any regression is found, so it can gate CI.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from core.schema import primary


def load_runs(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data["results"] if isinstance(data, dict) else data


def key(r: Dict[str, Any]) -> Tuple:
    return (r["framework"], r["model"], r["mode"], r["use_case"], r["precision"], r["batch_size"],
            r.get("execution_provider"))


def pct_change(base: Dict[str, Any], cur: Dict[str, Any]) -> Optional[float]:
    """Signed % improvement of cur over base (positive = better)."""
    if not base or not cur or base.get("value", 0) <= 0:
        return None
    delta = (cur["value"] - base["value"]) / base["value"] * 100.0
    return -delta if base.get("direction") == "lower_is_better" else delta


def compare(baseline: List[Dict], current: List[Dict], threshold: float) -> Tuple[List[Tuple], int]:
    base = {key(r): r for r in baseline if r["status"] == "PASS"}
    rows, regressions = [], 0
    for r in current:
        b = base.get(key(r))
        if b is None:
            continue
        bp, cp = primary(b["result"]), primary(r["result"]) if r["status"] == "PASS" else None
        change = pct_change(bp, cp)
        if cp is None:
            verdict = f"NOW {r['status']}"
            regressions += 1
        elif change is None:
            verdict = "n/a"
        elif change < -threshold:
            verdict = "REGRESSION"
            regressions += 1
        elif change > threshold:
            verdict = "improved"
        else:
            verdict = "same"
        label = f"{r['framework']}/{r['model']} {r['precision']} bs={r['batch_size']}"
        rows.append((label, bp, cp, change, verdict))
    return rows, regressions


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline")
    p.add_argument("current")
    p.add_argument("--threshold", type=float, default=5.0, help="percent change considered significant")
    args = p.parse_args(argv)

    rows, regressions = compare(load_runs(args.baseline), load_runs(args.current), args.threshold)
    if not rows:
        print("No matching passed tests between the two files.")
        return 0
    print(f"{'Test':<60} {'Baseline':>18} {'Current':>18} {'Change':>9}  Verdict")
    for label, bp, cp, change, verdict in rows:
        fmt = lambda m: f"{m['value']:,.2f} {m['unit'][:7]}" if m else "—"
        ch = f"{change:+.1f}%" if change is not None else "—"
        print(f"{label:<60} {fmt(bp):>18} {fmt(cp):>18} {ch:>9}  {verdict}")
    print(f"\n{len(rows)} compared, {regressions} regression(s) beyond {args.threshold}% "
          "(change is signed so + always means better)")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
