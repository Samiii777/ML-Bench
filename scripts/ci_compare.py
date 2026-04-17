#!/usr/bin/env python3
"""CI regression detector — compares current benchmark results against a baseline."""

import argparse
import json
import sys
from pathlib import Path


def load_results(filepath: str) -> list:
    with open(filepath) as f:
        data = json.load(f)
    return data.get("results", data if isinstance(data, list) else [data])


def make_key(r: dict) -> str:
    return f"{r.get('framework','')}/{r.get('model','')}/{r.get('precision','')}/bs{r.get('batch_size',1)}"


def extract_throughput(r: dict) -> float:
    metrics = r.get("metrics", {})
    if isinstance(metrics, list):
        for m in metrics:
            if isinstance(m, dict) and m.get("name") in ("throughput", "throughput_fps"):
                return float(m["value"])
    elif isinstance(metrics, dict):
        return float(metrics.get("throughput_fps", metrics.get("throughput", 0)))
    return 0.0


def compare(baseline_file: str, current_file: str, threshold: float = 5.0) -> int:
    baseline = {make_key(r): r for r in load_results(baseline_file)}
    current = {make_key(r): r for r in load_results(current_file)}

    regressions = []
    improvements = []

    for key, cur in current.items():
        if key not in baseline:
            continue
        base = baseline[key]
        base_tp = extract_throughput(base)
        cur_tp = extract_throughput(cur)

        if base_tp <= 0:
            continue

        change_pct = ((cur_tp - base_tp) / base_tp) * 100.0

        if change_pct < -threshold:
            regressions.append((key, base_tp, cur_tp, change_pct))
        elif change_pct > threshold:
            improvements.append((key, base_tp, cur_tp, change_pct))

    print("## Benchmark Comparison Report\n")

    if regressions:
        print(f"### Regressions (>{threshold}% slower)\n")
        for key, base_tp, cur_tp, pct in regressions:
            print(f"- **{key}**: {base_tp:.2f} → {cur_tp:.2f} ({pct:+.1f}%)")
        print()

    if improvements:
        print(f"### Improvements (>{threshold}% faster)\n")
        for key, base_tp, cur_tp, pct in improvements:
            print(f"- **{key}**: {base_tp:.2f} → {cur_tp:.2f} ({pct:+.1f}%)")
        print()

    matched = len(set(current.keys()) & set(baseline.keys()))
    print(f"Compared {matched} matching configs. "
          f"{len(regressions)} regression(s), {len(improvements)} improvement(s).")

    if regressions:
        print(f"\nFAIL: {len(regressions)} regression(s) exceed {threshold}% threshold")
        return 1
    print("\nPASS: No regressions detected")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI benchmark regression detector")
    parser.add_argument("--baseline", required=True, help="Baseline result JSON file")
    parser.add_argument("--current", required=True, help="Current result JSON file")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold percentage (default: 5.0)")
    args = parser.parse_args()
    sys.exit(compare(args.baseline, args.current, args.threshold))
