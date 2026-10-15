"""Save and display a run's results.

Per run, writes to the output dir:
  run_<ts>.json   full records (args, system info, every metric, latency stats, logs of failures)
  run_<ts>.csv    one flat row per test
  run_<ts>.md     human report grouped by use case
  history.jsonl   one line per test, appended (feeds the dashboard and trend checks)
Nothing is estimated: a value that wasn't measured is shown as "—".
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schema import metric_map, primary

NOISY_CV_PCT = 10.0
COLUMNS = ["Framework", "Model", "Precision", "BS", "Result", "p50 ms", "p99 ms", "CV %", "Mem GB", "Status"]


def _fmt(v: Optional[float], spec: str = ",.2f") -> str:
    return "—" if v is None else format(v, spec)


def row(record: Dict[str, Any]) -> List[str]:
    """Display cells for one record (shared by console and markdown)."""
    result = record.get("result") or {}
    stats = result.get("latency_stats") or {}
    metrics = metric_map(result)
    p = primary(result) if record["status"] == "PASS" else None
    headline = f"{p['value']:,.2f} {p['unit']}" if p else "—"
    precision = result.get("precision") or record["precision"]
    if precision != record["precision"]:
        precision = f"{precision} (req {record['precision']})"
    model = record["model"] + (f" [{record['execution_provider'].replace('ExecutionProvider', '')}]"
                               if record.get("execution_provider") else "")
    cv = stats.get("cv_pct")
    cv_text = _fmt(cv, ".1f") + (" ⚠" if cv is not None and cv > NOISY_CV_PCT else "")
    mem = metrics.get("peak_memory_gb", {}).get("value")
    status = record["status"] if record["status"] == "PASS" else f"{record['status']}: {record.get('error') or ''}"
    return [record["framework"], model, precision, str(record["batch_size"]), headline,
            _fmt(stats.get("median")), _fmt(stats.get("p99")), cv_text, _fmt(mem), status]


def by_use_case(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['use_case']} ({r['mode']})" if r["mode"] != "inference" else r["use_case"]].append(r)
    return dict(sorted(groups.items()))


def system_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    for r in records:
        si = (r.get("result") or {}).get("system_info")
        if si and si.get("device_name") not in (None, "unknown"):
            return si
    return {}


def counts(records: List[Dict[str, Any]]) -> str:
    c = defaultdict(int)
    for r in records:
        c[r["status"]] += 1
    return f"{c['PASS']} passed, {c['FAIL']} failed, {c['SKIP']} skipped"


def print_summary(records: List[Dict[str, Any]]) -> None:
    if not records:
        print("No results.")
        return
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console(width=None if Console().is_terminal else 200)
        for use_case, recs in by_use_case(records).items():
            table = Table(title=use_case, title_justify="left")
            for col in COLUMNS:
                table.add_column(col, overflow="fold", justify="right" if col in COLUMNS[3:9] else "left")
            for r in recs:
                cells = row(r)
                style = {"FAIL": "red", "SKIP": "dim"}.get(r["status"])
                table.add_row(*cells, style=style)
            console.print(table)
    except ImportError:
        print(to_markdown(records))
    print(f"\n{counts(records)}")


def to_markdown(records: List[Dict[str, Any]], title: str = "ML-Bench results") -> str:
    si = system_summary(records)
    lines = [f"# {title}", ""]
    if si:
        stack = f"ROCm {si['rocm_version']}" if si.get("rocm_version") else f"CUDA {si.get('cuda_version') or '—'}"
        lines += [f"- **Device:** {si.get('device_name')} ({_fmt(si.get('gpu_memory_total_gb'), '.0f')} GB)",
                  f"- **Stack:** PyTorch {si.get('torch_version')}, {stack}, driver {si.get('gpu_driver_version') or '—'}",
                  f"- **Host:** {si.get('hostname')} — {si.get('cpu_model')} — {si.get('platform')}", ""]
    lines += [f"**{counts(records)}**", "",
              "Latency is per batch (one iteration). CV = std/mean of iteration latency; "
              f"⚠ marks CV > {NOISY_CV_PCT:.0f}% (noisy, rerun with more iterations).", ""]
    for use_case, recs in by_use_case(records).items():
        lines += [f"## {use_case}", "", "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(c.replace("|", "\\|") for c in row(r)) + " |" for r in recs]
        lines.append("")
    return "\n".join(lines)


def flat_row(record: Dict[str, Any]) -> Dict[str, Any]:
    result = record.get("result") or {}
    stats = result.get("latency_stats") or {}
    p = primary(result) or {}
    out = {
        "timestamp": record.get("timestamp"), "framework": record["framework"], "model": record["model"],
        "mode": record["mode"], "use_case": record["use_case"], "precision": record["precision"],
        "effective_precision": result.get("precision"), "batch_size": record["batch_size"],
        "execution_provider": record.get("execution_provider"), "status": record["status"],
        "primary_metric": result.get("primary_metric"), "primary_value": p.get("value"),
        "primary_unit": p.get("unit"), "latency_p50_ms": stats.get("median"),
        "latency_p99_ms": stats.get("p99"), "latency_cv_pct": stats.get("cv_pct"),
        "duration_s": record.get("duration_s"), "error": record.get("error"),
    }
    for name, m in metric_map(result).items():
        out.setdefault(f"metric_{name}", m.get("value"))
    return out


def save_results(records: List[Dict[str, Any]], output_dir: str, args: Dict[str, Any]) -> Dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {ext: out / f"run_{ts}.{ext}" for ext in ("json", "csv", "md")}

    payload = {"metadata": {"timestamp": datetime.now().isoformat(timespec="seconds"), "args": args,
                            "system_info": system_summary(records)},
               "results": records}
    paths["json"].write_text(json.dumps(payload, indent=2, default=str))

    rows = [flat_row(r) for r in records]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with paths["csv"].open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    paths["md"].write_text(to_markdown(records))

    with (out / "history.jsonl").open("a") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != "log_tail"}, default=str) + "\n")

    print("\nResults saved:")
    for p in paths.values():
        print(f"  {p}")
    return paths
