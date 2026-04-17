"""Append-only JSONL results store for benchmark history tracking."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_HISTORY_PATH = "benchmark_results/history.jsonl"


def store_run(results: List[Dict[str, Any]], history_path: str = DEFAULT_HISTORY_PATH) -> None:
    """Append benchmark results to the JSONL history file."""
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    timestamp = datetime.now().isoformat()
    hostname = ""
    try:
        import platform
        hostname = platform.node()
    except Exception:
        pass

    with open(history_path, "a") as f:
        for r in results:
            entry = {
                "timestamp": timestamp,
                "hostname": hostname,
                **r,
            }
            f.write(json.dumps(entry, default=str) + "\n")


def load_history(history_path: str = DEFAULT_HISTORY_PATH) -> List[Dict[str, Any]]:
    """Load all entries from the JSONL history file."""
    p = Path(history_path)
    if not p.exists():
        return []

    entries = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def query_history(
    benchmark_key: str,
    metric: str = "throughput_fps",
    history_path: str = DEFAULT_HISTORY_PATH,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Query history for a specific benchmark configuration and metric."""
    entries = load_history(history_path)
    matching = []
    for e in entries:
        key = f"{e.get('framework','')}/{e.get('model','')}/{e.get('precision','')}/bs{e.get('batch_size',1)}"
        if key != benchmark_key:
            continue
        metrics = e.get("metrics", {})
        value = None
        if isinstance(metrics, dict):
            value = metrics.get(metric)
        elif isinstance(metrics, list):
            for m in metrics:
                if isinstance(m, dict) and m.get("name") == metric:
                    value = m.get("value")
                    break
        if value is not None:
            matching.append({
                "timestamp": e.get("timestamp", ""),
                "value": value,
                "hostname": e.get("hostname", ""),
            })
    return matching[-limit:]


def get_latest_baseline(
    benchmark_key: str,
    history_path: str = DEFAULT_HISTORY_PATH,
) -> Optional[Dict[str, Any]]:
    """Get the most recent result for a given benchmark configuration."""
    entries = load_history(history_path)
    for e in reversed(entries):
        key = f"{e.get('framework','')}/{e.get('model','')}/{e.get('precision','')}/bs{e.get('batch_size',1)}"
        if key == benchmark_key:
            return e
    return None
