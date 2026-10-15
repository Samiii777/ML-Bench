#!/usr/bin/env python3
"""ML-Bench dashboard: `python benchmark.py --dashboard` (needs streamlit, plotly, pandas).

Reads benchmark_results/history.jsonl, which every run appends to.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.results import flat_row

HISTORY = Path(__file__).resolve().parent / "benchmark_results" / "history.jsonl"


def load() -> pd.DataFrame:
    if not HISTORY.exists():
        return pd.DataFrame()
    rows = []
    for line in HISTORY.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        flat = flat_row(r)
        si = (r.get("result") or {}).get("system_info") or {}
        flat["device"] = si.get("device_name") or "—"
        flat["host"] = si.get("hostname") or "—"
        flat["test"] = f"{flat['framework']}/{flat['model']} {flat['precision']} bs={flat['batch_size']}"
        rows.append(flat)
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="ML-Bench", layout="wide")
    st.title("ML-Bench")
    df = load()
    if df.empty:
        st.warning(f"No history yet at {HISTORY}. Run `python benchmark.py` first.")
        return
    passed = df[df["status"] == "PASS"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tests run", len(df))
    c2.metric("Passed", int((df["status"] == "PASS").sum()))
    c3.metric("Failed", int((df["status"] == "FAIL").sum()))
    c4.metric("Devices", df["device"].nunique())

    use_case = st.selectbox("Use case", sorted(passed["use_case"].unique()))
    sub = passed[passed["use_case"] == use_case]
    unit = sub["primary_unit"].dropna().iloc[0] if not sub.empty else ""

    st.subheader("Latest result per test and device")
    latest = sub.sort_values("timestamp").groupby(["test", "device"], as_index=False).last()
    st.plotly_chart(px.bar(latest, x="test", y="primary_value", color="device", barmode="group",
                           labels={"primary_value": unit}), use_container_width=True)
    st.dataframe(latest[["test", "device", "effective_precision", "primary_value", "primary_unit",
                         "latency_p50_ms", "latency_p99_ms", "latency_cv_pct", "metric_peak_memory_gb"]])

    st.subheader("Trend")
    test = st.selectbox("Test", sorted(sub["test"].unique()))
    st.plotly_chart(px.line(sub[sub["test"] == test].sort_values("timestamp"), x="timestamp", y="primary_value",
                            color="device", markers=True, labels={"primary_value": unit}),
                    use_container_width=True)

    failed = df[df["status"] == "FAIL"].sort_values("timestamp", ascending=False)
    if not failed.empty:
        st.subheader("Recent failures")
        st.dataframe(failed[["timestamp", "test", "device", "error"]].head(50))


main()
