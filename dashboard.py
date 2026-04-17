#!/usr/bin/env python3
"""ML-Bench Results Dashboard — Streamlit-based interactive visualization."""

import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
except ImportError:
    print("Dashboard requires: pip install streamlit plotly pandas")
    sys.exit(1)

from core.results_db import load_history, DEFAULT_HISTORY_PATH


def load_all_results():
    """Load results from JSONL history and any JSON files in benchmark_results/."""
    entries = load_history()

    results_dir = Path("benchmark_results")
    if results_dir.exists():
        for f in sorted(results_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                results = data.get("results", [])
                ts = data.get("metadata", {}).get("timestamp", f.stem)
                for r in results:
                    r.setdefault("timestamp", ts)
                    entries.append(r)
            except Exception:
                continue

    return entries


def flatten_metrics(entries):
    """Convert entries to a flat DataFrame for analysis."""
    rows = []
    for e in entries:
        row = {
            "timestamp": e.get("timestamp", ""),
            "framework": e.get("framework", ""),
            "model": e.get("model", ""),
            "precision": e.get("precision", ""),
            "batch_size": e.get("batch_size", 1),
            "status": e.get("status", ""),
            "mode": e.get("mode", ""),
            "use_case": e.get("usecase", e.get("use_case", "")),
            "hostname": e.get("hostname", ""),
        }
        metrics = e.get("metrics", {})
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    row[k] = v
        elif isinstance(metrics, list):
            for m in metrics:
                if isinstance(m, dict) and "name" in m:
                    row[m["name"]] = m.get("value", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="ML-Bench Dashboard", layout="wide")
    st.title("ML-Bench Results Dashboard")

    entries = load_all_results()
    if not entries:
        st.warning("No benchmark results found. Run some benchmarks first.")
        return

    df = flatten_metrics(entries)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Hardware Comparison", "Regressions"])

    with tab1:
        st.header("Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Runs", len(df))
        col2.metric("Passed", len(df[df["status"] == "PASS"]))
        col3.metric("Failed", len(df[df["status"] == "FAIL"]))
        col4.metric("Models Tested", df["model"].nunique())

        if "throughput" in df.columns or "throughput_fps" in df.columns:
            tp_col = "throughput" if "throughput" in df.columns else "throughput_fps"
            st.subheader("Latest Results")
            latest = df.sort_values("timestamp", ascending=False).drop_duplicates(
                subset=["framework", "model", "precision", "batch_size"])
            st.dataframe(latest[["framework", "model", "precision", "batch_size", "status", tp_col]].head(20))

    with tab2:
        st.header("Performance Trends")
        tp_col = "throughput" if "throughput" in df.columns else "throughput_fps"
        if tp_col in df.columns and len(df) > 1:
            model = st.selectbox("Select model", sorted(df["model"].unique()))
            model_df = df[df["model"] == model].sort_values("timestamp")
            if not model_df.empty:
                fig = px.line(model_df, x="timestamp", y=tp_col, color="precision",
                              title=f"{model} Throughput Over Time")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need multiple runs to show trends.")

    with tab3:
        st.header("Hardware Comparison")
        if "hostname" in df.columns and df["hostname"].nunique() > 1:
            tp_col = "throughput" if "throughput" in df.columns else "throughput_fps"
            if tp_col in df.columns:
                fig = px.bar(df, x="model", y=tp_col, color="hostname", barmode="group",
                             title="Throughput by Hardware")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run benchmarks on multiple machines to compare hardware.")

    with tab4:
        st.header("Regression Detection")
        tp_col = "throughput" if "throughput" in df.columns else "throughput_fps"
        if tp_col in df.columns:
            grouped = df.groupby(["framework", "model", "precision", "batch_size"])
            regressions = []
            for name, group in grouped:
                if len(group) < 2:
                    continue
                sorted_g = group.sort_values("timestamp")
                prev = sorted_g[tp_col].iloc[-2]
                curr = sorted_g[tp_col].iloc[-1]
                if prev > 0:
                    change = ((curr - prev) / prev) * 100
                    if change < -5:
                        regressions.append({
                            "config": "/".join(str(x) for x in name),
                            "previous": f"{prev:.2f}",
                            "current": f"{curr:.2f}",
                            "change": f"{change:+.1f}%",
                        })
            if regressions:
                st.error(f"Found {len(regressions)} regression(s) (>5% slower)")
                st.table(pd.DataFrame(regressions))
            else:
                st.success("No regressions detected")
        else:
            st.info("No throughput data available for regression analysis.")


if __name__ == "__main__":
    main()
