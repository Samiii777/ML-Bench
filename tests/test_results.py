"""Reports and comparisons must show measured values only, and judge direction correctly."""

import json

import compare
from utils import results


def rec(model="resnet50", status="PASS", value=100.0, direction="higher_is_better", mem=None, cv=1.0, **kw):
    metrics = [{"name": "throughput", "value": value, "unit": "samples/sec", "direction": direction}]
    if mem is not None:
        metrics.append({"name": "peak_memory_gb", "value": mem, "unit": "GB", "direction": "lower_is_better"})
    result = {"status": status, "precision": kw.get("effective", "fp16"), "metrics": metrics,
              "primary_metric": "throughput", "latency_stats": {"median": 5.0, "p99": 6.0, "cv_pct": cv},
              "system_info": {"device_name": "Test GPU", "gpu_memory_total_gb": 24}}
    return {"framework": "pytorch", "model": model, "mode": "inference", "use_case": "classification",
            "precision": "fp16", "batch_size": 1, "execution_provider": None, "status": status,
            "error": None if status == "PASS" else "boom", "result": result if status != "SKIP" else None}


def test_report_never_invents_memory():
    md = results.to_markdown([rec(mem=None)])
    row = next(line for line in md.splitlines() if line.startswith("| pytorch"))
    assert row.split("|")[9].strip() == "—"
    assert "1.50" in results.to_markdown([rec(mem=1.5)])


def test_report_flags_noise_and_shows_requested_vs_effective_precision():
    cells = results.row(rec(cv=25.0, effective="amp_bf16"))
    assert "⚠" in cells[7]
    assert cells[2] == "amp_bf16 (req fp16)"


def test_save_results_writes_all_artifacts(tmp_path):
    paths = results.save_results([rec(), rec(model="x", status="FAIL"), rec(model="y", status="SKIP")],
                                 str(tmp_path), {"mode": "inference"})
    data = json.loads(paths["json"].read_text())
    assert len(data["results"]) == 3 and data["metadata"]["system_info"]["device_name"] == "Test GPU"
    assert "primary_value" in paths["csv"].read_text().splitlines()[0]
    assert "1 passed, 1 failed, 1 skipped" in paths["md"].read_text()
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 3


def test_compare_respects_metric_direction():
    base = [rec(value=100), rec(model="lat", value=10, direction="lower_is_better")]
    cur = [rec(value=80), rec(model="lat", value=8, direction="lower_is_better")]
    rows, regressions = compare.compare(base, cur, threshold=5)
    verdicts = {label.split()[0]: verdict for label, *_, verdict in rows}
    assert verdicts["pytorch/resnet50"] == "REGRESSION"   # throughput fell
    assert verdicts["pytorch/lat"] == "improved"          # latency fell
    assert regressions == 1


def test_compare_counts_new_failures(tmp_path):
    rows, regressions = compare.compare([rec()], [rec(status="FAIL")], threshold=5)
    assert regressions == 1 and rows[0][-1] == "NOW FAIL"
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    b.write_text(json.dumps({"results": [rec()]}))
    c.write_text(json.dumps({"results": [rec(value=101)]}))
    assert compare.main([str(b), str(c)]) == 0
