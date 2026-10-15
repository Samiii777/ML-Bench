"""End-to-end on a real GPU: run through the orchestrator and sanity-check magnitudes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("no CUDA/ROCm GPU", allow_module_level=True)


def run(tmp_path, *args):
    subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--output_dir", str(tmp_path), *args],
                   cwd=ROOT, check=False, capture_output=True, text=True, timeout=1200)
    return json.loads(next(tmp_path.glob("run_*.json")).read_text())["results"]


def test_gemm_reaches_plausible_tflops(tmp_path):
    (r,) = run(tmp_path, "--framework", "pytorch", "--model", "gemm_ops", "--precision", "fp16", "--num_runs", "5")
    assert r["status"] == "PASS", r.get("error")
    best = next(m["value"] for m in r["result"]["metrics"] if m["name"] == "best_gflops")
    assert 5_000 < best < 3_000_000  # >5 TFLOPS: real GPU math; <3 PFLOPS: no timing bug


def test_resnet50_classifies_correctly(tmp_path):
    (r,) = run(tmp_path, "--framework", "pytorch", "--model", "resnet50", "--precision", "fp16", "--batch_size", "8", "--num_runs", "10")
    assert r["status"] == "PASS", r.get("error")
    checks = {c["name"]: c["passed"] for c in r["result"]["validation_checks"]}
    assert checks["top1_class"]
