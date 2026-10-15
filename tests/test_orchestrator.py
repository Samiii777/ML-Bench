"""benchmark.py decides PASS/FAIL and which tests exist; check those decisions."""

import argparse
import json
import sys
import textwrap

import pytest

import benchmark
from core.schema import JSON_END, JSON_START
from utils import config


def cli(**kw):
    base = dict(framework=["pytorch"], model=None, mode="inference", usecase=None, precision=None, batch_size=None,
                device="auto", num_runs=None, num_warmup=None, power=False, execution_provider=None,
                script_args=None, skip_model=None, skip_usecase=None, skip_vram_check=True, timeout=1,
                verbose=False, debug=False)
    base.update(kw)
    return argparse.Namespace(**base)


def block(status="PASS", **extra):
    return f"noise\n{JSON_START}\n{json.dumps({'status': status, **extra})}\n{JSON_END}\n"


def test_parse_result_takes_last_block_and_rejects_garbage():
    assert benchmark.parse_result(block("FAIL") + block("PASS"))["status"] == "PASS"
    assert benchmark.parse_result("no json here") is None
    assert benchmark.parse_result(f"{JSON_START}\n{{broken\n{JSON_END}") is None


@pytest.fixture
def fake_script(tmp_path, monkeypatch):
    """Point every test at a throwaway script whose body we choose."""
    def make(body):
        path = tmp_path / "main.py"
        path.write_text(textwrap.dedent(body))
        monkeypatch.setattr(benchmark, "resolve_benchmark_path", lambda *a: path)
        return path
    return make


def test_json_status_wins_over_exit_code_zero(fake_script):
    fake_script(f"print('''{block('FAIL', error='top1 wrong')}''')")
    rec = benchmark.run_test(dict(framework="pytorch", model="m", mode="inference", use_case="classification",
                                  precision="fp32", batch_size=1, execution_provider=None, skip=None), cli())
    assert rec["status"] == "FAIL" and rec["error"] == "top1 wrong"


def test_nonzero_exit_fails_even_if_json_says_pass(fake_script):
    fake_script(f"import sys; print('''{block('PASS')}'''); sys.exit(3)")
    rec = benchmark.run_test(dict(framework="pytorch", model="m", mode="inference", use_case="classification",
                                  precision="fp32", batch_size=1, execution_provider=None, skip=None), cli())
    assert rec["status"] == "FAIL" and "exit code 3" in rec["error"]


def test_missing_result_is_fail_with_log_tail(fake_script):
    fake_script("print('Traceback: boom')")
    rec = benchmark.run_test(dict(framework="pytorch", model="m", mode="inference", use_case="classification",
                                  precision="fp32", batch_size=1, execution_provider=None, skip=None), cli())
    assert rec["status"] == "FAIL" and "no result" in rec["error"] and "boom" in rec["log_tail"]


def test_silent_hang_is_killed_by_timeout(fake_script):
    fake_script("import time; time.sleep(30)")
    rec = benchmark.run_test(dict(framework="pytorch", model="m", mode="inference", use_case="classification",
                                  precision="fp32", batch_size=1, execution_provider=None, skip=None),
                             cli(timeout=0.02))  # 1.2 s
    assert rec["status"] == "FAIL" and "timed out" in rec["error"]
    assert rec["duration_s"] < 10


def test_default_matrix_has_a_script_for_every_model(monkeypatch):
    monkeypatch.setattr(benchmark, "has_gpu", lambda: True)
    for mode in ("inference", "training"):
        tests = benchmark.build_matrix(cli(mode=mode))
        assert tests
        missing = [t for t in tests if t["skip"] and "no pytorch" in t["skip"]]
        assert not missing, missing


def test_matrix_expands_defaults_and_gpu_ops(monkeypatch):
    monkeypatch.setattr(benchmark, "has_gpu", lambda: True)
    tests = benchmark.build_matrix(cli(model=["resnet50"]))
    assert {(t["precision"], t["batch_size"]) for t in tests} == {
        (p, b) for p in config.USE_CASE_PRECISIONS["classification"] for b in config.USE_CASE_BATCH_SIZES["classification"]}
    ops = benchmark.build_matrix(cli(model=["gpu_ops"], precision=["fp32"]))
    assert [t["model"] for t in ops] == benchmark.GPU_OPS


def test_reduced_precision_skipped_without_gpu(monkeypatch):
    monkeypatch.setattr(benchmark, "has_gpu", lambda: False)
    tests = benchmark.build_matrix(cli(model=["resnet50"], precision=["fp32", "bf16"], batch_size=[1]))
    by_p = {t["precision"]: t["skip"] for t in tests}
    assert by_p["fp32"] is None and "GPU" in by_p["bf16"]


def test_unavailable_framework_is_skip_not_fail(monkeypatch):
    monkeypatch.setattr(benchmark, "framework_unavailable", lambda fw: "not installed")
    tests = benchmark.build_matrix(cli(framework=["onnx"]))
    assert tests and all(t["skip"] == "not installed" for t in tests)


def test_model_without_framework_runs_only_where_listed():
    pairs = benchmark.framework_models(cli(framework=None, model=["bert-base", "resnet50", "efficientnet_v2_s"]))
    assert pairs == [("pytorch", "bert-base"), ("pytorch", "resnet50"), ("onnx", "resnet50"),
                     ("pytorch", "efficientnet_v2_s")]


def test_script_skip_is_recorded_as_skip(fake_script):
    fake_script(f"print('''{block('SKIP', error='TensorRT listed but not usable')}''')")
    rec = benchmark.run_test(dict(framework="onnx", model="m", mode="inference", use_case="classification",
                                  precision="fp16", batch_size=1, execution_provider="X", skip=None), cli())
    assert rec["status"] == "SKIP" and "TensorRT" in rec["error"] and rec["result"] is None


def test_self_managed_frameworks_ignore_precision_and_batch_flags(monkeypatch):
    monkeypatch.setattr(benchmark, "framework_unavailable", lambda fw: None)
    tests = benchmark.build_matrix(cli(framework=["ollama"], model=["qwen3:8b"], precision=["fp16"], batch_size=[8]))
    assert [(t["precision"], t["batch_size"]) for t in tests] == [("auto", 1)]


def test_training_mode_only_schedules_pytorch(monkeypatch):
    monkeypatch.setattr(benchmark, "framework_unavailable", lambda fw: None)
    tests = benchmark.build_matrix(cli(framework=None, mode="training"))
    assert tests and {t["framework"] for t in tests} == {"pytorch"}


def test_timeout_kills_grandchildren(fake_script, tmp_path):
    pidfile = tmp_path / "child.pid"
    fake_script(f"""
        import subprocess, time
        child = subprocess.Popen(["sleep", "60"])
        open({str(pidfile)!r}, "w").write(str(child.pid))
        time.sleep(60)
    """)
    rec = benchmark.run_test(dict(framework="llamacpp", model="m", mode="inference", use_case="text_generation",
                                  precision="auto", batch_size=1, execution_provider=None, skip=None),
                             cli(timeout=0.02))
    assert rec["status"] == "FAIL" and "timed out" in rec["error"]
    import os, time
    time.sleep(0.3)  # orphaned child is reaped by init
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
