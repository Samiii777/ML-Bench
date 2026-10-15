"""The harness turns timings into the numbers we report — check that arithmetic end to end."""

import time

import pytest
import torch

from core.args import build_base_parser
from core.harness import ComputeHarness, InferenceHarness, TrainingHarness, run_main
from tests.conftest import emitted, make_args

STEP_S = 0.004


def spin(seconds):
    """Precise fake work (time.sleep overshoots by ~50% on macOS)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


class SleepHarness(InferenceHarness):
    def load_model(self):
        return None

    def prepare_inputs(self):
        return torch.zeros(self.batch_size, 3, 8, 8)

    def run_step(self, model, inputs):
        spin(STEP_S)
        return inputs


def test_throughput_and_latency_follow_from_step_time(capsys):
    SleepHarness(make_args(batch_size=8, num_runs=10)).run()
    r = emitted(capsys)
    m = {x["name"]: x for x in r["metrics"]}
    assert r["status"] == "PASS"
    assert r["primary_metric"] == "throughput"
    # throughput is batch / mean batch latency, not per-sample latency inverted twice
    assert m["throughput"]["value"] == pytest.approx(8 * 1000 / r["latency_stats"]["mean"])
    assert m["throughput"]["value"] == pytest.approx(8 / STEP_S, rel=0.1)
    assert m["latency_p50_ms"]["value"] == pytest.approx(STEP_S * 1000, rel=0.1)
    assert m["latency_p50_ms"]["direction"] == "lower_is_better"
    assert r["latency_stats"]["n"] == 10
    assert r["config"] == {"input_shape": [8, 3, 8, 8]}


class FailingValidation(SleepHarness):
    def validate_result(self, model, inputs, outputs, validator):
        validator.expect_equals("answer", 41, 42)


def test_failed_validation_marks_run_fail(capsys):
    result = FailingValidation(make_args(num_runs=2)).run()
    assert result.status == "FAIL"
    assert "answer" in result.error
    assert emitted(capsys)["status"] == "FAIL"


class Exploding(SleepHarness):
    def load_model(self):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")


def test_run_main_emits_structured_fail_on_exception(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--device", "cpu"])
    with pytest.raises(SystemExit) as exc:
        run_main(Exploding, build_base_parser())
    assert exc.value.code == 1
    r = emitted(capsys)
    assert r["status"] == "FAIL" and r["error"].startswith("OOM:")


class TinyTraining(TrainingHarness):
    def load_model(self):
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))

    def prepare_inputs(self):
        g = torch.Generator().manual_seed(0)
        return torch.randn(self.batch_size, 16, generator=g), torch.randint(0, 4, (self.batch_size,), generator=g)

    def compute_loss(self, model, inputs):
        x, y = inputs
        return torch.nn.functional.cross_entropy(model(x), y)


def test_training_steps_reduce_loss(capsys):
    result = TinyTraining(make_args(batch_size=32, num_runs=20)).run()
    assert result.status == "PASS", result.error
    assert result.mode == "training"
    assert result.get_metric("loss_last") < result.get_metric("loss_first")


def test_training_amp_precision_is_labelled_honestly():
    assert TinyTraining(make_args(precision="bf16")).effective_precision == "amp_bf16"
    assert TinyTraining(make_args(precision="fp32")).effective_precision == "fp32"


class SleepOps(ComputeHarness):
    def get_operations(self):
        yield "op_flops", (lambda: spin(0.002)), 1.0, "GFLOPS"   # 1 GFLOP per 2 ms -> 500 GFLOPS
        yield "op_bytes", (lambda: spin(0.002)), 0.5, "GB/s"     # 0.5 GB per 2 ms -> 250 GB/s


def test_compute_rates_are_work_over_time(capsys):
    result = SleepOps(make_args(model="sleep_ops", num_runs=5)).run()
    assert result.status == "PASS"
    assert result.get_metric("op_flops") == pytest.approx(500, rel=0.1)
    assert result.get_metric("op_bytes") == pytest.approx(250, rel=0.1)
    assert result.get_metric("best_gflops") == result.get_metric("op_flops")
    assert result.primary_metric in ("best_gflops", "best_bandwidth_gbs")


def test_precision_dtypes():
    assert SleepHarness(make_args(precision="bf16")).dtype == torch.bfloat16
    assert SleepHarness(make_args(precision="fp16")).dtype == torch.float16
    assert SleepHarness(make_args(precision="mixed")).dtype == torch.float32  # weights stay fp32
    assert SleepHarness(make_args(precision="tf32")).dtype == torch.float32


class Unsupported(SleepHarness):
    def load_model(self):
        from core.harness import SkipBenchmark
        raise SkipBenchmark("not on this hardware")


def test_skip_is_emitted_with_reason_and_exit_zero(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--device", "cpu"])
    with pytest.raises(SystemExit) as exc:
        run_main(Unsupported, build_base_parser())
    assert exc.value.code == 0
    r = emitted(capsys)
    assert r["status"] == "SKIP" and r["error"] == "not on this hardware"


def test_mixed_precision_label_names_the_autocast_dtype():
    assert SleepHarness(make_args(precision="mixed", device="cpu")).effective_precision == "mixed_bf16"
