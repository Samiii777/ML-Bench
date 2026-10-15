"""Benchmark harnesses: one timing/memory/stats/output path shared by all scripts.

Subclasses implement ``load_model``, ``prepare_inputs`` and ``run_step``; ``run()``
does warmup, timed iterations, memory + optional power sampling, validation and
emits a single ``BenchmarkResult``.

Metric conventions (all harnesses):
  throughput        samples/sec = batch_size / mean batch latency
  latency_p50_ms    median latency of one batch (one ``run_step`` call)
  latency_p99_ms    99th percentile batch latency
  peak_memory_gb    PyTorch allocator peak during the timed loop
"""

import argparse
import contextlib
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.benchmark_utils import (
    benchmark_loop, calibrated_reps, compute_stats, measure_peak_memory,
    reset_memory_tracking, setup_torch_backends, synchronize, warmup,
)
from utils.shared_device_utils import collect_system_fingerprint
from core.schema import BenchmarkResult, MetricEntry, SystemInfo
from core.output import emit_result, emit_error
from core.validation import ResultValidator

DTYPES = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "mixed": torch.float32,  # fp32 weights, fp16 autocast
}


class SkipBenchmark(Exception):
    """Raise when a config can't be measured meaningfully here (reported as SKIP, not FAIL)."""


def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_system_info(device: torch.device) -> SystemInfo:
    fp = collect_system_fingerprint()
    si = SystemInfo(device=str(device), **{k: v for k, v in fp.items() if k in SystemInfo.__dataclass_fields__})
    if device.type == "cuda":
        si.device_name = torch.cuda.get_device_name(device)
        si.gpu_memory_total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        si.gpu_vendor = "amd" if getattr(torch.version, "hip", None) else "nvidia"
    elif device.type == "mps":
        si.device_name, si.gpu_vendor = "Apple MPS", "apple"
    else:
        si.device_name, si.gpu_vendor = si.cpu_model or "CPU", "none"
    return si


class BenchmarkHarness:
    framework = "pytorch"
    mode = "inference"
    use_case = "classification"
    primary_metric = "throughput"
    throughput_unit = "samples/sec"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.precision = getattr(args, "precision", "fp32")
        self.batch_size = getattr(args, "batch_size", 1)
        self.device = resolve_device(getattr(args, "device", "auto"))
        self._model = None
        self._inputs = None

    # ── subclass API ──────────────────────────────────────────────

    def load_model(self) -> Any:
        raise NotImplementedError

    def prepare_inputs(self) -> Any:
        raise NotImplementedError

    def run_step(self, model: Any, inputs: Any) -> Any:
        raise NotImplementedError

    def get_extra_metrics(self, model: Any, inputs: Any, outputs: Any) -> List[MetricEntry]:
        return []

    def validate_result(self, model: Any, inputs: Any, outputs: Any, validator: ResultValidator) -> None:
        """Add correctness checks; any failed check marks the run FAIL."""

    def get_config(self) -> Dict[str, Any]:
        if isinstance(self._inputs, torch.Tensor):
            return {"input_shape": list(self._inputs.shape)}
        return {}

    # ── precision helpers ─────────────────────────────────────────

    @property
    def dtype(self) -> torch.dtype:
        """Dtype for weights and inputs."""
        return DTYPES.get(self.precision, torch.float32)

    @property
    def effective_precision(self) -> str:
        if self.precision == "mixed":
            return f"mixed_{'bf16' if self.device.type == 'cpu' else 'fp16'}"
        return self.precision

    def autocast(self):
        if self.precision == "mixed":
            dtype = torch.bfloat16 if self.device.type == "cpu" else torch.float16
            return torch.autocast(device_type=self.device.type, dtype=dtype)
        return contextlib.nullcontext()

    @property
    def timing_device(self) -> torch.device:
        """Device whose events/syncs bracket each iteration."""
        return self.device

    # ── run ───────────────────────────────────────────────────────

    def print_device_info(self) -> None:
        print("=" * 50)
        print(f"Device: {self.device}  |  PyTorch {torch.__version__}")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            stack = f"ROCm {torch.version.hip}" if getattr(torch.version, "hip", None) else f"CUDA {torch.version.cuda}"
            print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB, {stack})")
        print(f"Precision: {self.precision}  |  Batch size: {self.batch_size}")
        print("=" * 50)

    def _make_step_fn(self):
        model, inputs = self._model, self._inputs

        def step():
            with torch.inference_mode(), self.autocast():
                return self.run_step(model, inputs)

        return step

    def run(self) -> BenchmarkResult:
        setup_torch_backends(self.precision)
        self.print_device_info()

        print("Loading model...")
        self._model = self.load_model()
        print("Preparing inputs...")
        self._inputs = self.prepare_inputs()

        num_warmup = getattr(self.args, "num_warmup", 5)
        num_runs = getattr(self.args, "num_runs", 20)
        step_fn = self._make_step_fn()
        tdev = self.timing_device

        warmup(step_fn, num_warmup, tdev)
        reset_memory_tracking(self.device)

        power_sampler = None
        if getattr(self.args, "power", False):
            from utils.power import PowerSampler
            power_sampler = PowerSampler(interval_ms=100)
            power_sampler.start()

        latencies_ms = benchmark_loop(step_fn, num_runs, tdev)

        power_stats = power_sampler.stop() if power_sampler else None
        peak_mem = measure_peak_memory(self.device)
        stats = compute_stats(latencies_ms)

        outputs = step_fn()
        synchronize(tdev)

        metrics = latency_metrics(stats, self.batch_size, self.throughput_unit)
        if peak_mem.get("peak_allocated_gb") and self.framework == "pytorch":  # ORT etc. don't use torch's allocator
            metrics.append(MetricEntry("peak_memory_gb", peak_mem["peak_allocated_gb"], "GB", "lower_is_better"))
        if power_stats:
            tput = metrics[0].value
            metrics += [
                MetricEntry("avg_power_watts", power_stats.avg_power_watts, "W", "lower_is_better"),
                MetricEntry("peak_power_watts", power_stats.peak_power_watts, "W", "lower_is_better"),
                MetricEntry("samples_per_joule", tput / power_stats.avg_power_watts, "samples/J", "higher_is_better"),
            ]
        metrics += self.get_extra_metrics(self._model, self._inputs, outputs)

        validator = ResultValidator()
        self.validate_result(self._model, self._inputs, outputs, validator)
        return self.finish(metrics, stats, validator)

    def finish(self, metrics: List[MetricEntry], stats: Dict[str, float],
               validator: Optional[ResultValidator] = None) -> BenchmarkResult:
        """Validate, assemble and emit the result. Custom ``run()``s call this too."""
        validator = validator or ResultValidator()
        for m in metrics:
            validator.expect_finite(m.name, m.value)
        primary = next((m for m in metrics if m.name == self.primary_metric), None)
        validator.expect_greater_than(self.primary_metric, primary.value if primary else 0.0, 0)

        passed, checks = validator.validate()
        failed = [c for c in checks if not c.passed]
        if failed:
            print(f"\nValidation FAILED ({len(failed)} check(s)):")
            for c in failed:
                print(f"  {c.name}: {c.message}")

        result = BenchmarkResult(
            status="PASS" if passed else "FAIL",
            framework=self.framework,
            model=getattr(self.args, "model", None) or "unknown",
            mode=self.mode,
            use_case=self.use_case,
            precision=self.effective_precision,
            batch_size=self.batch_size,
            system_info=build_system_info(self.device),
            metrics=metrics,
            latency_stats=stats,
            primary_metric=self.primary_metric,
            config=self.get_config(),
            validation_checks=validator.to_dicts(),
            error="; ".join(f"{c.name}: {c.message}" for c in failed) or None,
        )
        emit_result(result)
        return result


InferenceHarness = BenchmarkHarness


def latency_metrics(stats: Dict[str, float], batch_size: int, unit: str = "samples/sec") -> List[MetricEntry]:
    mean = stats.get("mean", 0.0)
    return [
        MetricEntry("throughput", batch_size * 1000.0 / mean if mean > 0 else 0.0, unit, "higher_is_better"),
        MetricEntry("latency_p50_ms", stats.get("median", 0.0), "ms", "lower_is_better"),
        MetricEntry("latency_p99_ms", stats.get("p99", 0.0), "ms", "lower_is_better"),
    ]


class TrainingHarness(BenchmarkHarness):
    """Times full optimizer steps (forward + backward + update) on a fixed,
    device-resident synthetic batch, so the number is GPU training throughput,
    not dataloader/augmentation throughput.

    Weights stay fp32; fp16/bf16/mixed run under autocast (AMP), fp16 with a
    GradScaler. A fixed batch must be memorised, so loss has to go down.
    """

    mode = "training"

    def create_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def compute_loss(self, model, inputs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def effective_precision(self) -> str:
        return {"fp16": "amp_fp16", "mixed": "amp_fp16", "bf16": "amp_bf16"}.get(self.precision, self.precision)

    def autocast(self):
        amp = {"fp16": torch.float16, "mixed": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
        if amp is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=amp)

    def _make_step_fn(self):
        model, inputs = self._model, self._inputs
        model.train()
        opt = self.create_optimizer(model)
        use_scaler = self.precision in ("fp16", "mixed") and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        self._losses: List[torch.Tensor] = []

        def step():
            opt.zero_grad(set_to_none=True)
            with self.autocast():
                loss = self.compute_loss(model, inputs)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            self._losses.append(loss.detach())  # no host sync inside the timed loop
            return loss

        return step

    def get_extra_metrics(self, model, inputs, outputs):
        first, last = self._losses[0].item(), self._losses[-1].item()
        self._loss_first, self._loss_last = first, last
        return [
            MetricEntry("loss_first", first, "loss", "lower_is_better"),
            MetricEntry("loss_last", last, "loss", "lower_is_better"),
        ]

    def validate_result(self, model, inputs, outputs, validator):
        validator.expect_greater_than("loss_decrease", self._loss_first - self._loss_last, 0)


class OnnxHarness(BenchmarkHarness):
    """ONNX Runtime: ``session.run`` is synchronous and returns host arrays,
    so latency is host wall-clock including H2D/D2H copies."""

    framework = "onnx"

    def get_onnx_model_path(self) -> str:
        raise NotImplementedError

    def prepare_numpy_inputs(self) -> dict:
        raise NotImplementedError

    @property
    def timing_device(self) -> torch.device:
        return torch.device("cpu")

    def load_model(self):
        import onnxruntime as ort
        path = self.get_onnx_model_path()
        available = ort.get_available_providers()
        provider = getattr(self.args, "execution_provider", None)
        if provider and provider not in available:
            raise RuntimeError(f"{provider} not available (have: {available})")
        providers = [provider] if provider else available
        if provider == "TensorrtExecutionProvider" and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")  # nodes TensorRT can't take go to CUDA, not CPU
        session = ort.InferenceSession(path, providers=providers)
        self._provider = session.get_providers()[0]
        print(f"ONNX Runtime {ort.__version__}, active provider: {self._provider}")
        if provider and self._provider != provider:
            # ORT silently falls back when a provider is listed but its libraries are missing
            raise SkipBenchmark(f"{provider} is listed but not usable here (session fell back to {self._provider})")
        return session

    def prepare_inputs(self):
        return self.prepare_numpy_inputs()

    def run_step(self, session, inputs):
        return session.run(None, inputs)

    def _make_step_fn(self):
        session, inputs = self._model, self._inputs
        return lambda: session.run(None, inputs)

    def get_config(self):
        shapes = {k: list(v.shape) for k, v in (self._inputs or {}).items()}
        return {"execution_provider": getattr(self, "_provider", None), "input_shapes": shapes}


class ComputeHarness(BenchmarkHarness):
    """Micro-benchmarks. ``get_operations`` yields (name, fn, work, unit) where
    ``work`` is GFLOP (unit "GFLOPS") or GB moved (unit "GB/s") per call.

    Each sample times ``reps`` back-to-back calls (calibrated to ~5 ms) so tiny
    kernels measure the kernel, not launch + sync overhead. Yielding lazily lets
    each op's tensors be freed before the next op allocates.
    """

    use_case = "compute"

    def get_operations(self) -> Iterable[Tuple[str, Any, float, str]]:
        raise NotImplementedError

    def run(self) -> BenchmarkResult:
        setup_torch_backends(self.precision)
        self.print_device_info()
        num_runs = getattr(self.args, "num_runs", 20)

        metrics: List[MetricEntry] = []
        best: Dict[str, float] = {}
        for name, fn, work, unit in self.get_operations():
            with torch.inference_mode(), self.autocast():
                warmup(fn, 3, self.device, desc=f"Warmup {name}")
                reps = calibrated_reps(fn, self.device)

                def batch(fn=fn, reps=reps):
                    for _ in range(reps):
                        fn()

                lats = benchmark_loop(batch, num_runs, self.device, desc=f"{name} x{reps}", report_interval=0)
            per_call_ms = compute_stats(lats)["median"] / reps
            perf = work / (per_call_ms / 1000.0)
            print(f"  {name}: {perf:,.1f} {unit} ({per_call_ms * 1000:.1f} us/call)")
            metrics.append(MetricEntry(name, perf, unit, "higher_is_better"))
            best[unit] = max(best.get(unit, 0.0), perf)

        if "GFLOPS" in best:
            metrics.insert(0, MetricEntry("best_gflops", best["GFLOPS"], "GFLOPS", "higher_is_better"))
            self.primary_metric = "best_gflops"
        if "GB/s" in best:
            metrics.insert(0, MetricEntry("best_bandwidth_gbs", best["GB/s"], "GB/s", "higher_is_better"))
            self.primary_metric = "best_bandwidth_gbs"

        validator = ResultValidator()
        self.validate_result(None, None, None, validator)
        return self.finish(metrics, {}, validator)


def run_main(harness_cls, parser: argparse.ArgumentParser) -> None:
    """Script entry point: always emits a result block, FAIL + error on exceptions."""
    args = parser.parse_args()
    harness = None
    try:
        harness = harness_cls(args)
        result = harness.run()
    except SkipBenchmark as e:
        print(f"SKIP: {e}")
        emit_result(BenchmarkResult(status="SKIP", framework=getattr(harness_cls, "framework", "pytorch"),
                                    model=getattr(args, "model", None) or "unknown",
                                    mode=getattr(harness_cls, "mode", "inference"),
                                    use_case=getattr(harness_cls, "use_case", "unknown"),
                                    precision=getattr(args, "precision", "unknown"),
                                    batch_size=getattr(args, "batch_size", 1), error=str(e)))
        sys.exit(0)
    except Exception as e:  # report every failure as a structured FAIL
        traceback.print_exc()
        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        msg = f"OOM: {e}" if oom else f"{type(e).__name__}: {e}"
        emit_error(
            framework=getattr(harness_cls, "framework", "pytorch"),
            model=getattr(args, "model", None) or "unknown",
            error_msg=" ".join(msg.split())[:500],
            mode=getattr(harness_cls, "mode", "inference"),
            use_case=getattr(harness_cls, "use_case", "unknown"),
            precision=getattr(args, "precision", "unknown"),
            batch_size=getattr(args, "batch_size", 1),
        )
        sys.exit(1)
    sys.exit(0 if result.status == "PASS" else 1)

