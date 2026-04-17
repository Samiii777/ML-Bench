"""BenchmarkHarness — base class that eliminates duplicated device/timing/memory code."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure project root is on path so utils/ is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.benchmark_utils import (
    BenchmarkTimer, benchmark_loop, compute_stats, gc_disabled,
    measure_peak_memory, reset_memory_tracking, setup_torch_backends, warmup,
)
from utils.shared_device_utils import get_gpu_memory_efficient
from core.schema import BenchmarkResult, MetricEntry, SystemInfo
from core.output import emit_result


class BenchmarkHarness:
    """Base class for all benchmark scripts.

    Subclasses override ``load_model``, ``prepare_inputs``, and ``run_step``.
    The ``run()`` method handles device setup, warmup, timing, memory
    measurement, statistics, and structured output.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self._resolve_device()
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

    # ── provided by base ──────────────────────────────────────────

    @staticmethod
    def _resolve_device_from_str(device_str: str = "auto") -> torch.device:
        if device_str != "auto":
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_device(self) -> torch.device:
        return self._resolve_device_from_str(getattr(self.args, "device", "auto"))

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def get_gpu_memory(self) -> Optional[Dict[str, Any]]:
        return get_gpu_memory_efficient()

    def print_device_info(self) -> None:
        print("=" * 50)
        print("DEVICE INFORMATION")
        print("=" * 50)
        print(f"Selected device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: True")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA available: False")
        print("=" * 50)

    def _build_system_info(self) -> SystemInfo:
        si = SystemInfo(device=str(self.device), torch_version=torch.__version__)
        if torch.cuda.is_available():
            si.device_name = torch.cuda.get_device_name(0)
            si.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            si.gpu_vendor = "amd" if getattr(torch.version, "hip", None) else "nvidia"
        elif self.device.type == "mps":
            si.device_name = "Apple MPS"
            si.gpu_vendor = "apple"
        else:
            si.device_name = "CPU"
            si.gpu_vendor = "none"
        return si

    # ── framework / use_case metadata (override in subclass or set BENCHMARK_META) ──

    @property
    def framework(self) -> str:
        return "pytorch"

    @property
    def mode(self) -> str:
        return "inference"

    @property
    def use_case(self) -> str:
        return "classification"

    # ── main entry point ──────────────────────────────────────────

    def run(self) -> BenchmarkResult:
        setup_torch_backends(cudnn_benchmark=True)
        self.print_device_info()
        print(f"Using device: {self.device}")

        print("Loading model...")
        self._model = self.load_model()

        print("Preparing inputs...")
        self._inputs = self.prepare_inputs()

        num_warmup = getattr(self.args, "num_warmup", 5)
        num_runs = getattr(self.args, "num_runs", 20)
        batch_size = getattr(self.args, "batch_size", 1)
        precision = getattr(self.args, "precision", "fp32")
        model_name = getattr(self.args, "model", "unknown")

        step_fn = self._make_step_fn()

        warmup(step_fn, num_warmup, self.device)
        reset_memory_tracking(self.device)

        latencies_ms = benchmark_loop(step_fn, num_runs, self.device)

        peak_mem = measure_peak_memory(self.device)
        stats = compute_stats(latencies_ms)

        outputs = step_fn()
        self.synchronize()
        extra = self.get_extra_metrics(self._model, self._inputs, outputs)

        mean_ms = stats["mean"]
        per_sample_ms = mean_ms / batch_size
        throughput = (batch_size / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

        metrics = [
            MetricEntry("avg_latency_ms", per_sample_ms, "ms", "lower_is_better"),
            MetricEntry("throughput", throughput, "samples/sec", "higher_is_better"),
        ]
        if peak_mem:
            metrics.append(MetricEntry("peak_memory_gb", peak_mem.get("peak_allocated_gb", 0), "GB", "lower_is_better"))
        metrics.extend(extra)

        result = BenchmarkResult(
            status="PASS",
            framework=self.framework,
            model=model_name,
            mode=self.mode,
            use_case=self.use_case,
            precision=precision,
            batch_size=batch_size,
            system_info=self._build_system_info(),
            metrics=metrics,
            latency_stats=stats,
        )
        emit_result(result)
        return result

    def _make_step_fn(self):
        model = self._model
        inputs = self._inputs
        harness = self

        def step():
            return harness.run_step(model, inputs)

        return step


class InferenceHarness(BenchmarkHarness):
    """Wraps run_step in torch.inference_mode + optional autocast."""

    @property
    def mode(self) -> str:
        return "inference"

    def _make_step_fn(self):
        model = self._model
        inputs = self._inputs
        harness = self
        precision = getattr(self.args, "precision", "fp32")
        use_autocast = precision == "mixed" and self.device.type == "cuda"

        def step():
            with torch.inference_mode():
                if use_autocast:
                    with torch.autocast(device_type="cuda"):
                        return harness.run_step(model, inputs)
                return harness.run_step(model, inputs)

        return step


class ComputeHarness(BenchmarkHarness):
    """For GPU ops benchmarks that run multiple operation sizes and report GFLOPS/GB/s.

    Subclasses override ``get_operations()`` to return a list of
    (name, fn, flops_or_bytes, metric_unit) tuples.
    """

    @property
    def use_case(self) -> str:
        return "compute"

    def get_operations(self) -> list:
        raise NotImplementedError

    def load_model(self):
        return None

    def prepare_inputs(self):
        return None

    def run_step(self, model, inputs):
        return None

    def run(self) -> BenchmarkResult:
        setup_torch_backends(cudnn_benchmark=True)
        self.print_device_info()

        precision = getattr(self.args, "precision", "fp32")
        model_name = getattr(self.args, "model", "unknown")

        ops = self.get_operations()
        metrics: List[MetricEntry] = []
        all_latencies: List[float] = []

        for name, fn, theoretical, unit in ops:
            warmup(fn, 3, self.device, desc=f"Warmup {name}")
            lats = benchmark_loop(fn, 10, self.device, desc=f"Bench {name}", report_interval=0)
            stats = compute_stats(lats)
            all_latencies.extend(lats)

            perf = theoretical / (stats["median"] / 1000.0) if stats["median"] > 0 else 0.0
            metrics.append(MetricEntry(f"{name}_perf", perf, unit, "higher_is_better"))
            metrics.append(MetricEntry(f"{name}_median_ms", stats["median"], "ms", "lower_is_better"))

        overall = compute_stats(all_latencies) if all_latencies else {}

        result = BenchmarkResult(
            status="PASS",
            framework=self.framework,
            model=model_name,
            mode=self.mode,
            use_case=self.use_case,
            precision=precision,
            batch_size=1,
            system_info=self._build_system_info(),
            metrics=metrics,
            latency_stats=overall,
        )
        emit_result(result)
        return result
