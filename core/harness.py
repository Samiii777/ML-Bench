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
from utils.shared_device_utils import get_gpu_memory_efficient, collect_system_fingerprint
from core.schema import BenchmarkResult, MetricEntry, SystemInfo
from core.output import emit_result
from core.validation import ResultValidator


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

    def validate_result(self, model: Any, inputs: Any, outputs: Any, validator: ResultValidator) -> None:
        """Add validation checks. Override in subclass to verify correctness."""
        pass

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
        fp = collect_system_fingerprint()
        si = SystemInfo(
            device=str(self.device),
            torch_version=fp.get("torch_version", torch.__version__),
            python_version=fp.get("python_version", ""),
            platform=fp.get("platform", ""),
            kernel_version=fp.get("kernel_version", ""),
            cpu_model=fp.get("cpu_model", ""),
            hostname=fp.get("hostname", ""),
            rocm_version=fp.get("rocm_version", ""),
            cuda_version=fp.get("cuda_version", ""),
            gpu_driver_version=fp.get("gpu_driver_version", ""),
        )
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

        num_gpus = getattr(self.args, "num_gpus", 1)
        if num_gpus > 1 and torch.cuda.device_count() >= num_gpus and isinstance(self._model, torch.nn.Module):
            from core.multigpu import wrap_data_parallel
            device_ids = list(range(num_gpus))
            self._model = wrap_data_parallel(self._model, device_ids)

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

        power_sampler = None
        if getattr(self.args, "power", False):
            from utils.power import PowerSampler
            power_sampler = PowerSampler(interval_ms=100)
            power_sampler.start()

        latencies_ms = benchmark_loop(step_fn, num_runs, self.device)

        power_stats = power_sampler.stop() if power_sampler else None
        peak_mem = measure_peak_memory(self.device)
        stats = compute_stats(latencies_ms)

        outputs = step_fn()
        self.synchronize()
        extra = self.get_extra_metrics(self._model, self._inputs, outputs)

        validator = ResultValidator()
        validator.expect_greater_than("throughput", (batch_size / stats["mean"]) * 1000.0 if stats["mean"] > 0 else 0, 0)
        self.validate_result(self._model, self._inputs, outputs, validator)
        validation_passed, validation_checks = validator.validate()

        mean_ms = stats["mean"]
        per_sample_ms = mean_ms / batch_size
        throughput = (batch_size / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

        metrics = [
            MetricEntry("avg_latency_ms", per_sample_ms, "ms", "lower_is_better"),
            MetricEntry("throughput", throughput, "samples/sec", "higher_is_better"),
        ]
        if peak_mem:
            metrics.append(MetricEntry("peak_memory_gb", peak_mem.get("peak_allocated_gb", 0), "GB", "lower_is_better"))
        if power_stats:
            metrics.append(MetricEntry("avg_power_watts", power_stats.avg_power_watts, "W", "lower_is_better"))
            metrics.append(MetricEntry("peak_power_watts", power_stats.peak_power_watts, "W", "lower_is_better"))
            metrics.append(MetricEntry("energy_joules", power_stats.energy_joules, "J", "lower_is_better"))
            if throughput > 0 and power_stats.avg_power_watts > 0:
                metrics.append(MetricEntry("samples_per_joule", throughput / power_stats.avg_power_watts, "samples/J", "higher_is_better"))
        metrics.extend(extra)

        input_res = ""
        if isinstance(self._inputs, torch.Tensor) and self._inputs.dim() >= 3:
            h, w = self._inputs.shape[-2], self._inputs.shape[-1]
            input_res = f"{h}x{w}"

        failed_checks = [c for c in validation_checks if not c.passed]
        if failed_checks:
            print(f"\nValidation FAILED ({len(failed_checks)} check(s)):")
            for c in failed_checks:
                print(f"  {c.name}: {c.message}")

        result = BenchmarkResult(
            status="PASS" if validation_passed else "FAIL",
            framework=self.framework,
            model=model_name,
            mode=self.mode,
            use_case=self.use_case,
            precision=precision,
            batch_size=batch_size,
            system_info=self._build_system_info(),
            metrics=metrics,
            latency_stats=stats,
            model_architecture=getattr(self, "model_architecture", ""),
            input_resolution=input_res,
            validation_checks=validator.to_dicts(),
            error="; ".join(c.message for c in failed_checks) if failed_checks else None,
        )
        emit_result(result)
        return result


class TrainingHarness(BenchmarkHarness):
    """Base class for training benchmarks."""

    @property
    def mode(self) -> str:
        return "training"

    def create_datasets(self):
        raise NotImplementedError

    def create_optimizer(self, model):
        raise NotImplementedError

    def compute_loss(self, outputs, targets, criterion):
        raise NotImplementedError

    def compute_accuracy(self, outputs, targets) -> float:
        return 0.0

    def run(self) -> BenchmarkResult:
        import time as _time
        from torch.utils.data import DataLoader

        setup_torch_backends(cudnn_benchmark=True)
        self.print_device_info()

        precision = getattr(self.args, "precision", "fp32")
        model_name = getattr(self.args, "model", "unknown")
        batch_size = getattr(self.args, "batch_size", 32)
        num_epochs = getattr(self.args, "num_epochs", 1)
        use_amp = precision == "mixed" and self.device.type == "cuda"

        print("Loading model...")
        model = self.load_model()
        model.to(self.device)
        model.train()

        print("Creating datasets...")
        train_ds, val_ds = self.create_datasets()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=(self.device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=(self.device.type == "cuda"))

        optimizer, scheduler = self.create_optimizer(model)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        reset_memory_tracking(self.device)
        train_losses, val_losses = [], []
        best_val_acc = 0.0
        train_throughput = 0.0
        val_throughput = 0.0

        for epoch in range(num_epochs):
            model.train()
            epoch_loss, train_samples = 0.0, 0
            t0 = _time.perf_counter()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx >= 20:
                    break
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.autocast(device_type="cuda"):
                        outputs = model(inputs)
                        loss = self.compute_loss(outputs, targets, criterion)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = self.compute_loss(outputs, targets, criterion)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                train_samples += inputs.size(0)

            self.synchronize()
            train_time = _time.perf_counter() - t0
            train_throughput = train_samples / train_time if train_time > 0 else 0

            model.eval()
            val_loss_total, val_correct, val_samples = 0.0, 0.0, 0
            t0 = _time.perf_counter()

            with torch.inference_mode():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    if batch_idx >= 10:
                        break
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    outputs = model(inputs)
                    loss = self.compute_loss(outputs, targets, criterion)
                    val_loss_total += loss.item() * inputs.size(0)
                    val_correct += self.compute_accuracy(outputs, targets) * inputs.size(0)
                    val_samples += inputs.size(0)

            self.synchronize()
            val_time = _time.perf_counter() - t0
            val_throughput = val_samples / val_time if val_time > 0 else 0
            avg_val_loss = val_loss_total / max(val_samples, 1)
            val_acc = val_correct / max(val_samples, 1)
            best_val_acc = max(best_val_acc, val_acc)
            train_losses.append(epoch_loss / max(train_samples, 1))
            val_losses.append(avg_val_loss)

            if scheduler:
                scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_losses[-1]:.4f} "
                  f"val_loss={avg_val_loss:.4f} val_acc={val_acc*100:.1f}% "
                  f"train={train_throughput:.1f} samples/sec")

        peak_mem = measure_peak_memory(self.device)
        metrics = [
            MetricEntry("train_throughput", train_throughput, "samples/sec", "higher_is_better"),
            MetricEntry("val_throughput", val_throughput, "samples/sec", "higher_is_better"),
            MetricEntry("final_train_loss", train_losses[-1] if train_losses else 0, "loss", "lower_is_better"),
            MetricEntry("final_val_loss", val_losses[-1] if val_losses else 0, "loss", "lower_is_better"),
            MetricEntry("best_val_accuracy", best_val_acc * 100, "%", "higher_is_better"),
            MetricEntry("throughput", train_throughput, "samples/sec", "higher_is_better"),
        ]
        if peak_mem:
            metrics.append(MetricEntry("peak_memory_gb", peak_mem.get("peak_allocated_gb", 0), "GB", "lower_is_better"))

        result = BenchmarkResult(
            status="PASS", framework=self.framework, model=model_name,
            mode=self.mode, use_case=self.use_case, precision=precision,
            batch_size=batch_size, system_info=self._build_system_info(),
            metrics=metrics, latency_stats={"mean": (1000.0 / train_throughput) if train_throughput > 0 else 0},
        )
        emit_result(result)
        return result


class OnnxHarness(BenchmarkHarness):
    """Base class for ONNX Runtime benchmarks."""

    @property
    def framework(self) -> str:
        return "onnx"

    def get_onnx_model_path(self) -> str:
        raise NotImplementedError

    def export_to_onnx(self) -> str:
        raise NotImplementedError

    def prepare_numpy_inputs(self) -> dict:
        raise NotImplementedError

    def load_model(self):
        import onnxruntime as ort
        path = self.get_onnx_model_path()
        if not Path(path).exists():
            path = self.export_to_onnx()
        provider = getattr(self.args, "execution_provider", None)
        available = ort.get_available_providers()
        if provider and provider in available:
            providers = [provider]
        else:
            providers = available
        print(f"ONNX providers: {providers}")
        return ort.InferenceSession(path, providers=providers)

    def prepare_inputs(self):
        return self.prepare_numpy_inputs()

    def run_step(self, session, inputs):
        return session.run(None, inputs)

    def run(self) -> BenchmarkResult:
        import time as _time
        self.print_device_info()

        precision = getattr(self.args, "precision", "fp32")
        model_name = getattr(self.args, "model", "unknown")
        batch_size = getattr(self.args, "batch_size", 1)
        num_warmup = getattr(self.args, "num_warmup", 5)
        num_runs = getattr(self.args, "num_runs", 20)

        print("Loading ONNX model...")
        session = self.load_model()
        print("Preparing inputs...")
        inputs = self.prepare_numpy_inputs()

        def step():
            return session.run(None, inputs)

        print(f"Warming up ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            step()

        print(f"Benchmarking ({num_runs} iterations)...")
        latencies: List[float] = []
        for _ in range(num_runs):
            t0 = _time.perf_counter()
            step()
            latencies.append((_time.perf_counter() - t0) * 1000.0)

        stats = compute_stats(latencies)
        outputs = step()
        extra = self.get_extra_metrics(session, inputs, outputs)

        mean_ms = stats["mean"]
        per_sample_ms = mean_ms / batch_size
        throughput = (batch_size / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

        metrics = [
            MetricEntry("avg_latency_ms", per_sample_ms, "ms", "lower_is_better"),
            MetricEntry("throughput", throughput, "samples/sec", "higher_is_better"),
        ]
        metrics.extend(extra)

        result = BenchmarkResult(
            status="PASS", framework=self.framework, model=model_name,
            mode=self.mode, use_case=self.use_case, precision=precision,
            batch_size=batch_size, system_info=self._build_system_info(),
            metrics=metrics, latency_stats=stats,
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
