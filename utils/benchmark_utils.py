"""Shared timing, memory and statistics helpers used by every benchmark."""

import gc
import time
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class BenchmarkTimer:
    """Times one region. On CUDA/ROCm uses device events (no host jitter);
    elsewhere uses perf_counter around device syncs.

    ``stop()`` always synchronizes, so back-to-back start/stop pairs measure
    each iteration in isolation (latency, not pipelined throughput).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.use_events = device.type == "cuda"
        if self.use_events:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        if self.use_events:
            self._start.record()
        else:
            synchronize(self.device)
            self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Return elapsed milliseconds since ``start()``."""
        if self.use_events:
            self._end.record()
            self._end.synchronize()
            return self._start.elapsed_time(self._end)
        synchronize(self.device)
        return (time.perf_counter() - self._t0) * 1000.0


@contextmanager
def gc_disabled():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def reset_memory_tracking(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def measure_peak_memory(device: torch.device) -> Dict[str, float]:
    """Peak PyTorch-allocator memory since the last ``reset_memory_tracking``."""
    if device.type != "cuda":
        return {}
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def compute_stats(values_ms: List[float]) -> Dict[str, float]:
    """mean, std (sample), min, max, median, p90/p95/p99, cv_pct, n."""
    arr = np.asarray(values_ms, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "cv_pct": (std / mean * 100.0) if mean > 0 else 0.0,
        "n": n,
    }


def warmup(fn, num_warmup: int, device: torch.device, desc: str = "Warming up") -> None:
    if num_warmup > 0:
        print(f"{desc} ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        fn()
    synchronize(device)


def benchmark_loop(fn, num_runs: int, device: torch.device, desc: str = "Benchmarking",
                   report_interval: int = 20) -> List[float]:
    """Run ``fn`` ``num_runs`` times; return per-iteration latencies in ms."""
    timer = BenchmarkTimer(device)
    latencies: List[float] = []
    print(f"{desc} ({num_runs} iterations)...")
    with gc_disabled():
        for i in range(num_runs):
            timer.start()
            fn()
            latencies.append(timer.stop())
            if report_interval and (i + 1) % report_interval == 0:
                print(f"  Completed {i + 1}/{num_runs}")
    return latencies


def calibrated_reps(fn, device: torch.device, target_ms: float = 5.0, max_reps: int = 1000) -> int:
    """How many back-to-back calls of ``fn`` fill ~target_ms of device time.

    Timing a single 10 µs kernel with a sync per sample measures launch + sync
    overhead, not the kernel. Timing ``reps`` queued calls per sample amortises it.
    """
    timer = BenchmarkTimer(device)
    timer.start()
    fn()
    one_ms = max(timer.stop(), 1e-3)
    return int(min(max_reps, max(1, target_ms // one_ms)))


def setup_torch_backends(precision: str = "fp32", cudnn_benchmark: bool = True) -> None:
    """Make the math mode match the precision label.

    "fp32" means IEEE fp32: TF32 is disabled for both matmul and cuDNN conv
    (PyTorch enables cuDNN TF32 by default, which would silently make "fp32"
    conv numbers TF32 numbers on Ampere+). "tf32" opts in explicitly.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        tf32 = precision == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
