"""
Shared benchmarking utilities for accurate timing, memory measurement, and statistics.

Usage:
    from utils.benchmark_utils import BenchmarkTimer, measure_peak_memory, compute_stats

    timer = BenchmarkTimer(device)
    timer.sync()
    timer.start()
    # ... run inference ...
    elapsed_ms = timer.stop()

    stats = compute_stats(latencies_ms)
    # stats = {'mean': ..., 'std': ..., 'min': ..., 'max': ..., 'median': ..., 'p90': ..., 'p95': ..., 'p99': ...}
"""

import gc
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch


class BenchmarkTimer:
    """GPU-aware timer using CUDA Events when available, with wall-clock fallback.
    
    CUDA events measure elapsed time on the GPU stream directly, avoiding
    host-side jitter from OS scheduling, Python GIL, etc.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda_events = device.type == "cuda"
        
        if self.use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._start_time: float = 0.0
            self._end_time: float = 0.0

    def sync(self):
        """Synchronize the device (call before and after timed regions)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def start(self):
        """Record the start of a timed region."""
        if self.use_cuda_events:
            self._start_event.record()
        else:
            self.sync()
            self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Record the end of a timed region and return elapsed time in milliseconds."""
        if self.use_cuda_events:
            self._end_event.record()
            torch.cuda.synchronize(self.device)
            return self._start_event.elapsed_time(self._end_event)
        else:
            self.sync()
            self._end_time = time.perf_counter()
            return (self._end_time - self._start_time) * 1000.0


@contextmanager
def gc_disabled():
    """Context manager that disables garbage collection for the duration."""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


def measure_peak_memory(device: torch.device) -> Dict[str, float]:
    """Measure peak GPU memory using PyTorch allocator stats.
    
    Call torch.cuda.reset_peak_memory_stats() before the timed region,
    then call this after to get clean benchmark-only peak.
    
    Returns dict with keys: peak_allocated_gb, peak_reserved_gb, current_allocated_gb
    """
    if device.type != "cuda":
        return {}
    
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "current_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
    }


def reset_memory_tracking(device: torch.device):
    """Reset peak memory stats so subsequent measurement is clean."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def compute_stats(values_ms: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics from a list of latency measurements (in ms).
    
    Returns dict with: mean, std (sample), min, max, median, p90, p95, p99
    """
    arr = np.array(values_ms, dtype=np.float64)
    n = len(arr)
    
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n": n,
    }


def warmup(fn, num_warmup: int, device: torch.device, desc: str = "Warming up"):
    """Run warmup iterations with proper synchronization.
    
    Args:
        fn: callable that runs one iteration (no return value needed)
        num_warmup: number of warmup iterations
        device: torch device for synchronization
    """
    print(f"{desc} ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        fn()
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def benchmark_loop(
    fn,
    num_runs: int,
    device: torch.device,
    desc: str = "Benchmarking",
    report_interval: int = 20,
) -> List[float]:
    """Run a benchmark loop with proper timing and return per-iteration latencies (ms).
    
    Args:
        fn: callable that runs one iteration
        num_runs: number of timed iterations
        device: torch device for timer
        desc: description for progress reporting
        report_interval: how often to print progress
    
    Returns:
        List of latencies in milliseconds
    """
    timer = BenchmarkTimer(device)
    latencies: List[float] = []
    
    print(f"{desc} ({num_runs} iterations)...")
    
    with gc_disabled():
        for i in range(num_runs):
            timer.start()
            fn()
            elapsed_ms = timer.stop()
            latencies.append(elapsed_ms)
            
            if report_interval and (i + 1) % report_interval == 0:
                print(f"  Completed {i + 1}/{num_runs}")
    
    return latencies


def setup_torch_backends(compile_model: bool = False, cudnn_benchmark: bool = True):
    """Configure PyTorch backends for optimal benchmark performance.
    
    Args:
        compile_model: If True, callers should wrap their model with torch.compile()
        cudnn_benchmark: Enable cuDNN autotuning for fixed-size inputs (default True)
    """
    if cudnn_benchmark and torch.cuda.is_available():
        # Maps to MIOpen autotune on ROCm, cuDNN autotune on NVIDIA — safe on both.
        torch.backends.cudnn.benchmark = True
    
    # TF32 is NVIDIA Ampere+ only. On ROCm/AMD the flag is either absent or a
    # silent no-op; gate it to NVIDIA to avoid misleading the user about what's
    # actually active.
    is_nvidia = (
        torch.cuda.is_available()
        and getattr(torch.version, "hip", None) is None
    )
    if is_nvidia:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass


def maybe_compile(model: torch.nn.Module, compile_enabled: bool = False):
    """Optionally wrap a model with torch.compile().
    
    Returns the (possibly compiled) model. Safe to call even when compile
    is not supported — falls back gracefully.
    """
    if not compile_enabled:
        return model
    
    try:
        compiled = torch.compile(model)
        print("Model compiled with torch.compile()")
        return compiled
    except Exception as e:
        print(f"torch.compile() not available or failed: {e}, using eager mode")
        return model


def print_benchmark_stats(
    stats: Dict[str, float],
    batch_size: int = 1,
    prefix: str = "",
):
    """Print benchmark statistics in a standard format parseable by the framework."""
    mean_ms = stats["mean"]
    per_sample_ms = mean_ms / batch_size
    throughput = (batch_size / mean_ms) * 1000.0 if mean_ms > 0 else 0.0
    
    if prefix:
        print(f"\n{prefix}")
    print(f"Average Inference Time: {mean_ms:.2f} ms")
    print(f"Per-sample Latency: {per_sample_ms:.2f} ms/sample")
    print(f"Median Inference Time: {stats['median']:.2f} ms")
    print(f"P90 Inference Time: {stats['p90']:.2f} ms")
    print(f"P95 Inference Time: {stats['p95']:.2f} ms")
    print(f"P99 Inference Time: {stats['p99']:.2f} ms")
    print(f"Min Inference Time: {stats['min']:.2f} ms")
    print(f"Max Inference Time: {stats['max']:.2f} ms")
    print(f"Std Inference Time: {stats['std']:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Number of runs: {stats['n']}")
