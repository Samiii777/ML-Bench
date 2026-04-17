"""Emit benchmark results as JSON + human-readable output."""

import sys
from core.schema import BenchmarkResult, JSON_START, JSON_END


def emit_result(result: BenchmarkResult) -> None:
    """Print human-readable summary, then emit structured JSON block."""
    _print_summary(result)
    print(f"\n{JSON_START}")
    print(result.to_json())
    print(JSON_END)
    sys.stdout.flush()


def emit_error(framework: str, model: str, error_msg: str,
               mode: str = "inference", use_case: str = "unknown",
               precision: str = "unknown", batch_size: int = 1) -> None:
    result = BenchmarkResult(
        status="FAIL",
        framework=framework,
        model=model,
        mode=mode,
        use_case=use_case,
        precision=precision,
        batch_size=batch_size,
        error=error_msg,
    )
    emit_result(result)


def _print_summary(result: BenchmarkResult) -> None:
    """Print a concise human-readable summary."""
    tag = "PASS" if result.status == "PASS" else "FAIL"
    print(f"\n{'='*50}")
    print(f"[{tag}] {result.framework} | {result.model} | {result.precision} | bs={result.batch_size}")
    print(f"{'='*50}")

    if result.error:
        print(f"Error: {result.error}")
        return

    for m in result.metrics:
        print(f"  {m.name}: {m.value:.4g} {m.unit}")

    if result.latency_stats:
        s = result.latency_stats
        print(f"  latency: mean={s.get('mean',0):.2f} median={s.get('median',0):.2f} "
              f"p95={s.get('p95',0):.2f} ms")

    # Legacy parseable line for backward compat with unmigrated orchestrator consumers
    lat = result.get_metric("throughput")
    if lat is not None:
        print(f"Throughput: {lat:.2f} samples/sec")
    infer = result.get_metric("avg_latency_ms")
    if infer is not None:
        print(f"PyTorch Inference Time = {infer:.2f} ms")
