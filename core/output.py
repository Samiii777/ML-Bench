"""Emit benchmark results: a short human summary, then the JSON block."""

import sys
from core.schema import BenchmarkResult, JSON_START, JSON_END


def emit_result(result: BenchmarkResult) -> None:
    _print_summary(result)
    print(f"\n{JSON_START}")
    print(result.to_json())
    print(JSON_END)
    sys.stdout.flush()


def emit_error(framework: str, model: str, error_msg: str, mode: str = "inference",
               use_case: str = "unknown", precision: str = "unknown", batch_size: int = 1) -> None:
    emit_result(BenchmarkResult(status="FAIL", framework=framework, model=model, mode=mode,
                                use_case=use_case, precision=precision, batch_size=batch_size,
                                error=error_msg))


def _print_summary(result: BenchmarkResult) -> None:
    print(f"\n{'=' * 50}")
    print(f"[{result.status}] {result.framework} | {result.model} | {result.precision} | bs={result.batch_size}")
    print("=" * 50)
    if result.error:
        print(f"Error: {result.error}")
    for m in result.metrics:
        print(f"  {m.name}: {m.value:,.1f} {m.unit}" if abs(m.value) >= 100 else f"  {m.name}: {m.value:.4g} {m.unit}")
    s = result.latency_stats
    if s.get("n"):
        print(f"  latency/iter: mean={s['mean']:.2f} p50={s['median']:.2f} p99={s['p99']:.2f} ms "
              f"(cv={s['cv_pct']:.1f}%, n={s['n']})")
