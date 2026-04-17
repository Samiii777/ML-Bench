#!/usr/bin/env python3
"""GEMM (General Matrix Multiply) Operations Benchmark for PyTorch"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch

from core.harness import ComputeHarness
from core.schema import BenchmarkMeta
from core.args import build_base_parser

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="gpu_ops",
    supported_models=["gemm_ops"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="compute",
)


class GEMMBenchmark(ComputeHarness):

    @property
    def use_case(self):
        return "compute"

    def get_operations(self):
        precision = getattr(self.args, "precision", "fp32")
        dtype = torch.float32 if precision in ("fp32", "mixed") else torch.float16

        gemm_sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (1, 4096, 4096),
            (32, 2048, 2048),
            (128, 1024, 1024),
        ]

        ops = []
        for m, k, n in gemm_sizes:
            A = torch.randn(m, k, dtype=dtype, device=self.device)
            B = torch.randn(k, n, dtype=dtype, device=self.device)

            if precision == "mixed" and self.device.type == "cuda":
                def gemm_op(A=A, B=B):
                    with torch.cuda.amp.autocast():
                        return torch.mm(A, B)
            else:
                def gemm_op(A=A, B=B):
                    return torch.mm(A, B)

            # 2 FLOPs per multiply-add, result in GFLOPS
            flops = 2.0 * m * k * n
            gflops = flops / 1e9

            ops.append((f"gemm_{m}x{k}x{n}", gemm_op, gflops, "GFLOPS"))

        return ops


if __name__ == "__main__":
    parser = build_base_parser("GEMM Operations Benchmark")
    parser.set_defaults(model="gemm_ops")
    args = parser.parse_args()

    try:
        benchmark = GEMMBenchmark(args)
        benchmark.run()
        print("GEMM Operations Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
