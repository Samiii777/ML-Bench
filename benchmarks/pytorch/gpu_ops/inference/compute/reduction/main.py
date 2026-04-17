#!/usr/bin/env python3
"""Reduction Operations Benchmark for PyTorch
Tests sum, mean, max, min, softmax, layer_norm across dimensions."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch
import torch.nn.functional as F

from core.harness import ComputeHarness
from core.schema import BenchmarkMeta
from core.args import build_base_parser

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="gpu_ops",
    supported_models=["reduction_ops"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="compute",
)


class ReductionBenchmark(ComputeHarness):

    @property
    def use_case(self):
        return "compute"

    def get_operations(self):
        precision = getattr(self.args, "precision", "fp32")
        dtype = torch.float32 if precision in ("fp32", "mixed") else torch.float16

        sizes = [
            ((1024, 1024), "small"),
            ((4096, 4096), "large"),
            ((8192, 8192), "vlarge"),
            ((16384, 16384), "huge"),
            ((23170, 23170), "massive"),
            ((2048, 8192), "rect"),
            ((512, 16384), "wide"),
        ]

        reduction_ops = [
            ("sum",          lambda x: torch.sum(x)),
            ("mean",         lambda x: torch.mean(x)),
            ("max",          lambda x: torch.max(x)),
            ("min",          lambda x: torch.min(x)),
            ("sum_dim0",     lambda x: torch.sum(x, dim=0)),
            ("sum_dim1",     lambda x: torch.sum(x, dim=1)),
            ("mean_dim0",    lambda x: torch.mean(x, dim=0)),
            ("mean_dim1",    lambda x: torch.mean(x, dim=1)),
            ("softmax_dim0", lambda x: F.softmax(x, dim=0)),
            ("softmax_dim1", lambda x: F.softmax(x, dim=1)),
            ("layer_norm",   lambda x: F.layer_norm(x, x.shape[-1:])),
        ]

        ops = []
        for size, size_name in sizes:
            x = torch.randn(*size, dtype=dtype, device=self.device)
            input_bytes = x.numel() * x.element_size()
            gb = input_bytes / 1e9  # bandwidth: read input

            for op_name, op_func in reduction_ops:
                if precision == "mixed" and self.device.type == "cuda":
                    def fn(op_func=op_func, x=x):
                        with torch.cuda.amp.autocast():
                            return op_func(x)
                else:
                    def fn(op_func=op_func, x=x):
                        return op_func(x)

                ops.append((f"{op_name}_{size_name}", fn, gb, "GB/s"))

        return ops


if __name__ == "__main__":
    parser = build_base_parser("Reduction Operations Benchmark")
    parser.set_defaults(model="reduction_ops")
    args = parser.parse_args()

    try:
        benchmark = ReductionBenchmark(args)
        benchmark.run()
        print("Reduction Operations Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
