#!/usr/bin/env python3
"""Memory Operations Benchmark for PyTorch
Tests memory bandwidth: copy, transpose, fill, add."""

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
    supported_models=["memory_ops"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="compute",
)


class MemoryBenchmark(ComputeHarness):

    @property
    def use_case(self):
        return "compute"

    def get_operations(self):
        precision = getattr(self.args, "precision", "fp32")
        dtype = torch.float32 if precision in ("fp32", "mixed") else torch.float16

        memory_configs = [
            ((1024, 1024), "small"),
            ((2048, 2048), "medium"),
            ((4096, 4096), "large"),
            ((8192, 8192), "vlarge"),
            ((16384, 16384), "huge"),
            ((23170, 23170), "massive"),
            ((1024, 8192), "rect"),
            ((8192, 1024), "tall"),
        ]

        ops = []
        for size, name in memory_configs:
            src = torch.randn(*size, dtype=dtype, device=self.device)
            elem_size = src.element_size()
            numel = src.numel()

            # --- Copy (read + write) ---
            bytes_rw = numel * elem_size * 2  # read src + write dst
            gb_rw = bytes_rw / 1e9

            def copy_op(src=src):
                return src.clone()

            ops.append((f"copy_{name}", copy_op, gb_rw, "GB/s"))

            # --- Contiguous transpose (read + write) ---
            def transpose_op(src=src):
                return src.transpose(0, 1).contiguous()

            ops.append((f"transpose_{name}", transpose_op, gb_rw, "GB/s"))

            # --- Fill (write only) ---
            bytes_w = numel * elem_size
            gb_w = bytes_w / 1e9

            def fill_op(src=src):
                dst = torch.empty_like(src)
                dst.fill_(1.0)
                return dst

            ops.append((f"fill_{name}", fill_op, gb_w, "GB/s"))

            # --- Add (read + write) ---
            def add_op(src=src):
                return src + 1.0

            ops.append((f"add_{name}", add_op, gb_rw, "GB/s"))

        return ops


if __name__ == "__main__":
    parser = build_base_parser("Memory Operations Benchmark")
    parser.set_defaults(model="memory_ops")
    args = parser.parse_args()

    try:
        benchmark = MemoryBenchmark(args)
        benchmark.run()
        print("Memory Operations Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
