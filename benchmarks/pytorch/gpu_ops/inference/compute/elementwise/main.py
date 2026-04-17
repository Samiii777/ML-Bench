#!/usr/bin/env python3
"""Element-wise Operations Benchmark for PyTorch
Tests add, multiply, ReLU, GELU, sigmoid, tanh, exp, sqrt, sin, cos."""

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
    supported_models=["elementwise_ops"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="compute",
)


class ElementwiseBenchmark(ComputeHarness):

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
            ((1024, 8192), "rect"),
        ]

        # (name, lambda, op_type) -- op_type: "binary" reads 2 writes 1, "unary" reads 1 writes 1
        element_ops = [
            ("add",     lambda x, y: x + y,                              "binary"),
            ("multiply",lambda x, y: x * y,                              "binary"),
            ("relu",    lambda x, y: F.relu(x),                          "unary"),
            ("gelu",    lambda x, y: F.gelu(x),                          "unary"),
            ("sigmoid", lambda x, y: torch.sigmoid(x),                   "unary"),
            ("tanh",    lambda x, y: torch.tanh(x),                      "unary"),
            ("exp",     lambda x, y: torch.exp(x),                       "unary"),
            ("sqrt",    lambda x, y: torch.sqrt(torch.abs(x) + 1e-8),    "unary"),
            ("sin",     lambda x, y: torch.sin(x),                       "unary"),
            ("cos",     lambda x, y: torch.cos(x),                       "unary"),
        ]

        ops = []
        for size, size_name in sizes:
            x = torch.randn(*size, dtype=dtype, device=self.device)
            y = torch.randn(*size, dtype=dtype, device=self.device)
            elem_size = x.element_size()
            numel = x.numel()

            for op_name, op_func, op_type in element_ops:
                if op_type == "binary":
                    total_bytes = 3 * numel * elem_size  # read 2, write 1
                else:
                    total_bytes = 2 * numel * elem_size  # read 1, write 1

                gb = total_bytes / 1e9

                if precision == "mixed" and self.device.type == "cuda":
                    def fn(op_func=op_func, x=x, y=y):
                        with torch.cuda.amp.autocast():
                            return op_func(x, y)
                else:
                    def fn(op_func=op_func, x=x, y=y):
                        return op_func(x, y)

                ops.append((f"{op_name}_{size_name}", fn, gb, "GB/s"))

        return ops


if __name__ == "__main__":
    parser = build_base_parser("Element-wise Operations Benchmark")
    parser.set_defaults(model="elementwise_ops")
    args = parser.parse_args()

    try:
        benchmark = ElementwiseBenchmark(args)
        benchmark.run()
        print("Element-wise Operations Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
