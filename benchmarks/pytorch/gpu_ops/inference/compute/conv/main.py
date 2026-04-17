#!/usr/bin/env python3
"""Convolution Operations Benchmark for PyTorch"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch
import torch.nn as nn

from core.harness import ComputeHarness
from core.schema import BenchmarkMeta
from core.args import build_base_parser

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="gpu_ops",
    supported_models=["conv_ops"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="compute",
)


class ConvBenchmark(ComputeHarness):

    @property
    def use_case(self):
        return "compute"

    def get_operations(self):
        precision = getattr(self.args, "precision", "fp32")
        dtype = torch.float32 if precision in ("fp32", "mixed") else torch.float16

        # (batch, in_channels, height, width, out_channels, kernel_size, stride, padding, name)
        conv_configs = [
            (1, 3, 224, 224, 64, 3, 1, 1, "resnet_first"),
            (1, 64, 56, 56, 128, 3, 1, 1, "resnet_mid"),
            (1, 256, 14, 14, 512, 3, 1, 1, "resnet_deep"),
            (32, 64, 32, 32, 128, 3, 1, 1, "batch_processing"),
            (1, 512, 7, 7, 2048, 1, 1, 0, "conv_1x1"),
            (1, 128, 28, 28, 128, 5, 1, 2, "conv_5x5"),
            (16, 256, 16, 16, 256, 3, 1, 1, "medium_batch"),
        ]

        ops = []
        for batch, in_ch, h, w, out_ch, k, stride, padding, name in conv_configs:
            x = torch.randn(batch, in_ch, h, w, dtype=dtype, device=self.device)
            conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding,
                             dtype=dtype, device=self.device)

            if precision == "mixed" and self.device.type == "cuda":
                def conv_op(x=x, conv=conv):
                    with torch.cuda.amp.autocast():
                        return conv(x)
            else:
                def conv_op(x=x, conv=conv):
                    return conv(x)

            # FLOPS for convolution: batch * out_h * out_w * out_ch * in_ch * k * k * 2
            output_h = (h + 2 * padding - k) // stride + 1
            output_w = (w + 2 * padding - k) // stride + 1
            flops = batch * output_h * output_w * out_ch * in_ch * k * k * 2
            gflops = flops / 1e9

            ops.append((f"conv_{name}", conv_op, gflops, "GFLOPS"))

        return ops


if __name__ == "__main__":
    parser = build_base_parser("Convolution Operations Benchmark")
    parser.set_defaults(model="conv_ops")
    args = parser.parse_args()

    try:
        benchmark = ConvBenchmark(args)
        benchmark.run()
        print("Convolution Operations Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
