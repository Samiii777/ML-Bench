#!/usr/bin/env python3
"""GPU micro-benchmarks: GEMM / conv (GFLOPS) and memory / elementwise / reduction (GB/s).

--model picks the group: gemm_ops, conv_ops, memory_ops, elementwise_ops, reduction_ops.
Work per call is counted as useful FLOPs (2 per multiply-add) or bytes that must
cross DRAM at least once (reads + writes), so numbers compare against datasheet peaks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())))

import torch
import torch.nn.functional as F

from core.args import build_base_parser
from core.harness import ComputeHarness, SkipBenchmark, run_main

GB = 1e9

# (M, K, N): square sizes up to compute-bound, plus LLM-decode-like skinny shapes
GEMM_SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
               (1, 4096, 4096), (16, 4096, 4096), (128, 4096, 4096)]

# (N, Cin, H, W, Cout, k, stride, pad)
CONV_SHAPES = {
    "resnet_stem": (32, 3, 224, 224, 64, 7, 2, 3),
    "resnet_3x3_56": (32, 64, 56, 56, 64, 3, 1, 1),
    "resnet_3x3_14": (32, 256, 14, 14, 256, 3, 1, 1),
    "resnet_1x1_7": (32, 512, 7, 7, 2048, 1, 1, 0),
}

# 256M elements = 0.5-1 GB per tensor: far larger than any GPU last-level cache (4090 L2 72 MB,
# MI300X Infinity Cache 256 MB), so GB/s is DRAM bandwidth, not cache bandwidth.
SIZES = {"256M": (16384, 16384)}


def gemm_flops(m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n


def conv_flops(n, cin, h, w, cout, k, stride, pad) -> float:
    ho, wo = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    return 2.0 * n * ho * wo * cout * cin * k * k


def gemm_rel_error(dtype: torch.dtype, device) -> float:
    """Frobenius relative error of a 1024^3 matmul vs an fp64 reference."""
    g = torch.Generator(device="cpu").manual_seed(0)
    a, b = torch.randn(1024, 1024, generator=g), torch.randn(1024, 1024, generator=g)
    ref = a.double() @ b.double()
    out = (a.to(device, dtype) @ b.to(device, dtype)).double().cpu()
    return ((out - ref).norm() / ref.norm()).item()


class GpuOps(ComputeHarness):

    def get_operations(self):
        if self.precision == "mixed":
            raise SkipBenchmark("'mixed' would time fp32->fp16 casts; use fp16 or bf16 for kernel benchmarks")
        group = self.args.model
        dt, dev = self.dtype, self.device
        if group == "gemm_ops":
            for m, k, n in GEMM_SHAPES:
                a, b = torch.randn(m, k, dtype=dt, device=dev), torch.randn(k, n, dtype=dt, device=dev)
                yield f"gemm_{m}x{k}x{n}", (lambda a=a, b=b: a @ b), gemm_flops(m, k, n) / GB, "GFLOPS"
        elif group == "conv_ops":
            for name, (n, cin, h, w, cout, k, s, p) in CONV_SHAPES.items():
                x = torch.randn(n, cin, h, w, dtype=dt, device=dev)
                wt = torch.randn(cout, cin, k, k, dtype=dt, device=dev)
                yield (f"conv_{name}", (lambda x=x, wt=wt, s=s, p=p: F.conv2d(x, wt, stride=s, padding=p)),
                       conv_flops(n, cin, h, w, cout, k, s, p) / GB, "GFLOPS")
        elif group == "memory_ops":
            for size_name, shape in SIZES.items():
                x = torch.randn(*shape, dtype=dt, device=dev)
                nbytes = x.numel() * x.element_size()
                yield f"copy_{size_name}", (lambda x=x: x.clone()), 2 * nbytes / GB, "GB/s"
                yield f"transpose_{size_name}", (lambda x=x: x.t().contiguous()), 2 * nbytes / GB, "GB/s"
                yield f"fill_{size_name}", (lambda x=x: x.fill_(1.0)), nbytes / GB, "GB/s"
        elif group == "elementwise_ops":
            for size_name, shape in SIZES.items():
                x, y = torch.randn(*shape, dtype=dt, device=dev), torch.randn(*shape, dtype=dt, device=dev)
                pos = x.abs()
                nbytes = x.numel() * x.element_size()
                binary = {"add": lambda: x + y, "mul": lambda: x * y}
                unary = {"relu": lambda: F.relu(x), "gelu": lambda: F.gelu(x), "sigmoid": lambda: torch.sigmoid(x),
                         "exp": lambda: torch.exp(x), "sqrt": lambda: torch.sqrt(pos)}
                for name, fn in binary.items():
                    yield f"{name}_{size_name}", fn, 3 * nbytes / GB, "GB/s"  # 2 reads + 1 write
                for name, fn in unary.items():
                    yield f"{name}_{size_name}", fn, 2 * nbytes / GB, "GB/s"  # 1 read + 1 write
        elif group == "reduction_ops":
            for size_name, shape in SIZES.items():
                x = torch.randn(*shape, dtype=dt, device=dev)
                nbytes = x.numel() * x.element_size()
                reads = {"sum": lambda: x.sum(), "max": lambda: x.amax(),
                         "sum_dim0": lambda: x.sum(dim=0), "sum_dim1": lambda: x.sum(dim=1)}
                read_write = {"softmax": lambda: F.softmax(x, dim=-1),
                              "layer_norm": lambda: F.layer_norm(x, x.shape[-1:])}
                for name, fn in reads.items():
                    yield f"{name}_{size_name}", fn, nbytes / GB, "GB/s"
                for name, fn in read_write.items():
                    yield f"{name}_{size_name}", fn, 2 * nbytes / GB, "GB/s"
        else:
            raise ValueError(f"Unknown op group {group!r}")

    def validate_result(self, model, inputs, outputs, validator):
        if self.args.model == "gemm_ops":
            err = gemm_rel_error(self.dtype, self.device) if self.precision != "mixed" else None
            if err is not None:
                bound = 1e-5 if self.precision == "fp32" else 3e-2
                print(f"GEMM relative error vs fp64: {err:.2e} (bound {bound:g})")
                validator.expect_in_range("gemm_rel_error", err, 0.0, bound)


if __name__ == "__main__":
    parser = build_base_parser("GPU compute micro-benchmarks")
    parser.set_defaults(model="gemm_ops", num_runs=10)
    run_main(GpuOps, parser)
