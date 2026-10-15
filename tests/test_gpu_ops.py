"""FLOP / byte accounting behind the GFLOPS and GB/s numbers."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "gpu_ops", Path(__file__).resolve().parents[1] / "benchmarks/pytorch/gpu_ops/inference/compute/main.py")
gpu_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_ops)


def test_gemm_flops_counts_multiply_and_add():
    assert gpu_ops.gemm_flops(2, 3, 4) == 2 * 2 * 3 * 4


def test_conv_flops_matches_brute_force_mac_count():
    n, cin, h, w, cout, k, s, p = 1, 3, 8, 8, 5, 3, 2, 1
    x, wt = torch.randn(n, cin, h, w), torch.randn(cout, cin, k, k)
    out = torch.nn.functional.conv2d(x, wt, stride=s, padding=p)
    macs = out.numel() * cin * k * k  # every output element is a cin*k*k dot product
    assert gpu_ops.conv_flops(n, cin, h, w, cout, k, s, p) == 2 * macs


def test_gemm_reference_check_separates_fp32_from_reduced_precision():
    assert gpu_ops.gemm_rel_error(torch.float32, torch.device("cpu")) < 1e-5
    assert gpu_ops.gemm_rel_error(torch.bfloat16, torch.device("cpu")) > 1e-4  # would fail the fp32 bound


@pytest.mark.parametrize("group,unit", [("memory_ops", "GB/s"), ("elementwise_ops", "GB/s"),
                                         ("reduction_ops", "GB/s"), ("gemm_ops", "GFLOPS"), ("conv_ops", "GFLOPS")])
def test_every_group_yields_ops(group, unit, monkeypatch):
    monkeypatch.setattr(gpu_ops, "SIZES", {"tiny": (64, 64)})
    monkeypatch.setattr(gpu_ops, "GEMM_SHAPES", [(8, 8, 8)])
    monkeypatch.setattr(gpu_ops, "CONV_SHAPES", {"t": (1, 3, 8, 8, 4, 3, 1, 1)})
    from tests.conftest import make_args
    ops = list(gpu_ops.GpuOps(make_args(model=group)).get_operations())
    assert ops and all(u == unit and work > 0 for _, _, work, u in ops)
    for _, fn, _, _ in ops:
        fn()
