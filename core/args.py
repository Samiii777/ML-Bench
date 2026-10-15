"""Shared argparse builder for benchmark scripts."""

import argparse

PRECISIONS = ["fp32", "tf32", "fp16", "bf16", "mixed"]


def build_base_parser(description: str = "ML-Bench Benchmark", precisions=PRECISIONS) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", type=str, default=None, help="Model name to benchmark")
    p.add_argument("--precision", type=str, default=precisions[0], choices=precisions,
                   help="fp32 = IEEE fp32 (TF32 off); tf32; fp16; bf16; mixed = fp32 weights + fp16 autocast")
    p.add_argument("--batch_size", "--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num_warmup", "--num-warmup", type=int, default=5, help="Untimed warmup iterations")
    p.add_argument("--num_runs", "--num-runs", type=int, default=20, help="Timed iterations")
    p.add_argument("--power", action="store_true", help="Sample GPU power during the timed loop")
    return p


def add_onnx_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution_provider", type=str, default=None,
                        help="ONNX Runtime execution provider (default: best available)")


def add_text_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt_tokens", "--prompt-tokens", type=int, default=128,
                        help="Exact prompt length in tokens (default: 128)")
    parser.add_argument("--max_tokens", "--max-tokens", type=int, default=128,
                        help="Tokens to generate per sequence; EOS is ignored (default: 128)")
