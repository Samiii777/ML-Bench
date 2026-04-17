"""Shared argparse builder for benchmark scripts."""

import argparse


def build_base_parser(description: str = "ML-Bench Benchmark") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to benchmark")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16", "mixed"],
                        help="Precision mode (default: fp32)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use (default: auto)")
    parser.add_argument("--num_warmup", type=int, default=5,
                        help="Number of warmup iterations (default: 5)")
    parser.add_argument("--num_runs", type=int, default=20,
                        help="Number of benchmark iterations (default: 20)")
    return parser


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--height", type=int, default=512,
                        help="Image height (default: 512)")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width (default: 512)")
    parser.add_argument("--num-inference-steps", type=int, default=20,
                        help="Number of denoising steps (default: 20)")
    parser.add_argument("--guidance-scale", type=float, default=4.5,
                        help="Guidance scale (default: 4.5)")
    parser.add_argument("--sdp-backend", type=str, default="auto",
                        choices=["auto", "safe", "math", "mem_efficient", "flash"],
                        help="SDPA backend (default: auto)")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Enable CPU offload to save VRAM")
    parser.add_argument("--custom-prompt", type=str, default=None,
                        help="Custom prompt for generation")
    parser.add_argument("--force-fp16", action="store_true",
                        help="Force fp16 even for models that prefer bf16")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training_mode", type=str, default="scratch",
                        choices=["scratch", "finetune"],
                        help="Training mode (default: scratch)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")


def add_onnx_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution_provider", type=str, default=None,
                        choices=["CUDAExecutionProvider",
                                 "TensorrtExecutionProvider",
                                 "CPUExecutionProvider",
                                 "ROCMExecutionProvider",
                                 "MIGraphXExecutionProvider"],
                        help="ONNX Runtime execution provider")


def add_text_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Max tokens to generate (default: 50)")
    parser.add_argument("--custom-prompt", type=str, default=None,
                        help="Custom prompt for text generation")
