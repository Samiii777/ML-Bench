#!/usr/bin/env python3
"""PyTorch LLaMA Text Generation Inference Benchmark"""

import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch

torch.set_flush_denormal(True)

from core.harness import InferenceHarness
from core.schema import MetricEntry, BenchmarkMeta, BenchmarkResult
from core.args import build_base_parser, add_text_args
from core.validation import ResultValidator
from core.output import emit_result
from utils.benchmark_utils import (
    BenchmarkTimer, compute_stats, gc_disabled,
    measure_peak_memory, reset_memory_tracking, setup_torch_backends, warmup,
)
from utils.shared_device_utils import collect_system_fingerprint

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="llama",
    supported_models=[
        "llama3.1-8b", "llama3.2", "llama-3.2-1b", "llama-3.2-3b",
        "llama3.2-1b", "llama3.2-3b",
        "deepseek", "deepseek-r1", "deepseek-r1-7b",
        "deepseek-r1-1.5b", "deepseek-r1-8b", "deepseek-r1-0528-8b",
    ],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="text_generation",
)


def get_llama_model_name(model_arg):
    """Map model argument to actual Hugging Face model name or use directly if in HF format"""
    if "/" in model_arg:
        print(f"Using HuggingFace model directly: {model_arg}")
        return model_arg

    llama_models = {
        "llama3.1-8b": "meta-llama/Llama-3.1-8B",
        "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-r1-1.5b": "deepseek-ai/Deepseek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-r1-0528-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    }
    return llama_models.get(model_arg, "meta-llama/Llama-3.1-8B")


class LlamaTextGenerationBenchmark(InferenceHarness):
    """LLaMA / DeepSeek text-generation benchmark.

    Because text generation uses ``model.generate()`` with variable-length
    output and needs to track tokens_per_second, this class overrides
    ``run()`` entirely instead of relying on the base-class timing loop
    (which calls a fixed ``run_step``).
    """

    @property
    def use_case(self):
        return "text_generation"

    # -- model / inputs --------------------------------------------------

    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        name = get_llama_model_name(self.args.model or "meta-llama/Llama-3.2-1B-Instruct")
        self._hf_model_name = name
        self._is_deepseek = "deepseek" in name.lower()
        print(f"Loading model: {name}")

        self._tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

        torch_dtype = torch.float32
        if self.args.precision in ("fp16", "mixed"):
            torch_dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            use_cache=True,
            low_cpu_mem_usage=True,
        )
        if self.device.type != "cuda":
            model = model.to(self.device)
        model.eval()

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.add_special_tokens({"pad_token": self._tokenizer.eos_token})
            model.resize_token_embeddings(len(self._tokenizer))

        return model

    def _get_prompts(self):
        if self._is_deepseek:
            return [
                "Please reason step by step: What are the key advantages of renewable energy over fossil fuels?",
                "Please reason step by step: How does machine learning contribute to medical diagnosis improvements?",
                "Please reason step by step: What factors should be considered when designing sustainable cities?",
                "Please reason step by step: How can artificial intelligence help address climate change challenges?",
                "Please reason step by step: What are the ethical implications of autonomous vehicles?",
            ]
        return [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly",
            "Machine learning has transformed the way we",
            "The most important aspect of scientific research is",
            "Climate change represents one of the greatest challenges",
        ]

    def prepare_inputs(self):
        prompts = self._get_prompts()
        batch_prompts = [prompts[j % len(prompts)] for j in range(self.args.batch_size)]
        inputs = self._tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def run_step(self, model, inputs):
        return model.generate(
            **inputs,
            max_new_tokens=self._max_tokens,
            num_return_sequences=1,
            pad_token_id=self._tokenizer.pad_token_id,
            do_sample=False,
        )

    # -- custom run with token counting ---------------------------------

    def run(self) -> BenchmarkResult:
        setup_torch_backends(cudnn_benchmark=True)
        self.print_device_info()
        print(f"Using device: {self.device}")

        print("Loading model...")
        self._model = self.load_model()

        print("Preparing inputs...")
        self._inputs = self.prepare_inputs()

        batch_size = getattr(self.args, "batch_size", 1)
        precision = getattr(self.args, "precision", "fp16")
        model_name = getattr(self.args, "model", "unknown")
        num_warmup = getattr(self.args, "num_warmup", 5)
        num_runs = max(10, 20 // batch_size)
        self._max_tokens = getattr(self.args, "max_tokens", 30 if batch_size > 1 else 50)

        prompts = self._get_prompts()

        # Warmup
        print(f"Warming up ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            with torch.inference_mode():
                self.run_step(self._model, self._inputs)
            self.synchronize()

        reset_memory_tracking(self.device)

        # Benchmark with token counting
        print(f"Benchmarking ({num_runs} iterations)...")
        timer = BenchmarkTimer(self.device)
        inference_times = []
        tokens_generated = []

        with gc_disabled():
            for i in range(num_runs):
                batch_prompts = [prompts[j % len(prompts)] for j in range(batch_size)]
                inputs = self._tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                timer.start()
                with torch.inference_mode():
                    outputs = self.run_step(self._model, inputs)
                elapsed_ms = timer.stop()

                inference_times.append(elapsed_ms)
                input_length = inputs["input_ids"].shape[1]
                output_length = outputs.shape[1]
                tokens_gen = (output_length - input_length) * batch_size
                tokens_generated.append(tokens_gen)

                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/{num_runs} runs")

        peak_mem = measure_peak_memory(self.device)
        stats = compute_stats(inference_times)

        total_tokens = sum(tokens_generated)
        total_time_s = sum(t / 1000.0 for t in inference_times)
        tokens_per_second = total_tokens / total_time_s if total_time_s > 0 else 0
        avg_tokens_per_run = float(np.mean(tokens_generated))

        mean_ms = stats["mean"]
        per_sample_ms = mean_ms / batch_size
        throughput = (batch_size / mean_ms) * 1000.0 if mean_ms > 0 else 0.0

        print(f"\nTokens per second: {tokens_per_second:.1f}")
        print(f"Average tokens per run: {avg_tokens_per_run:.1f}")

        # Validation
        validator = ResultValidator()
        validator.expect_greater_than("throughput", throughput, 0)
        validator.expect_greater_than("tokens_generated", float(total_tokens), 0)
        validation_passed, validation_checks = validator.validate()

        failed_checks = [c for c in validation_checks if not c.passed]
        if failed_checks:
            print(f"\nValidation FAILED ({len(failed_checks)} check(s)):")
            for c in failed_checks:
                print(f"  {c.name}: {c.message}")

        metrics = [
            MetricEntry("avg_latency_ms", per_sample_ms, "ms", "lower_is_better"),
            MetricEntry("throughput", throughput, "samples/sec", "higher_is_better"),
            MetricEntry("tokens_per_second", tokens_per_second, "tokens/sec", "higher_is_better"),
            MetricEntry("avg_tokens_per_run", avg_tokens_per_run, "tokens", "higher_is_better"),
        ]
        if peak_mem:
            metrics.append(MetricEntry("peak_memory_gb", peak_mem.get("peak_allocated_gb", 0), "GB", "lower_is_better"))

        result = BenchmarkResult(
            status="PASS" if validation_passed else "FAIL",
            framework=self.framework,
            model=model_name,
            mode=self.mode,
            use_case=self.use_case,
            precision=precision,
            batch_size=batch_size,
            system_info=self._build_system_info(),
            metrics=metrics,
            latency_stats=stats,
            validation_checks=validator.to_dicts(),
            error="; ".join(c.message for c in failed_checks) if failed_checks else None,
        )
        emit_result(result)
        return result


if __name__ == "__main__":
    parser = build_base_parser("PyTorch LLaMA Text Generation Inference Benchmark")
    add_text_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct", precision="fp16")
    args = parser.parse_args()

    try:
        benchmark = LlamaTextGenerationBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
