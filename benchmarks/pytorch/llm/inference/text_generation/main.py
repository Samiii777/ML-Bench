#!/usr/bin/env python3
"""Hugging Face causal-LM generation benchmark (Qwen3, Llama, ...).

Workload: batch of identical prompts of exactly --prompt_tokens tokens, greedy
decode of exactly --max_tokens new tokens (EOS ignored), KV cache on. Fixed
shapes mean no padding and a token count that is exact, not estimated.

Per iteration two timings are taken:
  ttft    generate(max_new_tokens=1)          -> prefill + first token
  full    generate(max_new_tokens=N)          -> whole request
Metrics:
  decode_tokens_per_sec   batch * (N-1) / (full - ttft)   [primary; same meaning
                          as llama.cpp "tg" and Ollama "eval rate"]
  prefill_tokens_per_sec  batch * prompt_tokens / ttft
  ttft_ms, e2e latency (latency_stats), output_tokens_per_sec = batch * N / full
Correctness: perplexity of a fixed English passage must be sane (catches
fp16 overflow / broken kernels that still "generate" garbage fast).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())))

import math

import torch

from core.args import add_text_args, build_base_parser
from core.harness import InferenceHarness, run_main
from core.schema import MetricEntry
from core.validation import ResultValidator
from utils.benchmark_utils import (
    benchmark_loop, compute_stats, measure_peak_memory, reset_memory_tracking, setup_torch_backends, warmup,
)

PASSAGE = (
    "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, "
    "continental Europe, and the United States that occurred during the period from around 1760 to about "
    "1820. This transition included going from hand production methods to machines, new chemical "
    "manufacturing and iron production processes, the increasing use of steam power and water power, the "
    "development of machine tools and the rise of the mechanised factory system. Output greatly increased, "
    "and a result was an unprecedented rise in population and in the rate of population growth. "
)
MAX_PERPLEXITY = 40.0


class LLMGeneration(InferenceHarness):
    use_case = "text_generation"
    primary_metric = "decode_tokens_per_sec"

    def load_model(self):
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = self.args.model
        self._tok = AutoTokenizer.from_pretrained(name)
        kwargs = dict(dtype=self.dtype)
        try:
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        except ValueError:  # multimodal checkpoints (e.g. Qwen3.5) expose the LM via image-text-to-text
            model = transformers.AutoModelForImageTextToText.from_pretrained(name, **kwargs)
        model.generation_config.pad_token_id = self._tok.pad_token_id or self._tok.eos_token_id
        return model.eval().to(self.device)

    def prepare_inputs(self):
        n = self.args.prompt_tokens
        ids = self._tok(PASSAGE * (n // 40 + 2), return_tensors="pt", add_special_tokens=False).input_ids[0, :n]
        assert ids.numel() == n, f"could not build a {n}-token prompt"
        return ids.unsqueeze(0).repeat(self.batch_size, 1).to(self.device)

    def generate(self, model, ids, new_tokens):
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                             max_new_tokens=new_tokens, min_new_tokens=new_tokens)
        return out

    def run(self):
        setup_torch_backends(self.precision)
        self.print_device_info()
        print(f"Loading {self.args.model}...")
        model = self._model = self.load_model()
        ids = self._inputs = self.prepare_inputs()
        n_new, n_prompt, bs = self.args.max_tokens, ids.shape[1], self.batch_size
        num_runs, num_warmup = self.args.num_runs, self.args.num_warmup

        def ttft_step():
            with torch.inference_mode(), self.autocast():
                return self.generate(model, ids, 1)

        def full_step():
            with torch.inference_mode(), self.autocast():
                return self.generate(model, ids, n_new)

        warmup(full_step, num_warmup, self.device)
        ttft = compute_stats(benchmark_loop(ttft_step, num_runs, self.device, desc="TTFT"))
        reset_memory_tracking(self.device)
        full = compute_stats(benchmark_loop(full_step, num_runs, self.device, desc=f"Generate {n_new} tokens"))
        peak = measure_peak_memory(self.device)

        out = full_step()
        generated = out.shape[1] - n_prompt
        decode_ms = full["mean"] - ttft["mean"]
        metrics = [
            MetricEntry("decode_tokens_per_sec", bs * (n_new - 1) / (decode_ms / 1000) if decode_ms > 0 else 0.0,
                        "tokens/sec", "higher_is_better"),
            MetricEntry("decode_tokens_per_sec_per_seq", (n_new - 1) / (decode_ms / 1000) if decode_ms > 0 else 0.0,
                        "tokens/sec", "higher_is_better"),
            MetricEntry("prefill_tokens_per_sec", bs * n_prompt / (ttft["mean"] / 1000), "tokens/sec",
                        "higher_is_better"),
            MetricEntry("ttft_ms", ttft["median"], "ms", "lower_is_better"),
            MetricEntry("output_tokens_per_sec", bs * n_new / (full["mean"] / 1000), "tokens/sec", "higher_is_better"),
        ]
        if peak:
            metrics.append(MetricEntry("peak_memory_gb", peak["peak_allocated_gb"], "GB", "lower_is_better"))

        ppl = self.perplexity(model)
        sample = self._tok.decode(out[0, n_prompt:n_prompt + 24], skip_special_tokens=True)
        print(f"Perplexity on reference passage: {ppl:.2f}\nSample continuation: {sample!r}")
        metrics.append(MetricEntry("perplexity", ppl, "ppl", "lower_is_better"))

        validator = ResultValidator()
        validator.expect_equals("generated_tokens", generated, n_new)
        validator.expect_in_range("perplexity", ppl, 1.0, MAX_PERPLEXITY)
        return self.finish(metrics, full, validator)

    @torch.inference_mode()
    def perplexity(self, model) -> float:
        ids = self._tok(PASSAGE, return_tensors="pt").input_ids.to(self.device)
        with self.autocast():
            loss = model(input_ids=ids, labels=ids).loss
        return math.exp(loss.float().item())

    def get_config(self):
        return {"prompt_tokens": self.args.prompt_tokens, "new_tokens": self.args.max_tokens,
                "decoding": "greedy, EOS ignored", "attn_implementation": self._model.config._attn_implementation}


if __name__ == "__main__":
    parser = build_base_parser("Hugging Face causal LM generation")
    add_text_args(parser)
    parser.set_defaults(model="Qwen/Qwen3-1.7B", precision="bf16", num_runs=5, num_warmup=2)
    run_main(LLMGeneration, parser)
