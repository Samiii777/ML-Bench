#!/usr/bin/env python3
"""Ollama text generation benchmark (server must be running on localhost:11434).

Uses Ollama's own server-side timings (nanoseconds), not HTTP wall time:
  decode_tokens_per_sec   eval_count / eval_duration            [primary]
  prefill_tokens_per_sec  prompt_eval_count / prompt_eval_duration
  ttft_ms                 load + prompt eval of a request
Warmup requests load the model first. Every request gets a unique prefix so
Ollama's prompt cache can't turn the prefill measurement into a cache hit.
Greedy decoding with a fixed seed and --max_tokens cap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())))

import json
import urllib.request

import torch

from core.args import add_text_args, build_base_parser
from core.harness import BenchmarkHarness, run_main
from core.schema import MetricEntry
from core.validation import ResultValidator
from utils.benchmark_utils import compute_stats

HOST = "http://localhost:11434"
PROMPT = ("Explain, step by step and in detail, how a modern GPU executes a matrix multiplication, "
          "covering memory hierarchy, tiling, tensor cores, and how software libraries choose kernels.")


def api(path: str, payload=None, timeout: float = 600):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class OllamaGeneration(BenchmarkHarness):
    framework = "ollama"
    use_case = "text_generation"
    primary_metric = "decode_tokens_per_sec"

    def generate(self, i: int):
        return api("/api/generate", {
            "model": self.args.model, "prompt": f"[request {i}] {PROMPT}", "stream": False,
            "options": {"num_predict": self.args.max_tokens, "temperature": 0, "seed": 0},
        })

    def run(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # for system info only
        model = self.args.model
        self._version = api("/api/version").get("version", "?")
        if model not in {m["name"] for m in api("/api/tags").get("models", [])}:
            print(f"Pulling {model} (one-time download)...")
            api("/api/pull", {"model": model, "stream": False}, timeout=3600)
        print(f"Ollama {self._version}, model {model}")

        for i in range(max(1, self.args.num_warmup)):
            self.generate(-1 - i)
        runs = [self.generate(i) for i in range(self.args.num_runs)]
        try:
            api("/api/generate", {"model": model, "keep_alive": 0})  # free VRAM for the next test
        except Exception:
            pass

        ns = 1e-9
        evals = sum(r.get("eval_count", 0) for r in runs)
        eval_s = sum(r.get("eval_duration", 0) for r in runs) * ns
        prompts = sum(r.get("prompt_eval_count", 0) for r in runs)
        prompt_s = sum(r.get("prompt_eval_duration", 0) for r in runs) * ns
        stats = compute_stats([r.get("total_duration", 0) / 1e6 for r in runs])
        ttft = compute_stats([(r.get("load_duration", 0) + r.get("prompt_eval_duration", 0)) / 1e6 for r in runs])
        self._sample = runs[-1].get("response", "")
        print(f"Sample: {self._sample[:120]!r}")

        metrics = [
            MetricEntry("decode_tokens_per_sec", evals / eval_s if eval_s else 0.0, "tokens/sec", "higher_is_better"),
            MetricEntry("prefill_tokens_per_sec", prompts / prompt_s if prompt_s else 0.0, "tokens/sec",
                        "higher_is_better"),
            MetricEntry("ttft_ms", ttft.get("median", 0.0), "ms", "lower_is_better"),
            MetricEntry("avg_output_tokens", evals / len(runs), "tokens", "higher_is_better"),
        ]
        self._prompt_tokens = prompts / len(runs)
        validator = ResultValidator()
        validator.expect_greater_than("output_tokens", evals, 0)
        return self.finish(metrics, stats, validator)

    @property
    def effective_precision(self):
        return "auto"

    def get_config(self):
        return {"ollama_version": self._version, "max_tokens": self.args.max_tokens,
                "avg_prompt_tokens": self._prompt_tokens, "decoding": "temperature 0, seed 0"}


if __name__ == "__main__":
    parser = build_base_parser("Ollama text generation", precisions=["auto"])  # Ollama picks quantization
    add_text_args(parser)
    parser.set_defaults(model="llama3.1:8b", num_runs=5, num_warmup=1)
    run_main(OllamaGeneration, parser)
