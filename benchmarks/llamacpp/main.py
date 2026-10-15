#!/usr/bin/env python3
"""llama.cpp benchmark via `llama-bench -o json`.

  --model PATH.gguf (or a file name found in ./models, ~/models, benchmarks/llamacpp/llama.cpp/models)
  llama-bench is taken from --llama_bench, $LLAMA_BENCH, PATH, ~/llama.cpp/build/bin, or
  benchmarks/llamacpp/llama.cpp/build/bin (build one with utils/setup_llamacpp.py).

Runs the standard pp (prompt processing, --prompt_tokens) and tg (generation,
--max_tokens) tests, --num_runs repetitions each, all layers on GPU:
  decode_tokens_per_sec   tg rate  [primary; same meaning as Ollama eval rate]
  prefill_tokens_per_sec  pp rate
"""

import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())
sys.path.insert(0, str(ROOT))

import json
import os
import shutil
import subprocess

import torch

from core.args import add_text_args, build_base_parser
from core.harness import BenchmarkHarness, run_main
from core.schema import MetricEntry
from utils.benchmark_utils import compute_stats

HERE = Path(__file__).resolve().parent


def find_llama_bench(explicit=None) -> str:
    candidates = [explicit, os.environ.get("LLAMA_BENCH"), shutil.which("llama-bench"),
                  Path.home() / "llama.cpp/build/bin/llama-bench", HERE / "llama.cpp/build/bin/llama-bench"]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    raise FileNotFoundError("llama-bench not found; pass --llama_bench, set LLAMA_BENCH, "
                            "or build it with utils/setup_llamacpp.py")


def find_model(name: str) -> Path:
    for p in [Path(name), ROOT / "models" / name, Path.home() / "models" / name, HERE / "llama.cpp/models" / name]:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"GGUF model {name!r} not found (give a path or put it in ./models or ~/models)")


class LlamaCppBench(BenchmarkHarness):
    framework = "llamacpp"
    use_case = "text_generation"
    primary_metric = "decode_tokens_per_sec"

    def run(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # for system info only
        model = find_model(self.args.model)
        cmd = [find_llama_bench(self.args.llama_bench), "-m", str(model), "-ngl", str(self.args.ngl),
               "-p", str(self.args.prompt_tokens), "-n", str(self.args.max_tokens),
               "-r", str(self.args.num_runs), "-o", "json"]
        print("$ " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            lines = [l for l in proc.stderr.splitlines() if l.strip()]
            reason = next((l for l in reversed(lines) if "error" in l.lower()), lines[-1] if lines else "")
            raise RuntimeError(f"llama-bench exited {proc.returncode}: {reason.strip()}")
        entries = json.loads(proc.stdout)
        pp = next(e for e in entries if e["n_prompt"] > 0 and e["n_gen"] == 0)
        tg = next(e for e in entries if e["n_gen"] > 0 and e["n_prompt"] == 0)
        self._info = {k: tg.get(k) for k in ("build_commit", "backends", "gpu_info", "model_type", "model_size",
                                              "n_gpu_layers", "flash_attn", "n_batch", "n_ubatch", "type_k")}
        print(f"{tg.get('model_type')} on {tg.get('backends')}: pp{pp['n_prompt']} {pp['avg_ts']:.1f} t/s, "
              f"tg{tg['n_gen']} {tg['avg_ts']:.1f} t/s")
        metrics = [
            MetricEntry("decode_tokens_per_sec", tg["avg_ts"], "tokens/sec", "higher_is_better"),
            MetricEntry("prefill_tokens_per_sec", pp["avg_ts"], "tokens/sec", "higher_is_better"),
        ]
        stats = compute_stats([ns / 1e6 for ns in tg.get("samples_ns", [])])  # ms per tg request
        return self.finish(metrics, stats)

    @property
    def effective_precision(self):
        return "auto"

    def get_config(self):
        return {"prompt_tokens": self.args.prompt_tokens, "new_tokens": self.args.max_tokens, **self._info}


if __name__ == "__main__":
    parser = build_base_parser("llama.cpp (llama-bench)", precisions=["auto"])
    add_text_args(parser)
    parser.add_argument("--llama_bench", "--llama-bench", help="path to the llama-bench binary")
    parser.add_argument("--ngl", type=int, default=999, help="layers offloaded to GPU (default: all)")
    parser.set_defaults(prompt_tokens=512, num_runs=5)
    run_main(LlamaCppBench, parser)
