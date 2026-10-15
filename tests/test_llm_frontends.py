"""llama.cpp and Ollama numbers come from the tools' own timings; check we read them right."""

import importlib.util
import json
import stat
from pathlib import Path

import pytest

from tests.conftest import make_args

ROOT = Path(__file__).resolve().parents[1]


def load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_llamacpp_reads_pp_and_tg_from_llama_bench_json(tmp_path):
    llamacpp = load("benchmarks/llamacpp/main.py")
    entries = [
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 5000.0, "samples_ns": [102_400_000], "model_type": "llama 8B Q4_0",
         "backends": "CUDA", "build_commit": "abc"},
        {"n_prompt": 0, "n_gen": 128, "avg_ts": 160.0, "samples_ns": [800_000_000, 800_000_000],
         "model_type": "llama 8B Q4_0", "backends": "CUDA", "build_commit": "abc"},
    ]
    fake = tmp_path / "llama-bench"
    fake.write_text(f"#!/bin/sh\ncat <<'EOF'\n{json.dumps(entries)}\nEOF\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    args = make_args(model=str(gguf), precision="auto", llama_bench=str(fake), ngl=99, prompt_tokens=512,
                     max_tokens=128, num_runs=2)
    result = llamacpp.LlamaCppBench(args).run()
    assert result.status == "PASS"
    assert result.get_metric("decode_tokens_per_sec") == 160.0
    assert result.get_metric("prefill_tokens_per_sec") == 5000.0
    assert result.latency_stats["median"] == pytest.approx(800.0)
    assert result.config["backends"] == "CUDA"


def test_ollama_rates_use_server_side_durations(monkeypatch):
    ollama = load("benchmarks/ollama/text_generation/main.py")
    calls = []

    def fake_api(path, payload=None, timeout=600):
        calls.append((path, payload))
        if path == "/api/version":
            return {"version": "0.12.0"}
        if path == "/api/tags":
            return {"models": [{"name": "m:8b"}]}
        # 100 tokens in 1 s decode, 50 prompt tokens in 0.1 s
        return {"response": "hi", "eval_count": 100, "eval_duration": 1_000_000_000, "prompt_eval_count": 50,
                "prompt_eval_duration": 100_000_000, "load_duration": 0, "total_duration": 1_200_000_000}

    monkeypatch.setattr(ollama, "api", fake_api)
    result = ollama.OllamaGeneration(make_args(model="m:8b", precision="auto", max_tokens=100, num_runs=3)).run()
    assert result.get_metric("decode_tokens_per_sec") == pytest.approx(100.0)
    assert result.get_metric("prefill_tokens_per_sec") == pytest.approx(500.0)
    assert result.get_metric("ttft_ms") == pytest.approx(100.0)
    prompts = [p["prompt"] for path, p in calls if path == "/api/generate" and p and "prompt" in p]
    assert len(set(prompts)) == len(prompts)  # unique per request: no prompt-cache hits
