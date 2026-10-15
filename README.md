# ML-Bench

Benchmarks real ML workloads — vision, detection, NLP, LLMs, image generation, training and GPU
micro-kernels — across PyTorch, ONNX Runtime, Ollama, llama.cpp and ComfyUI, on NVIDIA (CUDA),
AMD (ROCm), Apple (MPS) and CPU.

Every result is **validated** (the model must produce a correct answer, not just run fast), labelled
with the **precision that actually ran**, and stored with full latency statistics and a system
fingerprint so runs can be compared over time and across machines.

## Quick start

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # or rocm7.1 / cpu
pip install -r requirements.txt

python benchmark.py                                   # default matrix: PyTorch + ONNX + Ollama
python benchmark.py --model resnet50 --batch_size 1 64
python benchmark.py --model gpu_ops --precision fp32 tf32 fp16 bf16
python benchmark.py --model Qwen/Qwen3-8B --script-args "--prompt_tokens 1024 --max_tokens 256"
python benchmark.py --mode training
python benchmark.py --framework llamacpp --model ~/models/Qwen3.5-9B-Q4_K_M.gguf
python benchmark.py --dry-run                         # show the test matrix
python benchmark.py --list-benchmarks
```

Each run prints a table and writes `benchmark_results/run_<timestamp>.{json,csv,md}` plus one line per
test to `benchmark_results/history.jsonl`. The exit code is 1 if any test failed.

```bash
python compare.py benchmark_results/run_A.json benchmark_results/run_B.json --threshold 5   # exit 1 on regression
python benchmark.py --dashboard                                                            # Streamlit, reads history
```

## Workloads

| Use case | Framework | Default models | Headline metric |
|---|---|---|---|
| Image classification | PyTorch, ONNX | `resnet50`, `vit_b_16`, `convnext_tiny` (any torchvision name works) | samples/sec |
| Object detection | PyTorch, ONNX | `yolo11s` (Ultralytics, incl. NMS) | samples/sec |
| Text classification | PyTorch | `bert-base` (seq len 128) | samples/sec |
| LLM generation | PyTorch (HF) | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B` (any HF causal LM id) | decode tokens/sec |
| LLM generation | Ollama | `llama3.1:8b`, `qwen3:8b`, `gemma3:4b`, `gpt-oss:20b` | decode tokens/sec |
| LLM generation | llama.cpp | any GGUF (`--model path.gguf`) | decode tokens/sec |
| Image generation | PyTorch (diffusers) | `stable_diffusion_1_5`, `stable_diffusion_3_5_medium`, `flux_2_klein_4b`, `z_image_turbo` | seconds/image |
| Image generation | ComfyUI | `comfyui_flux_schnell`, `comfyui_flux_dev` | seconds/image |
| Training | PyTorch | `resnet50`, `vit_b_16` | samples/sec |
| GPU kernels | PyTorch | `gemm_ops`, `conv_ops`, `memory_ops`, `elementwise_ops`, `reduction_ops` | best GFLOPS / GB/s |

Also available but not in the default matrix: `stable_diffusion_3_5_large_turbo`, `flux_1_schnell`,
`flux_1_dev` (these need >24 GB or `--script-args --cpu_offload`), and Qwen3.5 checkpoints
(`--model Qwen/Qwen3.5-9B`; install `flash-linear-attention` + `causal-conv1d` for the fast path).

Default precisions and batch sizes per use case live in `utils/config.py`.

## What the numbers mean

**Precision** — the label is what ran:

| `--precision` | Meaning |
|---|---|
| `fp32` | IEEE fp32. TF32 is explicitly **off** for matmul *and* cuDNN conv (PyTorch enables cuDNN TF32 by default). |
| `tf32` | fp32 tensors with TF32 math allowed. |
| `fp16`, `bf16` | weights and activations in that dtype. |
| `mixed` | fp32 weights, fp16 autocast. |
| training | weights stay fp32; `fp16`/`bf16` run as AMP and are reported as `amp_fp16`/`amp_bf16`. |

**Timing** — warmup iterations first, then each timed iteration is bracketed by device events
(CUDA/ROCm) or synchronised wall clock. GC is disabled during the timed loop.

| Metric | Definition |
|---|---|
| `throughput` | `batch_size / mean iteration latency` |
| `latency_p50_ms`, `latency_p99_ms` | per **batch** (one iteration), not per sample |
| CV % | std/mean of iteration latency; reports flag CV > 10% as noisy |
| `peak_memory_gb` | PyTorch allocator peak during the timed loop (omitted when the framework doesn't allocate through PyTorch) |
| `seconds_per_image` | full pipeline call (text encode + all steps + VAE) / batch size, at each model's reference resolution and step count |

**LLMs** — fixed workload: prompt of exactly `--prompt_tokens` (128) tokens, exactly `--max_tokens` (128)
new tokens, greedy, EOS ignored, so token counts are exact.
`ttft_ms` is a `max_new_tokens=1` call (prefill + first token);
`decode_tokens_per_sec = batch × (N−1) / (full − ttft)`;
`prefill_tokens_per_sec = batch × prompt_tokens / ttft`.
Ollama and llama.cpp report the same quantities from their own server-side timings, so
`decode_tokens_per_sec` means the same thing across all three.

**GPU kernels** — work is useful FLOPs (2 per multiply-add) or the minimum DRAM bytes (reads + writes).
Each sample times enough back-to-back calls to fill ~5 ms, so small kernels aren't measured as launch
overhead. Bandwidth tests use 256M-element tensors (0.5–1 GB) so they measure DRAM, not cache.
Compare against datasheet peaks: on an RTX 4090 expect ~160 TFLOPS fp16 GEMM and ~920 GB/s copy.

**Correctness checks** (a failed check makes the test FAIL):
classification top-1 = "Samoyed" on `data/dog.jpg`; YOLO detects the dog; BERT rates a positive review
≥4 stars and a negative one ≤2; LLM perplexity on a reference passage is sane and exactly N tokens are
generated; generated images are finite and not blank; training loss decreases; GEMM matches an fp64
reference (the fp32 bound is tight enough to catch accidental TF32); every metric is finite.

**Status** — the JSON block each script prints is the source of truth. A script that exits non-zero, prints
no result, or exceeds `--timeout` is a FAIL with the log tail attached (a timeout kills the script's whole
process group, so a stuck `llama-bench` or ComfyUI server can't keep the GPU busy for later tests).
SKIPs carry a reason: no GPU for fp16, framework not installed, ONNX provider listed but unusable,
model too large for free VRAM (override with `--skip-vram-check`).

## Layout

```
benchmark.py                  orchestrator: matrix -> subprocess per test -> collect JSON
compare.py                    compare two runs, gate on regressions
dashboard.py                  Streamlit view of history.jsonl
core/harness.py               shared timing / memory / validation / output for all scripts
core/schema.py                BenchmarkResult (the JSON contract)
utils/config.py               model registry and default matrix
utils/results.py              JSON / CSV / Markdown reports
benchmarks/<framework>/<family>/<mode>/<use_case>/main.py
tests/                        pytest; tests/test_gpu_smoke.py runs only with a GPU
```

## Adding a benchmark

Subclass a harness, implement three methods, and add the model to `utils/config.py`:

```python
class MyBench(InferenceHarness):
    use_case = "classification"
    def load_model(self):  return build().to(self.device, self.dtype).eval()
    def prepare_inputs(self):  return torch.randn(self.batch_size, 3, 224, 224, device=self.device, dtype=self.dtype)
    def run_step(self, model, x):  return model(x)
    def validate_result(self, model, x, out, v):  v.expect_finite("logit0", out[0, 0].item())

if __name__ == "__main__":
    run_main(MyBench, build_base_parser("My benchmark"))
```

`run_main` handles warmup, timing, stats, memory, `--power` sampling, and always emits a result
(a structured FAIL with the error, including `OOM:`, if anything raises).

## Tests

```bash
python -m pytest            # harness math, FLOP/byte accounting, status rules, reports; + GPU smoke tests if a GPU exists
python -m pytest -m gpu     # only the GPU end-to-end tests
```
