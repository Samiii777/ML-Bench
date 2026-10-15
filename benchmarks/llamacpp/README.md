# llama.cpp benchmark

Runs `llama-bench -o json` on a GGUF model and reports llama.cpp's own numbers:

| Metric | llama-bench test |
|---|---|
| `decode_tokens_per_sec` (headline) | `tg` — generate `--max_tokens` tokens (default 128) |
| `prefill_tokens_per_sec` | `pp` — process a `--prompt_tokens` prompt (default 512) |

```bash
python benchmark.py --framework llamacpp --model ~/models/Qwen3.5-9B-Q4_K_M.gguf
python benchmarks/llamacpp/main.py --model my-model.gguf --ngl 999 --num_runs 5
```

`--model` is a GGUF path, or a file name found in `./models`, `~/models` or `benchmarks/llamacpp/llama.cpp/models`.
`llama-bench` is taken from `--llama_bench`, `$LLAMA_BENCH`, `PATH`, `~/llama.cpp/build/bin`, or
`benchmarks/llamacpp/llama.cpp/build/bin`.

To build llama.cpp (and optionally download a model) into `benchmarks/llamacpp/llama.cpp`:

```bash
python utils/setup_llamacpp.py --gpu nvidia            # or amd-rocm (add --amdgpu-targets gfx1100,...), cpu
python utils/setup_llamacpp.py --gpu nvidia --model llama-3.1-8b-q4
```

The build commit, backend, GPU, model type/size, flash-attention and batch settings are recorded in each
result's `config` so runs from different llama.cpp builds aren't compared blindly.
