# llama.cpp Benchmark

Benchmarks language model inference using llama.cpp with GPU acceleration.

## Overview

This benchmark:
- Builds llama.cpp from source with GPU support
- Downloads GGUF format models from HuggingFace
- Runs the built-in `llama-bench` utility
- Reports tokens per second for text generation

## Requirements

- **CMake** (3.14+): Required for building llama.cpp
- **C++ Compiler**: GCC/Clang with C++17 support
- **GPU-specific requirements**:
  - **NVIDIA**: CUDA Toolkit (11.0+)
  - **AMD ROCm**: ROCm 5.0+ with `hipconfig` available
  - **AMD ROCK**: ROCK SDK with `rocm-sdk` command available

### Installing CMake

```bash
# Ubuntu/Debian
sudo apt install cmake

# Fedora
sudo dnf install cmake

# Arch
sudo pacman -S cmake
```

## Usage

### Basic Usage (Auto-detect GPU)

```bash
python benchmarks/llamacpp/main.py
```

This will:
1. Auto-detect your GPU type
2. Clone and build llama.cpp
3. Download the default model (Llama 3.1 8B Q4_0)
4. Run the benchmark

### Specify GPU Type

```bash
# NVIDIA GPU
python benchmarks/llamacpp/main.py --device nvidia

# AMD GPU (ROCm)
python benchmarks/llamacpp/main.py --device amd-rocm

# AMD GPU (ROCK SDK)
python benchmarks/llamacpp/main.py --device amd-rock

# CPU only
python benchmarks/llamacpp/main.py --device cpu
```

### Choose Model

```bash
# Q4_0 quantization (faster, less memory)
python benchmarks/llamacpp/main.py --model llama-3.1-8b-q4

# Q8_0 quantization (slower, more accurate)
python benchmarks/llamacpp/main.py --model llama-3.1-8b-q8
```

### AMD GPU Targets

For AMD GPUs, you can specify target architectures:

```bash
python benchmarks/llamacpp/main.py --device amd-rocm --amdgpu-targets gfx1100,gfx1151,gfx1201
```

Common AMD GPU targets:
- `gfx1100`: RX 7900 XTX, RX 7900 XT
- `gfx1151`: RX 7700 XT, RX 7800 XT
- `gfx1201`: RX 7600, RX 7600 XT

### GPU Layer Offloading

Control how many layers to offload to GPU:

```bash
# Offload all layers (default)
python benchmarks/llamacpp/main.py --ngl 999

# Offload 32 layers only
python benchmarks/llamacpp/main.py --ngl 32

# CPU only (no GPU layers)
python benchmarks/llamacpp/main.py --ngl 0
```

### Use Custom Model File

If you have your own GGUF model:

```bash
# Place model in llama.cpp/models/
cp my-model.gguf benchmarks/llamacpp/llama.cpp/models/

# Run with custom model
python benchmarks/llamacpp/main.py --model-file my-model.gguf
```

## Manual Setup

You can also run the setup script directly:

```bash
# Setup for NVIDIA
python utils/setup_llamacpp.py --gpu nvidia --model llama-3.1-8b-q4

# Setup for AMD ROCm
python utils/setup_llamacpp.py --gpu amd-rocm --model llama-3.1-8b-q4

# Setup for AMD ROCK with custom targets
python utils/setup_llamacpp.py --gpu amd-rock --amdgpu-targets gfx1100,gfx1151 --model llama-3.1-8b-q4

# Just build, no model download
python utils/setup_llamacpp.py --gpu nvidia

# Just download model (if already built)
python utils/setup_llamacpp.py --skip-clone --skip-build --model llama-3.1-8b-q4
```

## Build Types

### NVIDIA (CUDA)

Uses CUDA for GPU acceleration:
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### AMD ROCm

Uses ROCm/HIP for GPU acceleration:
```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j 16
```

### AMD ROCK SDK

Uses ROCK SDK for newer AMD drivers:
```bash
HIPCXX="$(rocm-sdk path --root)/llvm/bin/clang" \
  HIP_PATH="$(rocm-sdk path --root)" \
  HIP_PLATFORM=amd \
  CMAKE_PREFIX_PATH="$(rocm-sdk path --root):$CMAKE_PREFIX_PATH" \
  cmake --fresh -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_FLAGS:STRING="-I$(rocm-sdk path --root)/include"
cmake --build build --config Release -- -j 16
```

## Benchmark Output

The benchmark runs multiple tests:
- **pp512**: Prompt processing (512 tokens)
- **tg128**: Text generation (128 tokens)

Example output:
```
llama.cpp Benchmark
Model: llama-3.1-8b-q4
GPU Type: nvidia
GPU Layers: 999

✓ Using model: /path/to/llama.cpp/models/meta-llama-3.1-8b-instruct-q4_0.gguf

Running benchmark...

| model                          |       size |     params | backend    | ngl |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ---------------: |
| llama 8B Q4_0                  |   4.34 GiB |     8.03 B | CUDA       |  99 |         pp512 |    4885.46 ± 8.42|
| llama 8B Q4_0                  |   4.34 GiB |     8.03 B | CUDA       |  99 |         tg128 |     162.84 ± 0.21|

============================================================
LLAMA.CPP BENCHMARK RESULTS
============================================================

Test: pp512
  Model: llama 8B Q4_0
  Size: 4.34 GiB
  Params: 8.03 B
  Backend: CUDA
  GPU Layers: 99
  Tokens/sec: 4885.46 ± 8.42

Test: tg128
  Model: llama 8B Q4_0
  Size: 4.34 GiB
  Params: 8.03 B
  Backend: CUDA
  GPU Layers: 99
  Tokens/sec: 162.84 ± 0.21
============================================================

FINAL RESULT: 162.84 tokens/sec
```

## Troubleshooting

### CMake not found
```bash
sudo apt install cmake  # Ubuntu/Debian
```

### CUDA not found (NVIDIA)
- Install CUDA Toolkit from NVIDIA
- Add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

### hipconfig not found (AMD ROCm)
- Install ROCm: https://rocm.docs.amd.com/
- Source ROCm: `source /opt/rocm/setup.sh`

### rocm-sdk not found (AMD ROCK)
- Install ROCK SDK: https://www.amd.com/en/developer/resources/rocm-hub.html

### Build fails with GPU errors
Try CPU-only build:
```bash
python benchmarks/llamacpp/main.py --device cpu
```

### Model download fails
Download manually from HuggingFace:
```bash
# Visit: https://huggingface.co/ggml-org/Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF
# Download: meta-llama-3.1-8b-instruct-q4_0.gguf
# Place in: benchmarks/llamacpp/llama.cpp/models/
```

## Integration with Benchmark Framework

Use with the main benchmark framework:

```bash
# Run via framework
python benchmark.py --backend llamacpp --model llama-3.1-8b-q4

# Compare with other backends
python compare.py --backends llamacpp pytorch --model llama-3.1-8b-q4
```

## References

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Model Hub](https://huggingface.co/models?library=gguf)





