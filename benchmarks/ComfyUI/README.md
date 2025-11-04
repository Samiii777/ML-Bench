# ComfyUI Benchmark

This benchmark tests FLUX.1-schnell image generation using ComfyUI as the backend.

## Structure

```
benchmarks/ComfyUI/
├── main.py                    # Benchmark script
├── README.md                  # This file
├── ComfyUI/                   # ComfyUI installation (auto-setup)
│   ├── main.py
│   ├── models/
│   │   ├── diffusion_models/  # FLUX UNET models
│   │   ├── text_encoders/     # CLIP & T5 encoders
│   │   └── vae/               # VAE models
│   └── ...
└── workflows/                 # Saved workflows (auto-created)
    └── flux_schnell_fp16_benchmark.json

```

## How It Works

1. **Auto-setup**: On first run, automatically downloads ComfyUI and FLUX.1-schnell models
2. **Server Management**: Starts ComfyUI API server in background
3. **Benchmark Execution**: Runs workflow via API and measures performance
4. **Cleanup**: Stops server after benchmarking

## Usage

### Via Benchmark Framework

```bash
# Run ComfyUI FLUX benchmark
python benchmark.py --framework comfyui

# With custom parameters
python benchmark.py --framework comfyui --num_runs 20
```

### Standalone

```bash
# Direct execution
python benchmarks/ComfyUI/main.py --num_runs 10 --steps 4

# Custom resolution
python benchmarks/ComfyUI/main.py --width 512 --height 512 --steps 4
```

## Requirements

- PyTorch with CUDA support
- ~20GB disk space (for FLUX models)
- ~8GB VRAM minimum (recommended: 12GB+)
- All dependencies installed via requirements.txt

## Models Downloaded

On first run, the following files are automatically downloaded to `ComfyUI/models/`:

- **UNET**: `flux1-schnell.safetensors` (~17GB)
- **Text Encoders**: 
  - `t5xxl_fp16.safetensors` (~9GB)
  - `clip_l.safetensors` (~1GB)
- **VAE**: `ae.safetensors` (~330MB)

## Metrics

The benchmark reports:
- **Throughput**: Images per second
- **Latency**: Average, min, max, std deviation (in seconds)
- **Memory**: Peak GPU memory usage
- **Steps**: Number of diffusion steps (default: 4 for schnell)
- **Resolution**: Image dimensions (default: 1024x1024)

## Notes

- First run takes longer due to model download (~27GB total)
- FLUX.1-schnell is optimized for 4-step generation
- Server automatically starts/stops for each benchmark run
- Workflows are saved to `workflows/` for reference



