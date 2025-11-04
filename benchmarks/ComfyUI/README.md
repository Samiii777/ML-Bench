# ComfyUI Benchmark

This benchmark tests FLUX.1 image generation (schnell and dev variants) using ComfyUI as the backend.

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

## Supported Models

### FLUX.1-schnell (Default)
- **Optimized for:** 4-step generation
- **Speed:** ~4 seconds/image @ 1024x1024
- **Quality:** Good, fast iterations
- **Use case:** Rapid prototyping, batch generation

### FLUX.1-dev
- **Optimized for:** 20-50 steps
- **Speed:** ~20 seconds/image @ 1024x1024 (20 steps)
- **Quality:** Higher quality, more detailed
- **Use case:** Production, final renders

## Usage

### Via Benchmark Framework

```bash
# Run both FLUX models (schnell + dev)
python benchmark.py --framework comfyui

# Run schnell only (faster)
python benchmark.py --framework comfyui --model comfyui_flux_schnell

# Run dev only (higher quality)
python benchmark.py --framework comfyui --model comfyui_flux_dev
```

### Standalone

```bash
# FLUX.1-schnell (fast, 4 steps)
python benchmarks/ComfyUI/main.py --model comfyui_flux_schnell --num_runs 10

# FLUX.1-dev (high quality, 20 steps)
python benchmarks/ComfyUI/main.py --model comfyui_flux_dev --num_runs 5

# Custom resolution
python benchmarks/ComfyUI/main.py --model comfyui_flux_schnell --width 512 --height 512

# Custom steps (override defaults)
python benchmarks/ComfyUI/main.py --model comfyui_flux_dev --steps 50
```

## Requirements

- PyTorch with CUDA support
- ~20GB disk space (for FLUX models)
- ~8GB VRAM minimum (recommended: 12GB+)
- All dependencies installed via requirements.txt

## Models Downloaded

### FLUX.1-schnell (downloaded by default)
- **UNET**: `flux1-schnell.safetensors` (~17GB)
- **Text Encoders** (shared):
  - `t5xxl_fp16.safetensors` (~9GB)
  - `clip_l.safetensors` (~1GB)
- **VAE** (shared): `ae.safetensors` (~330MB)
- **Total**: ~27GB

### FLUX.1-dev (downloaded on first use)
- **UNET**: `flux1-dev.safetensors` (~17GB)
- **Text Encoders**: Shared with schnell
- **VAE**: Shared with schnell
- **Additional**: +17GB (only UNET is different)

All files downloaded to `ComfyUI/models/`

## Metrics

The benchmark reports:
- **Time per Image**: Seconds to generate one image (primary metric)
- **Throughput**: Images per second (calculated from time/image)
- **Latency**: Average, min, max, std deviation (in seconds)
- **Memory**: Peak GPU memory usage
- **Steps**: Number of diffusion steps (default: 4 for schnell, 20 for dev)
- **Resolution**: Image dimensions (default: 1024x1024)

### Performance Format
```
[1/1] comfyui/comfyui_flux_schnell[generation] fp16 BS=1  ✓ 4.04 s/img
[1/1] comfyui/comfyui_flux_dev[generation] fp16 BS=1      ✓ 18.50 s/img (est.)
```

## Notes

- First run takes longer due to model download (~27GB total)
- FLUX.1-schnell is optimized for 4-step generation
- Server automatically starts/stops for each benchmark run
- Workflows are saved to `workflows/` for reference



