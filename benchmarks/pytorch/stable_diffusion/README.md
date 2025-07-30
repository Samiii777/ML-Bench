# Stable Diffusion Benchmark

This benchmark provides a **unified interface** for testing **Stable Diffusion 1.5**, **Stable Diffusion 3 Medium**, and **Stable Diffusion 3.5 Large Turbo** models. When run, it automatically benchmarks the specified models with optimized settings and provides comprehensive performance comparisons.

## Key Features

- **Multi-Model Support**: SD1.5, SD3 Medium, and SD3.5 Large Turbo
- **Automatic Optimization**: Model-specific image sizes and inference steps
- **Multiple Precisions**: fp32, fp16, and mixed precision support
- **Memory Optimization**: Automatic memory management and CPU offload options
- **Comprehensive Metrics**: Detailed performance and memory usage statistics
- **Framework Integration**: Seamlessly integrates with the main ML-Bench framework
- **Flexible Configuration**: Customizable batch sizes, image dimensions, and inference steps

## Integration with ML-Bench

This benchmark is designed to work with the main ML-Bench framework. When you run:

```bash
python benchmark.py --use_case generation
```

It automatically executes this script and benchmarks the Stable Diffusion models with optimized parameters.

## Supported Models

### Stable Diffusion 1.5
- **Model aliases**: `sd15`, `sd1.5`, `stable_diffusion_1_5`
- **Hugging Face ID**: `runwayml/stable-diffusion-v1-5`
- **Default settings**: 512x512, 20 inference steps
- **Memory requirements**: ~4GB VRAM (fp16), ~8GB (fp32)

### Stable Diffusion 3 Medium
- **Model aliases**: `sd3`, `sd3_medium`, `stable_diffusion_3_medium`
- **Hugging Face ID**: `stabilityai/stable-diffusion-3-medium-diffusers`
- **Default settings**: 1024x1024, 28 inference steps
- **Memory requirements**: ~18GB VRAM (fp16), ~24GB (fp32)

### Stable Diffusion 3.5 Large Turbo
- **Model aliases**: `sd35_turbo`, `sd3.5_turbo`, `stable_diffusion_3_5_large_turbo`
- **Hugging Face ID**: `stabilityai/stable-diffusion-3.5-large-turbo`
- **Default settings**: 1024x1024, 4 inference steps (optimized for speed)
- **Memory requirements**: ~10GB VRAM (fp16), ~16GB (fp32)

## Automatic Model Optimization

The benchmark automatically applies optimal settings for each model:

- **SD1.5**: 512x512 resolution, 20 inference steps
- **SD3 Medium**: 1024x1024 resolution, 28 inference steps  
- **SD3.5 Turbo**: 1024x1024 resolution, 4 inference steps

These defaults can be overridden using command-line arguments.

## Usage

### Via Main Benchmark Framework (Recommended)

```bash
# Run specific model with optimal settings
python benchmark.py --framework pytorch --model sd35_turbo --precision fp16

# Run all stable diffusion models
python benchmark.py --use_case generation --precision fp16

# Run comprehensive benchmarks
python benchmark.py --use_case generation --comprehensive
```

### Direct Script Execution

```bash
# Run specific model with optimal settings
python benchmarks/pytorch/stable_diffusion/inference/generation/main.py --model sd35_turbo

# Run all models (default behavior)
python benchmarks/pytorch/stable_diffusion/inference/generation/main.py

# Custom configuration (overrides optimized defaults)
python benchmarks/pytorch/stable_diffusion/inference/generation/main.py \
    --model sd3 \
    --precision fp16 \
    --batch_size 1 \
    --height 768 \
    --width 768 \
    --num-inference-steps 15
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `None` | Specific model to benchmark (default: run all) |
| `--precision` | str | `fp16` | Precision mode (`fp32`, `fp16`, `mixed`) |
| `--batch_size` | int | `1` | Batch size for inference |
| `--height` | int | Auto | Image height (512 for SD1.5, 1024 for SD3+) |
| `--width` | int | Auto | Image width (512 for SD1.5, 1024 for SD3+) |
| `--num-inference-steps` | int | Auto | Inference steps (20/28/4 based on model) |
| `--guidance-scale` | float | `4.5` | Guidance scale (SD3: 4.5, Turbo: 1.0) |
| `--num-runs` | int | `5` | Number of benchmark runs |
| `--cpu-offload` | flag | `False` | Enable CPU offload for SD3+ |
| `--save-images` | flag | `False` | Save generated images |
| `--output-dir` | str | `None` | Output directory for images |
| `--custom-prompt` | str | `None` | Custom generation prompt |

## Performance Comparison

Typical performance on RTX 4090 (fp16 precision):

| Model | Resolution | Steps | Performance | VRAM Usage |
|-------|------------|-------|-------------|------------|
| SD1.5 | 512x512 | 20 | ~1.3 samples/sec | ~4GB |
| SD3 Medium | 1024x1024 | 28 | ~0.2 samples/sec | ~18GB |
| SD3.5 Turbo | 1024x1024 | 4 | ~0.5 samples/sec | ~10GB |

## Example Output

```
============================================================
STABLE DIFFUSION COMBINED BENCHMARK
============================================================
Running benchmarks for Stable Diffusion models
Precision: fp16
Batch size: 1
Auto-optimized settings per model

============================================================
STARTING MODEL: Stable Diffusion 3.5 Large Turbo
============================================================
Using default image size 1024x1024 for Stable Diffusion 3.5 Large Turbo
Using default inference steps 4 for Stable Diffusion 3.5 Large Turbo

============================================================
BENCHMARKING: Stable Diffusion 3.5 Large Turbo
============================================================
...
✅ Stable Diffusion 3.5 Large Turbo: 0.49 images/sec, 22.6 GB VRAM
```

## Memory Requirements

### Stable Diffusion 1.5
| Precision | Batch Size 1 | Batch Size 2 | Batch Size 4 |
|-----------|--------------|--------------|--------------|
| FP32      | ~8 GB        | ~12 GB       | ~20 GB       |
| FP16      | ~4 GB        | ~6 GB        | ~10 GB       |
| Mixed     | ~6 GB        | ~9 GB        | ~15 GB       |

### Stable Diffusion 3 Medium
| Precision | Batch Size 1 | Batch Size 2 | Batch Size 4 |
|-----------|--------------|--------------|--------------|
| FP32      | ~22 GB       | ~40 GB+      | OOM          |
| FP16      | ~12 GB       | ~20 GB       | ~35 GB+      |
| Mixed     | ~18 GB       | ~30 GB+      | OOM          |

## Performance Tips

1. **Use FP16 precision** for best speed/memory balance on CUDA GPUs
2. **Enable CPU offload** for SD3 if running out of VRAM
3. **Reduce batch size** if encountering OOM errors
4. **Lower inference steps** for faster generation (10-15 steps often sufficient)
5. **Use smaller image sizes** (256x256, 384x384) for faster testing

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch size to 1
- Use FP16 precision
- Enable CPU offload for SD3: `--cpu-offload`
- Reduce image resolution: `--height 256 --width 256`

### Slow Performance
- Ensure CUDA is available and being used
- Use FP16 precision
- Install xformers for memory efficient attention
- Close other GPU-intensive applications

### Model Download Issues
- Ensure stable internet connection
- Check Hugging Face Hub access
- Verify sufficient disk space (~10-20 GB per model)

## Architecture

The benchmark automatically:

1. **Loads both models sequentially** to avoid memory conflicts
2. **Runs warm-up iterations** for accurate timing
3. **Measures performance metrics** including throughput and latency
4. **Monitors memory usage** throughout execution
5. **Cleans up resources** between models to prevent memory leaks
6. **Outputs standardized metrics** for framework integration

## License

This benchmark follows the same license as the main ML-Bench project. 