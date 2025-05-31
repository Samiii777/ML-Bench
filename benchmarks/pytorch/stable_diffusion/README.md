# Stable Diffusion Benchmark

This benchmark provides a **unified interface** for testing both **Stable Diffusion 1.5** and **Stable Diffusion 3 Medium** models. When run, it automatically benchmarks both models and provides comprehensive performance comparisons.

## Key Features

- **Automatic Dual Model Testing**: Runs both SD 1.5 and SD3 models in a single execution
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

It automatically executes this script and benchmarks both Stable Diffusion models with the specified parameters.

## Supported Models

### Stable Diffusion 1.5
- Model aliases: `sd15`, `sd1.5`, `stable_diffusion_1_5`
- Hugging Face ID: `runwayml/stable-diffusion-v1-5`
- Memory requirements: ~4-8 GB VRAM (depending on precision)

### Stable Diffusion 3 Medium
- Model aliases: `sd3`, `sd3_medium`, `stable_diffusion_3_medium`
- Hugging Face ID: `stabilityai/stable-diffusion-3-medium-diffusers`
- Memory requirements: ~12-22 GB VRAM (depending on precision)

## Usage

### Via Main Benchmark Framework (Recommended)

```bash
# Run both models with default settings
python benchmark.py --use_case generation

# Run with specific precision and batch size
python benchmark.py --use_case generation --precision fp16 --batch_size 1

# Run comprehensive benchmarks
python benchmark.py --use_case generation --comprehensive
```

### Direct Script Execution

```bash
# Run both models (default behavior)
python benchmarks/pytorch/stable_diffusion/inference/generation/main.py

# Run specific model only
python benchmarks/pytorch/stable_diffusion/inference/generation/main.py --model sd15
python benchmarks/pytorch/stable_diffusion/inference/generation/main.py --model sd3

# Custom configuration
python benchmarks/pytorch/stable_diffusion/inference/generation/main.py \
    --precision fp16 \
    --batch_size 1 \
    --num-runs 10 \
    --num-inference-steps 20
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `None` | Specific model to benchmark (default: run both) |
| `--precision` | str | `fp16` | Precision mode (`fp32`, `fp16`, `mixed`) |
| `--batch_size` | int | `1` | Batch size for inference |
| `--height` | int | `512` | Image height in pixels |
| `--width` | int | `512` | Image width in pixels |
| `--num-inference-steps` | int | `20` | Number of denoising steps |
| `--guidance-scale` | float | `4.5` | Guidance scale (SD3 only) |
| `--num-runs` | int | `5` | Number of benchmark runs |
| `--cpu-offload` | flag | `False` | Enable CPU offload for SD3 |
| `--save-images` | flag | `False` | Save generated images |
| `--output-dir` | str | `None` | Output directory for images |
| `--custom-prompt` | str | `None` | Custom generation prompt |

## Example Output

```
============================================================
STABLE DIFFUSION COMBINED BENCHMARK
============================================================
Running benchmarks for both Stable Diffusion models
Precision: fp16
Batch size: 1
Image size: 512x512
Inference steps: 20
Number of runs per model: 5

============================================================
BENCHMARKING: Stable Diffusion 1.5
============================================================
...
✅ Stable Diffusion 1.5: 2.34 images/sec, 3.9 GB VRAM

============================================================
BENCHMARKING: Stable Diffusion 3 Medium
============================================================
...
✅ Stable Diffusion 3 Medium: 0.81 images/sec, 14.2 GB VRAM

============================================================
BENCHMARK SUMMARY
============================================================
✅ Stable Diffusion 1.5: 2.34 images/sec, 3.9 GB VRAM
✅ Stable Diffusion 3 Medium: 0.81 images/sec, 14.2 GB VRAM
============================================================
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