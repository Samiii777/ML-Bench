# Combined Stable Diffusion Benchmark

This benchmark provides a unified interface for testing both **Stable Diffusion 1.5** and **Stable Diffusion 3 Medium** models with various precision configurations.

## Features

- **Unified Interface**: Single script supports both SD 1.5 and SD3 models
- **Multiple Precisions**: fp32, fp16, and mixed precision support
- **Memory Optimization**: Automatic memory management and CPU offload options
- **Comprehensive Metrics**: Detailed performance and memory usage statistics
- **Flexible Configuration**: Customizable batch sizes, image dimensions, and inference steps
- **Image Generation**: Optional image saving with timestamped filenames

## Supported Models

### Stable Diffusion 1.5
- Model aliases: `sd15`, `sd1.5`, `stable_diffusion_1_5`
- Hugging Face ID: `runwayml/stable-diffusion-v1-5`
- Memory requirements: ~4-8 GB VRAM (depending on precision)

### Stable Diffusion 3 Medium
- Model aliases: `sd3`, `sd3_medium`, `stable_diffusion_3_medium`
- Hugging Face ID: `stabilityai/stable-diffusion-3-medium-diffusers`
- Memory requirements: ~12-22 GB VRAM (depending on precision)

## Installation

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

**Test Stable Diffusion 1.5:**
```bash
python benchmarks/pytorch/stable_diffusion_combined/inference/generation/main.py --model sd15
```

**Test Stable Diffusion 3:**
```bash
python benchmarks/pytorch/stable_diffusion_combined/inference/generation/main.py --model sd3
```

### Advanced Configuration

**Custom precision and batch size:**
```bash
python benchmarks/pytorch/stable_diffusion_combined/inference/generation/main.py \
    --model sd3 \
    --precision fp16 \
    --batch-size 2 \
    --num-runs 10
```

**Generate and save images:**
```bash
python benchmarks/pytorch/stable_diffusion_combined/inference/generation/main.py \
    --model sd15 \
    --save-images \
    --output-dir ./generated_images \
    --custom-prompt "A beautiful sunset over mountains"
```

**Memory optimization for SD3:**
```bash
python benchmarks/pytorch/stable_diffusion_combined/inference/generation/main.py \
    --model sd3 \
    --precision fp16 \
    --cpu-offload \
    --batch-size 1
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `sd15` | Model to benchmark (`sd15`, `sd3`, etc.) |
| `--precision` | str | `fp16` | Precision mode (`fp32`, `fp16`, `mixed`) |
| `--batch-size` | int | `1` | Batch size for inference |
| `--height` | int | `512` | Image height in pixels |
| `--width` | int | `512` | Image width in pixels |
| `--num-inference-steps` | int | `20` | Number of denoising steps |
| `--guidance-scale` | float | `4.5` | Guidance scale (SD3 only) |
| `--num-runs` | int | `5` | Number of benchmark runs |
| `--cpu-offload` | flag | `False` | Enable CPU offload for SD3 |
| `--save-images` | flag | `False` | Save generated images |
| `--output-dir` | str | `None` | Output directory for images |
| `--custom-prompt` | str | `None` | Custom generation prompt |

## Precision Modes

### FP32 (Full Precision)
- **Memory**: Highest usage
- **Quality**: Best quality
- **Speed**: Slowest
- **Compatibility**: All devices

### FP16 (Half Precision)
- **Memory**: ~50% reduction
- **Quality**: Minimal quality loss
- **Speed**: Faster inference
- **Compatibility**: CUDA GPUs only

### Mixed Precision
- **Memory**: Balanced usage
- **Quality**: Good quality
- **Speed**: Good performance
- **Compatibility**: CUDA GPUs with autocast

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

*Note: Memory usage varies based on image resolution and inference steps*

## Example Output

```
Running Stable Diffusion 3 Medium inference benchmark
Model ID: stabilityai/stable-diffusion-3-medium-diffusers
Model Type: SD3
Precision: fp16
Batch size: 1
Image size: 512x512
Inference steps: 20
Guidance scale: 4.5

==================================================
DEVICE INFORMATION
==================================================
Selected device: cuda
PyTorch version: 2.7.0+cu126
CUDA available: True
CUDA version: 12.6
Number of GPUs: 1
GPU 0: NVIDIA GeForce RTX 4090
  Memory: 23.5 GB
==================================================

==================================================
BENCHMARK RESULTS
==================================================
Model: Stable Diffusion 3 Medium
Model Type: SD3
Precision: fp16
Batch size: 1
Image size: 512x512
Inference steps: 20
Guidance scale: 4.5
Number of runs: 5

Average time per run: 1.234 ± 0.045 seconds
Min time: 1.189 seconds
Max time: 1.298 seconds
Average images per second: 0.81
Average time per image: 1.234 seconds

Memory usage: 14.2 GB
GPU utilization: 59.2%
==================================================
```

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
- Enable CPU offload for SD3
- Reduce image resolution

### Slow Performance
- Ensure CUDA is available and being used
- Use FP16 precision
- Install xformers for memory efficient attention
- Close other GPU-intensive applications

### Model Download Issues
- Ensure stable internet connection
- Check Hugging Face Hub access
- Verify sufficient disk space (~10-20 GB per model)

## Integration with Main Benchmark

This combined benchmark can be integrated with the main ML-Bench framework by adding the appropriate configuration to `utils/config.py`:

```python
'stable_diffusion_combined': {
    'sd15_fp16': {
        'command': 'python benchmarks/pytorch/stable_diffusion_combined/inference/generation/main.py --model sd15 --precision fp16',
        'expected_memory_gb': 4.0
    },
    'sd3_fp16': {
        'command': 'python benchmarks/pytorch/stable_diffusion_combined/inference/generation/main.py --model sd3 --precision fp16',
        'expected_memory_gb': 12.0
    }
}
```

## License

This benchmark follows the same license as the main ML-Bench project. 