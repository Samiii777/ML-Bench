#!/usr/bin/env python3
"""
PyTorch Combined Stable Diffusion Inference Benchmark
Automatically runs both Stable Diffusion 1.5 and Stable Diffusion 3 Medium models
with fp32, fp16, and mixed precision configurations
"""

import argparse
import sys
import os
import time
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from datetime import datetime

# Add project root to path for utils import
import sys
from pathlib import Path
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

# Clean import of utils - no ugly relative paths!
import utils
from utils.shared_device_utils import get_gpu_memory_efficient

def get_gpu_memory_nvidia_smi():
    """Get GPU memory using nvidia-smi directly"""
    try:
        import nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        nvidia_smi.nvmlShutdown()
        
        used_gb = info.used / (1024**3)
        total_gb = info.total / (1024**3)
        
        return {
            "total_gpu_used_gb": used_gb,
            "total_gpu_total_gb": total_gb,
            "gpu_utilization_percent": (used_gb / total_gb) * 100
        }
    except ImportError:
        print("Warning: nvidia-ml-py3 not installed, memory measurement unavailable")
        return None
    except Exception as e:
        print(f"Warning: GPU memory measurement failed: {e}")
        return None

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def synchronize():
    """Synchronize device operations for accurate timing"""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

def print_device_info():
    """Print device information"""
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    
    device = get_device()
    print(f"Selected device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA available: False")
    
    print("=" * 50)

def get_model_configs():
    """Get configurations for all SD / FLUX models"""
    return [
        {
            'name': 'stable_diffusion_1_5',
            'type': 'sd15',
            'model_id': 'runwayml/stable-diffusion-v1-5',
            'display_name': 'Stable Diffusion 1.5'
        },
        {
            'name': 'stable_diffusion_3_medium',
            'type': 'sd3',
            'model_id': 'stabilityai/stable-diffusion-3-medium-diffusers',
            'display_name': 'Stable Diffusion 3 Medium'
        },
        {
            'name': 'stable_diffusion_3_5_medium',
            'type': 'sd35_medium',
            'model_id': 'stabilityai/stable-diffusion-3.5-medium',
            'display_name': 'Stable Diffusion 3.5 Medium'
        },
        {
            'name': 'stable_diffusion_3_5_large_turbo',
            'type': 'sd3_turbo',
            'model_id': 'stabilityai/stable-diffusion-3.5-large-turbo',
            'display_name': 'Stable Diffusion 3.5 Large Turbo'
        },
        {
            'name': 'flux_1_schnell',
            'type': 'flux_schnell',
            'model_id': 'black-forest-labs/FLUX.1-schnell',
            'display_name': 'FLUX.1 Schnell'
        },
        {
            'name': 'flux_1_dev',
            'type': 'flux_dev',
            'model_id': 'black-forest-labs/FLUX.1-dev',
            'display_name': 'FLUX.1 Dev'
        }
    ]


def _half_dtype_for(model_type: str, force_fp16: bool = False) -> torch.dtype:
    """Pick the recommended half-precision dtype for a given model family.
    
    SD3 / SD3.5 / FLUX use MMDiT attention whose activations overflow fp16's
    ~6.5e4 range for many prompts, producing NaN or black-image outputs.
    Stability AI's reference code for SD3 and SD3.5 uses bf16, and FLUX is
    distributed in bf16. SD1.5 is numerically fine with fp16.
    
    Falls back to fp16 if:
      - the user passed --force-fp16 (explicit opt-out for A/B comparison), or
      - the current CUDA device lacks native bf16 support (pre-Ampere NVIDIA,
        very old ROCm), since emulated bf16 is much slower than native fp16.
    """
    is_bf16_family = model_type in (
        'sd3', 'sd3_medium', 'sd3_turbo', 'sd35_medium',
        'flux_schnell', 'flux_dev',
    )
    if not is_bf16_family or force_fp16:
        return torch.float16
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    print(f"Warning: {model_type} is recommended to run in bf16, but this "
          f"device does not support bf16 natively; falling back to fp16 "
          f"(may produce NaN / black outputs on some prompts)")
    return torch.float16

def get_test_prompts():
    """Get a set of test prompts suitable for both SD 1.5 and SD3"""
    return [
        "A cat holding a sign that says hello world",
        "A beautiful landscape with mountains and a lake at sunset",
        "A futuristic city with flying cars and neon lights",
        "A portrait of a person with intricate details",
        "An abstract art piece with vibrant colors"
    ]

def save_images(images, output_dir, prefix="generated"):
    """Save generated images to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            image = image.cpu().permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_paths

def get_benchmark_images_dir():
    """Get the benchmark results images directory"""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    
    # Navigate up to find the project root (where benchmark.py is located)
    for _ in range(10):  # Safety limit
        if os.path.exists(os.path.join(project_root, 'benchmark.py')):
            break
        project_root = os.path.dirname(project_root)
    
    # Create benchmark_results/images directory
    images_dir = os.path.join(project_root, "benchmark_results", "images")
    return images_dir

def load_sd15_pipeline(model_id, precision, device):
    """Load Stable Diffusion 1.5 pipeline"""
    from diffusers import StableDiffusionPipeline
    
    if precision == "fp16" and device.type == "cuda":
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
    elif precision == "mixed" and device.type == "cuda":
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Load in FP32 for true mixed precision
            safety_checker=None,
            requires_safety_checker=False
        )
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            requires_safety_checker=False
        )
        if precision == "fp16" and device.type != "cuda":
            print("Warning: FP16 not supported on CPU, using FP32")
        if precision == "mixed" and device.type != "cuda":
            print("Warning: Mixed precision not supported on CPU, using FP32")
    
    # Move pipeline to device
    pipeline = pipeline.to(device)
    
    # Enable optimizations for SD 1.5
    if device.type == "cuda":
        try:
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                pipeline.enable_xformers_memory_efficient_attention()
            elif hasattr(pipeline, 'enable_memory_efficient_attention'):
                pipeline.enable_memory_efficient_attention()
        except Exception as e:
            print(f"Note: Could not enable memory efficient attention: {e}")
        
        try:
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
        except Exception as e:
            print(f"Note: Could not enable attention slicing: {e}")
    
    return pipeline

def load_sd3_family_pipeline(model_id, precision, device, half_dtype, cpu_offload=False):
    """Load any SD3 / SD3.5 pipeline (Medium, 3.5 Medium, 3.5 Large Turbo).
    
    For single-GPU runs we use `.to(device)` rather than `device_map="balanced"`:
    - `"balanced"` is designed for multi-GPU sharding and on a single GPU just
      adds hook layers plus a `ValueError` path we previously had to catch.
    - `.to(device)` matches Stability AI's reference code and the SD1.5 path.
    
    Precision handling:
    - "fp16": load in `half_dtype` (bf16 for SD3-family by default — see
      `_half_dtype_for` — fp16 only if --force-fp16 or device lacks bf16).
    - "mixed": load in fp32; autocast is applied by the caller.
    - "fp32": load in fp32.
    Users who need multi-GPU sharding can pass --cpu-offload (sequential or
    model-level offload via accelerate).
    """
    from diffusers import StableDiffusion3Pipeline
    
    if precision == "fp16":
        torch_dtype = half_dtype
    else:
        torch_dtype = torch.float32
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    
    # Place on device unless CPU-offload will manage placement for us.
    if device.type == "cuda" and not cpu_offload:
        pipeline = pipeline.to(device)
    
    try:
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
    except Exception as e:
        print(f"Note: Could not enable memory efficient attention: {e}")
    
    if cpu_offload:
        try:
            if hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
                print("Enabled model CPU offload")
        except Exception as e:
            print(f"Note: Could not enable model CPU offload: {e}")
    
    return pipeline

def load_flux_pipeline(model_id, precision, device, half_dtype, cpu_offload=False):
    """Load any FLUX pipeline (Schnell or Dev).
    
    FLUX is released in bf16; `half_dtype` will normally be bf16 unless the
    user passed --force-fp16 or the GPU lacks native bf16 support.
    
    For single-GPU runs we use `.to(device)` rather than `device_map="balanced"`
    for the same reasons as SD3 (see `load_sd3_family_pipeline`). When
    --cpu-offload is set, placement is delegated to accelerate hooks via
    `enable_sequential_cpu_offload` (FLUX's preferred offload mode).
    """
    from diffusers import FluxPipeline
    
    if precision == "fp16":
        torch_dtype = half_dtype
    elif precision == "mixed":
        # Historically FLUX was loaded in bf16 for "mixed"; keep bf16 here too
        # since autocast to fp16 overflows on FLUX in exactly the same way it
        # does on SD3.
        torch_dtype = torch.bfloat16
    else:  # fp32
        torch_dtype = torch.float32
    
    pipeline = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    
    if device.type == "cuda" and not cpu_offload:
        pipeline = pipeline.to(device)
    
    try:
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
    except Exception as e:
        print(f"Note: Could not enable memory efficient attention: {e}")
    
    if cpu_offload:
        try:
            # FLUX benefits more from sequential offload than model-level
            # (weights are chunky and the text encoders are big).
            if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                pipeline.enable_sequential_cpu_offload()
                print("Enabled sequential CPU offload for FLUX")
            elif hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
                print("Enabled model CPU offload for FLUX")
        except Exception as e:
            print(f"Note: Could not enable CPU offload: {e}")
    
    return pipeline

def get_default_image_size(model_type):
    """Get default image size based on model type"""
    if model_type in ['sd3', 'sd35_medium', 'sd3_turbo', 'flux_schnell', 'flux_dev']:
        # SD3, SD3.5 (Medium/Large Turbo), and FLUX work best at 1024x1024
        return 1024, 1024
    else:
        # SD1.5 works best at 512x512
        return 512, 512

def get_default_inference_steps(model_type):
    """Get default inference steps based on model type"""
    if model_type == 'sd3_turbo':
        # SD3.5 Turbo is optimized for 4-step inference
        return 4
    elif model_type == 'flux_schnell':
        # FLUX Schnell is optimized for 4-step inference
        return 4
    elif model_type == 'flux_dev':
        # FLUX Dev requires more steps for higher quality
        return 20
    elif model_type in ('sd3', 'sd35_medium'):
        # SD3 Medium / SD3.5 Medium both recommend ~28 steps
        return 28
    else:
        # SD1.5 works well with 20-50 steps
        return 20

def run_single_model_benchmark(model_config, params, device=None):
    """Run benchmark for a single model"""
    model_type = model_config['type']
    model_id = model_config['model_id']
    display_name = model_config['display_name']
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {display_name}")
    print(f"{'='*60}")
    print(f"Model ID: {model_id}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Image size: {params.height}x{params.width}")
    print(f"Inference steps: {params.num_inference_steps}")
    if model_type == 'sd3':
        print(f"Guidance scale: {params.guidance_scale}")
    elif model_type == 'sd3_turbo':
        print(f"Guidance scale: 1.0 (optimized for turbo)")
    elif model_type == 'flux_schnell':
        print(f"Guidance scale: 0.0 (no guidance for FLUX Schnell)")
    elif model_type == 'flux_dev':
        print(f"Guidance scale: {params.guidance_scale} (guidance enabled for FLUX Dev)")
    
    if device is None:
        device = get_device()
    
    # Measure initial memory usage
    initial_memory = get_gpu_memory_nvidia_smi()
    
    print(f"Loading {display_name} pipeline...")
    
    # Pick the right half-precision dtype for this model family. For SD3 /
    # SD3.5 / FLUX this is bf16 by default (fp16 overflows MMDiT attention);
    # SD1.5 stays on fp16. Users can pass --force-fp16 for A/B comparison.
    force_fp16 = bool(getattr(params, 'force_fp16', False))
    half_dtype = _half_dtype_for(model_type, force_fp16=force_fp16)
    if params.precision in ("fp16", "mixed") and model_type != 'sd15':
        print(f"Using {half_dtype} for {model_type} "
              f"(bf16 is recommended for SD3 / SD3.5 / FLUX)")
    
    # Load the appropriate pipeline based on model type
    if model_type == 'sd15':
        pipeline = load_sd15_pipeline(model_id, params.precision, device)
    elif model_type in ('sd3', 'sd35_medium', 'sd3_turbo'):
        pipeline = load_sd3_family_pipeline(
            model_id, params.precision, device, half_dtype, params.cpu_offload)
    elif model_type in ('flux_schnell', 'flux_dev'):
        pipeline = load_flux_pipeline(
            model_id, params.precision, device, half_dtype, params.cpu_offload)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Measure memory after loading
    warmup_memory = get_gpu_memory_nvidia_smi()
    
    # Get test prompts
    test_prompts = get_test_prompts()
    prompt = test_prompts[0] if not params.custom_prompt else params.custom_prompt
    print(f"Test prompt: '{prompt}'")
    
    # Import shared benchmark utilities
    from utils.benchmark_utils import (
        BenchmarkTimer, compute_stats, gc_disabled,
        reset_memory_tracking, measure_peak_memory, setup_torch_backends
    )
    setup_torch_backends(cudnn_benchmark=True)
    
    # Determine guidance scale for this model
    def _guidance_scale():
        if model_type in ('sd3', 'sd35_medium'):
            return params.guidance_scale
        elif model_type == 'sd3_turbo':
            return 1.0
        elif model_type == 'flux_schnell':
            return 0.0
        elif model_type == 'flux_dev':
            return params.guidance_scale
        else:
            return 7.5
    
    use_mixed = params.precision == "mixed" and device.type == "cuda"
    # Autocast dtype must match the dtype we'd have loaded the model in for
    # pure-fp16 / bf16 runs, otherwise SD3/FLUX overflow fp16 even under
    # autocast (activations, not just weights, are the source of overflow).
    autocast_dtype = half_dtype
    guidance = _guidance_scale()
    
    # Silence diffusers' per-step tqdm during warmup + benchmark. Each pipeline
    # call already takes many seconds on large models and the per-step progress
    # bar adds noticeable host overhead plus log spam, especially on slower
    # backends (AMD ROCm without xformers, CPU). Failing silently if the
    # pipeline API doesn't expose this hook is fine.
    try:
        pipeline.set_progress_bar_config(disable=True)
    except Exception:
        pass
    
    # Warmup: a single full-shape iteration is enough to prime cuDNN/MIOpen
    # autotune. On slow devices each SD iteration can take minutes, so 5 warmup
    # iterations multiplies the wall-clock cost by 5× with no statistical gain.
    num_warmup = max(1, params.num_warmup) if hasattr(params, 'num_warmup') else 1
    print(f"Performing warm-up ({num_warmup} iteration, {params.num_inference_steps} steps)...")
    for i in range(num_warmup):
        with torch.inference_mode():
            gen_kwargs = {
                'prompt': prompt,
                'height': params.height,
                'width': params.width,
                'num_inference_steps': params.num_inference_steps,
                'guidance_scale': guidance,
                'num_images_per_prompt': 1
            }
            if use_mixed:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    _ = pipeline(**gen_kwargs).images
            else:
                _ = pipeline(**gen_kwargs).images
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    
    print("Warm-up completed. Starting benchmark...")
    
    # Reset peak memory after warmup for clean benchmark-only measurement
    reset_memory_tracking(device)
    
    # Benchmark runs. Fully respect --num-runs — each SD iteration can take
    # many seconds to minutes on slow backends, so forcing a floor here would
    # turn a slow backend into a multi-hour run with no way to escape.
    num_runs = max(1, params.num_runs)
    timer = BenchmarkTimer(device)
    times_ms = []
    all_images = []
    
    print(f"Benchmarking ({num_runs} iterations)...")
    with gc_disabled():
        for run in range(num_runs):
            batch_prompts = [prompt] * params.batch_size
            
            gen_kwargs = {
                'prompt': batch_prompts,
                'height': params.height,
                'width': params.width,
                'num_inference_steps': params.num_inference_steps,
                'guidance_scale': guidance,
                'num_images_per_prompt': 1
            }
            
            timer.start()
            
            with torch.inference_mode():
                if use_mixed:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        result = pipeline(**gen_kwargs)
                else:
                    result = pipeline(**gen_kwargs)
            
            elapsed_ms = timer.stop()
            times_ms.append(elapsed_ms)
            
            if run == 0:
                all_images = result.images
    
    # Print per-run summary AFTER the timed loop (out of band, no host overhead impact)
    for idx, elapsed_ms in enumerate(times_ms):
        run_time_s = elapsed_ms / 1000.0
        images_per_second = params.batch_size / run_time_s
        time_per_image = run_time_s / params.batch_size
        print(f"  Run {idx + 1}/{num_runs}: {run_time_s:.2f}s | {images_per_second:.2f} img/s | {time_per_image:.2f}s/img")
    
    # Measure peak memory (only covers the benchmark runs, not warmup)
    peak_mem = measure_peak_memory(device)
    
    # Compute comprehensive statistics (in milliseconds)
    stats = compute_stats(times_ms)
    
    # Convert to seconds for display
    avg_time = stats["mean"] / 1000.0
    std_time = stats["std"] / 1000.0
    min_time = stats["min"] / 1000.0
    max_time = stats["max"] / 1000.0
    
    avg_images_per_second = params.batch_size / avg_time
    avg_time_per_image = avg_time / params.batch_size
    avg_latency_ms = avg_time_per_image * 1000  # Convert to milliseconds
    
    # Measure final memory (nvidia-smi as secondary reference)
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Print results in both human-readable and parseable formats
    print(f"\n{'-'*50}")
    print(f"RESULTS: {display_name}")
    print(f"{'-'*50}")
    print(f"Model: {display_name}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Image size: {params.height}x{params.width}")
    print(f"Inference steps: {params.num_inference_steps}")
    print(f"Guidance scale: {guidance}")
    print(f"Number of runs: {num_runs}")
    print()
    print(f"Average time per run: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"Median time per run: {stats['median']/1000:.3f} seconds")
    print(f"P90 time: {stats['p90']/1000:.3f} seconds")
    print(f"P95 time: {stats['p95']/1000:.3f} seconds")
    print(f"P99 time: {stats['p99']/1000:.3f} seconds")
    print(f"Min time: {min_time:.3f} seconds")
    print(f"Max time: {max_time:.3f} seconds")
    print(f"Average images per second: {avg_images_per_second:.2f}")
    print(f"Average time per image: {avg_time_per_image:.3f} seconds")
    
    # Memory information (PyTorch allocator peak is primary)
    if peak_mem:
        print(f"\nGPU Memory Allocated: {peak_mem.get('peak_allocated_gb', 0):.2f} GB")
        print(f"GPU Memory Cached: {peak_mem.get('peak_reserved_gb', 0):.2f} GB")
    if final_memory:
        print(f"Total GPU Memory Used (nvidia-smi): {final_memory['total_gpu_used_gb']:.2f} GB")
    
    # Output in format expected by benchmark framework
    print(f"\n# Benchmark Framework Parseable Output for {display_name}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Throughput: {avg_images_per_second:.2f} samples/sec")
    print(f"Per-sample Latency: {avg_latency_ms:.2f} ms/sample")
    if final_memory:
        print(f"Total GPU Memory Used: {final_memory['total_gpu_used_gb']:.2f} GB")
    print(f"# End Parseable Output for {display_name}")
    
    # Always save images to benchmark_results for analysis
    if all_images:
        print(f"\nSaving {len(all_images)} generated images...")
        
        # Create timestamped directory in benchmark_results/images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_images_dir = get_benchmark_images_dir()
        run_dir = f"{model_config['name']}_{params.precision}_bs{params.batch_size}_{timestamp}"
        output_dir = os.path.join(base_images_dir, run_dir)
        
        prefix = f"{model_type}_{params.precision}_bs{params.batch_size}"
        saved_paths = save_images(all_images, output_dir, prefix)
        print(f"Images saved to: {output_dir}")
    
    # Clean up pipeline to free memory
    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return results for potential further processing
    return {
        'model': display_name,
        'model_type': model_type,
        'precision': params.precision,
        'batch_size': params.batch_size,
        'image_size': f"{params.height}x{params.width}",
        'inference_steps': params.num_inference_steps,
        'guidance_scale': guidance,
        'num_runs': num_runs,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'median_time': stats['median'] / 1000.0,
        'p90_time': stats['p90'] / 1000.0,
        'p95_time': stats['p95'] / 1000.0,
        'p99_time': stats['p99'] / 1000.0,
        'avg_images_per_second': avg_images_per_second,
        'avg_time_per_image': avg_time_per_image,
        'memory_usage_gb': peak_mem.get('peak_allocated_gb', None) if peak_mem else (final_memory['total_gpu_used_gb'] if final_memory else None),
        'gpu_utilization_percent': final_memory['gpu_utilization_percent'] if final_memory else None
    }

def _configure_sdp_backend(choice: str):
    """Configure PyTorch scaled_dot_product_attention backends.
    
    On ROCm, the flash-SDPA path (AOTriton) can hang or be extremely slow for
    long-sequence MMDiT attention (SD3, FLUX). The math + mem_efficient
    backends are reliable everywhere. This helper lets the user pick.
    
    choice values:
      'auto'          - leave PyTorch defaults
      'math'          - force pure-math SDPA (slowest but most portable)
      'mem_efficient' - memory-efficient kernel only
      'flash'         - flash-attention kernel only (NVIDIA recommended)
      'safe'          - disable flash on ROCm, keep it on NVIDIA (recommended default)
    """
    if not torch.cuda.is_available():
        return
    is_rocm = getattr(torch.version, "hip", None) is not None
    be = torch.backends.cuda
    
    # Map of setter availability by PyTorch version
    def _try(name, val):
        fn = getattr(be, name, None)
        if callable(fn):
            try:
                fn(val)
            except Exception:
                pass
    
    if choice == 'auto':
        return
    if choice == 'safe':
        if is_rocm:
            # Disable flash on ROCm to avoid known hangs; keep math + mem_efficient.
            _try('enable_flash_sdp', False)
            _try('enable_mem_efficient_sdp', True)
            _try('enable_math_sdp', True)
            _try('enable_cudnn_sdp', False)
            print("[SDP] ROCm detected; disabling flash-SDPA, keeping math + mem_efficient")
        else:
            print("[SDP] NVIDIA/CUDA — leaving all SDPA backends enabled")
        return
    if choice == 'math':
        _try('enable_flash_sdp', False)
        _try('enable_mem_efficient_sdp', False)
        _try('enable_math_sdp', True)
        _try('enable_cudnn_sdp', False)
        print("[SDP] forcing math-only SDPA")
        return
    if choice == 'mem_efficient':
        _try('enable_flash_sdp', False)
        _try('enable_mem_efficient_sdp', True)
        _try('enable_math_sdp', False)
        _try('enable_cudnn_sdp', False)
        print("[SDP] forcing mem_efficient SDPA")
        return
    if choice == 'flash':
        _try('enable_flash_sdp', True)
        _try('enable_mem_efficient_sdp', False)
        _try('enable_math_sdp', False)
        _try('enable_cudnn_sdp', False)
        print("[SDP] forcing flash SDPA (may hang on ROCm!)")
        return


def run_inference(params):
    """Main inference function that runs both SD 1.5 and SD3"""
    
    print("=" * 60)
    print("STABLE DIFFUSION COMBINED BENCHMARK")
    print("=" * 60)
    print(f"Running benchmarks for both Stable Diffusion models")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Image size: {params.height}x{params.width}")
    print(f"Inference steps: {params.num_inference_steps}")
    print(f"Number of runs per model: {params.num_runs}")
    
    # Configure SDPA backend before any pipeline work touches attention.
    sdp_choice = getattr(params, 'sdp_backend', 'safe')
    _configure_sdp_backend(sdp_choice)
    
    # Set device
    if params.device == 'auto':
        device = get_device()
    else:
        device = torch.device(params.device)
    
    print_device_info()
    print(f"Using device: {device}")
    
    # Get model configurations
    model_configs = get_model_configs()
    
    # If specific model is requested, filter to that model
    if hasattr(params, 'model') and params.model:
        # Map model names to configs
        model_name_mapping = {
            'stable_diffusion_1_5': 'stable_diffusion_1_5',
            'sd1.5': 'stable_diffusion_1_5', 
            'sd15': 'stable_diffusion_1_5',
            'stable_diffusion_3_medium': 'stable_diffusion_3_medium',
            'sd3_medium': 'stable_diffusion_3_medium',
            'sd3': 'stable_diffusion_3_medium',
            'stable_diffusion_3_5_medium': 'stable_diffusion_3_5_medium',
            'sd3.5_medium': 'stable_diffusion_3_5_medium',
            'sd35_medium': 'stable_diffusion_3_5_medium',
            'sd3.5': 'stable_diffusion_3_5_medium',
            'stable_diffusion_3_5_large_turbo': 'stable_diffusion_3_5_large_turbo',
            'sd3.5_turbo': 'stable_diffusion_3_5_large_turbo',
            'sd35_turbo': 'stable_diffusion_3_5_large_turbo',
            'flux_1_schnell': 'flux_1_schnell',
            'flux1_schnell': 'flux_1_schnell',
            'flux_schnell': 'flux_1_schnell',
            'flux.1-schnell': 'flux_1_schnell',
            'flux_1_dev': 'flux_1_dev',
            'flux1_dev': 'flux_1_dev',
            'flux_dev': 'flux_1_dev',
            'flux.1-dev': 'flux_1_dev'
        }
        
        target_model = model_name_mapping.get(params.model.lower())
        if target_model:
            model_configs = [config for config in model_configs if config['name'] == target_model]
        else:
            print(f"Warning: Unknown model '{params.model}', running all models")
    
    all_results = []
    
    # Run benchmarks for each model
    for i, model_config in enumerate(model_configs):
        try:
            print(f"\n{'='*60}")
            print(f"STARTING MODEL {i+1}/{len(model_configs)}: {model_config['display_name']}")
            print(f"{'='*60}")
            
            # Create a copy of params for this model to avoid modifying the original
            import copy
            model_params = copy.copy(params)
            
            # Set model-specific image size if using defaults (512x512)
            if params.height == 512 and params.width == 512:
                height, width = get_default_image_size(model_config['type'])
                model_params.height = height
                model_params.width = width
                print(f"Using default image size {height}x{width} for {model_config['display_name']}")
            
            # Set model-specific inference steps if using defaults (20)
            if params.num_inference_steps == 20:
                model_params.num_inference_steps = get_default_inference_steps(model_config['type'])
                print(f"Using default inference steps {model_params.num_inference_steps} for {model_config['display_name']}")
            
            result = run_single_model_benchmark(model_config, model_params, device)
            all_results.append(result)
            
        except Exception as e:
            print(f"Error benchmarking {model_config['display_name']}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add failed result
            all_results.append({
                'model': model_config['display_name'],
                'model_type': model_config['type'],
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        if 'status' in result and result['status'] == 'FAILED':
            print(f"[FAIL] {result['model']}: FAILED - {result['error']}")
        else:
            memory_str = f"{result['memory_usage_gb']:.1f} GB" if result['memory_usage_gb'] is not None else "N/A"
            print(f"[PASS] {result['model']}: {result['avg_images_per_second']:.2f} images/sec, {memory_str} VRAM")
    
    print(f"{'='*60}")
    print("Benchmark completed!")
    
    return all_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Combined Stable Diffusion Inference Benchmark')
    
    # Model selection (optional - if not specified, runs all models)
    parser.add_argument('--model', type=str, default=None,
                        choices=['stable_diffusion_1_5', 'sd1.5', 'sd15', 
                                'stable_diffusion_3_medium', 'sd3_medium', 'sd3',
                                'stable_diffusion_3_5_medium', 'sd3.5_medium', 'sd35_medium', 'sd3.5',
                                'stable_diffusion_3_5_large_turbo', 'sd3.5_turbo', 'sd35_turbo',
                                'flux_1_schnell', 'flux1_schnell', 'flux_schnell', 'flux.1-schnell',
                            'flux_1_dev', 'flux1_dev', 'flux_dev', 'flux.1-dev'],
                        help='Specific model to benchmark (default: run all models)')
    
    # Precision settings
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'mixed'],
                        help='Precision mode (default: fp16). Note: for SD3 / '
                             'SD3.5 / FLUX the "fp16" slot actually uses bf16 '
                             'under the hood — their MMDiT attention overflows '
                             'fp16. Use --force-fp16 to override.')
    parser.add_argument('--force-fp16', action='store_true',
                        help='Force actual torch.float16 for SD3 / SD3.5 / '
                             'FLUX when --precision fp16 or mixed is chosen. '
                             'Useful only for fp16-vs-bf16 A/B comparison; '
                             'expect NaN / black outputs on some prompts.')
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--height', type=int, default=512,
                        help='Image height (default: 512 for SD1.5, 1024 for SD3+)')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width (default: 512 for SD1.5, 1024 for SD3+)')
    parser.add_argument('--num-inference-steps', type=int, default=20,
                        help='Number of inference steps (default: 20 for SD1.5, 28 for SD3, 4 for SD3.5 Turbo)')
    parser.add_argument('--guidance-scale', type=float, default=4.5,
                        help='Guidance scale for SD3 (default: 4.5)')
    
    # Benchmark settings
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of benchmark runs (default: 3). Each run executes '
                             'num_inference_steps denoising steps, which can take minutes '
                             'on slow backends; bump for tighter percentiles.')
    parser.add_argument('--num-warmup', type=int, default=1,
                        help='Number of warmup runs before timing (default: 1). A single '
                             'full-shape warmup is enough to prime cuDNN/MIOpen autotune.')
    
    # Memory optimization
    parser.add_argument('--cpu-offload', action='store_true',
                        help='Enable CPU offload for SD3 (saves GPU memory)')
    
    # Output settings (images are automatically saved to benchmark_results/images/)
    parser.add_argument('--save-images', action='store_true',
                        help='Legacy flag - images are now automatically saved to benchmark_results/images/')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Legacy option - images are automatically saved to benchmark_results/images/')
    parser.add_argument('--custom-prompt', type=str, default=None,
                        help='Custom prompt for generation (default: use test prompt)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--sdp-backend', type=str, default='safe',
                        choices=['auto', 'safe', 'math', 'mem_efficient', 'flash'],
                        help='Scaled-dot-product-attention backend. '
                             '"safe" (default): disables flash on ROCm (known to '
                             'hang on long-seq MMDiT attention) but keeps it on '
                             'NVIDIA. "auto": PyTorch defaults. "math" / '
                             '"mem_efficient" / "flash": force a specific kernel.')
    
    args = parser.parse_args()
    
    try:
        results = run_inference(args)
        print("\nBenchmark completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 