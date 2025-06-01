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
    """Get configurations for both SD models"""
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
        }
    ]

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
        print(f"✓ Saved: {filepath}")
    
    return saved_paths

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

def load_sd3_pipeline(model_id, precision, device, cpu_offload=False):
    """Load Stable Diffusion 3 pipeline"""
    from diffusers import StableDiffusion3Pipeline
    
    if precision == "fp16":
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="balanced" if device.type == "cuda" else None
        )
    elif precision == "mixed":
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="balanced" if device.type == "cuda" else None
        )
    else:  # fp32
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="balanced" if device.type == "cuda" else None
        )
    
    # Move to device if not using device_map
    if device.type == "cuda":
        try:
            pipeline = pipeline.to(device)
        except ValueError as e:
            if "device mapping strategy" in str(e):
                print("✓ Using device mapping strategy, skipping manual device placement")
            else:
                raise e
    
    # Enable optimizations for SD3
    try:
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            pipeline.enable_xformers_memory_efficient_attention()
            print("✓ Enabled xformers memory efficient attention")
    except Exception as e:
        print(f"Note: Could not enable memory efficient attention: {e}")
    
    try:
        if hasattr(pipeline, 'enable_model_cpu_offload') and cpu_offload:
            pipeline.enable_model_cpu_offload()
            print("✓ Enabled model CPU offload")
    except Exception as e:
        print(f"Note: Could not enable model CPU offload: {e}")
    
    return pipeline

def run_single_model_benchmark(model_config, params):
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
    
    device = get_device()
    
    # Measure initial memory usage
    initial_memory = get_gpu_memory_nvidia_smi()
    
    print(f"Loading {display_name} pipeline...")
    
    # Load the appropriate pipeline based on model type
    if model_type == 'sd15':
        pipeline = load_sd15_pipeline(model_id, params.precision, device)
    elif model_type == 'sd3':
        pipeline = load_sd3_pipeline(model_id, params.precision, device, params.cpu_offload)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Measure memory after loading
    warmup_memory = get_gpu_memory_nvidia_smi()
    
    # Get test prompts
    test_prompts = get_test_prompts()
    prompt = test_prompts[0] if not params.custom_prompt else params.custom_prompt
    print(f"Test prompt: '{prompt}'")
    
    # Warm-up runs
    print("Performing warm-up runs...")
    for i in range(2):
        print(f"Warm-up {i+1}/2...")
        with torch.no_grad():
            if params.precision == "mixed" and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = pipeline(
                        prompt,
                        height=params.height,
                        width=params.width,
                        num_inference_steps=10,  # Fewer steps for warmup
                        guidance_scale=params.guidance_scale if model_type == 'sd3' else 7.5,
                        num_images_per_prompt=1
                    ).images
            else:
                generation_kwargs = {
                    'prompt': prompt,
                    'height': params.height,
                    'width': params.width,
                    'num_inference_steps': 10,  # Fewer steps for warmup
                    'num_images_per_prompt': 1
                }
                
                # Add guidance_scale only for SD3
                if model_type == 'sd3':
                    generation_kwargs['guidance_scale'] = params.guidance_scale
                else:
                    generation_kwargs['guidance_scale'] = 7.5
                
                _ = pipeline(**generation_kwargs).images
        
        synchronize()
    
    print("Warm-up completed. Starting benchmark...")
    
    # Benchmark runs
    times = []
    all_images = []
    
    for run in range(params.num_runs):
        print(f"Benchmark run {run + 1}/{params.num_runs}")
        
        # Prepare batch prompts
        batch_prompts = [prompt] * params.batch_size
        
        synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            if params.precision == "mixed" and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    generation_kwargs = {
                        'prompt': batch_prompts,
                        'height': params.height,
                        'width': params.width,
                        'num_inference_steps': params.num_inference_steps,
                        'num_images_per_prompt': 1
                    }
                    
                    # Add guidance_scale only for SD3
                    if model_type == 'sd3':
                        generation_kwargs['guidance_scale'] = params.guidance_scale
                    else:
                        generation_kwargs['guidance_scale'] = 7.5
                    
                    result = pipeline(**generation_kwargs)
            else:
                generation_kwargs = {
                    'prompt': batch_prompts,
                    'height': params.height,
                    'width': params.width,
                    'num_inference_steps': params.num_inference_steps,
                    'num_images_per_prompt': 1
                }
                
                # Add guidance_scale only for SD3
                if model_type == 'sd3':
                    generation_kwargs['guidance_scale'] = params.guidance_scale
                else:
                    generation_kwargs['guidance_scale'] = 7.5
                
                result = pipeline(**generation_kwargs)
        
        synchronize()
        end_time = time.time()
        
        run_time = end_time - start_time
        times.append(run_time)
        
        # Store images from first run for saving
        if run == 0:
            all_images = result.images
        
        images_per_second = params.batch_size / run_time
        time_per_image = run_time / params.batch_size
        
        print(f"  Time: {run_time:.2f}s | Images/sec: {images_per_second:.2f} | Time/image: {time_per_image:.2f}s")
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    avg_images_per_second = params.batch_size / avg_time
    avg_time_per_image = avg_time / params.batch_size
    avg_latency_ms = avg_time_per_image * 1000  # Convert to milliseconds
    
    # Measure final memory
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
    if model_type == 'sd3':
        print(f"Guidance scale: {params.guidance_scale}")
    print(f"Number of runs: {params.num_runs}")
    print()
    print(f"Average time per run: {avg_time:.3f} ± {std_time:.3f} seconds")
    print(f"Min time: {min_time:.3f} seconds")
    print(f"Max time: {max_time:.3f} seconds")
    print(f"Average images per second: {avg_images_per_second:.2f}")
    print(f"Average time per image: {avg_time_per_image:.3f} seconds")
    
    # Memory information
    if final_memory:
        print(f"\nMemory usage: {final_memory['total_gpu_used_gb']:.2f} GB")
        print(f"GPU utilization: {final_memory['gpu_utilization_percent']:.1f}%")
    
    # Output in format expected by benchmark framework
    print(f"\n# Benchmark Framework Parseable Output for {display_name}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Throughput: {avg_images_per_second:.2f} samples/sec")
    print(f"Per-sample Latency: {avg_latency_ms:.2f} ms/sample")
    if final_memory:
        print(f"Total GPU Memory Used: {final_memory['total_gpu_used_gb']:.2f} GB")
    print(f"# End Parseable Output for {display_name}")
    
    # Save images if requested
    if params.save_images and all_images:
        print(f"\nSaving {len(all_images)} generated images...")
        output_dir = params.output_dir or f"output_{model_type}_{params.precision}"
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
        'guidance_scale': params.guidance_scale if model_type == 'sd3' else 7.5,
        'num_runs': params.num_runs,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_images_per_second': avg_images_per_second,
        'avg_time_per_image': avg_time_per_image,
        'memory_usage_gb': final_memory['total_gpu_used_gb'] if final_memory else None,
        'gpu_utilization_percent': final_memory['gpu_utilization_percent'] if final_memory else None
    }

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
    
    # Get device and print info
    device = get_device()
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
            'sd3': 'stable_diffusion_3_medium'
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
            
            result = run_single_model_benchmark(model_config, params)
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
            print(f"❌ {result['model']}: FAILED - {result['error']}")
        else:
            print(f"✅ {result['model']}: {result['avg_images_per_second']:.2f} images/sec, {result['memory_usage_gb']:.1f} GB VRAM")
    
    print(f"{'='*60}")
    print("Benchmark completed!")
    
    return all_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Combined Stable Diffusion Inference Benchmark')
    
    # Model selection (optional - if not specified, runs both models)
    parser.add_argument('--model', type=str, default=None,
                        choices=['stable_diffusion_1_5', 'sd1.5', 'sd15', 
                                'stable_diffusion_3_medium', 'sd3_medium', 'sd3'],
                        help='Specific model to benchmark (default: run both models)')
    
    # Precision settings
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'mixed'],
                        help='Precision mode (default: fp16)')
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--height', type=int, default=512,
                        help='Image height (default: 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width (default: 512)')
    parser.add_argument('--num-inference-steps', type=int, default=20,
                        help='Number of inference steps (default: 20)')
    parser.add_argument('--guidance-scale', type=float, default=4.5,
                        help='Guidance scale for SD3 (default: 4.5)')
    
    # Benchmark settings
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of benchmark runs (default: 5)')
    
    # Memory optimization
    parser.add_argument('--cpu-offload', action='store_true',
                        help='Enable CPU offload for SD3 (saves GPU memory)')
    
    # Output settings
    parser.add_argument('--save-images', action='store_true',
                        help='Save generated images to disk')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for images (default: auto-generated)')
    parser.add_argument('--custom-prompt', type=str, default=None,
                        help='Custom prompt for generation (default: use test prompt)')
    
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