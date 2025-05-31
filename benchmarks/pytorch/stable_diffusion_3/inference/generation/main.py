#!/usr/bin/env python3
"""
PyTorch Stable Diffusion 3 Medium Inference Benchmark
Supports fp32, fp16, and mixed precision with configurable parameters
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

# Add utils to path for shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'utils'))

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
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def synchronize():
    """Synchronize device operations"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_test_prompts():
    """Get a list of test prompts for SD3"""
    return [
        "A cat holding a sign that says hello world",
        "A beautiful landscape with mountains and a lake at sunset",
        "A futuristic city with flying cars and neon lights",
        "A portrait of a person with intricate details",
        "An abstract art piece with vibrant colors"
    ]

def save_images(images, output_dir, prefix="generated"):
    """Save generated images to disk"""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, image in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths

def run_inference(params):
    """Main inference function"""
    
    # Map model names to Hugging Face model IDs
    model_mapping = {
        "stable_diffusion_3_medium": "stabilityai/stable-diffusion-3-medium-diffusers",
        "sd3_medium": "stabilityai/stable-diffusion-3-medium-diffusers",
        "sd3": "stabilityai/stable-diffusion-3-medium-diffusers"
    }
    
    model_name = params.model.lower()
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {params.model}. Available models: {list(model_mapping.keys())}")
    
    model_id = model_mapping[model_name]
    
    print(f"Running {model_name} Stable Diffusion 3 Medium inference benchmark")
    print(f"Model ID: {model_id}")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Image size: {params.height}x{params.width}")
    print(f"Inference steps: {params.num_inference_steps}")
    print(f"Guidance scale: {params.guidance_scale}")
    
    device = get_device()
    
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"Selected device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {memory_gb:.1f} GB")
    
    print("=" * 50)
    print(f"Using device: {device}")
    
    # Measure initial memory
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # Import diffusers here to avoid import overhead if not needed
    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError:
        raise ImportError("diffusers library is required. Install with: pip install diffusers")
    
    print("Loading Stable Diffusion 3 Medium pipeline...")
    
    # Load the pipeline with appropriate precision
    if params.precision == "fp16":
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="balanced" if device.type == "cuda" else None
        )
    elif params.precision == "mixed":
        # For mixed precision, load in fp32 but use autocast during inference
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
    # When device_map is used, the pipeline handles device placement automatically
    if device.type == "cuda":
        # Check if device_map was actually applied
        try:
            # If device_map is active, this will raise an error
            pipeline = pipeline.to(device)
        except ValueError as e:
            if "device mapping strategy" in str(e):
                print("✓ Using device mapping strategy, skipping manual device placement")
            else:
                raise e
    
    # Enable memory efficient attention if available
    try:
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            pipeline.enable_xformers_memory_efficient_attention()
            print("✓ Enabled xformers memory efficient attention")
    except Exception as e:
        print(f"Note: Could not enable memory efficient attention: {e}")
    
    # Try to enable model CPU offload for memory efficiency
    try:
        if hasattr(pipeline, 'enable_model_cpu_offload') and params.cpu_offload:
            pipeline.enable_model_cpu_offload()
            print("✓ Enabled model CPU offload")
    except Exception as e:
        print(f"Note: Could not enable model CPU offload: {e}")

    # Measure memory after loading
    warmup_memory = get_gpu_memory_nvidia_smi()
    
    # Get test prompts
    test_prompts = get_test_prompts()
    
    # Use the first prompt for benchmarking (SD3's example prompt)
    prompt = test_prompts[0] if not params.custom_prompt else params.custom_prompt
    print(f"Test prompt: '{prompt}'")
    
    # Warm-up runs
    print("Performing warm-up runs...")
    for i in range(2):
        print(f"Warm-up {i+1}/2...")
        with torch.no_grad():
            if params.precision == "mixed" and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = pipeline(
                        prompt,
                        negative_prompt=params.negative_prompt,
                        height=params.height,
                        width=params.width,
                        num_inference_steps=params.num_inference_steps,
                        num_images_per_prompt=1,
                        guidance_scale=params.guidance_scale
                    )
            else:
                _ = pipeline(
                    prompt,
                    negative_prompt=params.negative_prompt,
                    height=params.height,
                    width=params.width,
                    num_inference_steps=params.num_inference_steps,
                    num_images_per_prompt=1,
                    guidance_scale=params.guidance_scale
                )
        synchronize()
    
    # Benchmark runs
    print("Running benchmark...")
    latencies = []
    num_runs = params.num_runs
    generated_images = []
    
    for i in range(num_runs):
        synchronize()
        start = time.time()
        
        with torch.no_grad():
            if params.precision == "mixed" and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    result = pipeline(
                        prompt,
                        negative_prompt=params.negative_prompt,
                        height=params.height,
                        width=params.width,
                        num_inference_steps=params.num_inference_steps,
                        num_images_per_prompt=params.batch_size,
                        guidance_scale=params.guidance_scale
                    )
            else:
                result = pipeline(
                    prompt,
                    negative_prompt=params.negative_prompt,
                    height=params.height,
                    width=params.width,
                    num_inference_steps=params.num_inference_steps,
                    num_images_per_prompt=params.batch_size,
                    guidance_scale=params.guidance_scale
                )
        
        synchronize()
        latency = time.time() - start
        latencies.append(latency)
        
        # Store images from first run for saving
        if i == 0:
            generated_images = result.images
        
        print(f"Run {i+1}/{num_runs}: {latency:.2f} seconds")
    
    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = np.std(latencies)
    
    # Calculate throughput (images per second)
    images_per_run = params.batch_size
    throughput = images_per_run / avg_latency
    
    # Calculate per-image latency
    per_image_latency = avg_latency / images_per_run  # seconds per image
    
    # Measure final memory usage
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Save generated images
    if generated_images and params.save_images:
        output_dir = f"generated_images_{model_name.replace('_', '-')}"
        saved_paths = save_images(generated_images, output_dir, f"{model_name}_benchmark")
        print(f"Generated images saved to: {output_dir}")
    
    # Print results in a format that can be parsed by the main benchmark framework
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Image size: {params.height}x{params.width}")
    print(f"Inference steps: {params.num_inference_steps}")
    print(f"Guidance scale: {params.guidance_scale}")
    print(f"Average Generation Time: {avg_latency:.2f} seconds")
    print(f"Per-image Latency: {per_image_latency:.2f} seconds/image")
    print(f"Per-sample Latency: {per_image_latency*1000:.2f} ms/sample")
    print(f"Min Generation Time: {min_latency:.2f} seconds")
    print(f"Max Generation Time: {max_latency:.2f} seconds")
    print(f"Std Generation Time: {std_latency:.2f} seconds")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    # Memory information
    if final_memory:
        print(f"Total GPU Memory Used: {final_memory.get('total_gpu_used_gb', 0):.3f} GB")
        print(f"Total GPU Memory Available: {final_memory.get('total_gpu_total_gb', 0):.3f} GB")
        print(f"GPU Memory Utilization: {final_memory.get('gpu_utilization_percent', 0):.1f}%")
    
    print(f"PyTorch Inference Time = {avg_latency*1000:.2f} ms")
    
    return {
        "avg_latency_ms": avg_latency * 1000,
        "per_image_latency_seconds": per_image_latency,
        "min_latency_ms": min_latency * 1000,
        "max_latency_ms": max_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "throughput_fps": throughput,
        "images_per_sec": throughput,
        "device": str(device),
        "framework": "PyTorch",
        "model_id": model_id,
        "image_size": f"{params.height}x{params.width}",
        "inference_steps": params.num_inference_steps,
        "guidance_scale": params.guidance_scale,
        "memory_usage": final_memory,
        "generated_images": len(generated_images) if generated_images else 0
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PyTorch Stable Diffusion 3 Medium Inference Benchmark")
    parser.add_argument("--model", type=str, default="stable_diffusion_3_medium", 
                       help="Model name for Stable Diffusion 3")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
                       help="Precision for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Number of images to generate per batch")
    parser.add_argument("--height", type=int, default=1024,
                       help="Height of generated images (SD3 default: 1024)")
    parser.add_argument("--width", type=int, default=1024,
                       help="Width of generated images (SD3 default: 1024)")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                       help="Number of denoising steps (SD3 default: 28)")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                       help="Guidance scale for classifier-free guidance (SD3 default: 7.0)")
    parser.add_argument("--negative_prompt", type=str, default="",
                       help="Negative prompt for generation")
    parser.add_argument("--custom_prompt", type=str, default="",
                       help="Custom prompt for generation (overrides default)")
    parser.add_argument("--num_runs", type=int, default=3,
                       help="Number of benchmark runs")
    parser.add_argument("--save_images", action="store_true",
                       help="Save generated images to disk")
    parser.add_argument("--cpu_offload", action="store_true",
                       help="Enable model CPU offload for memory efficiency")
    args = parser.parse_args()

    try:
        results = run_inference(args)
        print("Benchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 