import torch
from diffusers import StableDiffusionPipeline
import argparse
import time
import sys
import os
import numpy as np
from PIL import Image

# Simple device utilities - everything in one place
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

def get_gpu_memory_nvidia_smi():
    """Get GPU memory using nvidia-smi directly"""
    try:
        import nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        nvidia_smi.nvmlShutdown()
        
        used_gb = info.used / 1024**3
        total_gb = info.total / 1024**3
        
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

def get_model_id(model_name):
    """Get the Hugging Face model ID for the given model name"""
    model_mapping = {
        'stable_diffusion_1_5': 'runwayml/stable-diffusion-v1-5',
        'sd1.5': 'runwayml/stable-diffusion-v1-5',
        'sd15': 'runwayml/stable-diffusion-v1-5',
    }
    return model_mapping.get(model_name.lower(), 'runwayml/stable-diffusion-v1-5')

def get_test_prompts():
    """Get a set of test prompts for benchmarking"""
    return [
        "A beautiful landscape with mountains and a lake at sunset",
        "A cute cat sitting on a windowsill",
        "A futuristic city with flying cars",
        "A portrait of a person with blue eyes",
        "A colorful abstract painting"
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
        
        filename = f"{prefix}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        saved_paths.append(filepath)
        print(f"✓ Saved: {filepath}")
    
    return saved_paths

def run_inference(params):
    """Main inference function"""
    model_name = params.model.lower() if params.model else "stable_diffusion_1_5"
    model_id = get_model_id(model_name)
    
    print(f"Running {model_name} Stable Diffusion inference benchmark")
    print(f"Model ID: {model_id}")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Image size: {params.height}x{params.width}")
    print(f"Inference steps: {params.num_inference_steps}")
    
    # Get device and print info
    device = get_device()
    print_device_info()
    print(f"Using device: {device}")
    
    # Measure initial memory usage
    initial_memory = get_gpu_memory_nvidia_smi()
    
    print("Loading Stable Diffusion pipeline...")
    
    # Load the pipeline with appropriate precision
    if params.precision == "fp16" and device.type == "cuda":
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
    elif params.precision == "mixed" and device.type == "cuda":
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Load in FP32 for true mixed precision
            safety_checker=None,
            requires_safety_checker=False
        )
        # Enable memory efficient attention
        try:
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
        except Exception as e:
            print(f"Note: Could not enable attention slicing: {e}")
        
        try:
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                pipeline.enable_xformers_memory_efficient_attention()
            elif hasattr(pipeline, 'enable_memory_efficient_attention'):
                pipeline.enable_memory_efficient_attention()
        except Exception as e:
            print(f"Note: Could not enable memory efficient attention: {e}")
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            requires_safety_checker=False
        )
        if params.precision == "fp16" and device.type != "cuda":
            print("Warning: FP16 not supported on CPU, using FP32")
        if params.precision == "mixed" and device.type != "cuda":
            print("Warning: Mixed precision not supported on CPU, using FP32")
    
    # Move pipeline to device
    pipeline = pipeline.to(device)
    
    # Enable optimizations
    if device.type == "cuda":
        # Try to enable memory efficient attention (API changed in newer diffusers)
        try:
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                pipeline.enable_xformers_memory_efficient_attention()
            elif hasattr(pipeline, 'enable_memory_efficient_attention'):
                pipeline.enable_memory_efficient_attention()
        except Exception as e:
            print(f"Note: Could not enable memory efficient attention: {e}")
        
        # Try to enable attention slicing
        try:
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
        except Exception as e:
            print(f"Note: Could not enable attention slicing: {e}")
    
    # Measure memory after loading
    warmup_memory = get_gpu_memory_nvidia_smi()
    
    # Get test prompts
    test_prompts = get_test_prompts()
    
    # Use the first prompt for benchmarking
    prompt = test_prompts[0]
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
                        height=params.height,
                        width=params.width,
                        num_inference_steps=params.num_inference_steps,
                        num_images_per_prompt=1,
                        guidance_scale=7.5
                    )
            else:
                _ = pipeline(
                    prompt,
                    height=params.height,
                    width=params.width,
                    num_inference_steps=params.num_inference_steps,
                    num_images_per_prompt=1,
                    guidance_scale=7.5
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
                        height=params.height,
                        width=params.width,
                        num_inference_steps=params.num_inference_steps,
                        num_images_per_prompt=params.batch_size,
                        guidance_scale=7.5
                    )
            else:
                result = pipeline(
                    prompt,
                    height=params.height,
                    width=params.width,
                    num_inference_steps=params.num_inference_steps,
                    num_images_per_prompt=params.batch_size,
                    guidance_scale=7.5
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
        output_dir = f"generated_images_{model_name}"
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
        "memory_usage": final_memory,
        "generated_images": len(generated_images) if generated_images else 0
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PyTorch Stable Diffusion Inference Benchmark")
    parser.add_argument("--model", type=str, default="stable_diffusion_1_5", 
                       help="Model name for Stable Diffusion")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
                       help="Precision for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Number of images to generate per batch")
    parser.add_argument("--height", type=int, default=512,
                       help="Height of generated images")
    parser.add_argument("--width", type=int, default=512,
                       help="Width of generated images")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                       help="Number of denoising steps")
    parser.add_argument("--num_runs", type=int, default=3,
                       help="Number of benchmark runs")
    parser.add_argument("--save_images", action="store_true",
                       help="Save generated images to disk")
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