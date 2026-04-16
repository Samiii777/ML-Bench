import torch
from torchvision import transforms
from PIL import Image
import urllib.request
import os
import argparse
import time
import sys
import numpy as np

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
from utils.download import get_imagenet_classes_path, get_sample_image_path
from utils.safe_print import safe_print, format_success_message

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

# Simple download utilities
def download_file(url, filename):
    """Download file if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        safe_print(format_success_message(f"{filename} downloaded"))
    else:
        safe_print(format_success_message(f"{filename} already exists"))

def get_imagenet_classes_path():
    """Get path to ImageNet classes file"""
    # Use the clean utils function instead of ugly relative paths
    from utils.download import get_imagenet_classes_path as utils_get_path
    return utils_get_path()

def get_sample_image_path():
    """Get path to sample image"""
    # Use the clean utils function instead of ugly relative paths  
    from utils.download import get_sample_image_path as utils_get_path
    return utils_get_path()

def load_categories(filename):
    """Load the categories from the given file"""
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def preprocess_image(image_path, batch_size=1):
    """Preprocess the input image for InceptionV3 inference (299x299 input size)"""
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(342),  # InceptionV3 uses larger resize
        transforms.CenterCrop(299),  # InceptionV3 uses 299x299 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    
    # Create batch
    if batch_size > 1:
        input_batch = input_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        input_batch = input_tensor.unsqueeze(0)
    
    return input_batch

def run_inference(params):
    """Main inference function"""
    model_weights = {
        'inceptionv3': "Inception_V3_Weights.DEFAULT",
        'inception_v3': "Inception_V3_Weights.DEFAULT",
    }
    
    inception_model = params.model.lower() if params.model else "inceptionv3"
    
    if inception_model not in model_weights:
        raise ValueError(f"Unsupported model: {inception_model}. Supported models: {list(model_weights.keys())}")
    
    print(f"Running {inception_model} inference benchmark")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    
    # Download required files
    classes_file = get_imagenet_classes_path()
    image_file = get_sample_image_path()
    
    categories = load_categories(classes_file)
    print(f"Loading model: {inception_model}")
    
    # Load model - InceptionV3 from torchvision
    from torchvision import models
    if inception_model in ['inceptionv3', 'inception_v3']:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.eval()
    
    # Load and preprocess input image
    input_batch = preprocess_image(image_file, params.batch_size)
    
    # Set device
    if params.device == 'auto':
        device = get_device()
    else:
        device = torch.device(params.device)
    
    print_device_info()
    print(f"Using device: {device}")
    
    input_batch = input_batch.to(device)
    model.to(device)
    
    # Measure initial memory usage
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # Apply precision settings
    use_mixed_precision = False
    if params.precision == "fp16":
        if device.type == "cuda" or device.type == "mps":
            model = model.half()
            input_batch = input_batch.half()
        elif device.type == "cpu":
            print("Warning: FP16 on CPU may be slower than FP32, but proceeding as requested...")
            model = model.half()
            input_batch = input_batch.half()
        else:
            model = model.half()
            input_batch = input_batch.half()
    elif params.precision == "mixed":
        if device.type == "cuda":
            use_mixed_precision = True
            print("Using mixed precision (AMP)")
        else:
            print("Warning: Mixed precision not supported on CPU, using FP32")
    elif params.precision == "int8":
        print("Warning: INT8 quantization not implemented in this benchmark")
    
    # Import shared benchmark utilities
    from utils.benchmark_utils import (
        benchmark_loop, warmup as warmup_fn, compute_stats, setup_torch_backends,
        reset_memory_tracking, measure_peak_memory
    )
    setup_torch_backends(cudnn_benchmark=True)
    
    # Define one inference step
    def inference_step():
        with torch.inference_mode():
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    return model(input_batch)
            else:
                return model(input_batch)
    
    # Warmup
    warmup_fn(inference_step, num_warmup=params.num_warmup, device=device)
    
    # Reset peak memory after warmup for clean benchmark-only measurement
    reset_memory_tracking(device)
    
    # Benchmark runs with CUDA Events timing and GC disabled
    latencies_ms = benchmark_loop(inference_step, num_runs=params.num_runs, device=device)
    
    # Measure peak memory (only covers the benchmark runs, not warmup)
    peak_mem = measure_peak_memory(device)
    
    # Also get nvidia-smi as secondary reference
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Compute comprehensive statistics
    stats = compute_stats(latencies_ms)
    avg_latency = stats["mean"] / 1000.0  # seconds for backward compat
    std_latency = stats["std"] / 1000.0
    min_latency = stats["min"] / 1000.0
    max_latency = stats["max"] / 1000.0
    
    throughput = (params.batch_size / stats["mean"]) * 1000.0
    
    memory_used_gb = peak_mem.get('peak_allocated_gb', 0.0) if peak_mem else 0.0
    
    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    
    # Run one extra forward pass to get predictions for display
    output = inference_step()
    probabilities = torch.nn.functional.softmax(output[0].float(), dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Print results
    print(f"\n=== {inception_model.upper()} INFERENCE BENCHMARK RESULTS ===")
    print(f"Framework: PyTorch")
    print(f"Model: {inception_model}")
    print(f"Device: {device}")
    print(f"Precision: {params.precision}")
    print(f"Batch Size: {params.batch_size}")
    print(f"Input Shape: {list(input_batch.shape)}")
    print(f"Model Parameters: {total_params:,}")
    print(f"Mixed Precision: {'Enabled' if use_mixed_precision else 'Disabled'}")
    if peak_mem:
        print(f"GPU Memory Allocated: {memory_used_gb:.3f} GB")
        print(f"GPU Memory Cached: {peak_mem.get('peak_reserved_gb', 0):.3f} GB")
    print()
    print("Performance Metrics:")
    print(f"Average Inference Time: {stats['mean']:.2f} ms")
    print(f"Median Inference Time: {stats['median']:.2f} ms")
    print(f"P90 Inference Time: {stats['p90']:.2f} ms")
    print(f"P95 Inference Time: {stats['p95']:.2f} ms")
    print(f"P99 Inference Time: {stats['p99']:.2f} ms")
    print(f"Min Inference Time: {stats['min']:.2f} ms")
    print(f"Max Inference Time: {stats['max']:.2f} ms")
    print(f"Std Inference Time: {stats['std']:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print()
    print("Top 5 Predictions:")
    for i in range(min(5, len(top5_catid))):
        print(f"{i+1}: {categories[top5_catid[i]]} ({top5_prob[i]*100:.2f}%)")
    print(f"PyTorch Inference Time = {stats['mean']:.2f} ms")
    print("=" * 60)
    
    # Print final result in expected format
    print(f"\nFINAL RESULT: {throughput:.2f} samples/sec")
    
    return {
        'throughput_fps': throughput,
        'avg_latency_ms': stats['mean'],
        'std_latency_ms': stats['std'],
        'min_latency_ms': stats['min'],
        'max_latency_ms': stats['max'],
        'median_latency_ms': stats['median'],
        'p90_latency_ms': stats['p90'],
        'p95_latency_ms': stats['p95'],
        'p99_latency_ms': stats['p99'],
        'memory_used_gb': memory_used_gb,
        'total_params': total_params
    }

def main():
    parser = argparse.ArgumentParser(description='PyTorch InceptionV3 Inference Benchmark')
    parser.add_argument('--model', type=str, default='inceptionv3',
                       choices=['inceptionv3', 'inception_v3'],
                       help='InceptionV3 model variant')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'mixed', 'int8'],
                       help='Inference precision')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--num_warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Run benchmark
    results = run_inference(args)

if __name__ == "__main__":
    main() 