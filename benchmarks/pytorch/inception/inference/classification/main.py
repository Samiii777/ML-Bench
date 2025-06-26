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
    
    # Move to device
    device = get_device()
    print_device_info()
    print(f"Using device: {device}")
    
    input_batch = input_batch.to(device)
    model.to(device)
    
    # Measure initial memory usage
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # Apply precision settings
    use_mixed_precision = False
    if params.precision == "fp16":
        if device.type == "cuda":
            model = model.half()
            input_batch = input_batch.half()
        else:
            print("Warning: FP16 not supported on CPU, using FP32")
    elif params.precision == "mixed":
        if device.type == "cuda":
            use_mixed_precision = True
            print("Using mixed precision (AMP)")
        else:
            print("Warning: Mixed precision not supported on CPU, using FP32")
    elif params.precision == "int8":
        print("Warning: INT8 quantization not implemented in this benchmark")
    
    # Warmup runs
    print(f"\nStarting warmup ({params.num_warmup} iterations)...")
    synchronize()
    
    for i in range(params.num_warmup):
        with torch.no_grad():
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    _ = model(input_batch)
            else:
                _ = model(input_batch)
        synchronize()
    
    print(f"Warmup completed")
    
    # Benchmark runs
    print(f"\nStarting benchmark ({params.num_runs} iterations)...")
    synchronize()
    
    latencies = []
    start_time = time.time()
    
    for i in range(params.num_runs):
        iteration_start = time.time()
        
        with torch.no_grad():
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    output = model(input_batch)
            else:
                output = model(input_batch)
        
        synchronize()
        iteration_end = time.time()
        
        latency = iteration_end - iteration_start
        latencies.append(latency)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{params.num_runs} iterations")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # Throughput calculation
    total_samples = params.num_runs * params.batch_size
    throughput = total_samples / total_time
    
    # Memory usage
    final_memory = get_gpu_memory_nvidia_smi()
    memory_used_gb = 0
    if initial_memory and final_memory:
        memory_used_gb = final_memory["total_gpu_used_gb"] - initial_memory["total_gpu_used_gb"]
        memory_used_gb = max(0, memory_used_gb)  # Ensure non-negative
    
    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    
    # Get top predictions for the first sample
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
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
    if memory_used_gb > 0:
        print(f"GPU Memory Used: {memory_used_gb:.2f} GB")
    print()
    print("Performance Metrics:")
    print(f"Average Latency: {avg_latency*1000:.2f} ms")
    print(f"Std Latency: {std_latency*1000:.2f} ms")
    print(f"Min Latency: {min_latency*1000:.2f} ms")
    print(f"Max Latency: {max_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print()
    print("Top 5 Predictions:")
    for i in range(min(5, len(top5_catid))):
        print(f"{i+1}: {categories[top5_catid[i]]} ({top5_prob[i]*100:.2f}%)")
    print("=" * 60)
    
    # Print final result in expected format
    print(f"\nFINAL RESULT: {throughput:.2f} samples/sec")
    
    return {
        'throughput_fps': throughput,
        'avg_latency_ms': avg_latency * 1000,
        'std_latency_ms': std_latency * 1000,
        'min_latency_ms': min_latency * 1000,
        'max_latency_ms': max_latency * 1000,
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