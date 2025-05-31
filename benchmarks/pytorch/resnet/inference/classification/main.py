import torch
from torchvision import transforms
from PIL import Image
import urllib.request
import os
import argparse
import time
import sys
import numpy as np

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
        print(f"✓ {filename} downloaded")
    else:
        print(f"✓ {filename} already exists")

def get_imagenet_classes_path():
    """Get path to ImageNet classes file"""
    # Use shared data directory at project root
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data")
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    filename = os.path.join(data_dir, "imagenet_classes.txt")
    download_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", filename)
    return filename

def get_sample_image_path():
    """Get path to sample image"""
    # Use shared data directory at project root
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data")
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    filename = os.path.join(data_dir, "dog.jpg")
    download_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", filename)
    return filename

def load_categories(filename):
    """Load the categories from the given file"""
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def preprocess_image(image_path, batch_size=1):
    """Preprocess the input image for inference"""
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
        'resnet18': "ResNet18_Weights.DEFAULT",
        'resnet34': "ResNet34_Weights.DEFAULT", 
        'resnet50': "ResNet50_Weights.DEFAULT",
        'resnet101': "ResNet101_Weights.DEFAULT",
        'resnet152': "ResNet152_Weights.DEFAULT",
    }
    
    resnet_model = params.model.lower() if params.model else "resnet50"
    
    if resnet_model not in model_weights:
        raise ValueError(f"Unsupported model: {resnet_model}. Supported models: {list(model_weights.keys())}")
    
    print(f"Running {resnet_model} inference benchmark")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    
    # Download required files
    classes_file = get_imagenet_classes_path()
    image_file = get_sample_image_path()
    
    categories = load_categories(classes_file)
    print(f"Loading model: {resnet_model}")
    
    # Load model
    model = torch.hub.load('pytorch/vision:v0.10.0', resnet_model, weights=model_weights[resnet_model])
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
    
    # Warm-up runs
    print("Performing warm-up runs...")
    with torch.no_grad():
        for _ in range(3):
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model(input_batch)
            else:
                output = model(input_batch)
            synchronize()
    
    # Measure memory after warmup
    warmup_memory = get_gpu_memory_nvidia_smi()
    
    # Benchmark runs
    print("Running benchmark...")
    latencies = []
    num_runs = 10
    
    for i in range(num_runs):
        synchronize()
        start = time.time()
        with torch.no_grad():
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model(input_batch)
            else:
                output = model(input_batch)
        synchronize()
        latency = time.time() - start
        latencies.append(latency)
        print(f"Run {i+1}/{num_runs}: {latency*1000:.2f} ms")
    
    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = np.std(latencies)
    
    # Calculate throughput (samples per second)
    throughput = params.batch_size / avg_latency
    
    # Calculate per-sample latency
    per_sample_latency = avg_latency * 1000 / params.batch_size  # ms per sample
    
    # Measure final memory usage
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Get predictions for the first image in batch (ensure output is in FP32 for softmax)
    if use_mixed_precision or params.precision == "fp16":
        output_fp32 = output.float()
    else:
        output_fp32 = output
    
    probabilities = torch.nn.functional.softmax(output_fp32[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    print("\nTop 5 predictions:")
    for i in range(top5_prob.size(0)):
        print(f"{categories[top5_catid[i]]}: {top5_prob[i].item():.4f}")
    
    # Print results in a format that can be parsed by the main benchmark framework
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Model: {resnet_model}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Average Inference Time: {avg_latency*1000:.2f} ms")
    print(f"Per-sample Latency: {per_sample_latency:.2f} ms/sample")
    print(f"Min Inference Time: {min_latency*1000:.2f} ms")
    print(f"Max Inference Time: {max_latency*1000:.2f} ms")
    print(f"Std Inference Time: {std_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    # Memory information
    if final_memory:
        print(f"Total GPU Memory Used: {final_memory.get('total_gpu_used_gb', 0):.3f} GB")
        print(f"Total GPU Memory Available: {final_memory.get('total_gpu_total_gb', 0):.3f} GB")
        print(f"GPU Memory Utilization: {final_memory.get('gpu_utilization_percent', 0):.1f}%")
    
    print(f"PyTorch Inference Time = {avg_latency*1000:.2f} ms")
    
    return {
        "avg_latency_ms": avg_latency * 1000,
        "per_sample_latency_ms": per_sample_latency,
        "min_latency_ms": min_latency * 1000,
        "max_latency_ms": max_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "throughput_fps": throughput,
        "samples_per_sec": throughput,
        "device": str(device),
        "framework": "PyTorch",
        "top1_prediction": categories[top5_catid[0]],
        "top1_confidence": top5_prob[0].item(),
        "memory_usage": final_memory
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PyTorch ResNet Inference Benchmark")
    parser.add_argument("--model", type=str, default="resnet50", 
                       help="Model ID for the RESNET Pipeline")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed", "int8"],
                       help="Precision for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    args = parser.parse_args()

    try:
        results = run_inference(args)
        print("Benchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
