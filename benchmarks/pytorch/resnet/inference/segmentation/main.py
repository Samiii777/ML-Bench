import torch
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from PIL import Image
import numpy as np
import argparse
import time
import sys
import os
import urllib.request

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
from utils.download import get_sample_image_path

# Simple device utilities - everything auto-detects!
def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_segmentation_model(model_name, num_classes=21):
    """Create a semantic segmentation model with ResNet backbone"""
    model_name = model_name.lower()
    
    if "resnet50" in model_name:
        # DeepLabV3 with ResNet-50 backbone
        model = deeplabv3_resnet50(weights='DEFAULT', num_classes=num_classes)
    elif "resnet101" in model_name:
        # DeepLabV3 with ResNet-101 backbone  
        model = deeplabv3_resnet101(weights='DEFAULT', num_classes=num_classes)
    else:
        # Default to ResNet-50 for other ResNet variants
        print(f"Using ResNet-50 backbone for {model_name} (closest available)")
        model = deeplabv3_resnet50(weights='DEFAULT', num_classes=num_classes)
    
    # Always set to eval mode for inference
    model.eval()
    return model

def get_cityscapes_colors():
    """Get color palette for Cityscapes dataset visualization"""
    return [
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk  
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
        [0, 0, 0],        # background
        [255, 255, 255]   # void
    ]

def preprocess_image(image_path, target_size=(520, 520)):
    """Load and preprocess image for segmentation"""
    image = Image.open(image_path).convert('RGB')
    
    # Standard ImageNet preprocessing for segmentation
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)

def run_segmentation(model, image_tensor, device):
    """Run semantic segmentation inference"""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        input_batch = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        output = model(input_batch)
        
        # Get the segmentation mask
        if isinstance(output, dict):
            # DeepLabV3 returns a dictionary with 'out' key
            segmentation_mask = output['out']
        else:
            segmentation_mask = output
        
        # Convert to class predictions
        predictions = torch.argmax(segmentation_mask, dim=1)
        
        return predictions.cpu().numpy()

def colorize_segmentation(seg_mask, colors):
    """Convert segmentation mask to colored image"""
    h, w = seg_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(colors)):
        mask = seg_mask == class_id
        colored[mask] = colors[class_id]
    
    return colored

def benchmark_segmentation_model(model_name, device, batch_size=1, warmup_runs=3, benchmark_runs=10, precision="fp32"):
    """Benchmark semantic segmentation model performance"""
    print(f"Creating {model_name} segmentation model...")
    
    # Create model
    model = create_segmentation_model(model_name)
    model = model.to(device)
    
    # Set precision
    if precision == "fp16" and device.type == "cuda":
        model = model.half()
        print("Using FP16 precision")
    elif precision == "mixed":
        print("Using mixed precision")
    else:
        print("Using FP32 precision")
    
    # Get sample image
    image_path = get_sample_image_path()
    image_tensor = preprocess_image(image_path)
    
    if precision == "fp16" and device.type == "cuda":
        image_tensor = image_tensor.half()
    
    # Create batch
    batch_tensors = [image_tensor for _ in range(batch_size)]
    input_batch = torch.stack(batch_tensors).to(device)
    
    print(f"Input shape: {input_batch.shape}")
    print(f"Running on device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Warmup
    print("Warming up...")
    for i in range(warmup_runs):
        with torch.no_grad():
            if precision == "mixed" and device.type == "cuda":
                with torch.autocast(device_type='cuda'):
                    output = model(input_batch)
            else:
                output = model(input_batch)
    
    # Synchronize before benchmarking
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking...")
    times = []
    
    for i in range(benchmark_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if precision == "mixed" and device.type == "cuda":
                with torch.autocast(device_type='cuda'):
                    output = model(input_batch)
            else:
                output = model(input_batch)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate throughput
    samples_per_second = batch_size / avg_time
    
    # Memory usage
    if device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        memory_cached = torch.cuda.max_memory_reserved() / 1024**3  # GB
    else:
        memory_used = 0
        memory_cached = 0
    
    # Get segmentation results for the first image
    sample_input = image_tensor.unsqueeze(0).to(device)
    if precision == "fp16" and device.type == "cuda":
        sample_input = sample_input.half()
    
    segmentation_result = run_segmentation(model, image_tensor, device)
    unique_classes = np.unique(segmentation_result)
    
    print(f"\n=== {model_name.upper()} SEGMENTATION BENCHMARK RESULTS ===")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Precision: {precision}")
    print(f"Input Resolution: {input_batch.shape[2]}x{input_batch.shape[3]}")
    print(f"Detected Classes: {len(unique_classes)} classes")
    print(f"Average Time: {avg_time*1000:.2f} ms")
    print(f"Std Dev: {std_time*1000:.2f} ms")
    print(f"Min Time: {min_time*1000:.2f} ms") 
    print(f"Max Time: {max_time*1000:.2f} ms")
    print(f"Throughput: {samples_per_second:.2f} samples/sec")
    
    if device.type == "cuda":
        print(f"GPU Memory Used: {memory_used:.2f} GB")
        print(f"GPU Memory Cached: {memory_cached:.2f} GB")
    
    print("="*60)
    
    return {
        'model': model_name,
        'device': str(device),
        'batch_size': batch_size,
        'precision': precision,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'samples_per_second': samples_per_second,
        'memory_used_gb': memory_used,
        'memory_cached_gb': memory_cached,
        'detected_classes': len(unique_classes)
    }

def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Semantic Segmentation Benchmark')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model name (resnet18, resnet34, resnet50, resnet101, resnet152)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'mixed'],
                       help='Precision mode')
    parser.add_argument('--warmup_runs', type=int, default=3,
                       help='Number of warmup runs')
    parser.add_argument('--benchmark_runs', type=int, default=10,
                       help='Number of benchmark runs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Check CUDA availability for GPU-specific precisions
    if args.precision in ['fp16', 'mixed'] and device.type != 'cuda':
        print(f"Warning: {args.precision} precision not supported on {device}, falling back to fp32")
        args.precision = 'fp32'
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    if device.type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run benchmark
    results = benchmark_segmentation_model(
        args.model, device, args.batch_size, 
        args.warmup_runs, args.benchmark_runs, args.precision
    )
    
    # Print final result in format expected by benchmark script
    print(f"\nFINAL RESULT: {results['samples_per_second']:.2f} samples/sec")

if __name__ == "__main__":
    main() 