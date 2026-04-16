import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fcos_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
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
from utils.download import get_sample_image_path, get_coco_classes_path

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

def get_sample_image_path():
    """Get path to sample image"""
    # Use the clean utils function instead of ugly relative paths  
    from utils.download import get_sample_image_path as utils_get_path
    return utils_get_path()

def get_coco_classes_path():
    """Get path to COCO classes file"""
    # Use the clean utils function instead of ugly relative paths  
    from utils.download import get_coco_classes_path as utils_get_path
    return utils_get_path()

def load_coco_categories(filename):
    """Load COCO categories from the given file"""
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    return coco_classes

def preprocess_image(image_path, batch_size=1):
    """Preprocess the input image for object detection"""
    input_image = Image.open(image_path).convert('RGB')
    
    # Object detection models typically expect images in their original size
    # but we need to convert to tensor format
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    input_tensor = transform(input_image)
    
    # Create batch - for detection, we need a list of tensors
    if batch_size > 1:
        input_batch = [input_tensor] * batch_size
    else:
        input_batch = [input_tensor]
    
    return input_batch, input_image

def load_detection_model(model_name, device):
    """Load pre-trained object detection model with ResNet backbone"""
    model_options = {
        'resnet18': fcos_resnet50_fpn,  # Use ResNet50 as backbone (ResNet18 not available for FCOS)
        'resnet34': fcos_resnet50_fpn,  # Use ResNet50 as backbone
        'resnet50': fcos_resnet50_fpn,
        'resnet101': fcos_resnet50_fpn,  # Use ResNet50 as backbone (ResNet101 not standard for FCOS)
        'resnet152': fcos_resnet50_fpn,  # Use ResNet50 as backbone
    }
    
    model_func = model_options.get(model_name, fcos_resnet50_fpn)
    
    if model_name != 'resnet50':
        print(f"NOTE: Detection uses FCOS-ResNet50-FPN for all variants (requested '{model_name}', actual backbone: ResNet50)")
    else:
        print(f"Loading FCOS-ResNet50-FPN detection model")
    model = model_func(pretrained=True)
    model.eval()
    model.to(device)
    
    return model

def run_inference(params):
    """Main inference function for object detection"""
    print(f"Running {params.model} object detection benchmark")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    
    # Load image and classes
    image_file = get_sample_image_path()
    coco_classes = load_coco_categories("")
    
    print(f"Loading object detection model with {params.model} backbone")
    
    # Set device
    if params.device == 'auto':
        device = get_device()
    else:
        device = torch.device(params.device)
    
    print_device_info()
    print(f"Using device: {device}")
    
    # Load model
    model = load_detection_model(params.model, device)
    
    # Load and preprocess input image
    input_batch, original_image = preprocess_image(image_file, params.batch_size)
    
    # Move input to device
    input_batch = [img.to(device) for img in input_batch]
    
    # Measure initial memory usage
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # Apply precision settings
    use_mixed_precision = False
    if params.precision == "fp16":
        if device.type == "cuda" or device.type == "mps":
            model = model.half()
            input_batch = [img.half() for img in input_batch]
        elif device.type == "cpu":
            print("Warning: FP16 on CPU may be slower than FP32, but proceeding as requested...")
            model = model.half()
            input_batch = [img.half() for img in input_batch]
        else:
            model = model.half()
            input_batch = [img.half() for img in input_batch]
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
        benchmark_loop, warmup, compute_stats, setup_torch_backends,
        reset_memory_tracking, measure_peak_memory
    )
    setup_torch_backends(cudnn_benchmark=True)
    
    # Define one inference step
    def inference_step():
        with torch.inference_mode():
            if use_mixed_precision:
                with torch.amp.autocast(device_type=device.type):
                    return model(input_batch)
            else:
                return model(input_batch)
    
    # Warmup
    warmup(inference_step, num_warmup=5, device=device)
    
    # Reset peak memory after warmup for clean benchmark-only measurement
    reset_memory_tracking(device)
    
    # Benchmark runs with CUDA Events timing and GC disabled
    num_runs = 50
    latencies_ms = benchmark_loop(inference_step, num_runs=num_runs, device=device)
    
    # Measure peak memory (only covers the benchmark runs, not warmup)
    peak_mem = measure_peak_memory(device)
    
    # Also get nvidia-smi as secondary reference
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Compute comprehensive statistics
    stats = compute_stats(latencies_ms)
    avg_inference_time = stats["mean"]
    
    # Calculate per-sample metrics
    samples_per_batch = params.batch_size
    avg_latency_per_sample = avg_inference_time / samples_per_batch
    throughput_fps = 1000.0 / avg_latency_per_sample  # samples per second
    
    # Get predictions for display
    predictions = inference_step()
    
    # Print results
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: FCOS with {params.model} backbone")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Use case: Object Detection")
    print()
    print("Performance Metrics:")
    print(f"Average Inference Time: {stats['mean']:.2f} ms")
    print(f"Per-sample Latency: {avg_latency_per_sample:.2f} ms/sample")
    print(f"Median Inference Time: {stats['median']:.2f} ms")
    print(f"P90 Inference Time: {stats['p90']:.2f} ms")
    print(f"P95 Inference Time: {stats['p95']:.2f} ms")
    print(f"P99 Inference Time: {stats['p99']:.2f} ms")
    print(f"Min Inference Time: {stats['min']:.2f} ms")
    print(f"Max Inference Time: {stats['max']:.2f} ms")
    print(f"Std Inference Time: {stats['std']:.2f} ms")
    print(f"Throughput: {throughput_fps:.2f} samples/sec")
    print(f"PyTorch Inference Time = {avg_inference_time:.2f} ms")
    print()
    
    # Memory information (PyTorch allocator peak is primary, nvidia-smi is secondary)
    if peak_mem:
        print("Memory Usage:")
        print(f"GPU Memory Allocated: {peak_mem.get('peak_allocated_gb', 0):.3f} GB")
        print(f"GPU Memory Cached: {peak_mem.get('peak_reserved_gb', 0):.3f} GB")
    if final_memory:
        print(f"Total GPU Memory Used: {final_memory['total_gpu_used_gb']:.2f} GB")
    print()
    
    # Show detection results for first image
    if len(predictions) > 0 and len(predictions[0]['boxes']) > 0:
        pred = predictions[0]
        scores = pred['scores'].cpu().numpy()
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter detections with confidence > 0.5
        high_conf_indices = scores > 0.5
        if np.any(high_conf_indices):
            print("Detection Results (confidence > 0.5):")
            for i, (score, box, label) in enumerate(zip(scores[high_conf_indices], 
                                                       boxes[high_conf_indices], 
                                                       labels[high_conf_indices])):
                class_name = coco_classes[label-1] if label-1 < len(coco_classes) else f"class_{label}"
                print(f"  {class_name}: {score:.3f} confidence")
        else:
            print("No high-confidence detections found")
    else:
        print("No objects detected")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="ResNet Object Detection Benchmark")
    parser.add_argument("--model", type=str, default="resnet50",
                       choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                       help="ResNet model to use as backbone")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed", "int8"],
                       help="Precision for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    try:
        run_inference(args)
    except Exception as e:
        print(f"Error during benchmark: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 