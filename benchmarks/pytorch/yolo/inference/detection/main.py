#!/usr/bin/env python3
"""
YOLOv5 Detection Inference Benchmark for PyTorch
Real YOLOv5 implementation using Ultralytics
"""

import torch
import time
import argparse
import numpy as np
import sys
import os
from pathlib import Path
import subprocess

# Add project root to path for utils import
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from utils.download import get_sample_image_path

def get_gpu_memory_usage():
    """Get GPU memory usage using cross-platform method"""
    try:
        from utils.shared_device_utils import get_gpu_memory_efficient
        memory_info = get_gpu_memory_efficient()
        return memory_info.get('total_gpu_used_gb', 0.0)
    except Exception as e:
        print(f"Warning: Could not get GPU memory usage: {e}")
        # Fallback to PyTorch memory tracking if available
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024**3
        return 0.0

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_synthetic_image(batch_size=1, height=640, width=640):
    """Create synthetic image data for benchmarking"""
    # Create random RGB image data (YOLOv5 expects 0-255 range)
    images = torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.float32)
    return images

def load_yolo_model(model_name, device, precision="fp32"):
    """Load actual YOLOv5 model using ultralytics"""
    try:
        from ultralytics import YOLO
        
        # Map model names to Ultralytics model files
        model_map = {
            "yolov5s": "yolov5s.pt",
            "yolov5m": "yolov5m.pt", 
            "yolov5l": "yolov5l.pt",
            "yolov5x": "yolov5x.pt",
            "yolov5": "yolov5s.pt"  # Default to small if generic "yolov5"
        }
        
        model_file = model_map.get(model_name, "yolov5s.pt")
        print(f"Loading YOLOv5 model: {model_file}")
        
        # Load model (will download if not exists)
        model = YOLO(model_file)
        
        # Move to device
        model.to(device)
        
        # Fuse conv+bn BEFORE converting to fp16. Ultralytics' fuse_conv_and_bn
        # synthesizes a zero-bias tensor with the default fp32 dtype when the
        # conv has no bias; if we've already called .half() on the model, the
        # conv weights are fp16 but that synthesized bias is fp32, and the
        # subsequent torch.mm(w_bn, b_conv) fails with a dtype mismatch on
        # ROCm/MIOpen (NVIDIA cuBLAS happens to silently promote).
        # Fusing first in fp32 sidesteps the issue entirely; .half() then
        # converts the fused model uniformly.
        try:
            if hasattr(model, "fuse"):
                model.fuse()
            elif hasattr(model, "model") and hasattr(model.model, "fuse"):
                model.model.fuse()
        except Exception as fuse_err:
            # Non-fatal: ultralytics may still lazily fuse inside predict(),
            # but at least we tried. Log so the user can see it.
            print(f"Note: explicit model.fuse() failed ({fuse_err}); "
                  f"relying on ultralytics' lazy fuse")

        # Set precision
        if precision == "fp16" and device.type == "cuda":
            model.half()
            print("Using FP16 precision")
        else:
            print(f"Using {precision} precision")
        
        return model, True  # True indicates real model loaded
        
    except ImportError:
        print("=" * 60)
        print("WARNING: ultralytics not installed!")
        print("YOLO benchmark will use a ResNet placeholder instead.")
        print("Results are NOT representative of real YOLOv5 performance.")
        print("Install with: pip install ultralytics")
        print("=" * 60)
        return load_resnet_placeholder(model_name, device, precision), False
    except Exception as e:
        print("=" * 60)
        print(f"WARNING: Failed to load YOLOv5: {e}")
        print("YOLO benchmark will use a ResNet placeholder instead.")
        print("Results are NOT representative of real YOLOv5 performance.")
        print("=" * 60)
        return load_resnet_placeholder(model_name, device, precision), False

def load_resnet_placeholder(model_name, device, precision="fp32"):
    """Fallback ResNet placeholder if YOLOv5 unavailable"""
    import torchvision.models as models
    
    print("Note: Using ResNet as YOLOv5 placeholder (install ultralytics for real YOLOv5)")
    
    if model_name == "yolov5s":
        model = models.resnet18(weights='DEFAULT')
    elif model_name == "yolov5m":
        model = models.resnet34(weights='DEFAULT')
    elif model_name == "yolov5l":
        model = models.resnet50(weights='DEFAULT')
    elif model_name == "yolov5x":
        model = models.resnet101(weights='DEFAULT')
    else:
        model = models.resnet18(weights='DEFAULT')
    
    # Modify final layer for detection-like output
    model.fc = torch.nn.Linear(model.fc.in_features, 1000)
    model = model.to(device)
    
    if precision == "fp16" and device.type == "cuda":
        model.half()
    
    model.eval()
    return model

def benchmark_yolo_inference(model_name, precision, batch_size, num_warmup=10, num_runs=100, device_str='auto'):
    """Benchmark YOLOv5 inference performance"""
    
    print(f"Starting {model_name} detection benchmark")
    print(f"Precision: {precision}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup runs: {num_warmup}")
    print(f"Benchmark runs: {num_runs}")
    
    if device_str == 'auto':
        device = get_device()
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")
    
    try:
        # Load YOLOv5 model (real or placeholder)
        model, is_real_yolo = load_yolo_model(model_name, device, precision)
        
        # Prepare input data
        if is_real_yolo:
            # For real YOLOv5, we can use the model's built-in preprocessing
            # Create synthetic images that look more realistic
            input_data = create_synthetic_image(batch_size, 640, 640)
            
            # YOLOv5 expects images in 0-255 range, normalized to 0-1 internally
            input_data = input_data / 255.0
            
            if precision == "fp16" and device.type == "cuda":
                input_data = input_data.half()
            
            input_data = input_data.to(device)
            
        else:
            # For ResNet placeholder, use standard preprocessing
            input_data = create_synthetic_image(batch_size, 640, 640).to(device)
            if precision == "fp16" and device.type == "cuda":
                input_data = input_data.half()
        
        # Import shared benchmark utilities
        from utils.benchmark_utils import (
            BenchmarkTimer, benchmark_loop, warmup as warmup_fn, compute_stats,
            setup_torch_backends, reset_memory_tracking, measure_peak_memory
        )
        setup_torch_backends(cudnn_benchmark=True)
        
        print(f"Input data shape: {input_data.shape}")
        
        use_mixed = precision == "mixed" and device.type == "cuda"
        
        # Define one inference step
        def inference_step():
            with torch.inference_mode():
                if is_real_yolo:
                    if use_mixed:
                        with torch.amp.autocast('cuda'):
                            return model(input_data, verbose=False)
                    else:
                        return model(input_data, verbose=False)
                else:
                    if use_mixed:
                        with torch.amp.autocast('cuda'):
                            return model(input_data)
                    else:
                        return model(input_data)
        
        # Warmup
        warmup_fn(inference_step, num_warmup=num_warmup, device=device)
        
        # Reset peak memory after warmup for clean benchmark-only measurement
        reset_memory_tracking(device)
        
        # Benchmark runs with CUDA Events timing and GC disabled
        latencies_ms = benchmark_loop(inference_step, num_runs=num_runs, device=device)
        
        # Measure peak memory (only covers the benchmark runs, not warmup)
        peak_mem = measure_peak_memory(device)
        
        # Run one extra pass to get detection results for display
        results = inference_step()
        
        # Compute comprehensive statistics
        stats = compute_stats(latencies_ms)
        avg_latency_ms = stats["mean"]
        
        # Throughput calculation
        throughput = (batch_size / avg_latency_ms) * 1000.0
        
        # Memory usage from PyTorch allocator
        memory_used_gb = peak_mem.get('peak_allocated_gb', 0.0) if peak_mem else 0.0
        
        # Count model parameters
        if is_real_yolo:
            # For YOLOv5, get parameter count from the underlying model
            total_params = sum(p.numel() for p in model.model.parameters())
        else:
            # For ResNet placeholder
            total_params = sum(p.numel() for p in model.parameters())
        
        # Additional detection metrics for real YOLOv5
        detection_info = ""
        if is_real_yolo and len(results) > 0 and hasattr(results[0], 'boxes'):
            # Extract detection information
            boxes = results[0].boxes
            if boxes is not None:
                num_detections = len(boxes)
                detection_info = f"Detections per image: {num_detections / batch_size:.1f}"
        
        model_type = "Real YOLOv5" if is_real_yolo else "ResNet Placeholder"
        print(f"\n=== {model_name.upper()} DETECTION BENCHMARK RESULTS ===")
        print(f"Framework: PyTorch")
        print(f"Model Type: {model_type}")
        print(f"Device: {device}")
        print(f"Precision: {precision}")
        print(f"Batch Size: {batch_size}")
        print(f"Model Parameters: {total_params:,}")
        print(f"Input Shape: {input_data.shape}")
        print(f"Total GPU Memory Used: {memory_used_gb:.2f} GB")
        if detection_info:
            print(f"{detection_info}")
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
            'total_params': total_params,
            'model_type': model_type,
            'is_real_yolo': is_real_yolo
        }
        
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='PyTorch YOLOv5 Detection Benchmark')
    parser.add_argument('--model', type=str, default='yolov5s',
                       choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'yolov5'],
                       help='YOLOv5 model variant')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'mixed'],
                       help='Inference precision')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--num_warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Check for ultralytics
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics not installed - will use ResNet placeholder")
    print()
    
    # Run benchmark
    results = benchmark_yolo_inference(
        args.model, args.precision, args.batch_size, 
        args.num_warmup, args.num_runs, args.device
    )
    
    if results is None:
        sys.exit(1)

if __name__ == "__main__":
    main() 