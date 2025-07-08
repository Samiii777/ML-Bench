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
    """Get GPU memory usage from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        memory_used_mb = int(result.stdout.strip())
        return memory_used_mb / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # Fallback to PyTorch memory tracking if nvidia-smi fails
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
        
        # Set precision
        if precision == "fp16" and device.type == "cuda":
            model.half()
            print("Using FP16 precision")
        else:
            print(f"Using {precision} precision")
        
        return model, True  # True indicates real model loaded
        
    except ImportError:
        print("Warning: ultralytics not available, falling back to ResNet placeholder")
        return load_resnet_placeholder(model_name, device, precision), False
    except Exception as e:
        print(f"Warning: Failed to load YOLOv5 ({e}), falling back to ResNet placeholder")
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

def benchmark_yolo_inference(model_name, precision, batch_size, num_warmup=10, num_runs=100):
    """Benchmark YOLOv5 inference performance"""
    
    print(f"Starting {model_name} detection benchmark")
    print(f"Precision: {precision}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup runs: {num_warmup}")
    print(f"Benchmark runs: {num_runs}")
    
    device = get_device()
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
        
        # Track memory before inference
        initial_memory = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated()
        
        print(f"Input data shape: {input_data.shape}")
        print(f"Initial GPU memory: {initial_memory / 1024**3:.2f} GB")
        
        # Warmup
        print(f"\nRunning {num_warmup} warmup iterations...")
        for i in range(num_warmup):
            with torch.no_grad():
                if is_real_yolo:
                    # Real YOLOv5 inference
                    if precision == "mixed" and device.type == "cuda":
                        with torch.amp.autocast('cuda'):
                            _ = model(input_data, verbose=False)
                    else:
                        _ = model(input_data, verbose=False)
                else:
                    # ResNet placeholder inference
                    if precision == "mixed" and device.type == "cuda":
                        with torch.amp.autocast('cuda'):
                            _ = model(input_data)
                    else:
                        _ = model(input_data)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
        
        # Benchmark
        print(f"\nRunning {num_runs} benchmark iterations...")
        latencies = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                if is_real_yolo:
                    # Real YOLOv5 inference
                    if precision == "mixed" and device.type == "cuda":
                        with torch.amp.autocast('cuda'):
                            results = model(input_data, verbose=False)
                    else:
                        results = model(input_data, verbose=False)
                else:
                    # ResNet placeholder inference
                    if precision == "mixed" and device.type == "cuda":
                        with torch.amp.autocast('cuda'):
                            results = model(input_data)
                    else:
                        results = model(input_data)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_runs} iterations")
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        # Throughput calculation
        throughput = batch_size / avg_latency
        
        # Memory usage
        memory_used_gb = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used_gb = peak_memory / 1024**3
        
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
        print(f"Average Latency: {avg_latency*1000:.2f} ms")
        print(f"Std Latency: {std_latency*1000:.2f} ms")
        print(f"Min Latency: {min_latency*1000:.2f} ms")
        print(f"Max Latency: {max_latency*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print(f"Throughput (images/sec): {throughput:.2f}")
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
        args.num_warmup, args.num_runs
    )
    
    if results is None:
        sys.exit(1)

if __name__ == "__main__":
    main() 