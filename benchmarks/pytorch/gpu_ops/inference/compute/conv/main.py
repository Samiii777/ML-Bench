#!/usr/bin/env python3
"""
Convolution Operations Benchmark for PyTorch
Tests 2D convolution performance across different configurations and precisions
"""

import torch
import torch.nn as nn
import argparse
import time
import sys
import os
import numpy as np

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def synchronize():
    """Synchronize device operations for accurate timing"""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.synchronize()

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

def benchmark_operation(operation_func, *args, num_warmup=5, num_runs=20, **kwargs):
    """Benchmark a GPU operation with proper warmup and timing"""
    # Warmup runs
    for _ in range(num_warmup):
        result = operation_func(*args, **kwargs)
        synchronize()
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        synchronize()
        start = time.time()
        result = operation_func(*args, **kwargs)
        synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        "avg_time_ms": np.mean(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "std_time_ms": np.std(times),
        "result_shape": result.shape if hasattr(result, 'shape') else None
    }

def run_conv_benchmark(precision="fp32"):
    """Run convolution benchmark"""
    device = get_device()
    print(f"Running Convolution Operations Benchmark with {precision.upper()} precision")
    print(f"Device: {device}")
    
    # Measure initial memory
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # For mixed precision, use fp32 tensors but apply autocast during operations
    dtype = torch.float32 if precision in ["fp32", "mixed"] else torch.float16
    
    # Different convolution configurations
    conv_configs = [
        # (batch, in_channels, height, width, out_channels, kernel_size, stride, padding, name)
        (1, 3, 224, 224, 64, 3, 1, 1, "ResNet First Layer"),
        (1, 64, 56, 56, 128, 3, 1, 1, "ResNet Mid Layer"),
        (1, 256, 14, 14, 512, 3, 1, 1, "ResNet Deep Layer"),
        (32, 64, 32, 32, 128, 3, 1, 1, "Batch Processing"),
        (1, 512, 7, 7, 2048, 1, 1, 0, "1x1 Convolution"),
        (1, 128, 28, 28, 128, 5, 1, 2, "5x5 Convolution"),
        (16, 256, 16, 16, 256, 3, 1, 1, "Medium Batch"),
    ]
    
    print(f"\n{'='*60}")
    print("CONVOLUTION OPERATIONS BENCHMARKS")
    print(f"{'='*60}")
    
    best_gflops = 0
    best_config = ""
    total_time = 0
    
    for batch, in_ch, h, w, out_ch, k, stride, padding, name in conv_configs:
        print(f"\nTesting {name}: B={batch}, C_in={in_ch}, H={h}, W={w} -> C_out={out_ch}, K={k}x{k}")
        
        # Create input tensor and conv layer
        x = torch.randn(batch, in_ch, h, w, dtype=dtype, device=device)
        conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, dtype=dtype, device=device)
        
        def conv_op():
            if precision == "mixed" and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    return conv(x)
            else:
                return conv(x)
        
        result = benchmark_operation(conv_op)
        
        # Calculate FLOPS for convolution
        output_h = (h + 2*padding - k) // stride + 1
        output_w = (w + 2*padding - k) // stride + 1
        flops = batch * output_h * output_w * out_ch * in_ch * k * k * 2  # 2 ops per MAC
        gflops = flops / (result["avg_time_ms"] * 1e6)
        
        print(f"  Average time: {result['avg_time_ms']:.2f} ms")
        print(f"  Performance: {gflops:.2f} GFLOPS")
        print(f"  Output shape: ({batch}, {out_ch}, {output_h}, {output_w})")
        
        # Track best performance
        if gflops > best_gflops:
            best_gflops = gflops
            best_config = name
        
        total_time += result["avg_time_ms"]
    
    # Measure final memory
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Calculate average metrics for framework integration
    avg_time = total_time / len(conv_configs)
    throughput = 1000 / avg_time  # samples/sec (treating each conv as one sample)
    
    # Print results in format expected by main benchmark framework
    print(f"\n{'='*50}")
    print("CONVOLUTION BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Best Conv Performance: {best_gflops:.2f} GFLOPS ({best_config})")
    print(f"Average Operation Time: {avg_time:.2f} ms")
    print(f"Per-sample Latency: {avg_time:.2f} ms/sample")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    # Memory information
    if final_memory:
        print(f"Total GPU Memory Used: {final_memory.get('total_gpu_used_gb', 0):.3f} GB")
        print(f"Total GPU Memory Available: {final_memory.get('total_gpu_total_gb', 0):.3f} GB")
        print(f"GPU Memory Utilization: {final_memory.get('gpu_utilization_percent', 0):.1f}%")
    
    print(f"PyTorch Inference Time = {avg_time:.2f} ms")
    
    return {
        "avg_latency_ms": avg_time,
        "throughput_fps": throughput,
        "best_gflops": best_gflops,
        "best_config": best_config,
        "device": str(device),
        "framework": "PyTorch",
        "precision": precision,
        "memory_usage": final_memory
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convolution Operations Benchmark")
    parser.add_argument("--model", type=str, default="conv_ops",
                       help="Model name (for compatibility)")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
                       help="Precision for operations")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (for compatibility)")
    
    args = parser.parse_args()
    
    try:
        results = run_conv_benchmark(args.precision)
        print("\nConvolution Operations Benchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 