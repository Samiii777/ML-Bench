#!/usr/bin/env python3
"""
Memory Operations Benchmark for PyTorch
Tests memory bandwidth operations like copy, transpose, and memory access patterns
"""

import torch
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

def run_memory_benchmark(precision="fp32"):
    """Run memory operations benchmark"""
    device = get_device()
    print(f"Running Memory Operations Benchmark with {precision.upper()} precision")
    print(f"Device: {device}")
    
    # Measure initial memory
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # For mixed precision, use fp32 tensors but apply autocast during operations
    dtype = torch.float32 if precision in ["fp32", "mixed"] else torch.float16
    
    # Different memory sizes to test
    memory_configs = [
        # (size, name, description)
        ((1024, 1024), "Small", "4/8 MB"),
        ((2048, 2048), "Medium", "16/32 MB"),
        ((4096, 4096), "Large", "64/128 MB"),
        ((8192, 8192), "Very Large", "256/512 MB"),
        ((16384, 16384), "Huge", "1/2 GB"),
        ((23170, 23170), "Massive", "2/4 GB"),  # ~2GB tensor to avoid cache
        ((1024, 8192), "Rectangular", "32/64 MB"),
        ((8192, 1024), "Tall", "32/64 MB"),
    ]
    
    print(f"\n{'='*60}")
    print("MEMORY OPERATIONS BENCHMARKS")
    print(f"{'='*60}")
    print("Note: Only measuring actual memory transfer operations")
    print("Note: Best bandwidth calculated from massive tensors (1000MB+) to avoid cache effects")
    
    best_bandwidth = 0
    best_config = ""
    total_time = 0
    operation_count = 0
    
    for size, name, description in memory_configs:
        print(f"\nTesting {name} tensors {size} ({description}):")
        
        # Create source tensor
        src = torch.randn(*size, dtype=dtype, device=device)
        
        # Memory copy test (actual memory transfer)
        def copy_op():
            return src.clone()
        
        result = benchmark_operation(copy_op)
        
        # Calculate bandwidth
        bytes_total = src.numel() * src.element_size() * 2  # read + write
        bandwidth_gbs = bytes_total / (result["avg_time_ms"] * 1e6)
        
        print(f"  Copy:      {result['avg_time_ms']:6.2f} ms, {bandwidth_gbs:6.1f} GB/s")
        
        # Only track best performance from massive tensors (1000MB+) to avoid cache effects
        tensor_size_mb = (src.numel() * src.element_size()) / (1024 * 1024)
        if tensor_size_mb >= 1000 and bandwidth_gbs > best_bandwidth:
            best_bandwidth = bandwidth_gbs
            best_config = f"Copy {name}"
        
        total_time += result["avg_time_ms"]
        operation_count += 1
        
        # Contiguous transpose test (forces actual memory transfer)
        def transpose_contiguous_op():
            return src.transpose(0, 1).contiguous()
        
        result = benchmark_operation(transpose_contiguous_op)
        bandwidth_gbs = bytes_total / (result["avg_time_ms"] * 1e6)
        
        print(f"  Transpose: {result['avg_time_ms']:6.2f} ms, {bandwidth_gbs:6.1f} GB/s")
        
        # Only track best performance from massive tensors (1000MB+) to avoid cache effects
        if tensor_size_mb >= 1000 and bandwidth_gbs > best_bandwidth:
            best_bandwidth = bandwidth_gbs
            best_config = f"Transpose {name}"
        
        total_time += result["avg_time_ms"]
        operation_count += 1
        
        # Memory fill test (write-only operation)
        def fill_op():
            dst = torch.empty_like(src)
            dst.fill_(1.0)
            return dst
        
        result = benchmark_operation(fill_op)
        bytes_total_write = src.numel() * src.element_size()  # write only
        bandwidth_gbs = bytes_total_write / (result["avg_time_ms"] * 1e6)
        
        print(f"  Fill:      {result['avg_time_ms']:6.2f} ms, {bandwidth_gbs:6.1f} GB/s")
        
        # Only track best performance from massive tensors (1000MB+) to avoid cache effects
        if tensor_size_mb >= 1000 and bandwidth_gbs > best_bandwidth:
            best_bandwidth = bandwidth_gbs
            best_config = f"Fill {name}"
        
        total_time += result["avg_time_ms"]
        operation_count += 1
        
        # Add/subtract test (read-write operation)
        def add_op():
            return src + 1.0
        
        result = benchmark_operation(add_op)
        bandwidth_gbs = bytes_total / (result["avg_time_ms"] * 1e6)
        
        print(f"  Add:       {result['avg_time_ms']:6.2f} ms, {bandwidth_gbs:6.1f} GB/s")
        
        # Only track best performance from massive tensors (1000MB+) to avoid cache effects
        if tensor_size_mb >= 1000 and bandwidth_gbs > best_bandwidth:
            best_bandwidth = bandwidth_gbs
            best_config = f"Add {name}"
        
        total_time += result["avg_time_ms"]
        operation_count += 1
    
    # Measure final memory
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Calculate average metrics for framework integration
    avg_time = total_time / operation_count
    throughput = 1000 / avg_time  # samples/sec (treating each operation as one sample)
    
    # Print results in format expected by main benchmark framework
    print(f"\n{'='*50}")
    print("MEMORY BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Best Memory Bandwidth: {best_bandwidth:.1f} GB/s ({best_config})")
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
        "best_bandwidth_gbs": best_bandwidth,
        "best_config": best_config,
        "device": str(device),
        "framework": "PyTorch",
        "precision": precision,
        "memory_usage": final_memory
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Memory Operations Benchmark")
    parser.add_argument("--model", type=str, default="memory_ops",
                       help="Model name (for compatibility)")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
                       help="Precision for operations")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (for compatibility)")
    
    args = parser.parse_args()
    
    try:
        results = run_memory_benchmark(args.precision)
        print("\nMemory Operations Benchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 