#!/usr/bin/env python3
"""
Reduction Operations Benchmark for PyTorch
Tests reduction operations like sum, mean, max, min, softmax across dimensions
"""

import torch
import torch.nn.functional as F
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

def run_reduction_benchmark(precision="fp32"):
    """Run reduction operations benchmark"""
    device = get_device()
    print(f"Running Reduction Operations Benchmark with {precision.upper()} precision")
    print(f"Device: {device}")
    
    # Measure initial memory
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # For mixed precision, use fp32 tensors but apply autocast during operations
    dtype = torch.float32 if precision in ["fp32", "mixed"] else torch.float16
    
    # Test different tensor sizes
    sizes = [
        ((1024, 1024), "Small", "4/8 MB"),
        ((4096, 4096), "Large", "64/128 MB"),
        ((8192, 8192), "Very Large", "256/512 MB"),
        ((16384, 16384), "Huge", "1/2 GB"),
        ((23170, 23170), "Massive", "2/4 GB"),  # ~2GB tensor to avoid cache
        ((2048, 8192), "Rectangular", "64/128 MB"),
        ((512, 16384), "Wide", "32/64 MB"),
    ]
    
    # Reduction operations to test
    operations = {
        "sum": (lambda x: torch.sum(x), "Global"),
        "mean": (lambda x: torch.mean(x), "Global"),
        "max": (lambda x: torch.max(x), "Global"),
        "min": (lambda x: torch.min(x), "Global"),
        "sum_dim0": (lambda x: torch.sum(x, dim=0), "Dimension 0"),
        "sum_dim1": (lambda x: torch.sum(x, dim=1), "Dimension 1"),
        "mean_dim0": (lambda x: torch.mean(x, dim=0), "Dimension 0"),
        "mean_dim1": (lambda x: torch.mean(x, dim=1), "Dimension 1"),
        "softmax_dim0": (lambda x: F.softmax(x, dim=0), "Dimension 0"),
        "softmax_dim1": (lambda x: F.softmax(x, dim=1), "Dimension 1"),
        "layer_norm": (lambda x: F.layer_norm(x, x.shape[-1:]), "Layer Norm"),
    }
    
    print(f"\n{'='*60}")
    print("REDUCTION OPERATIONS BENCHMARKS")
    print(f"{'='*60}")
    print("Note: Best bandwidth calculated from massive tensors (1000MB+) and simple reductions only")
    
    best_bandwidth = 0
    best_config = ""
    total_time = 0
    operation_count = 0
    
    for size, name, description in sizes:
        print(f"\nTesting reduction ops on {name} tensors {size} ({description}):")
        
        # Create test tensor
        x = torch.randn(*size, dtype=dtype, device=device)
        
        for op_name, (op_func, op_type) in operations.items():
            def mixed_precision_reduction():
                if precision == "mixed" and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        return op_func(x)
                else:
                    return op_func(x)
            
            try:
                result = benchmark_operation(mixed_precision_reduction)
                
                # Calculate bandwidth
                input_bytes = x.numel() * x.element_size()
                bandwidth_gbs = input_bytes / (result["avg_time_ms"] * 1e6)
                
                print(f"  {op_name:12s}: {result['avg_time_ms']:6.2f} ms, {bandwidth_gbs:6.1f} GB/s ({op_type})")
                
                # Only track best performance from massive tensors (1000MB+) to avoid cache effects
                # For simple reductions only (not compute-bound operations like softmax)
                tensor_size_mb = (x.numel() * x.element_size()) / (1024 * 1024)
                is_simple_reduction = op_name in ["sum", "mean", "max", "min", "sum_dim0", "sum_dim1", "mean_dim0", "mean_dim1"]
                
                if tensor_size_mb >= 1000 and is_simple_reduction and bandwidth_gbs > best_bandwidth:
                    best_bandwidth = bandwidth_gbs
                    best_config = f"{op_name} {name}"
                
                total_time += result["avg_time_ms"]
                operation_count += 1
                
            except Exception as e:
                print(f"  {op_name:12s}: FAILED - {str(e)}")
                continue
    
    # Measure final memory
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Calculate average metrics for framework integration
    if operation_count > 0:
        avg_time = total_time / operation_count
        throughput = 1000 / avg_time  # samples/sec (treating each operation as one sample)
    else:
        avg_time = 0
        throughput = 0
    
    # Print results in format expected by main benchmark framework
    print(f"\n{'='*50}")
    print("REDUCTION BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Best Reduction Bandwidth: {best_bandwidth:.1f} GB/s ({best_config})")
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
    parser = argparse.ArgumentParser(description="Reduction Operations Benchmark")
    parser.add_argument("--model", type=str, default="reduction_ops",
                       help="Model name (for compatibility)")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
                       help="Precision for operations")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (for compatibility)")
    
    args = parser.parse_args()
    
    try:
        results = run_reduction_benchmark(args.precision)
        print("\nReduction Operations Benchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 