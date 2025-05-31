#!/usr/bin/env python3
"""
GEMM (General Matrix Multiply) Operations Benchmark for PyTorch
Tests matrix multiplication performance across different sizes and precisions
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

def run_gemm_benchmark(precision="fp32"):
    """Run GEMM benchmark"""
    device = get_device()
    print(f"Running GEMM Operations Benchmark with {precision.upper()} precision")
    print(f"Device: {device}")
    
    # Measure initial memory
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # For mixed precision, use fp32 tensors but apply autocast during operations
    dtype = torch.float32 if precision in ["fp32", "mixed"] else torch.float16
    
    # Different matrix sizes to test
    gemm_sizes = [
        (1024, 1024, 1024),    # Small
        (2048, 2048, 2048),    # Medium  
        (4096, 4096, 4096),    # Large
        (8192, 8192, 8192),    # Very Large
        (1, 4096, 4096),       # Batch=1 (common in inference)
        (32, 2048, 2048),      # Small batch
        (128, 1024, 1024),     # Medium batch
    ]
    
    print(f"\n{'='*60}")
    print("GEMM (General Matrix Multiply) BENCHMARKS")
    print(f"{'='*60}")
    
    best_gflops = 0
    best_config = ""
    total_time = 0
    
    for m, k, n in gemm_sizes:
        print(f"\nTesting GEMM: ({m}, {k}) x ({k}, {n}) -> ({m}, {n})")
        
        # Create matrices
        A = torch.randn(m, k, dtype=dtype, device=device)
        B = torch.randn(k, n, dtype=dtype, device=device)
        
        def gemm_op():
            if precision == "mixed" and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    return torch.mm(A, B)
            else:
                return torch.mm(A, B)
        
        result = benchmark_operation(gemm_op)
        
        # Calculate FLOPS
        flops = 2 * m * k * n  # 2 operations per multiply-add
        gflops = flops / (result["avg_time_ms"] * 1e6)  # GFLOPS
        
        print(f"  Average time: {result['avg_time_ms']:.2f} ms")
        print(f"  Performance: {gflops:.2f} GFLOPS")
        print(f"  Memory: {(A.numel() + B.numel() + m*n) * A.element_size() / 1024**3:.3f} GB")
        
        # Track best performance
        if gflops > best_gflops:
            best_gflops = gflops
            best_config = f"{m}x{k}x{n}"
        
        total_time += result["avg_time_ms"]
    
    # Measure final memory
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Calculate average metrics for framework integration
    avg_time = total_time / len(gemm_sizes)
    throughput = 1000 / avg_time  # samples/sec (treating each GEMM as one sample)
    
    # Print results in format expected by main benchmark framework
    print(f"\n{'='*50}")
    print("GEMM BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Best GEMM Performance: {best_gflops:.2f} GFLOPS ({best_config})")
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
    parser = argparse.ArgumentParser(description="GEMM Operations Benchmark")
    parser.add_argument("--model", type=str, default="gemm_ops",
                       help="Model name (for compatibility)")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
                       help="Precision for operations")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (for compatibility)")
    
    args = parser.parse_args()
    
    try:
        results = run_gemm_benchmark(args.precision)
        print("\nGEMM Operations Benchmark completed successfully!")
        return 0
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 