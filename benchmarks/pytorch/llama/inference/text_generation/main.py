#!/usr/bin/env python3
"""
LLAMA Text Generation Inference Benchmark for PyTorch
"""

import torch
import argparse
import time
import sys
import numpy as np
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

# Import utilities
from utils.shared_device_utils import get_gpu_memory_efficient

# Simple device utilities - everything in one place
def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def synchronize_device(device=None):
    """Synchronize device operations for accurate timing"""
    if device is None:
        device = get_device()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

def get_llama_model_name(model_arg):
    """Map model argument to actual Hugging Face model name"""
    llama_models = {
        "llama": "meta-llama/Llama-2-7b-hf",
        "llama-2": "meta-llama/Llama-2-7b-hf",
        "llama2": "meta-llama/Llama-2-7b-hf",
        "llama-3": "meta-llama/Llama-3.1-8B",
        "llama3": "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-2-7b": "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b": "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b": "meta-llama/Llama-2-70b-hf",
        # DeepSeek reasoning models
        "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    }
    return llama_models.get(model_arg, "meta-llama/Llama-2-7b-hf")

def run_inference(args):
    """Run LLAMA text generation inference benchmark"""
    
    device = get_device()
    model_name = get_llama_model_name(args.model)
    
    print(f"Running LLAMA Text Generation Benchmark")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Precision: {args.precision}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        # Import transformers here to catch import errors
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Check model type first
        is_deepseek = "deepseek" in model_name.lower()
        
        # Load tokenizer and model
        print(f"Loading tokenizer and model...")
        # Load tokenizer (using default padding)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        
        # Set dtype based on precision
        torch_dtype = torch.float32
        if args.precision in ["fp16", "mixed"]:
            torch_dtype = torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if str(device) == "cuda" else None,
            trust_remote_code=True,  # Some LLAMA models may need this
            use_cache=True,  # Enable KV cache for efficiency
            low_cpu_mem_usage=True  # Reduce CPU memory during loading
        )
        
        # Move to device and set eval mode
        if str(device) != "cuda":  # device_map="auto" handles CUDA placement
            model = model.to(device)
        model.eval()
        
        # Memory optimization for larger batch sizes
        if is_deepseek and args.batch_size > 1:
            # Enable gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print(f"Enabled gradient checkpointing for batch size {args.batch_size}")
        
        # Set pad token if not present (LLAMA models often don't have pad tokens)
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            # If we added a pad token, resize embeddings
            model.resize_token_embeddings(len(tokenizer))
        
        # Sample prompts for benchmarking
        
        if is_deepseek:
            # DeepSeek reasoning prompts - include reasoning directive
            prompts = [
                "Please reason step by step: What are the key advantages of renewable energy over fossil fuels?",
                "Please reason step by step: How does machine learning contribute to medical diagnosis improvements?",
                "Please reason step by step: What factors should be considered when designing sustainable cities?",
                "Please reason step by step: How can artificial intelligence help address climate change challenges?",
                "Please reason step by step: What are the ethical implications of autonomous vehicles?"
            ]
        else:
            # Standard LLaMA prompts  
            prompts = [
                "The future of artificial intelligence is",
                "In a world where technology advances rapidly",
                "Machine learning has transformed the way we",
                "The most important aspect of scientific research is",
                "Climate change represents one of the greatest challenges"
            ]
        
        # Get initial GPU memory
        initial_memory = get_gpu_memory_efficient()
        
        # Warmup runs
        print("Warming up...")
        for _ in range(3):
            prompt = prompts[0]
            inputs = tokenizer(
                prompt, 
                padding=True,           # pad to longest in batch
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if is_deepseek:
                    # DeepSeek specific generation parameters (warmup with fewer tokens)
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=20,  # Smaller for warmup
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95
                    )
                else:
                    # Standard LLaMA generation (warmup with fewer tokens)
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=20,  # Smaller for warmup
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False
                    )
            synchronize_device(device)
        
        # Benchmark runs
        print("Running benchmark...")
        inference_times = []
        tokens_generated = []
        
        # Reduce number of runs for LLAMA models (they're slower)
        num_runs = max(10, 50 // args.batch_size)
        
        for i in range(num_runs):
            # Create batch of prompts
            batch_prompts = [prompts[j % len(prompts)] for j in range(args.batch_size)]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_prompts, 
                padding=True,           # pad to longest in batch
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Time the inference
            synchronize_device(device)
            start_time = time.time()
            
            with torch.no_grad():
                if is_deepseek:
                    # DeepSeek specific generation parameters
                    # Reduce tokens for larger batch sizes to save memory
                    max_tokens = 30 if args.batch_size > 1 else 50
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95
                    )
                else:
                    # Standard LLaMA generation
                    max_tokens = 30 if args.batch_size > 1 else 50
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False
                    )
            
            synchronize_device(device)
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Count tokens generated (difference from input)
            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.shape[1]
            tokens_gen = (output_length - input_length) * args.batch_size
            tokens_generated.append(tokens_gen)
            
            if (i + 1) % 5 == 0:  # Report less frequently for slower models
                print(f"Completed {i+1}/{num_runs} runs")
        
        # Get final GPU memory
        final_memory = get_gpu_memory_efficient()
        
        # Calculate metrics
        inference_times = np.array(inference_times)
        avg_latency = np.mean(inference_times)
        min_latency = np.min(inference_times)
        max_latency = np.max(inference_times)
        std_latency = np.std(inference_times)
        
        per_sample_latency = (avg_latency * 1000) / args.batch_size  # ms per sample
        throughput = args.batch_size / avg_latency  # samples per second
        
        total_tokens = sum(tokens_generated)
        total_time = sum(inference_times)
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        avg_tokens_per_run = np.mean(tokens_generated)
        
        # Print results
        print(f"\nBenchmark Results:")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Precision: {args.precision}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of runs: {num_runs}")
        print(f"Average Inference Time: {avg_latency*1000:.2f} ms")
        print(f"Per-sample Latency: {per_sample_latency:.2f} ms/sample")
        print(f"Min Inference Time: {min_latency*1000:.2f} ms")
        print(f"Max Inference Time: {max_latency*1000:.2f} ms")
        print(f"Std Inference Time: {std_latency*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print(f"Tokens per second: {tokens_per_second:.1f}")
        print(f"Average tokens per run: {avg_tokens_per_run:.1f}")
        
        # Memory information
        if final_memory:
            total_memory_used = final_memory.get('total_gpu_used_gb', 0)
            total_memory_available = final_memory.get('total_gpu_total_gb', 0)
            gpu_utilization = final_memory.get('gpu_utilization_percent', 0)
            
            print(f"Total GPU Memory Used: {total_memory_used:.2f} GB")
            print(f"Total GPU Memory Available: {total_memory_available:.2f} GB")
            print(f"GPU Memory Utilization: {gpu_utilization:.1f}%")
        
        # Framework compatibility output (expected by benchmark runner)
        print(f"PyTorch Inference Time = {avg_latency*1000:.2f} ms")
        
        return {
            "avg_latency_ms": avg_latency * 1000,
            "per_sample_latency_ms": per_sample_latency,
            "min_latency_ms": min_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "std_latency_ms": std_latency * 1000,
            "throughput_fps": throughput,
            "tokens_per_second": tokens_per_second,
            "avg_tokens_per_run": avg_tokens_per_run,
            "device": str(device),
            "framework": "PyTorch",
            "model_name": model_name,
            "memory_usage": final_memory
        }
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install transformers: pip install transformers")
        raise
    except Exception as e:
        print(f"Error during LLAMA text generation benchmark: {str(e)}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PyTorch LLAMA Text Generation Inference Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b",
                       help="Model name (llama, llama-2, llama-3, meta-llama/Llama-3.1-8B, deepseek-r1, etc.)")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
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