import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Clean import of utils - no ugly relative paths!
import utils
from utils.safe_print import safe_print, format_success_message

# Simple device utilities
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

def run_llm_inference(params):
    """Main LLM inference function"""
    model_id = params.model

    print(f"Running {model_id} LLM benchmark")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")

    # Load tokenizer and model
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    model.eval()

    # Move to device
    device = get_device()
    print_device_info()
    print(f"Using device: {device}")

    model.to(device)

    # Prepare dummy input
    # Use a fixed prompt for reproducibility
    prompt = "Once upon a time, in a land far, far away"
    # Ensure the tokenizer has a padding token defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Create batch
    if params.batch_size > 1:
        input_ids = input_ids.repeat(params.batch_size, 1)
        attention_mask = attention_mask.repeat(params.batch_size, 1)

    # Measure initial memory usage
    initial_memory = get_gpu_memory_nvidia_smi()

    # Apply precision settings
    use_mixed_precision = False
    if params.precision == "fp16":
        if device.type == "cuda":
            model = model.half()
        else:
            print("Warning: FP16 not supported on this device, using FP32")
    elif params.precision == "mixed":
        if device.type == "cuda":
            use_mixed_precision = True
            print("Using mixed precision (AMP)")
        else:
            print("Warning: Mixed precision not supported on this device, using FP32")
    elif params.precision == "int8":
        print("Warning: INT8 quantization not yet implemented for this LLM benchmark")

    # Warm-up runs
    print("Performing warm-up runs...")
    num_warmup_runs = 2
    max_new_tokens_warmup = 10
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens_warmup, do_sample=False)
            else:
                model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens_warmup, do_sample=False)
            synchronize()

    # Measure memory after warmup
    warmup_memory = get_gpu_memory_nvidia_smi()

    # Benchmark runs
    print("Running benchmark...")
    latencies = []
    total_tokens_generated = 0
    num_benchmark_runs = 5  # Number of generation runs for benchmarking
    max_new_tokens_benchmark = params.max_new_tokens if hasattr(params, 'max_new_tokens') else 50 # Number of tokens to generate per run

    for i in range(num_benchmark_runs):
        synchronize()
        start_time = time.time()
        with torch.no_grad():
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Ensure pad_token_id is set for generate
                    output_sequences = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens_benchmark,
                        do_sample=False, # Use greedy decoding for consistent token count
                        pad_token_id=tokenizer.pad_token_id
                    )
            else:
                output_sequences = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens_benchmark,
                    do_sample=False, # Use greedy decoding for consistent token count
                    pad_token_id=tokenizer.pad_token_id
                )
        synchronize()
        latency = time.time() - start_time
        latencies.append(latency)

        # Calculate tokens generated in this run
        # output_sequences includes input_ids, so subtract their length
        num_generated_tokens_in_run = (output_sequences.shape[1] - input_ids.shape[1]) * params.batch_size
        total_tokens_generated += num_generated_tokens_in_run

        print(f"Run {i+1}/{num_benchmark_runs}: {latency*1000:.2f} ms, Generated tokens: {num_generated_tokens_in_run}")

    # Calculate statistics
    avg_latency_per_run = sum(latencies) / len(latencies) if latencies else 0

    # Calculate tokens per second
    # total_tokens_generated is the sum of (new_tokens_per_run * batch_size) across all runs
    # sum(latencies) is the total time for all benchmark runs
    tokens_per_second = total_tokens_generated / sum(latencies) if sum(latencies) > 0 else 0

    # Per-sample latency (latency to generate max_new_tokens_benchmark for one sample in a batch)
    # This is tricky because generation time depends on output length.
    # We report average latency per run, and tokens/sec as the primary throughput metric.
    avg_latency_ms_per_run = avg_latency_per_run * 1000

    # Measure final memory usage
    final_memory = get_gpu_memory_nvidia_smi()

    # Decode one example from the last batch for verification (optional)
    if params.batch_size > 0 and output_sequences is not None:
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        #print(f"
Sample generated text:
{generated_text}") # Potentially long, uncomment for debugging

    print(f"
{'='*50}")
    print("LLM BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Model: {model_id}")
    print(f"Framework: PyTorch")
    print(f"Device: {device}")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Max new tokens per run: {max_new_tokens_benchmark}")
    print(f"Average latency per run: {avg_latency_ms_per_run:.2f} ms")
    print(f"Total tokens generated during benchmark: {total_tokens_generated}")
    print(f"Throughput (tokens/sec): {tokens_per_second:.2f} tokens/sec")

    if final_memory:
        print(f"Total GPU Memory Used: {final_memory.get('total_gpu_used_gb', 0):.3f} GB")
        print(f"Total GPU Memory Available: {final_memory.get('total_gpu_total_gb', 0):.3f} GB")
        print(f"GPU Memory Utilization: {final_memory.get('gpu_utilization_percent', 0):.1f}%")

    # Standardized output for benchmark.py to parse
    # Note: "Inference Time" and "Throughput" are common keys
    print(f"PyTorch Inference Time = {avg_latency_ms_per_run:.2f} ms") # This is latency per generation call
    print(f"Throughput: {tokens_per_second:.2f} tokens/sec") # This is the primary metric for LLMs

    return {
        "model_id": model_id,
        "avg_latency_ms_per_run": avg_latency_ms_per_run,
        "tokens_per_second": tokens_per_second,
        "total_tokens_generated": total_tokens_generated,
        "device": str(device),
        "framework": "PyTorch",
        "precision": params.precision,
        "batch_size": params.batch_size,
        "max_new_tokens": max_new_tokens_benchmark,
        "memory_usage": final_memory
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PyTorch LLM Inference Benchmark")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model ID from Hugging Face Model Hub (e.g., gpt2, EleutherAI/gpt-neo-125M)")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed", "int8"],
                       help="Precision for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum number of new tokens to generate in each benchmark run")
    # Add any other LLM specific parameters here

    args = parser.parse_args()

    try:
        results = run_llm_inference(args)
        print("LLM benchmark completed successfully!")
        # Print results in a way benchmark.py can parse easily
        # It looks for "Throughput: VALUE samples/sec" or similar patterns.
        # For LLMs, "tokens/sec" is more appropriate.
        # The _parse_benchmark_output in benchmark.py might need adjustment for "tokens/sec".
        # For now, we rely on the specific "Throughput: X tokens/sec" line.
        return 0
    except Exception as e:
        print(f"LLM benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    # This ensures that the script can be run directly for testing
    # Example: python benchmarks/pytorch/llm/inference/generation/main.py --model gpt2 --precision fp16 --batch_size 2
    exit_code = main()
    sys.exit(exit_code)
