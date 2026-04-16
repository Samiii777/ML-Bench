#!/usr/bin/env python3
"""
BERT Text Classification Inference Benchmark for PyTorch
Tests BERT for its intended purpose: text classification (sentiment analysis)
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
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

def get_bert_model_name(model_arg):
    """Map model argument to actual Hugging Face model name for classification or use directly if in HF format"""
    
    # If model_arg contains "/" it's likely a HuggingFace model ID - use directly
    if "/" in model_arg:
        print(f"Using HuggingFace model directly: {model_arg}")
        return model_arg
    
    # Otherwise, use predefined mappings for classification models
    bert_models = {
        "bert": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-base-uncased": "nlptown/bert-base-multilingual-uncased-sentiment", 
        "bert-base-cased": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-large-uncased": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-large-cased": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-sentiment": "nlptown/bert-base-multilingual-uncased-sentiment"
    }
    return bert_models.get(model_arg, "nlptown/bert-base-multilingual-uncased-sentiment")

def get_sample_texts():
    """Get diverse sample texts for sentiment classification benchmarking"""
    return [
        # Positive sentiment
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the best experience I've ever had. Highly recommended!",
        "Fantastic quality and excellent customer service. Very satisfied!",
        "Outstanding performance and great value for money. Love it!",
        "Incredible results! This exceeded all my expectations.",
        
        # Negative sentiment  
        "This is terrible quality and completely broke after one day.",
        "Worst purchase ever. Don't waste your money on this garbage.",
        "Extremely disappointed with this product. Poor build quality.",
        "This doesn't work at all and customer service is unhelpful.",
        "Overpriced and underdelivered. Completely unsatisfied.",
        
        # Neutral sentiment
        "The product works as described. Nothing special but does the job.",
        "Average quality for the price. Could be better, could be worse.",
        "It's okay, meets basic expectations but nothing more.",
        "Standard functionality, typical for this type of product.",
        "Decent enough, though there's room for improvement.",
        
        # Mixed/Complex sentiment
        "Great design but poor durability. Mixed feelings about this.",
        "Love the concept but execution could be better.",
        "Good features but the price is too high for what you get.",
        "Works well most of the time but occasionally has issues.",
        "Beautiful aesthetics but functionality is lacking."
    ]

def run_inference(args):
    """Run BERT text classification inference benchmark"""
    
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    model_name = get_bert_model_name(args.model)
    
    print(f"Running BERT Text Classification Benchmark")
    print(f"Model: {model_name}")
    print(f"Task: Sentiment Analysis")
    print(f"Device: {device}")
    print(f"Precision: {args.precision}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        # Import transformers here to catch import errors
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F
        
        # Load tokenizer and model
        print(f"Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set dtype based on precision
        torch_dtype = torch.float32
        if args.precision in ["fp16", "mixed"]:
            torch_dtype = torch.float16
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if str(device) == "cuda" else None
        )
        
        # Move to device and set eval mode
        if str(device) != "cuda":  # device_map="auto" handles CUDA placement
            model = model.to(device)
        model.eval()
        
        # Get sample texts for benchmarking
        sample_texts = get_sample_texts()
        
        # Get initial GPU memory
        initial_memory = get_gpu_memory_efficient()
        
        # Import shared benchmark utilities
        from utils.benchmark_utils import BenchmarkTimer, compute_stats, reset_memory_tracking, measure_peak_memory, gc_disabled, setup_torch_backends
        setup_torch_backends(cudnn_benchmark=True)
        
        use_mixed = args.precision == "mixed" and device.type == "cuda"
        
        # Warmup with batch-sized input (matching benchmark workload)
        print("Warming up...")
        for _ in range(5):
            batch_texts = [sample_texts[j % len(sample_texts)] for j in range(args.batch_size)]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                if use_mixed:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
            
            synchronize_device(device)
        
        # Reset peak memory after warmup
        reset_memory_tracking(device)
        
        # Benchmark runs
        print("Running benchmark...")
        inference_times_ms = []
        all_predictions = []
        
        num_runs = max(50, 200 // args.batch_size)
        timer = BenchmarkTimer(device)
        
        with gc_disabled():
            for i in range(num_runs):
                batch_texts = []
                for j in range(args.batch_size):
                    text_idx = (i * args.batch_size + j) % len(sample_texts)
                    batch_texts.append(sample_texts[text_idx])
                
                inputs = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                timer.start()
                
                with torch.inference_mode():
                    if use_mixed:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    
                    predictions = F.softmax(outputs.logits, dim=-1)
                    predicted_classes = torch.argmax(predictions, dim=-1)
                
                elapsed_ms = timer.stop()
                inference_times_ms.append(elapsed_ms)
                
                all_predictions.append(predicted_classes.cpu().numpy())
                
                if (i + 1) % 20 == 0:
                    print(f"Completed {i+1}/{num_runs} runs")
        
        # Get memory stats
        peak_mem = measure_peak_memory(device)
        final_memory = get_gpu_memory_efficient()
        
        # Calculate metrics using shared stats
        stats = compute_stats(inference_times_ms)
        avg_latency = stats["mean"] / 1000.0
        min_latency = stats["min"] / 1000.0
        max_latency = stats["max"] / 1000.0
        std_latency = stats["std"] / 1000.0
        
        per_sample_latency = (avg_latency * 1000) / args.batch_size  # ms per sample
        throughput = args.batch_size / avg_latency  # samples per second
        
        # Analyze predictions (basic stats)
        all_preds = np.concatenate(all_predictions)
        unique_classes, class_counts = np.unique(all_preds, return_counts=True)
        
        # Print results
        print(f"\nBenchmark Results:")
        print(f"Model: {model_name}")
        print(f"Task: Sentiment Analysis")
        print(f"Device: {device}")
        print(f"Precision: {args.precision}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of runs: {num_runs}")
        print(f"Total samples processed: {num_runs * args.batch_size}")
        print(f"Average Inference Time: {stats['mean']:.2f} ms")
        print(f"Per-sample Latency: {per_sample_latency:.2f} ms/sample")
        print(f"Median Inference Time: {stats['median']:.2f} ms")
        print(f"P90 Inference Time: {stats['p90']:.2f} ms")
        print(f"P95 Inference Time: {stats['p95']:.2f} ms")
        print(f"P99 Inference Time: {stats['p99']:.2f} ms")
        print(f"Min Inference Time: {min_latency*1000:.2f} ms")
        print(f"Max Inference Time: {max_latency*1000:.2f} ms")
        print(f"Std Inference Time: {std_latency*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        
        # Prediction distribution
        print(f"\nPrediction Distribution:")
        for cls, count in zip(unique_classes, class_counts):
            percentage = (count / len(all_preds)) * 100
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
        
        # Memory information (PyTorch allocator peak is primary)
        if peak_mem:
            print(f"\nGPU Memory Usage:")
            print(f"GPU Memory Allocated: {peak_mem.get('peak_allocated_gb', 0):.2f} GB")
            print(f"GPU Memory Cached: {peak_mem.get('peak_reserved_gb', 0):.2f} GB")
        if final_memory:
            total_memory_used = final_memory.get('total_gpu_used_gb', 0)
            print(f"Total GPU Memory Used: {total_memory_used:.2f} GB")
        
        # Framework compatibility output (expected by benchmark runner)
        print(f"\n# Benchmark Framework Output")
        print(f"Framework: PyTorch")
        print(f"Device: {device}")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print(f"Per-sample Latency: {per_sample_latency:.2f} ms/sample")
        if peak_mem:
            print(f"GPU Memory Allocated: {peak_mem.get('peak_allocated_gb', 0):.2f} GB")
        elif final_memory:
            print(f"Total GPU Memory Used: {final_memory.get('total_gpu_used_gb', 0):.2f} GB")
        print(f"PyTorch Inference Time = {stats['mean']:.2f} ms")
        print(f"# End Benchmark Framework Output")
        
        return {
            "avg_latency_ms": avg_latency * 1000,
            "per_sample_latency_ms": per_sample_latency,
            "min_latency_ms": min_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "std_latency_ms": std_latency * 1000,
            "throughput_fps": throughput,
            "samples_processed": num_runs * args.batch_size,
            "prediction_distribution": dict(zip(unique_classes.tolist(), class_counts.tolist())),
            "device": str(device),
            "framework": "PyTorch",
            "model_name": model_name,
            "task": "sentiment_analysis",
            "memory_usage": final_memory
        }
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install transformers: pip install transformers")
        raise
    except Exception as e:
        print(f"Error during BERT text classification benchmark: {str(e)}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PyTorch BERT Text Classification Inference Benchmark")
    parser.add_argument("--model", type=str, default="bert",
                       help="BERT model name (bert, bert-base-uncased) or HuggingFace model ID (e.g., cardiffnlp/twitter-roberta-base-sentiment-latest)")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
                       help="Precision for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for inference")
    
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