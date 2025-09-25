#!/usr/bin/env python3
"""
Ollama Text Generation Benchmark

This script benchmarks Ollama models for text generation tasks.
It measures throughput, latency, and other performance metrics.
"""

import sys
import os
import argparse
import requests
import time
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

def get_model_configs():
    """Get configurations for available Ollama models"""
    return [
        {
            'name': 'llama3.1:8b',
            'model_id': 'llama3.1:8b',
            'type': 'text_generation'
        },
        {
            'name': 'qwen2.5:7b',
            'model_id': 'qwen2.5:7b', 
            'type': 'text_generation'
        }
    ]

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except Exception:
        return False

def ensure_model_available(model_name):
    """Ensure the model is available (pull if necessary)"""
    try:
        # Check if model is already available
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            available_models = [model['name'] for model in response.json().get('models', [])]
            if model_name in available_models:
                return True
        
        # Try to pull the model
        print(f"Pulling model {model_name}...")
        pull_url = 'http://localhost:11434/api/pull'
        pull_payload = {"name": model_name}
        response = requests.post(pull_url, json=pull_payload)
        return response.status_code == 200
    except Exception as e:
        print(f"Error ensuring model availability: {e}")
        return False

def run_generation(model_name, prompt):
    """Run text generation with Ollama"""
    url = 'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            result['wall_time'] = end_time - start_time
            return result
        else:
            print(f"Error: HTTP {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

def run_single_model_benchmark(model_config, params):
    """Run benchmark for a single model"""
    model_name = model_config['model_id']
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_config['name']}")
    print(f"{'='*60}")
    print(f"Model ID: {model_name}")
    print(f"Number of runs: {params.num_runs}")
    
    # Check if Ollama server is running
    if not check_ollama_server():
        return {
            'model': model_config['name'],
            'status': 'FAILED',
            'error': 'Ollama server not running. Please start Ollama server.'
        }
    
    # Ensure model is available
    if not ensure_model_available(model_name):
        return {
            'model': model_config['name'],
            'status': 'FAILED',
            'error': f'Model {model_name} not available and could not be pulled.'
        }
    
    # Define test prompts
    prompts = [
        "Write me a C++ code that solves the fibonacci without recursion",
        "Explain the concept of quantum entanglement in simple terms", 
        "Compose a poem inspired by the idea of artificial intelligence"
    ]
    
    if params.custom_prompt:
        prompts = [params.custom_prompt] * params.num_runs
    
    # Run benchmark
    results = []
    all_times = []
    all_token_counts = []
    all_throughputs = []
    
    for i in range(params.num_runs):
        prompt = prompts[i % len(prompts)]
        print(f"Benchmark run {i+1}/{params.num_runs}")
        
        result = run_generation(model_name, prompt)
        if result is None:
            return {
                'model': model_config['name'],
                'status': 'FAILED',
                'error': f'Generation failed on run {i+1}'
            }
        
        results.append(result)
        
        # Extract metrics
        eval_count = result.get('eval_count', 0)
        eval_duration = result.get('eval_duration', 0) / 1e9  # Convert nanoseconds to seconds
        wall_time = result.get('wall_time', 0)
        
        all_times.append(wall_time)
        all_token_counts.append(eval_count)
        
        # Calculate throughput - prefer eval_duration, fallback to wall_time
        if eval_duration > 0 and eval_count > 0:
            throughput = eval_count / eval_duration
            all_throughputs.append(throughput)
            print(f"  Time: {wall_time:.2f}s | Tokens: {eval_count} | Throughput: {throughput:.2f} tokens/sec")
        elif wall_time > 0 and eval_count > 0:
            # Fallback calculation using wall time
            throughput = eval_count / wall_time
            all_throughputs.append(throughput)
            print(f"  Time: {wall_time:.2f}s | Tokens: {eval_count} | Throughput: {throughput:.2f} tokens/sec (wall time)")
        else:
            print(f"  Time: {wall_time:.2f}s | Tokens: {eval_count} | Throughput: N/A")
    
    # Calculate averages
    avg_time = sum(all_times) / len(all_times)
    avg_tokens = sum(all_token_counts) / len(all_token_counts)
    avg_throughput = sum(all_throughputs) / len(all_throughputs) if all_throughputs else 0
    
    print(f"\n{'-'*50}")
    print(f"RESULTS: {model_config['name']}")
    print(f"{'-'*50}")
    print(f"Average time per run: {avg_time:.3f} seconds")
    print(f"Average tokens generated: {avg_tokens:.1f}")
    print(f"Average throughput: {avg_throughput:.2f} tokens/sec")
    print(f"Number of runs: {params.num_runs}")
    
    # Framework parseable output
    print(f"\n# Benchmark Framework Parseable Output for {model_config['name']}")
    print(f"Framework: Ollama")
    print(f"Task: text_generation")
    print(f"Tokens per second: {avg_throughput:.2f}")
    print(f"Per-sample Latency: {avg_time*1000:.2f} ms/sample")
    print(f"Average Tokens: {avg_tokens:.1f}")
    print(f"# End Parseable Output for {model_config['name']}")
    
    return {
        'model': model_config['name'],
        'status': 'PASSED',
        'avg_time': avg_time,
        'avg_tokens': avg_tokens,
        'avg_throughput': avg_throughput,
        'num_runs': params.num_runs
    }

def run_inference(params):
    """Main function to run Ollama benchmarks"""
    print("="*60)
    print("OLLAMA TEXT GENERATION BENCHMARK")
    print("="*60)
    print(f"Number of runs per model: {params.num_runs}")
    if params.custom_prompt:
        print(f"Custom prompt: {params.custom_prompt}")
    
    # Get model configurations
    model_configs = get_model_configs()
    
    # Filter models if specified
    if params.model:
        model_configs = [config for config in model_configs if config['model_id'] == params.model]
        if not model_configs:
            print(f"Error: Model '{params.model}' not found in available models.")
            available_models = [config['model_id'] for config in get_model_configs()]
            print(f"Available models: {', '.join(available_models)}")
            return [{'model': params.model, 'status': 'FAILED', 'error': 'Model not found'}]
    
    results = []
    
    for i, model_config in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"STARTING MODEL {i+1}/{len(model_configs)}: {model_config['name']}")
        print(f"{'='*60}")
        
        try:
            result = run_single_model_benchmark(model_config, params)
            results.append(result)
        except KeyboardInterrupt:
            print(f"\nBenchmark interrupted by user")
            result = {'model': model_config['name'], 'status': 'INTERRUPTED'}
            results.append(result)
            break
        except Exception as e:
            print(f"Error benchmarking {model_config['name']}: {e}")
            result = {'model': model_config['name'], 'status': 'FAILED', 'error': str(e)}
            results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        if result['status'] == 'PASSED':
            print(f"[PASS] {result['model']}: {result['avg_throughput']:.2f} tokens/sec")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"[FAIL] {result['model']}: FAILED - {error_msg}")
    
    print(f"{'='*60}")
    print("Benchmark completed!")
    
    return results

def main():
    """Main function for Ollama text generation benchmark"""
    parser = argparse.ArgumentParser(description='Ollama Text Generation Benchmark')
    
    # Model selection
    available_models = [config['model_id'] for config in get_model_configs()]
    parser.add_argument('--model', type=str, default=None,
                        help=f'Specific model to benchmark (available: {", ".join(available_models)})')
    
    # Benchmark settings
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of benchmark runs per model (default: 3)')
    
    # Framework compatibility arguments (may not be used but needed for integration)
    parser.add_argument('--precision', type=str, default='auto',
                        choices=['fp32', 'fp16', 'mixed', 'auto'],
                        help='Precision mode (not used by Ollama, for framework compatibility)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (not used by Ollama, for framework compatibility)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (not used by Ollama, for framework compatibility)')
    
    # Custom prompt
    parser.add_argument('--custom-prompt', type=str, default=None,
                        help='Custom prompt for generation (default: use test prompts)')
    
    args = parser.parse_args()
    
    try:
        results = run_inference(args)
        
        # Check if any benchmarks failed
        failed_results = [r for r in results if r.get('status') == 'FAILED']
        if failed_results:
            print(f"\nOllama benchmark failed! {len(failed_results)} out of {len(results)} benchmarks failed.")
            for failed in failed_results:
                print(f"Failed: {failed['model']} - {failed.get('error', 'Unknown error')}")
            return 1
        else:
            print("\nOllama benchmark completed successfully!")
            return 0
    except KeyboardInterrupt:
        print("\nOllama benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during Ollama benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
