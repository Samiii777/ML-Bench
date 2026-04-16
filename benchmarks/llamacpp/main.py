#!/usr/bin/env python3
"""
llama.cpp Benchmark
Benchmarks language model inference using llama.cpp
"""

import sys
import os
import argparse
import subprocess
import re
from pathlib import Path


# Add project root to path
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break


# Model configurations matching utils/setup_llamacpp.py
MODELS = {
    'llama-3.1-8b-q4': {
        'filename': 'meta-llama-3.1-8b-instruct-q4_0.gguf',
    },
    'llama-3.1-8b-q8': {
        'filename': 'meta-llama-3.1-8b-instruct-q8_0.gguf',
    },
    'gpt-oss-20b': {
        'filename': 'gpt-oss-20b-mxfp4.gguf',
    }
}


def setup_llamacpp(benchmark_dir, gpu_type='auto', model_name=None, amdgpu_targets=None):
    """Setup llama.cpp if not already installed
    
    Args:
        benchmark_dir: Directory containing this benchmark
        gpu_type: 'auto', 'nvidia', 'amd-rocm', 'amd-rock', or 'cpu'
        model_name: Model to download (e.g., 'llama-3.1-8b-q4')
        amdgpu_targets: Comma-separated AMD GPU targets
    """
    llamacpp_path = benchmark_dir / "llama.cpp"
    
    # Check if llama.cpp exists and is built
    llama_bench = llamacpp_path / "build" / "bin" / "llama-bench"
    
    if llama_bench.exists():
        print(f"✓ llama.cpp already built at {llamacpp_path}")
        
        # Check if model exists
        if model_name:
            models_dir = llamacpp_path / "models"
            
            # Determine expected filename
            expected_filename = None
            if model_name in MODELS:
                expected_filename = MODELS[model_name]['filename']
            
            # Check if the specific model exists
            if expected_filename and (models_dir / expected_filename).exists():
                print(f"✓ Found model {expected_filename} in {models_dir}")
                return llamacpp_path
                
            # Fallback logic for backward compatibility or if model not in dict
            if models_dir.exists():
                gguf_files = list(models_dir.glob("*.gguf"))
                
                # Filter out known vocab files if we're looking for a model
                if expected_filename:
                    # If we expect a specific file and it's not there, we should probably download it
                    # But let's check if we can find it by partial match or just download it
                    print(f"Model {expected_filename} not found, downloading {model_name}...")
                    download_model_only(llamacpp_path, model_name)
                    return llamacpp_path

                if not gguf_files:
                    print(f"No models found, downloading {model_name}...")
                    download_model_only(llamacpp_path, model_name)
                else:
                    print(f"✓ Found {len(gguf_files)} model(s) in {models_dir}")
            else:
                print(f"Models directory not found, downloading {model_name}...")
                download_model_only(llamacpp_path, model_name)
        
        return llamacpp_path
    
    print("Setting up llama.cpp...")
    project_root = benchmark_dir.parent.parent
    setup_script = project_root / "utils" / "setup_llamacpp.py"
    
    if not setup_script.exists():
        raise FileNotFoundError(f"Setup script not found: {setup_script}")
    
    # Build command
    cmd = [
        sys.executable,
        str(setup_script),
        "--gpu", gpu_type,
        "--dir", str(llamacpp_path)
    ]
    
    # Add model if specified
    if model_name:
        cmd.extend(["--model", model_name])
    
    # Add AMD GPU targets if specified
    if amdgpu_targets:
        cmd.extend(["--amdgpu-targets", amdgpu_targets])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        raise RuntimeError("llama.cpp setup failed")
    
    print(f"✓ llama.cpp setup complete")
    return llamacpp_path


def download_model_only(llamacpp_path, model_name):
    """Download model using the setup script"""
    project_root = llamacpp_path.parent.parent
    setup_script = project_root / "utils" / "setup_llamacpp.py"
    
    cmd = [
        sys.executable,
        str(setup_script),
        "--skip-clone",
        "--skip-build",
        "--model", model_name,
        "--dir", str(llamacpp_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        raise RuntimeError(f"Model download failed")


def find_model(llamacpp_path, model_name=None):
    """Find a model file in the models directory"""
    models_dir = llamacpp_path / "models"
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # If specific model name provided, look for it
    if model_name:
        model_path = models_dir / model_name
        if model_path.exists():
            return model_path
        
        # Try with .gguf extension
        if not model_name.endswith('.gguf'):
            model_path = models_dir / f"{model_name}.gguf"
            if model_path.exists():
                return model_path
    
    # Find any .gguf file
    gguf_files = list(models_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf model files found in {models_dir}")
    
    # Return the first one
    return gguf_files[0]


def parse_llama_bench_output(output):
    """Parse llama-bench output to extract metrics"""
    results = {}
    
    # Look for lines like:
    # | model                          |       size |     params | backend    | ngl |          test |              t/s |
    # | ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ---------------: |
    # | llama 8B Q4_0                  |   4.34 GiB |     8.03 B | CUDA       |  99 |         pp512 |    4885.46 ± 8.42|
    # | llama 8B Q4_0                  |   4.34 GiB |     8.03 B | CUDA       |  99 |         tg128 |     162.84 ± 0.21|
    
    lines = output.split('\n')
    
    for line in lines:
        # Extract model info
        if 'model' in line and 'size' in line:
            continue  # Header line
        
        if '|' in line and not line.strip().startswith('|---'):
            parts = [p.strip() for p in line.split('|') if p.strip()]
            
            if len(parts) >= 7:
                model_name = parts[0]
                size = parts[1]
                params = parts[2]
                backend = parts[3]
                ngl = parts[4]
                test = parts[5]
                tokens_per_sec = parts[6]
                
                # Parse tokens/sec (format: "162.84 ± 0.21")
                match = re.search(r'([\d.]+)\s*±\s*([\d.]+)', tokens_per_sec)
                if match:
                    avg_tokens = float(match.group(1))
                    std_tokens = float(match.group(2))
                    
                    # Store results by test type
                    if test not in results:
                        results[test] = {}
                    
                    results[test] = {
                        'model': model_name,
                        'size': size,
                        'params': params,
                        'backend': backend,
                        'ngl': ngl,
                        'tokens_per_sec': avg_tokens,
                        'std_tokens_per_sec': std_tokens
                    }
    
    return results


def benchmark_llamacpp(model='llama-3.1-8b-q4', gpu_type='auto', ngl=999, 
                       model_file=None, amdgpu_targets=None):
    """Benchmark llama.cpp inference
    
    Args:
        model: Model name to download (if not exists)
        gpu_type: GPU type for building
        ngl: Number of GPU layers (-ngl flag)
        model_file: Specific model file to use (overrides auto-detection)
        amdgpu_targets: AMD GPU targets for building
    """
    print(f"llama.cpp Benchmark")
    print(f"Model: {model}")
    print(f"GPU Type: {gpu_type}")
    print(f"GPU Layers: {ngl}")
    
    benchmark_dir = Path(__file__).parent
    
    # Setup llama.cpp if needed
    llamacpp_path = setup_llamacpp(
        benchmark_dir, 
        gpu_type=gpu_type, 
        model_name=model,
        amdgpu_targets=amdgpu_targets
    )
    
    # Find model file
    if model_file:
        model_path = llamacpp_path / "models" / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    elif model in MODELS:
        # Use mapped filename
        model_filename = MODELS[model]['filename']
        model_path = llamacpp_path / "models" / model_filename
        if not model_path.exists():
             # Try finding with find_model as fallback
             try:
                 model_path = find_model(llamacpp_path, model_filename)
             except FileNotFoundError:
                 # If not found, try downloading again or fail? 
                 # setup_llamacpp should have handled it, but let's be safe
                 raise FileNotFoundError(f"Model {model} (file: {model_filename}) not found in {llamacpp_path}/models")
    else:
        if model_file:
            model_path = find_model(llamacpp_path, model_file)
        else:
            available = list((llamacpp_path / "models").glob("*.gguf")) if (llamacpp_path / "models").exists() else []
            available_names = [f.name for f in available]
            raise ValueError(
                f"Unknown model '{model}'. Known models: {list(MODELS.keys())}. "
                f"Use --model-file to specify a GGUF file directly. "
                f"Files found in models/: {available_names}"
            )
    
    print(f"\n✓ Using model: {model_path}")
    
    # Run llama-bench
    llama_bench_path = llamacpp_path / "build" / "bin" / "llama-bench"
    
    if not llama_bench_path.exists():
        raise FileNotFoundError(f"llama-bench not found: {llama_bench_path}")
    
    cmd = [
        str(llama_bench_path),
        "-m", str(model_path),
        "-ngl", str(ngl)
    ]
    
    print(f"\nRunning benchmark...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(llamacpp_path),
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Parse results
        results = parse_llama_bench_output(result.stdout)
        
        if not results:
            print("\n⚠ Warning: Could not parse benchmark results")
            return None
        
        # Print summary
        print("\n" + "=" * 60)
        print("LLAMA.CPP BENCHMARK RESULTS")
        print("=" * 60)
        
        for test_name, metrics in results.items():
            print(f"\nTest: {test_name}")
            print(f"  Model: {metrics['model']}")
            print(f"  Size: {metrics['size']}")
            print(f"  Params: {metrics['params']}")
            print(f"  Backend: {metrics['backend']}")
            print(f"  GPU Layers: {metrics['ngl']}")
            print(f"  Tokens/sec: {metrics['tokens_per_sec']:.2f} ± {metrics['std_tokens_per_sec']:.2f}")
        
        # Get primary metric (text generation - tg128)
        if 'tg128' in results:
            primary_tokens_per_sec = results['tg128']['tokens_per_sec']
        elif results:
            # Use first result if tg128 not found
            primary_tokens_per_sec = list(results.values())[0]['tokens_per_sec']
        else:
            primary_tokens_per_sec = 0
        
        print("=" * 60)
        print(f"\nFINAL RESULT: {primary_tokens_per_sec:.2f} tokens/sec")
        
        return {
            'tests': results,
            'primary_tokens_per_sec': primary_tokens_per_sec,
            'model': model,
            'gpu_layers': ngl
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed with exit code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='llama.cpp Benchmark')
    
    # Standard benchmark framework arguments
    parser.add_argument('--model', type=str, default='llama-3.1-8b-q4',
                       choices=['llama-3.1-8b-q4', 'llama-3.1-8b-q8', 'gpt-oss-20b'],
                       help='Model to use (will download if not exists)')
    parser.add_argument('--model-file', type=str, default=None,
                       help='Specific model file to use (in models/ directory)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device type (auto/nvidia/amd-rocm/amd-rock/cpu)')
    parser.add_argument('--gpu-type', type=str, default=None,
                       help='Alias for --device')
    
    # llama.cpp specific arguments
    parser.add_argument('--ngl', type=int, default=999,
                       help='Number of GPU layers to offload (default: 999 = all)')
    parser.add_argument('--amdgpu-targets', type=str, default=None,
                       help='AMD GPU targets for building (e.g., gfx1100,gfx1151,gfx1201)')
    
    # Compatibility arguments (not used by llama.cpp but accepted for framework compatibility)
    parser.add_argument('--precision', type=str, default='fp16',
                       help='Precision (not used by llama.cpp)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (not used by llama.cpp)')
    
    args = parser.parse_args()
    
    # Determine GPU type
    gpu_type = args.gpu_type if args.gpu_type else args.device
    
    print(f"llama.cpp version: git clone from ggml-org/llama.cpp")
    
    results = benchmark_llamacpp(
        model=args.model,
        gpu_type=gpu_type,
        ngl=args.ngl,
        model_file=args.model_file,
        amdgpu_targets=args.amdgpu_targets
    )
    
    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()





