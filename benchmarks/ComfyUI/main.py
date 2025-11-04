#!/usr/bin/env python3
"""
ComfyUI FLUX Benchmark
Benchmarks FLUX.1-schnell model using ComfyUI backend
"""

import sys
import os
import argparse
import time
import json
import subprocess
import signal
import requests
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from utils.shared_device_utils import get_gpu_memory_efficient


class ComfyUIServer:
    """Manages ComfyUI server lifecycle"""
    
    def __init__(self, comfyui_dir, port=8188):
        self.comfyui_dir = Path(comfyui_dir)
        self.port = port
        self.server_address = f"127.0.0.1:{port}"
        self.process = None
        self.server_logs = []  # Store server logs
        
    def start(self):
        """Start ComfyUI server in background"""
        print(f"Starting ComfyUI server on port {self.port}...")
        
        # Get python from current venv
        python_exe = sys.executable
        
        # Start server process
        server_cmd = [
            python_exe,
            str(self.comfyui_dir / "main.py"),
            "--listen", "127.0.0.1",
            "--port", str(self.port)
        ]
        
        self.process = subprocess.Popen(
            server_cmd,
            cwd=str(self.comfyui_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Wait for server to be ready
        max_wait = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"http://{self.server_address}/queue")
                if response.status_code == 200:
                    print(f"✓ ComfyUI server ready")
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        
        print("Warning: ComfyUI server may not be ready")
        return False
    
    def stop(self):
        """Stop ComfyUI server"""
        if self.process:
            print("Stopping ComfyUI server...")
            try:
                # Kill process group if possible
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None


class ComfyUIAPI:
    """ComfyUI API client for workflow execution"""
    
    def __init__(self, server_address='127.0.0.1:8188'):
        self.server_address = server_address
    
    def queue_prompt(self, workflow):
        """Queue a workflow for execution"""
        data = json.dumps({"prompt": workflow}).encode('utf-8')
        response = requests.post(
            f"http://{self.server_address}/prompt",
            data=data
        )
        return response.json()
    
    def wait_for_completion(self, prompt_id, check_interval=0.1):
        """Wait for workflow to complete and return execution info"""
        while True:
            queue = requests.get(f"http://{self.server_address}/queue").json()
            
            # Check if still in queue
            queue_remaining = queue['queue_running'] + queue['queue_pending']
            if len(queue_remaining) == 0:
                # Check if in history
                history = requests.get(f"http://{self.server_address}/history/{prompt_id}").json()
                if prompt_id in history:
                    # Extract execution time from history
                    history_entry = history[prompt_id]
                    
                    # ComfyUI execution time is in status.completed_at - status.started_at
                    # But the status might have execution time directly
                    execution_time = None
                    if 'status' in history_entry:
                        status = history_entry['status']
                        # Check for completed and started times
                        if 'completed_at' in status and 'started_at' in status:
                            execution_time = status['completed_at'] - status['started_at']
                    
                    return {
                        'history': history_entry,
                        'execution_time': execution_time
                    }
            
            time.sleep(check_interval)


def setup_comfyui(benchmark_dir):
    """Setup ComfyUI if not already installed"""
    comfyui_path = benchmark_dir / "ComfyUI"
    
    if comfyui_path.exists() and (comfyui_path / "main.py").exists():
        print(f"✓ ComfyUI already installed at {comfyui_path}")
        return comfyui_path
    
    print("Setting up ComfyUI...")
    project_root = benchmark_dir.parent.parent
    setup_script = project_root / "utils" / "setup_comfyui.py"
    
    if not setup_script.exists():
        raise FileNotFoundError(f"Setup script not found: {setup_script}")
    
    # Run setup with flux-schnell model
    cmd = [
        sys.executable,
        str(setup_script),
        "--model", "flux-schnell",
        "--dir", str(comfyui_path)
        # Don't skip requirements - we need ComfyUI dependencies
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        raise RuntimeError("ComfyUI setup failed")
    
    print(f"✓ ComfyUI setup complete")
    return comfyui_path


def create_flux_workflow(prompt_text, seed=42, steps=4, width=1024, height=1024, randomize_seed=False):
    """Create FLUX.1-schnell workflow
    
    Args:
        randomize_seed: If True, uses random seed for each generation
                        If False, uses fixed seed (faster due to caching)
    """
    # Use random seed if requested, otherwise use fixed seed
    actual_seed = int(time.time() * 1000) % (2**31) if randomize_seed else seed
    
    workflow = {
        "12": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-schnell.safetensors",
                "weight_dtype": "default"
            }
        },
        "11": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "t5xxl_fp16.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux"
            }
        },
        "10": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ae.safetensors"
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt_text,
                "clip": ["11", 0]
            }
        },
        "25": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": actual_seed
            }
        },
        "16": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler"
            }
        },
        "17": {
            "class_type": "BasicScheduler",
            "inputs": {
                "scheduler": "normal",
                "steps": steps,
                "denoise": 1.0,
                "model": ["12", 0]
            }
        },
        "22": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["12", 0],
                "conditioning": ["6", 0]
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["25", 0],
                "guider": ["22", 0],
                "sampler": ["16", 0],
                "sigmas": ["17", 0],
                "latent_image": ["5", 0]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["10", 0]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "benchmark",
                "images": ["8", 0]
            }
        }
    }
    
    return workflow


def benchmark_comfyui_flux(num_warmup=3, num_runs=10, steps=4, width=1024, height=1024):
    """Benchmark FLUX.1-schnell using ComfyUI"""
    
    print(f"ComfyUI FLUX Benchmark")
    print(f"Steps: {steps}, Resolution: {width}x{height}")
    print(f"Warmup: {num_warmup}, Runs: {num_runs}")
    
    benchmark_dir = Path(__file__).parent
    
    # Setup ComfyUI if needed
    comfyui_path = setup_comfyui(benchmark_dir)
    
    # Store workflows
    workflow_dir = benchmark_dir / "workflows"
    workflow_dir.mkdir(exist_ok=True)
    
    # Note: Using randomize_seed=False for consistent benchmarking
    # This enables caching and shows best-case performance
    # Set to True for realistic varied image generation
    workflow = create_flux_workflow(
        "a beautiful sunset over mountains",
        seed=42,
        steps=steps,
        width=width,
        height=height,
        randomize_seed=False  # Fixed seed for consistent benchmarking
    )
    
    workflow_file = workflow_dir / "flux_schnell_fp16_benchmark.json"
    with open(workflow_file, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"Workflow saved to: {workflow_file}")
    
    # Start ComfyUI server
    server = ComfyUIServer(comfyui_path)
    api = ComfyUIAPI(server.server_address)
    
    try:
        if not server.start():
            raise RuntimeError("Failed to start ComfyUI server")
        
        # Get device info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Warmup
        print(f"\nWarmup ({num_warmup} runs)...")
        for i in range(num_warmup):
            result = api.queue_prompt(workflow)
            completion_info = api.wait_for_completion(result['prompt_id'])
            exec_time = completion_info.get('execution_time')
            if exec_time:
                print(f"  Warmup {i+1}/{num_warmup} complete ({exec_time:.2f}s)")
            else:
                print(f"  Warmup {i+1}/{num_warmup} complete")
        
        # Benchmark  
        # Note: Same prompt but different seeds to measure actual sampling performance
        # Text encoding cached (realistic), but diffusion sampling runs fresh each time
        print(f"\nBenchmarking ({num_runs} runs)...")
        latencies = []
        comfyui_execution_times = []
        
        for i in range(num_runs):
            # Create workflow with same prompt but incrementing seed
            # This caches text encoding but forces actual diffusion sampling
            run_workflow = create_flux_workflow(
                "a beautiful sunset over mountains",  # Same prompt
                seed=42 + i,  # Different seed each run
                steps=steps,
                width=width,
                height=height,
                randomize_seed=False
            )
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            result = api.queue_prompt(run_workflow)
            completion_info = api.wait_for_completion(result['prompt_id'])
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            wall_time = end_time - start_time
            
            # Get actual ComfyUI execution time from history
            comfyui_exec_time = completion_info.get('execution_time')
            if comfyui_exec_time:
                latencies.append(comfyui_exec_time)
                comfyui_execution_times.append(comfyui_exec_time)
                print(f"  Run {i+1}/{num_runs}: {comfyui_exec_time:.3f}s (wall: {wall_time:.3f}s)")
            else:
                # Use wall time
                latencies.append(wall_time)
                print(f"  Run {i+1}/{num_runs}: {wall_time:.3f}s")
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        # Primary metric: seconds per image (more intuitive for generation)
        seconds_per_image = avg_latency
        throughput = 1.0 / avg_latency  # images per second (for reference)
        
        # Note which timing method was used
        timing_method = "ComfyUI execution time" if comfyui_execution_times else "Wall-clock time"
        print(f"\nTiming method: {timing_method}")
        
        # Get memory usage
        memory_used_gb = 0
        if device == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used_gb = peak_memory / 1024**3
        
        # Print results
        print(f"\n{'='*60}")
        print(f"COMFYUI FLUX BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Model: FLUX.1-schnell")
        print(f"Framework: ComfyUI")
        print(f"Device: {device}")
        print(f"Steps: {steps}")
        print(f"Resolution: {width}x{height}")
        print(f"GPU Memory Used: {memory_used_gb:.2f} GB")
        print()
        print(f"Performance Metrics:")
        print(f"  Time per Image: {seconds_per_image:.3f}s ({seconds_per_image*1000:.2f}ms)")
        print(f"  Std Deviation: {std_latency:.3f}s")
        print(f"  Min Time: {min_latency:.3f}s")
        print(f"  Max Time: {max_latency:.3f}s")
        print(f"  Throughput: {throughput:.2f} images/sec")
        print(f"{'='*60}")
        
        print(f"\nFINAL RESULT: {seconds_per_image:.3f} seconds/image")
        
        # Show where images are saved
        comfyui_output = comfyui_path / "output"
        print(f"\n✓ Generated images saved to: {comfyui_output}")
        
        return {
            'throughput_images_per_sec': throughput,
            'avg_latency_s': avg_latency,
            'std_latency_s': std_latency,
            'min_latency_s': min_latency,
            'max_latency_s': max_latency,
            'memory_used_gb': memory_used_gb,
            'steps': steps,
            'resolution': f"{width}x{height}"
        }
        
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description='ComfyUI FLUX Benchmark')
    # Standard benchmark framework arguments
    parser.add_argument('--model', type=str, default='comfyui_flux_schnell', help='Model name - only FLUX schnell supported')
    parser.add_argument('--precision', type=str, default='fp32', help='Precision (fp32/fp16/mixed) - not used by ComfyUI')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size - not used by ComfyUI')  
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu) - not used by ComfyUI')
    # ComfyUI-specific arguments
    parser.add_argument('--num_warmup', type=int, default=3, help='Number of warmup runs')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of benchmark runs')
    parser.add_argument('--steps', type=int, default=4, help='Number of sampling steps')
    parser.add_argument('--width', type=int, default=1024, help='Image width')
    parser.add_argument('--height', type=int, default=1024, help='Image height')
    
    args = parser.parse_args()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    results = benchmark_comfyui_flux(
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        steps=args.steps,
        width=args.width,
        height=args.height
    )
    
    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

