#!/usr/bin/env python3
"""
YOLOv5 Detection Inference Benchmark for ONNX
Real YOLOv5 implementation using Ultralytics
"""

import onnxruntime as ort
import numpy as np
import time
import argparse
import sys
import os
from pathlib import Path
import subprocess
import tempfile
import torch

# Add project root to path for utils import
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

def get_gpu_memory_usage():
    """Get GPU memory usage from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        memory_used_mb = int(result.stdout.strip())
        return memory_used_mb / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0

def create_synthetic_image(batch_size=1, height=640, width=640):
    """Create synthetic image data for benchmarking"""
    # Create random RGB image data in NCHW format (0-255 range like YOLOv5 expects)
    images = np.random.randint(0, 256, (batch_size, 3, height, width), dtype=np.uint8)
    # Convert to float32 and normalize to 0-1 range as expected by YOLOv5
    images = images.astype(np.float32) / 255.0
    return images

def convert_yolo_pytorch_to_onnx(model_name, onnx_path, precision="fp32"):
    """Convert YOLOv5 PyTorch model to ONNX format using ultralytics"""
    print(f"Converting {model_name} to ONNX format (precision: {precision}) with dynamic batch size...")
    
    try:
        # Try to use ultralytics first (modern approach)
        try:
            from ultralytics import YOLO
            
            # Map model names to Ultralytics model files
            model_map = {
                "yolov5s": "yolov5s.pt",
                "yolov5m": "yolov5m.pt", 
                "yolov5l": "yolov5l.pt",
                "yolov5x": "yolov5x.pt",
                "yolov5": "yolov5s.pt"  # Default to small if generic "yolov5"
            }
            
            model_file = model_map.get(model_name, "yolov5s.pt")
            print(f"Loading YOLOv5 model using ultralytics: {model_file}")
            
            # Load model (will download if not exists)
            model = YOLO(model_file)
            
            # Export to ONNX with dynamic batch size
            print("Exporting to ONNX...")
            onnx_path_from_export = model.export(
                format='onnx',
                dynamic=True,  # Enable dynamic batch size
                half=(precision == "fp16"),  # Use FP16 if requested
                opset=11,
                simplify=True,
                verbose=False
            )
            
            # Move the exported file to our desired location
            import shutil
            if onnx_path_from_export != onnx_path:
                shutil.move(onnx_path_from_export, onnx_path)
            
            print(f"✓ Real YOLOv5 model converted and saved to {onnx_path}")
            return True, "Real YOLOv5"
            
        except ImportError:
            print("Ultralytics not available, trying torch.hub...")
            # Fallback to torch.hub (older method)
            return convert_yolo_pytorch_to_onnx_torch_hub(model_name, onnx_path, precision)
            
    except Exception as e:
        print(f"Error loading/converting {model_name}: {str(e)}")
        print("Falling back to synthetic model...")
        return False, "Synthetic"

def convert_yolo_pytorch_to_onnx_torch_hub(model_name, onnx_path, precision="fp32"):
    """Fallback method using torch.hub for YOLOv5 conversion"""
    try:
        # Load YOLOv5 model from torch.hub
        print(f"Loading {model_name} from torch.hub...")
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, trust_repo=True)
        model.eval()
        
        print(f"✓ Successfully loaded {model_name}")
        
        # Handle precision conversion
        if precision == "fp16":
            model = model.half()
            dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float16)
            print("Converting model to FP16 precision")
        elif precision == "mixed":
            # For mixed precision, export as FP32 model - ONNX Runtime will optimize automatically
            dummy_input = torch.randn(1, 3, 640, 640)
            print("Exporting FP32 model for mixed precision optimization by ONNX Runtime")
        else:
            dummy_input = torch.randn(1, 3, 640, 640)
        
        # Create directory for ONNX file if it doesn't exist
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        # Export to ONNX with dynamic batch size support
        print("Exporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],  # YOLOv5 uses 'images' as input name
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},    # Variable batch dimension
                'output': {0: 'batch_size'}     # Variable output batch dimension
            }
        )
        
        print(f"✓ Model converted and saved to {onnx_path} with dynamic batch size support")
        return True, "Real YOLOv5 (torch.hub)"
        
    except Exception as e:
        print(f"Error loading/converting {model_name}: {str(e)}")
        print("This might be due to missing dependencies (opencv-python, etc.)")
        print("Falling back to synthetic model...")
        return False, "Synthetic"

def create_synthetic_onnx_model():
    """Create a synthetic ONNX model for benchmarking (fallback)"""
    print("Creating synthetic ONNX model as fallback...")
    import torch.nn as nn
    
    # Create a simple CNN model as placeholder
    class SimpleDetectionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 1000)  # Detection-like output
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Export model to ONNX with dynamic batch size
    model = SimpleDetectionModel()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            model, 
            dummy_input, 
            f.name, 
            input_names=['images'], 
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},    # Variable batch size
                'output': {0: 'batch_size'}     # Variable batch size for output too
            }
        )
        return f.name, "Synthetic"

def get_yolo_onnx_model_path(model_name, precision):
    """Get the path where the ONNX model should be stored"""
    # Create models directory in project root
    models_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create ONNX subdirectory
    onnx_dir = models_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    
    # Model filename includes precision for proper caching
    model_filename = f"{model_name}_{precision}.onnx"
    return onnx_dir / model_filename

def benchmark_yolo_onnx_inference(model_name, precision, batch_size, execution_provider, num_warmup=10, num_runs=100):
    """Benchmark YOLOv5 ONNX inference performance"""
    
    print(f"Starting {model_name} ONNX detection benchmark")
    print(f"Precision: {precision}")
    print(f"Batch size: {batch_size}")
    print(f"Execution Provider: {execution_provider}")
    print(f"Warmup runs: {num_warmup}")
    print(f"Benchmark runs: {num_runs}")
    
    try:
        # Get ONNX model path
        onnx_model_path = get_yolo_onnx_model_path(model_name, precision)
        
        # Convert PyTorch model to ONNX if not exists or force real model
        model_type = "Unknown"
        if not onnx_model_path.exists():
            print(f"ONNX model not found at {onnx_model_path}")
            success, model_type = convert_yolo_pytorch_to_onnx(model_name, str(onnx_model_path), precision)
            if not success:
                model_path, model_type = create_synthetic_onnx_model()
                print(f"Using synthetic model as fallback")
            else:
                model_path = str(onnx_model_path)
        else:
            model_path = str(onnx_model_path)
            model_type = "Real YOLOv5 (cached)"
            print(f"Using existing ONNX model: {model_path}")
        
        print(f"Model type: {model_type}")
        
        print(f"Creating ONNX Runtime session with {execution_provider}...")
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set execution provider with options
        providers = []
        if execution_provider == 'CUDAExecutionProvider':
            providers = [('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            })]
        elif execution_provider == 'TensorrtExecutionProvider':
            providers = [('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,  # 2GB
                'trt_fp16_enable': precision == 'fp16',
            })]
        elif execution_provider == 'ROCMExecutionProvider':
            providers = [('ROCMExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'do_copy_in_default_stream': True,
            })]
        elif execution_provider == 'MIGraphXExecutionProvider':
            providers = [('MIGraphXExecutionProvider', {
                'device_id': 0,
                'fp16_enable': precision == 'fp16',
            })]
        else:
            providers = [execution_provider]
        
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Model input: {input_name} {input_shape}")
        print(f"Model outputs: {output_names}")
        
        # Prepare synthetic data
        if len(input_shape) == 4:  # NCHW format
            # Handle dynamic batch size (when dimension is a string like 'batch_size' or -1)
            if input_shape[0] == 1 or input_shape[0] == -1 or isinstance(input_shape[0], str):
                input_data = create_synthetic_image(batch_size, input_shape[2], input_shape[3])
            else:
                input_data = create_synthetic_image(input_shape[0], input_shape[2], input_shape[3])
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
        
        # Convert precision based on model's expected input type
        model_input_type = session.get_inputs()[0].type
        if model_input_type == 'tensor(float16)' or (precision == "fp16" and 'Real YOLOv5' in model_type):
            input_data = input_data.astype(np.float16)
        elif model_input_type == 'tensor(float)':
            input_data = input_data.astype(np.float32)
        else:
            # Default to float32
            input_data = input_data.astype(np.float32)
        
        print(f"Input data shape: {input_data.shape}")
        print(f"Input data dtype: {input_data.dtype}")
        
        # Track initial memory
        initial_memory = get_gpu_memory_usage()
        print(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Warmup
        print(f"\nRunning {num_warmup} warmup iterations...")
        for i in range(num_warmup):
            _ = session.run(output_names, {input_name: input_data})
        
        # Benchmark
        print(f"\nRunning {num_runs} benchmark iterations...")
        latencies = []
        
        for i in range(num_runs):
            start_time = time.time()
            results = session.run(output_names, {input_name: input_data})
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_runs} iterations")
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        # Throughput calculation
        effective_batch_size = input_data.shape[0]
        throughput = effective_batch_size / avg_latency
        
        # Memory usage
        final_memory = get_gpu_memory_usage()
        memory_used_gb = final_memory - initial_memory if final_memory > initial_memory else final_memory
        
        print(f"\n=== {model_name.upper()} ONNX DETECTION BENCHMARK RESULTS ===")
        print(f"Framework: ONNX Runtime")
        print(f"Model Type: {model_type}")
        print(f"Execution Provider: {execution_provider}")
        print(f"Precision: {precision}")
        print(f"Batch Size: {effective_batch_size}")
        print(f"Input Shape: {input_data.shape}")
        print(f"Total GPU Memory Used: {memory_used_gb:.2f} GB")
        print()
        print("Performance Metrics:")
        print(f"Average Latency: {avg_latency*1000:.2f} ms")
        print(f"Std Latency: {std_latency*1000:.2f} ms")
        print(f"Min Latency: {min_latency*1000:.2f} ms")
        print(f"Max Latency: {max_latency*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print(f"Throughput (images/sec): {throughput:.2f}")
        print("=" * 60)
        
        # Print final result in expected format
        print(f"\nFINAL RESULT: {throughput:.2f} samples/sec")
        
        # Clean up temporary file only if using synthetic model
        if model_type == "Synthetic":
            try:
                os.unlink(model_path)
            except:
                pass
        
        return {
            'throughput_fps': throughput,
            'avg_latency_ms': avg_latency * 1000,
            'std_latency_ms': std_latency * 1000,
            'min_latency_ms': min_latency * 1000,
            'max_latency_ms': max_latency * 1000,
            'memory_used_gb': memory_used_gb,
            'execution_provider': execution_provider,
            'model_type': model_type
        }
        
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='ONNX YOLOv5 Detection Benchmark')
    parser.add_argument('--model', type=str, default='yolov5s',
                       choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'yolov5'],
                       help='YOLOv5 model variant')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'mixed'],
                       help='Inference precision')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--execution_provider', type=str, default='auto',
                       choices=['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider', 
                               'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'auto'],
                       help='ONNX Runtime execution provider (auto = select best available)')
    parser.add_argument('--num_warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--force_convert', action='store_true',
                       help='Force reconversion of PyTorch model to ONNX')
    
    args = parser.parse_args()
    
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Auto-select execution provider if requested
    available_providers = ort.get_available_providers()
    if args.execution_provider == 'auto':
        # Priority order: TensorRT > CUDA > ROCm > MIGraphX > CPU
        provider_priority = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 
                           'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'CPUExecutionProvider']
        for provider in provider_priority:
            if provider in available_providers:
                args.execution_provider = provider
                print(f"Auto-selected execution provider: {provider}")
                break
    else:
        # Validate specific execution provider
        if args.execution_provider not in available_providers:
            print(f"Error: {args.execution_provider} not available")
            print(f"Available providers: {available_providers}")
            sys.exit(1)
    
    # Force reconversion if requested
    if args.force_convert:
        onnx_model_path = get_yolo_onnx_model_path(args.model, args.precision)
        if onnx_model_path.exists():
            onnx_model_path.unlink()
            print(f"Removed existing ONNX model for forced reconversion")
    
    # Run benchmark
    results = benchmark_yolo_onnx_inference(
        args.model, args.precision, args.batch_size, args.execution_provider,
        args.num_warmup, args.num_runs
    )
    
    if results is None:
        sys.exit(1)

if __name__ == "__main__":
    main() 