#!/usr/bin/env python3
"""
InceptionV3 Classification Inference Benchmark for ONNX
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
from torchvision import models
from PIL import Image
from torchvision import transforms

# Add project root to path for utils import
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from utils.download import get_imagenet_classes_path, get_sample_image_path

def get_gpu_memory_usage():
    """Get GPU memory usage from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        memory_used_mb = int(result.stdout.strip())
        return memory_used_mb / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0

def load_categories(filename):
    """Load the categories from the given file"""
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def preprocess_image(image_path, batch_size=1):
    """Preprocess image for InceptionV3 (299x299 input size)"""
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(342),  # InceptionV3 uses larger resize
        transforms.CenterCrop(299),  # InceptionV3 uses 299x299 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    
    # Create batch and convert to numpy
    if batch_size > 1:
        input_batch = input_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        input_batch = input_tensor.unsqueeze(0)
    
    return input_batch.numpy()

def convert_pytorch_to_onnx(model_name, onnx_path, precision="fp32"):
    """Convert PyTorch InceptionV3 to ONNX format with specified precision and dynamic batch size"""
    print(f"Converting {model_name} to ONNX format (precision: {precision}) with dynamic batch size...")
    
    try:
        # Load PyTorch model
        print(f"Loading {model_name} from torchvision...")
        if model_name.lower() in ['inceptionv3', 'inception_v3']:
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model.eval()
        print(f"✓ Successfully loaded {model_name}")
        
        # Handle precision conversion
        if precision == "fp16":
            model = model.half()
            dummy_input = torch.randn(1, 3, 299, 299, dtype=torch.float16)
            print("Converting model to FP16 precision")
        elif precision == "mixed":
            # For mixed precision, export as FP32 model - ONNX Runtime will optimize automatically
            dummy_input = torch.randn(1, 3, 299, 299)
            print("Exporting FP32 model for mixed precision optimization by ONNX Runtime")
        else:
            dummy_input = torch.randn(1, 3, 299, 299)
        
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
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},    # Variable batch dimension
                'output': {0: 'batch_size'}    # Variable output batch dimension
            }
        )
        
        print(f"✓ Model converted and saved to {onnx_path} with dynamic batch size support")
        return True
        
    except Exception as e:
        print(f"Error converting {model_name}: {str(e)}")
        return False

def get_inception_onnx_model_path(model_name, precision):
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

def benchmark_inception_onnx_inference(model_name, precision, batch_size, execution_provider, num_warmup=10, num_runs=100):
    """Benchmark InceptionV3 ONNX inference performance"""
    
    print(f"Starting {model_name} ONNX classification benchmark")
    print(f"Precision: {precision}")
    print(f"Batch size: {batch_size}")
    print(f"Execution Provider: {execution_provider}")
    print(f"Warmup runs: {num_warmup}")
    print(f"Benchmark runs: {num_runs}")
    
    try:
        # Get ONNX model path
        onnx_model_path = get_inception_onnx_model_path(model_name, precision)
        
        # Convert PyTorch model to ONNX if not exists
        if not onnx_model_path.exists():
            print(f"ONNX model not found at {onnx_model_path}")
            success = convert_pytorch_to_onnx(model_name, str(onnx_model_path), precision)
            if not success:
                print("Failed to convert model to ONNX")
                return None
        else:
            print(f"Using existing ONNX model: {onnx_model_path}")
        
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
        else:
            providers = [execution_provider]
        
        session = ort.InferenceSession(str(onnx_model_path), sess_options=sess_options, providers=providers)
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Model input: {input_name} {input_shape}")
        print(f"Model outputs: {output_names}")
        
        # Load and preprocess sample image
        image_file = get_sample_image_path()
        input_data = preprocess_image(image_file, batch_size)
        
        # Convert precision if needed
        if precision == "fp16":
            input_data = input_data.astype(np.float16)
        
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
        
        # Get top predictions for the first sample
        classes_file = get_imagenet_classes_path()
        categories = load_categories(classes_file)
        
        output = results[0][0]  # First sample from batch
        probabilities = np.exp(output) / np.sum(np.exp(output))  # Softmax
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        
        print(f"\n=== {model_name.upper()} ONNX CLASSIFICATION BENCHMARK RESULTS ===")
        print(f"Framework: ONNX Runtime")
        print(f"Model Type: Real InceptionV3")
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
        print()
        print("Top 5 Predictions:")
        for i, idx in enumerate(top5_indices):
            print(f"{i+1}: {categories[idx]} ({probabilities[idx]*100:.2f}%)")
        print("=" * 60)
        
        # Print final result in expected format
        print(f"\nFINAL RESULT: {throughput:.2f} samples/sec")
        
        return {
            'throughput_fps': throughput,
            'avg_latency_ms': avg_latency * 1000,
            'std_latency_ms': std_latency * 1000,
            'min_latency_ms': min_latency * 1000,
            'max_latency_ms': max_latency * 1000,
            'memory_used_gb': memory_used_gb,
            'execution_provider': execution_provider,
            'model_type': "Real InceptionV3"
        }
        
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='ONNX InceptionV3 Classification Benchmark')
    parser.add_argument('--model', type=str, default='inceptionv3',
                       choices=['inceptionv3', 'inception_v3'],
                       help='InceptionV3 model variant')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'mixed'],
                       help='Inference precision')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--execution_provider', type=str, default='CUDAExecutionProvider',
                       choices=['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider'],
                       help='ONNX Runtime execution provider')
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
    
    # Validate execution provider
    available_providers = ort.get_available_providers()
    if args.execution_provider not in available_providers:
        print(f"Error: {args.execution_provider} not available")
        print(f"Available providers: {available_providers}")
        sys.exit(1)
    
    # Force reconversion if requested
    if args.force_convert:
        onnx_model_path = get_inception_onnx_model_path(args.model, args.precision)
        if onnx_model_path.exists():
            onnx_model_path.unlink()
            print(f"Removed existing ONNX model for forced reconversion")
    
    # Run benchmark
    results = benchmark_inception_onnx_inference(
        args.model, args.precision, args.batch_size, args.execution_provider,
        args.num_warmup, args.num_runs
    )
    
    if results is None:
        sys.exit(1)

if __name__ == "__main__":
    main() 