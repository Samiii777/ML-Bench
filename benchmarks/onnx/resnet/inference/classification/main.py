import onnxruntime as ort
import onnx
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import time
import sys
import os
import urllib.request

# Clean import of utils - no ugly relative paths!
import utils
from utils.download import get_imagenet_classes_path, get_sample_image_path

# Simple device utilities - everything in one place
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

def get_available_providers():
    """Get list of available ONNX execution providers on this system"""
    return ort.get_available_providers()

def get_optimal_providers():
    """Get optimal execution providers in order of preference"""
    available = get_available_providers()
    preferred_order = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider", 
        "CPUExecutionProvider"
    ]
    
    optimal = []
    for provider in preferred_order:
        if provider in available:
            optimal.append(provider)
    
    return optimal

def create_session(model_path, providers=None, precision="fp32"):
    """Create ONNX Runtime inference session with specified providers and precision"""
    if providers is None:
        providers = get_optimal_providers()
    
    # Ensure providers is a list
    if isinstance(providers, str):
        providers = [providers]
    
    # Configure session options for mixed precision
    session_options = ort.SessionOptions()
    
    if precision == "mixed":
        # Enable mixed precision optimizations
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable mixed precision for CUDA providers
        if any("CUDA" in provider or "Tensorrt" in provider for provider in providers):
            print("Enabling mixed precision optimizations for GPU execution")
            # These optimizations help with mixed precision performance
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_reuse = True
    
    try:
        session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
        return session
    except Exception as e:
        print(f"Failed to create session with providers {providers}: {e}")
        # Fallback to CPU
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=session_options)

def get_session_providers(session):
    """Get the actual providers used by the session"""
    return session.get_providers()

def print_provider_info():
    """Print information about available execution providers"""
    available = get_available_providers()
    optimal = get_optimal_providers()
    
    print(f"Available ONNX Execution Providers: {available}")
    print(f"Optimal Provider Order: {optimal}")

def synchronize_provider(provider_name):
    """Synchronize execution for timing (provider-specific)"""
    if "CUDA" in provider_name:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
    # For CPU and other providers, no explicit sync needed
    time.sleep(0.001)  # Small delay to ensure completion

# Simple download utilities
def download_file(url, filename):
    """Download file if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ {filename} downloaded")
    else:
        print(f"✓ {filename} already exists")

def get_imagenet_classes_path():
    """Get path to ImageNet classes file"""
    # Use the clean utils function instead of ugly relative paths
    from utils.download import get_imagenet_classes_path as utils_get_path
    return utils_get_path()

def get_sample_image_path():
    """Get path to sample image"""
    # Use the clean utils function instead of ugly relative paths
    from utils.download import get_sample_image_path as utils_get_path
    return utils_get_path()

def load_categories(filename):
    """Load the categories from the given file"""
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def preprocess_image(image_path, batch_size=1, precision="fp32"):
    """Preprocess the input image for inference"""
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    
    # Convert to numpy and create batch
    input_numpy = input_tensor.numpy()
    if batch_size > 1:
        input_batch = np.repeat(input_numpy[np.newaxis, :], batch_size, axis=0)
    else:
        input_batch = input_numpy[np.newaxis, :]
    
    # Convert input data to the specified precision
    if precision == "fp16":
        input_batch = input_batch.astype(np.float16)
    elif precision == "mixed":
        # For mixed precision, keep input as FP32 - ONNX Runtime handles the optimization
        input_batch = input_batch.astype(np.float32)
    else:  # fp32
        input_batch = input_batch.astype(np.float32)
    
    return input_batch

def convert_pytorch_to_onnx(model_name, onnx_path, precision="fp32"):
    """Convert PyTorch model to ONNX format with specified precision and dynamic batch size"""
    print(f"Converting {model_name} to ONNX format (precision: {precision}) with dynamic batch size...")
    
    # Load PyTorch model
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(weights="ResNet34_Weights.DEFAULT")
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(weights="ResNet101_Weights.DEFAULT")
    elif model_name == "resnet152":
        model = torchvision.models.resnet152(weights="ResNet152_Weights.DEFAULT")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.eval()
    
    # Convert model to FP16 if requested
    # For mixed precision, we keep the model in FP32 and let ONNX Runtime handle the optimization
    # Use batch size 1 for export, but enable dynamic batch size
    if precision == "fp16":
        model = model.half()
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float16)
    elif precision == "mixed":
        # For mixed precision, export as FP32 model - ONNX Runtime will optimize automatically
        dummy_input = torch.randn(1, 3, 224, 224)
        print("Exporting FP32 model for mixed precision optimization by ONNX Runtime")
    else:
        dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX with dynamic batch size support
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
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model converted and saved to {onnx_path} with dynamic batch size support")

def test_provider(onnx_model_path, provider, image_file, categories, params):
    """Test a specific execution provider"""
    # Create session with specific provider and precision settings
    session = create_session(onnx_model_path, providers=[provider], precision=params.precision)
    actual_providers = get_session_providers(session)
    
    print(f"Session created with providers: {actual_providers}")
    if params.precision == "mixed":
        print(f"Mixed precision optimizations enabled for {provider}")
    
    # Load and preprocess input image
    input_batch = preprocess_image(image_file, params.batch_size, params.precision)
    
    # Measure initial memory usage
    initial_memory = get_gpu_memory_nvidia_smi()
    
    # Warm-up runs
    print("Performing warm-up runs...")
    for _ in range(3):
        outputs = session.run(None, {"input": input_batch})
        synchronize_provider(provider)
    
    # Measure memory after warmup
    warmup_memory = get_gpu_memory_nvidia_smi()
    
    # Benchmark runs
    print("Running benchmark...")
    latencies = []
    num_runs = 10
    
    for i in range(num_runs):
        synchronize_provider(provider)
        start = time.time()
        outputs = session.run(None, {"input": input_batch})
        synchronize_provider(provider)
        latency = time.time() - start
        latencies.append(latency)
        print(f"Run {i+1}/{num_runs}: {latency*1000:.2f} ms")
    
    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = np.std(latencies)
    
    # Calculate throughput (samples per second)
    throughput = params.batch_size / avg_latency
    
    # Calculate per-sample latency
    per_sample_latency = avg_latency * 1000 / params.batch_size  # ms per sample
    
    # Measure final memory usage
    final_memory = get_gpu_memory_nvidia_smi()
    
    # Get predictions for the first image in batch
    output = outputs[0][0]  # First output, first batch item
    
    # Use numerically stable softmax to handle FP16 precision
    output_max = np.max(output)
    exp_output = np.exp(output - output_max)
    probabilities = exp_output / np.sum(exp_output)
    
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    
    print("\nTop 5 predictions:")
    for i, idx in enumerate(top5_indices):
        print(f"{categories[idx]}: {probabilities[idx]:.4f}")
    
    # Print results
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Model: {params.model}")
    print(f"Framework: ONNX Runtime")
    print(f"Execution Provider: {provider}")
    
    # Determine device based on execution provider
    if "CUDA" in provider or "Tensorrt" in provider:
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    print(f"Average Inference Time: {avg_latency*1000:.2f} ms")
    print(f"Per-sample Latency: {per_sample_latency:.2f} ms/sample")
    print(f"Min Inference Time: {min_latency*1000:.2f} ms")
    print(f"Max Inference Time: {max_latency*1000:.2f} ms")
    print(f"Std Inference Time: {std_latency*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
    
    # Memory information
    if final_memory:
        print(f"Total GPU Memory Used: {final_memory.get('total_gpu_used_gb', 0):.3f} GB")
        print(f"Total GPU Memory Available: {final_memory.get('total_gpu_total_gb', 0):.3f} GB")
        print(f"GPU Memory Utilization: {final_memory.get('gpu_utilization_percent', 0):.1f}%")
    
    print(f"ONNX Inference Time = {avg_latency*1000:.2f} ms")
    
    return {
        "avg_latency_ms": avg_latency * 1000,
        "per_sample_latency_ms": per_sample_latency,
        "min_latency_ms": min_latency * 1000,
        "max_latency_ms": max_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "throughput_fps": throughput,
        "samples_per_sec": throughput,
        "execution_provider": provider,
        "framework": "ONNX Runtime",
        "device": device,
        "top1_prediction": categories[top5_indices[0]],
        "top1_confidence": float(probabilities[top5_indices[0]]),
        "memory_usage": final_memory
    }

def print_provider_results(results, model_name, params):
    """Print comparison of results across execution providers"""
    print(f"\n{'='*80}")
    print("EXECUTION PROVIDER COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Provider':<25}{'Latency (ms)':<15}{'Throughput (samp/s)':<20}{'Status':<10}")
    print("-" * 70)
    
    for provider, result in results.items():
        if "error" in result:
            print(f"{provider:<25}{'N/A':<15}{'N/A':<20}{'FAILED':<10}")
        else:
            latency = result["avg_latency_ms"]
            throughput = result["throughput_fps"]
            print(f"{provider:<25}{latency:<15.2f}{throughput:<20.2f}{'PASS':<10}")

def run_inference(params):
    """Main inference function"""
    model_name = params.model.lower() if params.model else "resnet50"
    
    if model_name not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: resnet18, resnet34, resnet50, resnet101, resnet152")
    
    print(f"Running {model_name} ONNX inference benchmark")
    print(f"Precision: {params.precision}")
    print(f"Batch size: {params.batch_size}")
    
    # Download required files
    classes_file = get_imagenet_classes_path()
    image_file = get_sample_image_path()
    
    categories = load_categories(classes_file)
    print(f"Loading model: {model_name}")
    
    # Create ONNX model path with precision info only (batch size is dynamic)
    onnx_model_path = f"{model_name}_{params.precision}.onnx"
    
    # Convert PyTorch model to ONNX if not exists
    if not os.path.exists(onnx_model_path):
        convert_pytorch_to_onnx(model_name, onnx_model_path, params.precision)
    
    # Print provider information
    print_provider_info()
    
    # Get available providers
    available_providers = get_available_providers()
    optimal_providers = get_optimal_providers()
    
    # Determine which providers to test
    if hasattr(params, 'execution_provider') and params.execution_provider:
        # Test only the specified provider
        providers_to_test = [params.execution_provider]
        print(f"Testing specific execution provider: {params.execution_provider}")
    else:
        # Test only the best available provider (not all providers)
        providers_to_test = [optimal_providers[0]] if optimal_providers else ["CPUExecutionProvider"]
        print(f"Testing best execution provider: {providers_to_test[0]}")
    
    # Test each execution provider
    results = {}
    
    for provider in providers_to_test:
        if provider in available_providers:
            print(f"\n{'='*60}")
            print(f"Testing with {provider}")
            print(f"{'='*60}")
            
            try:
                result = test_provider(onnx_model_path, provider, image_file, categories, params)
                results[provider] = result
            except Exception as e:
                print(f"Failed to test {provider}: {e}")
                results[provider] = {"error": str(e)}
        else:
            print(f"Provider {provider} not available on this system")
            results[provider] = {"error": f"Provider not available"}
    
    # Print summary results if testing multiple providers
    if len(providers_to_test) > 1:
        print_provider_results(results, model_name, params)
    
    # Return the best result for the main framework
    best_provider = None
    best_result = None
    
    for provider, result in results.items():
        if "error" not in result and (best_result is None or result["avg_latency_ms"] < best_result["avg_latency_ms"]):
            best_provider = provider
            best_result = result
    
    if best_result:
        best_result["best_provider"] = best_provider
        return best_result
    else:
        # If all failed, return the first error
        for provider, result in results.items():
            if "error" in result:
                raise Exception(f"Execution provider {provider} failed: {result['error']}")
        raise Exception("All execution providers failed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ONNX ResNet Inference Benchmark")
    parser.add_argument("--model", type=str, default="resnet50", 
                       help="Model ID for the RESNET Pipeline")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "mixed"],
                       help="Precision for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--execution_provider", type=str, default=None,
                       choices=["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"],
                       help="Specific ONNX execution provider to test")
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