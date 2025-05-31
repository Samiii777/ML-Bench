"""
Simple configuration for the benchmarking framework
"""

# Model family mappings
MODEL_FAMILIES = {
    'resnet18': 'resnet',
    'resnet34': 'resnet', 
    'resnet50': 'resnet',
    'resnet101': 'resnet',
    'resnet152': 'resnet',
    'stable_diffusion_1_5': 'stable_diffusion',
    'sd1.5': 'stable_diffusion',
    'sd15': 'stable_diffusion',
    'stable_diffusion_3_medium': 'stable_diffusion',
    'sd3_medium': 'stable_diffusion',
    'sd3': 'stable_diffusion',
    'gpu_ops': 'gpu_ops',
    'gemm_ops': 'gpu_ops',
    'conv_ops': 'gpu_ops',
    'memory_ops': 'gpu_ops',
    'elementwise_ops': 'gpu_ops',
    'reduction_ops': 'gpu_ops',
}

# Memory requirements in GB for different models and configurations
# Format: model_name -> {precision -> {batch_size -> memory_gb}}
MODEL_MEMORY_REQUIREMENTS = {
    # ResNet models (approximate VRAM usage)
    'resnet18': {
        'fp32': {1: 0.5, 4: 1.2, 8: 2.0, 16: 3.5, 32: 6.5},
        'fp16': {1: 0.3, 4: 0.7, 8: 1.2, 16: 2.0, 32: 3.5},
        'mixed': {1: 0.4, 4: 0.9, 8: 1.6, 16: 2.8, 32: 5.0}
    },
    'resnet34': {
        'fp32': {1: 0.7, 4: 1.8, 8: 3.2, 16: 6.0, 32: 11.0},
        'fp16': {1: 0.4, 4: 1.0, 8: 1.8, 16: 3.2, 32: 6.0},
        'mixed': {1: 0.5, 4: 1.4, 8: 2.5, 16: 4.6, 32: 8.5}
    },
    'resnet50': {
        'fp32': {1: 1.0, 4: 2.5, 8: 4.5, 16: 8.5, 32: 16.0},
        'fp16': {1: 0.6, 4: 1.4, 8: 2.5, 16: 4.5, 32: 8.5},
        'mixed': {1: 0.8, 4: 1.9, 8: 3.5, 16: 6.5, 32: 12.0}
    },
    'resnet101': {
        'fp32': {1: 1.8, 4: 4.5, 8: 8.5, 16: 16.0, 32: 30.0},
        'fp16': {1: 1.0, 4: 2.5, 8: 4.5, 16: 8.5, 32: 16.0},
        'mixed': {1: 1.4, 4: 3.5, 8: 6.5, 16: 12.0, 32: 23.0}
    },
    'resnet152': {
        'fp32': {1: 2.5, 4: 6.0, 8: 11.0, 16: 21.0, 32: 40.0},
        'fp16': {1: 1.4, 4: 3.2, 8: 6.0, 16: 11.0, 32: 21.0},
        'mixed': {1: 1.9, 4: 4.6, 8: 8.5, 16: 16.0, 32: 30.0}
    },
    
    # Stable Diffusion models (high VRAM usage)
    'stable_diffusion_1_5': {
        'fp32': {1: 8.0, 2: 12.0, 4: 20.0, 8: 35.0},
        'fp16': {1: 4.0, 2: 6.0, 4: 10.0, 8: 18.0},
        'mixed': {1: 6.0, 2: 9.0, 4: 15.0, 8: 26.0}
    },
    'sd1.5': {  # Alias
        'fp32': {1: 8.0, 2: 12.0, 4: 20.0, 8: 35.0},
        'fp16': {1: 4.0, 2: 6.0, 4: 10.0, 8: 18.0},
        'mixed': {1: 6.0, 2: 9.0, 4: 15.0, 8: 26.0}
    },
    'sd15': {  # Alias
        'fp32': {1: 8.0, 2: 12.0, 4: 20.0, 8: 35.0},
        'fp16': {1: 4.0, 2: 6.0, 4: 10.0, 8: 18.0},
        'mixed': {1: 6.0, 2: 9.0, 4: 15.0, 8: 26.0}
    },
    'stable_diffusion_3_medium': {
        'fp32': {1: 22.0, 2: 40.0, 4: 70.0},  # Very high memory usage
        'fp16': {1: 12.0, 2: 20.0, 4: 35.0},
        'mixed': {1: 18.0, 2: 30.0, 4: 52.0}
    },
    'sd3_medium': {  # Alias
        'fp32': {1: 22.0, 2: 40.0, 4: 70.0},
        'fp16': {1: 12.0, 2: 20.0, 4: 35.0},
        'mixed': {1: 18.0, 2: 30.0, 4: 52.0}
    },
    'sd3': {  # Alias
        'fp32': {1: 22.0, 2: 40.0, 4: 70.0},
        'fp16': {1: 12.0, 2: 20.0, 4: 35.0},
        'mixed': {1: 18.0, 2: 30.0, 4: 52.0}
    },
    
    # GPU operations (moderate VRAM usage)
    'gemm_ops': {
        'fp32': {1: 2.0, 4: 4.0, 8: 7.0, 16: 12.0, 32: 22.0},
        'fp16': {1: 1.2, 4: 2.5, 8: 4.0, 16: 7.0, 32: 12.0},
        'mixed': {1: 1.6, 4: 3.2, 8: 5.5, 16: 9.5, 32: 17.0}
    },
    'conv_ops': {
        'fp32': {1: 1.5, 4: 3.5, 8: 6.0, 16: 10.0, 32: 18.0},
        'fp16': {1: 1.0, 4: 2.0, 8: 3.5, 16: 6.0, 32: 10.0},
        'mixed': {1: 1.2, 4: 2.7, 8: 4.7, 16: 8.0, 32: 14.0}
    },
    'memory_ops': {
        'fp32': {1: 1.0, 4: 2.0, 8: 3.5, 16: 6.0, 32: 10.0},
        'fp16': {1: 0.7, 4: 1.2, 8: 2.0, 16: 3.5, 32: 6.0},
        'mixed': {1: 0.8, 4: 1.6, 8: 2.7, 16: 4.7, 32: 8.0}
    },
    'elementwise_ops': {
        'fp32': {1: 0.8, 4: 1.5, 8: 2.5, 16: 4.5, 32: 8.0},
        'fp16': {1: 0.5, 4: 1.0, 8: 1.5, 16: 2.5, 32: 4.5},
        'mixed': {1: 0.6, 4: 1.2, 8: 2.0, 16: 3.5, 32: 6.2}
    },
    'reduction_ops': {
        'fp32': {1: 0.6, 4: 1.2, 8: 2.0, 16: 3.5, 32: 6.0},
        'fp16': {1: 0.4, 4: 0.7, 8: 1.2, 16: 2.0, 32: 3.5},
        'mixed': {1: 0.5, 4: 0.9, 8: 1.6, 16: 2.7, 32: 4.7}
    },
    'gpu_ops': {  # Generic GPU ops fallback
        'fp32': {1: 1.5, 4: 3.0, 8: 5.0, 16: 8.5, 32: 15.0},
        'fp16': {1: 1.0, 4: 1.8, 8: 3.0, 16: 5.0, 32: 8.5},
        'mixed': {1: 1.2, 4: 2.4, 8: 4.0, 16: 6.7, 32: 11.7}
    }
}

# Available models per framework
PYTORCH_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "stable_diffusion_1_5", "sd1.5", "sd15",
    "stable_diffusion_3_medium", "sd3_medium", "sd3",
    "gpu_ops", "gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"
]
ONNX_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
]

# ONNX Execution Providers (in order of preference)
ONNX_EXECUTION_PROVIDERS = [
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider", 
    "CPUExecutionProvider"
]

# All ONNX execution providers (same as default for now)
ONNX_EXECUTION_PROVIDERS_ALL = [
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider", 
    "CPUExecutionProvider"
]

# Default settings
DEFAULT_FRAMEWORKS = ["pytorch", "onnx"]
DEFAULT_PRECISIONS = ["fp32", "fp16", "mixed"]
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
DEFAULT_FRAMEWORK = "pytorch"
DEFAULT_MODE = "inference"
DEFAULT_USE_CASE = "classification"
DEFAULT_USE_CASES = ["classification", "generation", "compute"]

def get_model_family(model_name):
    """Get the model family for a given model"""
    return MODEL_FAMILIES.get(model_name, model_name)

def get_unique_models(framework="pytorch"):
    """Get list of unique models for a framework, removing aliases"""
    if framework == "pytorch":
        # Only include the canonical model names, not aliases
        return [
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "stable_diffusion_1_5", "stable_diffusion_3_medium",  # Both SD models as separate entries
            "gpu_ops", "gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"  # GPU operations benchmark
        ]
    elif framework == "onnx":
        return [
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ]
    else:
        return get_unique_models("pytorch")  # Default to pytorch

def get_available_models(framework="pytorch"):
    """Get list of all available models for a framework (including aliases)"""
    if framework == "pytorch":
        return PYTORCH_MODELS.copy()
    elif framework == "onnx":
        return ONNX_MODELS.copy()
    else:
        return PYTORCH_MODELS.copy()  # Default to pytorch

def get_onnx_execution_providers():
    """Get list of ONNX execution providers"""
    return ONNX_EXECUTION_PROVIDERS.copy()

def get_all_onnx_execution_providers():
    """Get list of all ONNX execution providers for comprehensive testing"""
    return ONNX_EXECUTION_PROVIDERS_ALL.copy()

def get_default_frameworks():
    """Get list of default frameworks to test when none specified"""
    return DEFAULT_FRAMEWORKS.copy()

def get_default_use_case_for_model(model_name):
    """Get the default use case for a given model"""
    model_family = get_model_family(model_name)
    
    # Map model families to their appropriate use cases
    if model_family == "stable_diffusion":
        return "generation"
    elif model_family == "resnet":
        return "classification"
    elif model_family == "gpu_ops":
        return "compute"
    else:
        return "classification"  # Default fallback

def get_available_frameworks_for_model(model_name):
    """Get list of available frameworks for a specific model"""
    model_family = get_model_family(model_name)
    
    # Map model families to their available frameworks
    if model_family == "stable_diffusion":
        return ["pytorch"]  # Only PyTorch for Stable Diffusion
    elif model_family == "resnet":
        return ["pytorch", "onnx"]  # Both frameworks for ResNet
    elif model_family == "gpu_ops":
        return ["pytorch"]  # Only PyTorch for GPU ops
    else:
        return ["pytorch"]  # Default to PyTorch only

def get_models_for_use_case(use_case, framework="pytorch"):
    """Get list of models that are compatible with a specific use case"""
    all_models = get_unique_models(framework)
    compatible_models = []
    
    for model in all_models:
        model_use_case = get_default_use_case_for_model(model)
        if model_use_case == use_case:
            compatible_models.append(model)
    
    return compatible_models

def get_available_frameworks_for_use_case(use_case):
    """Get list of available frameworks for a specific use case"""
    if use_case == "generation":
        return ["pytorch"]  # Only PyTorch supports Stable Diffusion
    elif use_case == "classification":
        return ["pytorch", "onnx"]  # Both frameworks support ResNet
    elif use_case == "compute":
        return ["pytorch"]  # Only PyTorch for GPU ops
    else:
        return ["pytorch"]  # Default to PyTorch only 

def get_memory_requirement(model_name: str, precision: str, batch_size: int) -> float:
    """
    Get estimated memory requirement for a model configuration
    
    Args:
        model_name: Name of the model
        precision: Precision mode (fp32, fp16, mixed)
        batch_size: Batch size for inference
        
    Returns:
        Estimated memory requirement in GB
    """
    # Normalize model name
    model_key = model_name.lower()
    
    # Check if we have specific requirements for this model
    if model_key in MODEL_MEMORY_REQUIREMENTS:
        model_reqs = MODEL_MEMORY_REQUIREMENTS[model_key]
        
        # Check if we have requirements for this precision
        if precision in model_reqs:
            precision_reqs = model_reqs[precision]
            
            # Find closest batch size (use exact match or interpolate)
            if batch_size in precision_reqs:
                return precision_reqs[batch_size]
            else:
                # Interpolate or extrapolate based on available batch sizes
                available_batch_sizes = sorted(precision_reqs.keys())
                
                if batch_size < min(available_batch_sizes):
                    # Extrapolate down (assume linear scaling)
                    min_bs = min(available_batch_sizes)
                    return precision_reqs[min_bs] * (batch_size / min_bs)
                elif batch_size > max(available_batch_sizes):
                    # Extrapolate up (assume linear scaling)
                    max_bs = max(available_batch_sizes)
                    return precision_reqs[max_bs] * (batch_size / max_bs)
                else:
                    # Interpolate between two values
                    lower_bs = max([bs for bs in available_batch_sizes if bs < batch_size])
                    upper_bs = min([bs for bs in available_batch_sizes if bs > batch_size])
                    
                    lower_mem = precision_reqs[lower_bs]
                    upper_mem = precision_reqs[upper_bs]
                    
                    # Linear interpolation
                    ratio = (batch_size - lower_bs) / (upper_bs - lower_bs)
                    return lower_mem + ratio * (upper_mem - lower_mem)
        else:
            # Fallback to fp32 requirements with precision scaling
            if 'fp32' in model_reqs:
                fp32_req = get_memory_requirement(model_name, 'fp32', batch_size)
                if precision == 'fp16':
                    return fp32_req * 0.6  # FP16 uses ~60% of FP32 memory
                elif precision == 'mixed':
                    return fp32_req * 0.8  # Mixed uses ~80% of FP32 memory
    
    # Fallback: estimate based on model family
    model_family = get_model_family(model_name)
    
    if model_family == 'stable_diffusion':
        # Conservative estimates for unknown SD models
        base_memory = 15.0 if 'sd3' in model_name.lower() else 6.0
    elif model_family == 'resnet':
        # Conservative estimates for ResNet models
        base_memory = 2.0
    elif model_family == 'gpu_ops':
        # Conservative estimates for GPU operations
        base_memory = 2.0
    else:
        # Very conservative fallback
        base_memory = 3.0
    
    # Scale by batch size (assume roughly linear)
    memory_estimate = base_memory * batch_size
    
    # Scale by precision
    if precision == 'fp16':
        memory_estimate *= 0.6
    elif precision == 'mixed':
        memory_estimate *= 0.8
    
    return memory_estimate

def check_memory_availability(model_name: str, precision: str, batch_size: int, 
                            available_memory_gb: float, safety_margin: float = 2.0) -> tuple[bool, float, str]:
    """
    Check if a model configuration will fit in available GPU memory
    
    Args:
        model_name: Name of the model
        precision: Precision mode (fp32, fp16, mixed)
        batch_size: Batch size for inference
        available_memory_gb: Available GPU memory in GB
        safety_margin: Safety margin in GB to account for overhead
        
    Returns:
        Tuple of (will_fit: bool, required_memory: float, recommendation: str)
    """
    required_memory = get_memory_requirement(model_name, precision, batch_size)
    effective_available = available_memory_gb - safety_margin
    
    will_fit = required_memory <= effective_available
    
    if will_fit:
        recommendation = f"✅ Configuration should fit (requires {required_memory:.1f}GB, {effective_available:.1f}GB available)"
    else:
        # Generate recommendations
        recommendations = []
        
        # Try smaller batch size
        if batch_size > 1:
            smaller_bs = max(1, batch_size // 2)
            smaller_req = get_memory_requirement(model_name, precision, smaller_bs)
            if smaller_req <= effective_available:
                recommendations.append(f"reduce batch size to {smaller_bs}")
        
        # Try FP16 if using FP32
        if precision == 'fp32':
            fp16_req = get_memory_requirement(model_name, 'fp16', batch_size)
            if fp16_req <= effective_available:
                recommendations.append(f"use fp16 precision")
        
        # Try FP16 + smaller batch size
        if precision == 'fp32' and batch_size > 1:
            smaller_bs = max(1, batch_size // 2)
            fp16_smaller_req = get_memory_requirement(model_name, 'fp16', smaller_bs)
            if fp16_smaller_req <= effective_available:
                recommendations.append(f"use fp16 precision with batch size {smaller_bs}")
        
        if recommendations:
            rec_text = f"❌ Insufficient memory (requires {required_memory:.1f}GB, {effective_available:.1f}GB available). Try: " + " or ".join(recommendations)
        else:
            rec_text = f"❌ Insufficient memory (requires {required_memory:.1f}GB, {effective_available:.1f}GB available). Model may not fit on this GPU."
        
        recommendation = rec_text
    
    return will_fit, required_memory, recommendation 