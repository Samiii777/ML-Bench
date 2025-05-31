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