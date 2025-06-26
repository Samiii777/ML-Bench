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
    # InceptionV3 models
    'inceptionv3': 'inception',
    'inception_v3': 'inception',
    # YOLOv5 models  
    'yolov5s': 'yolo',
    'yolov5m': 'yolo',
    'yolov5l': 'yolo',
    'yolov5x': 'yolo',
    'yolov5': 'yolo',
    # BERT models
    'bert-base-uncased': 'bert',
    'bert-base-cased': 'bert',
    'bert-large-uncased': 'bert',
    'bert-large-cased': 'bert',
    'bert': 'bert',
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
    'llama': 'llama',
    'llama-2': 'llama',
    'llama2': 'llama',
    'llama-3': 'llama',
    'llama3': 'llama',
    'meta-llama/Llama-3.1-8B': 'llama',
    'meta-llama/Llama-2-7b': 'llama',
    'meta-llama/Llama-2-13b': 'llama',
    'meta-llama/Llama-2-70b': 'llama',
}

# Available models per framework
PYTORCH_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "inceptionv3", "inception_v3",
    "yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov5",
    "bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased", "bert",
    "stable_diffusion_1_5", "sd1.5", "sd15",
    "stable_diffusion_3_medium", "sd3_medium", "sd3",
    "gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops",
    "llama", "llama-2", "llama2", "llama-3", "llama3",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-2-7b",
    "meta-llama/Llama-2-13b", "meta-llama/Llama-2-70b"
]
ONNX_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "inceptionv3", "inception_v3",
    "yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov5",
    "bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased", "bert"
]

# ONNX Execution Providers (in order of preference)
ONNX_EXECUTION_PROVIDERS = [
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider", 
    "CPUExecutionProvider"
]

# Default settings
DEFAULT_FRAMEWORKS = ["pytorch", "onnx"]
DEFAULT_PRECISIONS = ["fp32", "fp16", "mixed"]
DEFAULT_TRAINING_PRECISIONS = ["fp32", "mixed"]  # No pure fp16 for training
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_TRAINING_BATCH_SIZES = {
    "classification": [64],      # Large batch size works for classification
    "detection": [8],           # Smaller batch size needed for detection  
    "segmentation": [16],       # Medium batch size for segmentation
    "generation": [4],          # Very small for Stable Diffusion
    "compute": [64],            # Large for GPU compute operations
    "text_generation": [1]      # Small batch size for text generation
}
DEFAULT_FRAMEWORK = "pytorch"
DEFAULT_MODE = "inference"
DEFAULT_USE_CASE = "classification"
DEFAULT_USE_CASES = ["classification", "detection", "segmentation", "generation", "compute", "text_generation"]

# Simple VRAM requirement estimates (GB) - only for large models that need checking
VRAM_REQUIREMENTS = {
    # BERT models
    'bert-base-uncased': {'fp32': 2.0, 'fp16': 1.0, 'mixed': 1.5},
    'bert-base-cased': {'fp32': 2.0, 'fp16': 1.0, 'mixed': 1.5},
    'bert-large-uncased': {'fp32': 4.0, 'fp16': 2.0, 'mixed': 3.0},
    'bert-large-cased': {'fp32': 4.0, 'fp16': 2.0, 'mixed': 3.0},
    'bert': {'fp32': 2.0, 'fp16': 1.0, 'mixed': 1.5},
    'stable_diffusion_1_5': {'fp32': 12.0, 'fp16': 6.0, 'mixed': 9.0},
    'sd1.5': {'fp32': 12.0, 'fp16': 6.0, 'mixed': 9.0},
    'sd15': {'fp32': 12.0, 'fp16': 6.0, 'mixed': 9.0},
    'stable_diffusion_3_medium': {'fp32': '>24GB', 'fp16': 20.0, 'mixed': '>24GB'},
    'sd3_medium': {'fp32': '>24GB', 'fp16': 20.0, 'mixed': '>24GB'},
    'sd3': {'fp32': '>24GB', 'fp16': 20.0, 'mixed': '>24GB'},
    'llama': {'fp32': 16.0, 'fp16': 8.0, 'mixed': 12.0},
    'llama-2': {'fp32': 16.0, 'fp16': 8.0, 'mixed': 12.0},
    'llama2': {'fp32': 16.0, 'fp16': 8.0, 'mixed': 12.0},
    'llama-3': {'fp32': 16.0, 'fp16': 8.0, 'mixed': 12.0},
    'llama3': {'fp32': 16.0, 'fp16': 8.0, 'mixed': 12.0},
    'meta-llama/Llama-3.1-8B': {'fp32': 16.0, 'fp16': 8.0, 'mixed': 12.0},
    'meta-llama/Llama-2-7b': {'fp32': 14.0, 'fp16': 7.0, 'mixed': 10.0},
    'meta-llama/Llama-2-13b': {'fp32': 26.0, 'fp16': 13.0, 'mixed': 20.0},
    'meta-llama/Llama-2-70b': {'fp32': '>24GB', 'fp16': 35.0, 'mixed': '>24GB'},
}

def get_model_family(model_name):
    """Get the model family for a given model"""
    return MODEL_FAMILIES.get(model_name, model_name)

def get_unique_models(framework="pytorch"):
    """Get list of unique models for a framework, removing aliases"""
    if framework == "pytorch":
        # Only include the canonical model names, not aliases
        return [
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "inceptionv3",  # InceptionV3 model
            "yolov5s", "yolov5m", "yolov5l", "yolov5x",  # YOLOv5 variants
            "bert-base-uncased", "bert-large-uncased",  # BERT models
            "stable_diffusion_1_5", "stable_diffusion_3_medium",  # Both SD models as separate entries
            "gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops",  # GPU operations benchmark
            "meta-llama/Llama-3.1-8B",  # Latest LLaMA model
            "meta-llama/Llama-2-7b",    # LLaMA 2 models
            "meta-llama/Llama-2-13b",
            "meta-llama/Llama-2-70b"
        ]
    elif framework == "onnx":
        return [
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "inceptionv3",  # InceptionV3 model
            "yolov5s", "yolov5m", "yolov5l", "yolov5x",  # YOLOv5 variants
            "bert-base-uncased", "bert-large-uncased"  # BERT models
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

def get_default_frameworks():
    """Get list of default frameworks to test when none specified"""
    return DEFAULT_FRAMEWORKS.copy()

def get_default_use_cases():
    """Get list of default use cases to test when none specified"""
    return DEFAULT_USE_CASES.copy()

def get_default_use_case_for_model(model_name):
    """Get the default use case for a given model"""
    model_family = get_model_family(model_name)
    
    if model_family == "stable_diffusion":
        return "generation"
    elif model_family == "resnet":
        return "classification"
    elif model_family == "inception":
        return "classification"
    elif model_family == "yolo":
        return "detection"
    elif model_family == "bert":
        return "text_generation"
    elif model_family == "gpu_ops":
        return "compute"
    elif model_family == "llama":
        return "text_generation"
    else:
        return "classification"  # Default fallback

def get_available_frameworks_for_model(model_name):
    """Get list of available frameworks for a specific model"""
    model_family = get_model_family(model_name)
    
    if model_family == "stable_diffusion":
        return ["pytorch"]  # Only PyTorch for Stable Diffusion
    elif model_family == "resnet":
        return ["pytorch", "onnx"]  # Both frameworks for ResNet
    elif model_family == "inception":
        return ["pytorch", "onnx"]  # Both frameworks for InceptionV3
    elif model_family == "yolo":
        return ["pytorch", "onnx"]  # Both frameworks for YOLOv5
    elif model_family == "bert":
        return ["pytorch", "onnx"]  # Both frameworks for BERT
    elif model_family == "gpu_ops":
        return ["pytorch"]  # Only PyTorch for GPU ops
    elif model_family == "llama":
        return ["pytorch"]  # Only PyTorch for llama
    else:
        return ["pytorch"]  # Default to PyTorch only

def get_models_for_use_case(use_case, framework="pytorch"):
    """Get list of models that are compatible with a specific use case"""
    all_models = get_unique_models(framework)
    compatible_models = []
    
    for model in all_models:
        model_family = get_model_family(model)
        if use_case == "classification" and model_family in ["resnet", "inception"]:
            compatible_models.append(model)
        elif use_case == "detection" and model_family in ["resnet", "yolo"]:
            compatible_models.append(model)
        elif use_case == "segmentation" and model_family == "resnet":
            compatible_models.append(model)
        elif use_case == "generation" and model_family == "stable_diffusion":
            compatible_models.append(model)
        elif use_case == "compute" and model_family == "gpu_ops":
            compatible_models.append(model)
        elif use_case == "text_generation" and model_family in ["llama", "bert"]:
            compatible_models.append(model)
    
    return compatible_models

def get_available_frameworks_for_use_case(use_case):
    """Get list of available frameworks for a specific use case"""
    if use_case == "generation":
        return ["pytorch"]  # Only PyTorch supports Stable Diffusion
    elif use_case == "classification":
        return ["pytorch", "onnx"]  # Both frameworks support ResNet
    elif use_case == "detection":
        return ["pytorch", "onnx"]  # Both frameworks support ResNet for detection
    elif use_case == "segmentation":
        return ["pytorch", "onnx"]  # Both frameworks support ResNet for segmentation
    elif use_case == "compute":
        return ["pytorch"]  # Only PyTorch for GPU ops
    elif use_case == "text_generation":
        return ["pytorch"]  # Only PyTorch for text generation
    else:
        return ["pytorch"]  # Default to PyTorch only

def get_vram_requirement(model: str, precision: str = 'fp32', batch_size: int = 1) -> str:
    """
    Get VRAM requirement estimate for a model configuration
    Returns string like "4.5GB" or ">24GB"
    """
    model_key = model.lower()
    
    if model_key in VRAM_REQUIREMENTS:
        req = VRAM_REQUIREMENTS[model_key].get(precision, VRAM_REQUIREMENTS[model_key].get('fp32', 4.0))
        
        if isinstance(req, str):  # Already formatted like ">24GB"
            return req
        else:
            # Scale by batch size (rough approximation)
            scaled_req = req * (1 + (batch_size - 1) * 0.8)  # Each additional batch adds ~80% more
            if scaled_req > 24:
                return ">24GB"
            else:
                return f"{scaled_req:.1f}GB"
    else:
        # Small model not in requirements table - return minimal requirement
        return "1.0GB"

def get_available_use_cases_for_training(framework="pytorch"):
    """Get list of use cases that have training implementations for a framework"""
    if framework == "pytorch":
        # Only PyTorch has training implementations
        return ["classification", "detection", "segmentation", "text_generation"]
    elif framework == "onnx":
        # ONNX doesn't have training scripts (inference only)
        return []
    else:
        return []

def get_training_batch_sizes_for_use_case(use_case):
    """Get training batch sizes for a specific use case"""
    return DEFAULT_TRAINING_BATCH_SIZES.get(use_case, [32])  # Default fallback

def should_skip_use_case_for_mode(use_case, mode, framework):
    """Check if a use case should be skipped for a specific mode and framework"""
    if mode == "training":
        available_training_use_cases = get_available_use_cases_for_training(framework)
        return use_case not in available_training_use_cases
    else:
        # For inference, use the existing logic
        available_frameworks = get_available_frameworks_for_use_case(use_case)
        return framework not in available_frameworks

def should_skip_for_vram(model: str, precision: str, batch_size: int, available_vram_gb: float) -> tuple[bool, str]:
    """
    Check if a configuration should be skipped due to VRAM constraints
    Returns (should_skip, reason)
    """
    model_key = model.lower()
    
    # Only check VRAM for models in the requirements table (large models like Stable Diffusion)
    if model_key not in VRAM_REQUIREMENTS:
        return False, f"Small model - no VRAM check needed"
    
    requirement = get_vram_requirement(model, precision, batch_size)
    
    if requirement == ">24GB":
        return True, f"Requires >24GB VRAM (available: {available_vram_gb:.1f}GB)"
    
    try:
        required_gb = float(requirement.replace('GB', ''))
        if required_gb > available_vram_gb * 0.9:  # 90% safety margin
            return True, f"Requires {requirement} VRAM (available: {available_vram_gb:.1f}GB)"
        else:
            return False, f"Should fit: {requirement} required, {available_vram_gb:.1f}GB available"
    except:
        return False, "Unknown VRAM requirement" 