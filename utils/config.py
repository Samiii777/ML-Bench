"""
Simple configuration for the benchmarking framework
"""

# Configuration flags
SKIP_VRAM_CHECK = False  # Set to True to disable VRAM requirement checking

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
    # DeepSeek reasoning models
    'deepseek-r1': 'llama',
    'deepseek-r1-7b': 'llama',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': 'llama',
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
    "meta-llama/Llama-2-13b", "meta-llama/Llama-2-70b",
    "deepseek-r1", "deepseek-r1-7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
]
ONNX_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "inceptionv3", "inception_v3",
    "yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov5",
    "bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased", "bert"
]

# ONNX Execution Providers (dynamically detected)
def _get_available_onnx_execution_providers():
    """Get list of available ONNX execution providers for this system"""
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        
        # Priority order: TensorRT > CUDA > ROCm > MIGraphX > CPU
        provider_priority = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider", 
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        # Return providers in priority order, only if available
        return [provider for provider in provider_priority if provider in available_providers]
        
    except ImportError:
        # Fallback if onnxruntime not available
        return ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"]

ONNX_EXECUTION_PROVIDERS = _get_available_onnx_execution_providers()

# Default settings
DEFAULT_FRAMEWORKS = ["pytorch", "onnx"]
DEFAULT_PRECISIONS = ["fp32", "fp16", "mixed"]
DEFAULT_TRAINING_PRECISIONS = ["fp32", "mixed"]  # No pure fp16 for training
DEFAULT_USE_CASE_PRECISIONS = {
    "classification": ["fp32", "fp16", "mixed"],
    "detection": ["fp32", "fp16", "mixed"],
    "segmentation": ["fp32", "fp16", "mixed"],
    "generation": ["fp32", "fp16", "mixed"],
    "compute": ["fp32", "fp16", "mixed"],
    "text_generation": ["fp16"],  # Skip fp32 for LLMs - slower and uses more memory
    "text_classification": ["fp32", "fp16", "mixed"]
}
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_TRAINING_BATCH_SIZES = {
    "classification": [64],      # Large batch size works for classification
    "detection": [8],           # Smaller batch size needed for detection  
    "segmentation": [16],       # Medium batch size for segmentation
    "generation": [4],          # Very small for Stable Diffusion
    "compute": [64],            # Large for GPU compute operations
    "text_generation": [1],     # Small batch size for text generation
    "text_classification": [32] # Medium batch size for text classification
}
DEFAULT_FRAMEWORK = "pytorch"
DEFAULT_MODE = "inference"
DEFAULT_USE_CASE = "classification"
DEFAULT_USE_CASES = ["classification", "detection", "segmentation", "generation", "compute", "text_generation", "text_classification"]

# VRAM requirements (GB) based on actual benchmark results - only for large models that need checking
# Values are for batch size 1; actual usage scales with batch size
VRAM_REQUIREMENTS = {
    # BERT models
    'bert-base-uncased': {'fp32': 2.0, 'fp16': 1.0, 'mixed': 1.5},
    'bert-base-cased': {'fp32': 2.0, 'fp16': 1.0, 'mixed': 1.5},
    'bert-large-uncased': {'fp32': 4.0, 'fp16': 2.0, 'mixed': 3.0},
    'bert-large-cased': {'fp32': 4.0, 'fp16': 2.0, 'mixed': 3.0},
    'bert': {'fp32': 2.0, 'fp16': 1.0, 'mixed': 1.5},
    'stable_diffusion_1_5': {'fp32': 6.5, 'fp16': 4.0, 'mixed': 8.0},
    'sd1.5': {'fp32': 6.5, 'fp16': 4.0, 'mixed': 8.0},
    'sd15': {'fp32': 6.5, 'fp16': 4.0, 'mixed': 8.0},
    'stable_diffusion_3_medium': {'fp32': 24.0, 'fp16': 18.5, 'mixed': '>24GB'},
    'sd3_medium': {'fp32': 24.0, 'fp16': 18.5, 'mixed': '>24GB'},
    'sd3': {'fp32': 24.0, 'fp16': 18.5, 'mixed': '>24GB'},
    'llama': {'fp32': 16.0, 'fp16': 8.0},
    'llama-2': {'fp32': 16.0, 'fp16': 8.0},
    'llama2': {'fp32': 16.0, 'fp16': 8.0},
    'llama-3': {'fp32': 16.0, 'fp16': 8.0},
    'llama3': {'fp32': 16.0, 'fp16': 8.0},
    'meta-llama/Llama-3.1-8B': {'fp32': 16.0, 'fp16': 8.0},
    'meta-llama/Llama-2-7b': {'fp32': 14.0, 'fp16': 7.0},
    'meta-llama/Llama-2-13b': {'fp32': 26.0, 'fp16': 13.0},
    'meta-llama/Llama-2-70b': {'fp32': '>24GB', 'fp16': 35.0},
    # DeepSeek reasoning models (7B parameters)
    'deepseek-r1': {'fp32': 14.0, 'fp16': 7.0},
    'deepseek-r1-7b': {'fp32': 14.0, 'fp16': 7.0},
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': {'fp32': 14.0, 'fp16': 7.0},
}

def get_model_family(model_name):
    """Get the model family for a given model"""
    # First check explicit mappings
    if model_name in MODEL_FAMILIES:
        return MODEL_FAMILIES[model_name]
    
    # For HuggingFace models (contain "/"), try to infer family from name
    if "/" in model_name:
        model_lower = model_name.lower()
        
        # Common model family patterns
        if any(pattern in model_lower for pattern in ['llama', 'llama-2', 'llama-3']):
            return 'llama'
        elif any(pattern in model_lower for pattern in ['bert', 'roberta', 'distilbert']):
            return 'bert'
        elif any(pattern in model_lower for pattern in ['deepseek', 'deepseek-r1']):
            return 'llama'  # DeepSeek reasoning models use LLaMA-like architecture
        elif any(pattern in model_lower for pattern in ['gpt', 'gpt2', 'gpt-2']):
            return 'llama'  # GPT models use similar architecture for our purposes
        elif any(pattern in model_lower for pattern in ['stable-diffusion', 'sd']):
            return 'stable_diffusion'
        elif any(pattern in model_lower for pattern in ['resnet']):
            return 'resnet'
        elif any(pattern in model_lower for pattern in ['yolo']):
            return 'yolo'
        elif any(pattern in model_lower for pattern in ['inception']):
            return 'inception'
        else:
            # Default to LLaMA family for unknown text generation models
            print(f"Unknown model family for '{model_name}', defaulting to 'llama'")
            return 'llama'
    
    # Fallback to the model name itself
    return model_name

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
            "meta-llama/Llama-2-70b",
            "deepseek-r1", "deepseek-r1-7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # DeepSeek reasoning model
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

def is_model_available(model_name, framework="pytorch"):
    """Check if a model is available for a framework (includes HuggingFace models)"""
    available_models = get_available_models(framework)
    
    # Check if it's in the predefined list
    if model_name in available_models:
        return True
    
    # If it contains "/", assume it's a HuggingFace model and it's available
    if "/" in model_name:
        return True
    
    return False

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
        return "text_classification"
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
        elif use_case == "text_generation" and model_family in ["llama"]:
            compatible_models.append(model)
        elif use_case == "text_classification" and model_family == "bert":
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
    elif use_case == "text_classification":
        return ["pytorch", "onnx"]  # Both frameworks support BERT
    else:
        return ["pytorch"]  # Default to PyTorch only

def get_vram_requirement(model: str, precision: str = 'fp32', batch_size: int = 1) -> str:
    """
    Get VRAM requirement for a model configuration based on actual benchmark results
    Returns string like "4.5GB" or ">24GB"
    Values are empirically measured for batch size 1 and scaled up for larger batches
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

def get_precisions_for_use_case(use_case, mode="inference"):
    """Get precisions for a specific use case and mode"""
    if mode == "training":
        return DEFAULT_TRAINING_PRECISIONS
    else:
        return DEFAULT_USE_CASE_PRECISIONS.get(use_case, DEFAULT_PRECISIONS)

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
    # Check if VRAM checking is disabled
    if SKIP_VRAM_CHECK:
        return False, "VRAM checking disabled"
    
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

def set_skip_vram_check(skip: bool):
    """
    Enable or disable VRAM requirement checking
    
    Args:
        skip (bool): True to disable VRAM checking, False to enable it
    """
    global SKIP_VRAM_CHECK
    SKIP_VRAM_CHECK = skip

def get_skip_vram_check() -> bool:
    """
    Get the current state of VRAM checking
    
    Returns:
        bool: True if VRAM checking is disabled, False if enabled
    """
    return SKIP_VRAM_CHECK

def disable_vram_check():
    """Convenience function to disable VRAM checking"""
    set_skip_vram_check(True)

def enable_vram_check():
    """Convenience function to enable VRAM checking"""
    set_skip_vram_check(False) 