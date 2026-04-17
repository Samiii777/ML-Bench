"""
Simple configuration for the benchmarking framework
"""

# Configuration flags
SKIP_VRAM_CHECK = False  # Set to True to disable VRAM requirement checking

# Model family mappings
MODEL_FAMILIES = {
    # Classification — one canonical model per architecture
    'resnet50': 'resnet',
    'inceptionv3': 'inception',
    'inception_v3': 'inception',
    'vit_b_16': 'vit',
    'convnext_tiny': 'convnext',
    # Detection — one real-world model
    'yolov5s': 'yolo',
    # NLP
    'bert-base-uncased': 'bert',
    'bert': 'bert',
    # Image generation
    'stable_diffusion_1_5': 'stable_diffusion',
    'sd1.5': 'stable_diffusion',
    'stable_diffusion_3_medium': 'stable_diffusion',
    'sd3': 'stable_diffusion',
    'flux_1_schnell': 'flux',
    'flux_schnell': 'flux',
    'flux_1_dev': 'flux',
    'flux_dev': 'flux',
    # GPU compute
    'gemm_ops': 'gpu_ops',
    'conv_ops': 'gpu_ops',
    'memory_ops': 'gpu_ops',
    'elementwise_ops': 'gpu_ops',
    'reduction_ops': 'gpu_ops',
    # LLMs
    'meta-llama/Llama-3.1-8B': 'llama',
    'meta-llama/Llama-3.2-1B-Instruct': 'llama',
    'meta-llama/Llama-3.2-3B-Instruct': 'llama',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': 'llama',
    # Ollama
    'llama3.1:8b': 'ollama',
    'qwen2.5:7b': 'ollama',
    'deepseek-r1:8b': 'ollama',
    'gemma3:4b': 'ollama',
    'qwen3:8b': 'ollama',
    'llama3.2:3b': 'ollama',
    # ComfyUI
    'comfyui_flux_schnell': 'comfyui',
    'comfyui_flux_dev': 'comfyui',
}

# Available models per framework
PYTORCH_MODELS = [
    # Classification — one canonical model per architecture
    "resnet50",
    "inceptionv3",
    "vit_b_16",
    "convnext_tiny",
    # Detection
    "yolov5s",
    # NLP
    "bert-base-uncased",
    # Image generation
    "stable_diffusion_1_5",
    "stable_diffusion_3_medium",
    "flux_1_schnell",
    "flux_1_dev",
    # GPU compute
    "gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops",
    # LLMs
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
OLLAMA_MODELS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "qwen3:8b",
    "deepseek-r1:8b",
    "gemma3:4b",
]

COMFYUI_MODELS = [
    "comfyui_flux_schnell",
    "comfyui_flux_dev",
]

ONNX_MODELS = [
    "resnet50",
    "inceptionv3",
    "yolov5s",
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
DEFAULT_FRAMEWORKS = ["pytorch", "onnx", "ollama"]
DEFAULT_PRECISIONS = ["fp32", "fp16", "mixed"]
DEFAULT_TRAINING_PRECISIONS = ["fp32", "mixed"]  # No pure fp16 for training
DEFAULT_USE_CASE_PRECISIONS = {
    "classification": ["fp16"],
    "detection": ["fp16"],
    "generation": ["fp16"],         # SD/FLUX auto-select bf16 where needed
    "compute": ["fp32", "fp16"],    # Both matter — different hardware paths
    "text_generation": ["fp16"],
    "text_classification": ["fp16"],
}

# Framework-specific precision overrides
FRAMEWORK_PRECISION_OVERRIDES = {
    "ollama": {
        "text_generation": ["auto"]  # Ollama handles precision internally
    },
    "comfyui": {
        "generation": ["fp16"]  # ComfyUI FLUX models use FP16
    }
}
DEFAULT_BATCH_SIZES = [1, 4, 16]
DEFAULT_TRAINING_BATCH_SIZES = {
    "classification": [64],      # Large batch size works for classification
    "detection": [8],           # Smaller batch size needed for detection  
    "segmentation": [16],       # Medium batch size for segmentation
    "generation": [4],          # Very small for Stable Diffusion
    "compute": [64],            # Large for GPU compute operations
    "text_generation": [1],     # Small batch size for text generation
    "text_classification": [32] # Medium batch size for text classification
}

USE_CASE_BATCH_SIZES = {
    "classification": [1, 4, 16],
    "detection": [1, 4],
    "generation": [1, 2],           # VRAM-heavy, bs>2 OOMs on most GPUs for SD3/FLUX
    "compute": [1],                  # Raw ops, batch size is baked into the operation
    "text_generation": [1],          # Autoregressive, batching = concurrent sequences
    "text_classification": [1, 4, 16],
}

FRAMEWORK_BATCH_SIZE_OVERRIDES = {
    "ollama": {
        "text_generation": [1]
    },
    "comfyui": {
        "generation": [1]
    }
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
    # SD3.5 Medium - 2.5B MMDiT + T5 XXL + dual CLIP; fp16 slot actually runs bf16 (see benchmark script)
    'stable_diffusion_3_5_medium': {'fp32': 20.0, 'fp16': 10.0, 'mixed': 12.0},
    'sd3.5_medium': {'fp32': 20.0, 'fp16': 10.0, 'mixed': 12.0},
    'sd35_medium': {'fp32': 20.0, 'fp16': 10.0, 'mixed': 12.0},
    'sd3.5': {'fp32': 20.0, 'fp16': 10.0, 'mixed': 12.0},
    # SD3.5 Large Turbo - 8B params, optimized for 4-step inference (more efficient than regular SD3.5)
    'stable_diffusion_3_5_large_turbo': {'fp32': 16.0, 'fp16': 10.0, 'mixed': 14.0},
    'sd3.5_turbo': {'fp32': 16.0, 'fp16': 10.0, 'mixed': 14.0},
    'sd35_turbo': {'fp32': 16.0, 'fp16': 10.0, 'mixed': 14.0},
    'llama': {'fp32': 16.0, 'fp16': 8.0},
    'llama2': {'fp32': 16.0, 'fp16': 8.0},
    'llama-3': {'fp32': 16.0, 'fp16': 8.0},
    'llama3': {'fp32': 16.0, 'fp16': 8.0},
    'meta-llama/Llama-3.1-8B': {'fp32': 16.0, 'fp16': 8.0},
    # LLaMA 3.2 models
    'meta-llama/Llama-3.2-1B-Instruct': {'fp32': 3.0, 'fp16': 1.5},  # 1B parameters
    'meta-llama/Llama-3.2-3B-Instruct': {'fp32': 6.0, 'fp16': 3.0},  # 3B parameters
    # DeepSeek reasoning models (7B parameters)
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': {'fp32': 14.0, 'fp16': 7.0},
    'deepseek-ai/Deepseek-R1-Distill-Qwen-1.5B': {'fp32': 3.0, 'fp16': 1.5},  # 1.5B parameters
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': {'fp32': 16.0, 'fp16': 8.0},   # 8B parameters  
    'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': {'fp32': 16.0, 'fp16': 8.0},     # 8B parameters
    # FLUX Schnell - 12B params, optimized for 1-4 step inference (very fast)
    'flux_1_schnell': {'fp32': 48.0, 'fp16': 24.0, 'mixed': 33.0},
    'flux1_schnell': {'fp32': 48.0, 'fp16': 24.0, 'mixed': 33.0},
    'flux_schnell': {'fp32': 48.0, 'fp16': 24.0, 'mixed': 33.0},
    'flux.1-schnell': {'fp32': 48.0, 'fp16': 24.0, 'mixed': 33.0},
    # FLUX Dev - 12B params, full model requiring more steps and guidance (higher quality)
    'flux_1_dev': {'fp32': 50.0, 'fp16': 26.0, 'mixed': 35.0},
    'flux1_dev': {'fp32': 50.0, 'fp16': 26.0, 'mixed': 35.0},
    'flux_dev': {'fp32': 50.0, 'fp16': 26.0, 'mixed': 35.0},
    'flux.1-dev': {'fp32': 50.0, 'fp16': 26.0, 'mixed': 35.0},
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
        if any(pattern in model_lower for pattern in ['llama', 'llama-3']):
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
        elif any(pattern in model_lower for pattern in ['flux']):
            return 'flux'
        else:
            # Default to LLaMA family for unknown text generation models
            print(f"Unknown model family for '{model_name}', defaulting to 'llama'")
            return 'llama'
    
    # Prefix-based fallback for model variants (e.g. resnet18 -> resnet)
    model_lower = model_name.lower()
    for prefix, family in [('resnet', 'resnet'), ('yolov5', 'yolo'), ('yolov8', 'yolo'),
                           ('bert', 'bert'), ('vit_', 'vit'), ('convnext', 'convnext'),
                           ('inception', 'inception')]:
        if model_lower.startswith(prefix):
            return family

    return model_name

def get_unique_models(framework="pytorch"):
    """Get list of unique models for a framework, removing aliases.

    Derives from the canonical model lists (PYTORCH_MODELS, ONNX_MODELS, etc.)
    so there is only one place to maintain the model inventory.
    """
    source = {
        "pytorch": PYTORCH_MODELS,
        "onnx": ONNX_MODELS,
        "ollama": OLLAMA_MODELS,
        "comfyui": COMFYUI_MODELS,
    }.get(framework, PYTORCH_MODELS)

    return list(source)

def get_available_models(framework="pytorch"):
    """Get list of all available models for a framework (including aliases)"""
    if framework == "pytorch":
        return PYTORCH_MODELS.copy()
    elif framework == "onnx":
        return ONNX_MODELS.copy()
    elif framework == "ollama":
        return OLLAMA_MODELS.copy()
    elif framework == "comfyui":
        return COMFYUI_MODELS.copy()
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
    elif model_family == "ollama":
        return "text_generation"
    elif model_family == "flux":
        return "generation"  # FLUX Schnell is a generation model
    elif model_family == "comfyui":
        return "generation"
    elif model_family in ("vit", "convnext"):
        return "classification"
    else:
        return "classification"

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
    elif model_family == "flux":
        return ["pytorch"]  # Only PyTorch for FLUX Schnell
    elif model_family == "comfyui":
        return ["comfyui"]  # Only ComfyUI framework for ComfyUI models
    elif model_family == "ollama":
        return ["ollama"]
    elif model_family in ("vit", "convnext"):
        return ["pytorch"]
    else:
        return ["pytorch"]

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
        elif use_case == "text_generation" and model_family in ["llama", "ollama"]:
            compatible_models.append(model)
        elif use_case == "text_classification" and model_family == "bert":
            compatible_models.append(model)
        elif use_case == "generation" and model_family == "flux":
            compatible_models.append(model)
    
    return compatible_models

def get_available_frameworks_for_use_case(use_case):
    """Get list of available frameworks for a specific use case"""
    if use_case == "generation":
        return ["pytorch", "comfyui"]  # PyTorch and ComfyUI support image generation (Stable Diffusion, FLUX)
    elif use_case == "classification":
        return ["pytorch", "onnx"]  # Both frameworks support ResNet
    elif use_case == "detection":
        return ["pytorch", "onnx"]  # Both frameworks support ResNet for detection
    elif use_case == "segmentation":
        return ["pytorch", "onnx"]  # Both frameworks support ResNet for segmentation
    elif use_case == "compute":
        return ["pytorch"]  # Only PyTorch for GPU ops
    elif use_case == "text_generation":
        return ["pytorch", "ollama"]  # PyTorch and Ollama for text generation
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
        # bf16 and fp16 have identical memory footprint (2 bytes/param), so
        # fall back to the fp16 entry if no explicit bf16 value is provided
        # rather than jumping all the way to fp32.
        entry = VRAM_REQUIREMENTS[model_key]
        if precision == 'bf16':
            req = entry.get('bf16', entry.get('fp16', entry.get('fp32', 4.0)))
        else:
            req = entry.get(precision, entry.get('fp32', 4.0))
        
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

def get_precisions_for_use_case(use_case, mode="inference", framework="pytorch"):
    """Get precisions for a specific use case and mode"""
    # Check for framework-specific overrides first
    if framework in FRAMEWORK_PRECISION_OVERRIDES:
        if use_case in FRAMEWORK_PRECISION_OVERRIDES[framework]:
            return FRAMEWORK_PRECISION_OVERRIDES[framework][use_case]
    
    if mode == "training":
        return DEFAULT_TRAINING_PRECISIONS
    else:
        return DEFAULT_USE_CASE_PRECISIONS.get(use_case, DEFAULT_PRECISIONS)

def get_batch_sizes_for_use_case(use_case, mode="inference", framework="pytorch"):
    """Get batch sizes for a specific use case, mode, and framework"""
    # Check for framework-specific overrides first
    if framework in FRAMEWORK_BATCH_SIZE_OVERRIDES:
        if use_case in FRAMEWORK_BATCH_SIZE_OVERRIDES[framework]:
            return FRAMEWORK_BATCH_SIZE_OVERRIDES[framework][use_case]
    
    if mode == "training":
        return get_training_batch_sizes_for_use_case(use_case)
    else:
        return USE_CASE_BATCH_SIZES.get(use_case, DEFAULT_BATCH_SIZES)

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