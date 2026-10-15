"""Model registry and default test matrix.

A model maps to a *family*, which is the directory under benchmarks/<framework>/
holding its script. The default matrix is: every model in a framework's list ×
the precisions and batch sizes for that model's use case.
"""

from typing import Dict, List, Optional, Tuple

# ── models ────────────────────────────────────────────────────────────────

TORCHVISION_PREFIXES = ("resnet", "vit_", "convnext", "efficientnet", "regnet", "swin", "mobilenet", "maxvit")

MODEL_FAMILIES: Dict[str, str] = {
    # GPU micro-benchmarks
    **{g: "gpu_ops" for g in ("gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops")},
    # Detection (Ultralytics)
    "yolo11n": "yolo", "yolo11s": "yolo", "yolo11m": "yolo",
    # Encoder NLP
    "bert-base": "bert",
    # Image generation (diffusers)
    "stable_diffusion_1_5": "diffusion",
    "stable_diffusion_3_5_medium": "diffusion",
    "stable_diffusion_3_5_large_turbo": "diffusion",
    "flux_1_schnell": "diffusion",
    "flux_1_dev": "diffusion",
    "flux_2_klein_4b": "diffusion",
    "z_image_turbo": "diffusion",
}

USE_CASE_BY_FAMILY = {
    "torchvision": "classification",
    "yolo": "detection",
    "bert": "text_classification",
    "llm": "text_generation",
    "diffusion": "generation",
    "gpu_ops": "compute",
}

PYTORCH_MODELS = [
    "resnet50", "vit_b_16", "convnext_tiny",
    "yolo11s",
    "bert-base",
    "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B",
    "stable_diffusion_1_5", "stable_diffusion_3_5_medium", "flux_2_klein_4b", "z_image_turbo",
    "gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops",
]
PYTORCH_TRAINING_MODELS = ["resnet50", "vit_b_16"]
ONNX_MODELS = ["resnet50", "yolo11s"]
OLLAMA_MODELS = ["llama3.1:8b", "qwen3:8b", "gemma3:4b", "gpt-oss:20b"]
COMFYUI_MODELS = ["comfyui_flux_schnell", "comfyui_flux_dev"]
LLAMACPP_MODELS: List[str] = []  # pass GGUF paths with --model

MODELS_BY_FRAMEWORK = {
    "pytorch": PYTORCH_MODELS,
    "onnx": ONNX_MODELS,
    "ollama": OLLAMA_MODELS,
    "comfyui": COMFYUI_MODELS,
    "llamacpp": LLAMACPP_MODELS,
}

DEFAULT_FRAMEWORKS = ["pytorch", "onnx", "ollama"]

# ── default matrix ────────────────────────────────────────────────────────

USE_CASE_PRECISIONS = {
    "classification": ["fp16"],
    "detection": ["fp16"],
    "text_classification": ["fp16"],
    "text_generation": ["bf16"],
    "generation": ["bf16"],
    "compute": ["fp32", "fp16"],
}
TRAINING_PRECISIONS = ["bf16"]  # AMP

USE_CASE_BATCH_SIZES = {
    "classification": [1, 16, 64],
    "detection": [1, 8],
    "text_classification": [1, 16, 64],
    "text_generation": [1, 8],
    "generation": [1],
    "compute": [1],
}
TRAINING_BATCH_SIZES = [64]

# Frameworks that manage precision / batching themselves
FRAMEWORK_PRECISION = {"ollama": "auto", "llamacpp": "auto", "comfyui": "fp16"}

# ── VRAM pre-check ──────────────────────────────────────────────────────
# Approximate peak GB in 16-bit (weights + runtime) for models big enough to
# matter; fp32 doubles it. Only skips configs that clearly cannot fit, to avoid
# minutes of loading before an OOM. Batch size is ignored on purpose: for LLMs
# at these lengths the KV cache is small (Llama-3.1-8B: 15.0 GB at bs=1, 15.2 at
# bs=8 on an RTX 4090); anything that still OOMs is reported as an "OOM:" FAIL.
VRAM_GB_16BIT = {
    "Qwen/Qwen3-8B": 17.0,
    "meta-llama/Llama-3.1-8B": 15.5,
    "stable_diffusion_3_5_large_turbo": 28.0,
    "flux_1_schnell": 33.0,
    "flux_1_dev": 33.0,
}


def get_model_family(model: str) -> str:
    if model in MODEL_FAMILIES:
        return MODEL_FAMILIES[model]
    if "/" in model:  # Hugging Face causal LM id
        return "llm"
    if model.lower().startswith(TORCHVISION_PREFIXES):
        return "torchvision"
    return model


def get_use_case(model: str) -> str:
    return USE_CASE_BY_FAMILY.get(get_model_family(model), "text_generation")


def get_models(framework: str, mode: str = "inference") -> List[str]:
    if mode == "training":
        return list(PYTORCH_TRAINING_MODELS) if framework == "pytorch" else []
    return list(MODELS_BY_FRAMEWORK.get(framework, []))


def get_precisions(framework: str, use_case: str, mode: str = "inference") -> List[str]:
    if framework in FRAMEWORK_PRECISION:
        return [FRAMEWORK_PRECISION[framework]]
    if mode == "training":
        return list(TRAINING_PRECISIONS)
    return list(USE_CASE_PRECISIONS.get(use_case, ["fp32"]))


def get_batch_sizes(framework: str, use_case: str, mode: str = "inference") -> List[int]:
    if framework in FRAMEWORK_PRECISION:
        return [1]
    if mode == "training":
        return list(TRAINING_BATCH_SIZES)
    return list(USE_CASE_BATCH_SIZES.get(use_case, [1]))


def vram_skip_reason(model: str, precision: str, batch_size: int, free_gb: float) -> Optional[str]:
    """Reason string if the config clearly won't fit, else None."""
    base = VRAM_GB_16BIT.get(model)
    if base is None:
        return None
    need = base * (2 if precision in ("fp32", "tf32", "mixed") else 1)
    if need > free_gb:
        return f"needs ~{need:.0f} GB VRAM, {free_gb:.1f} GB free (override with --skip-vram-check)"
    return None
