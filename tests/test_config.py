from core.discovery import resolve_benchmark_path
from utils import config


def test_every_default_model_resolves_to_a_script():
    for framework, models in config.MODELS_BY_FRAMEWORK.items():
        for model in models:
            assert resolve_benchmark_path(framework, model, "inference", config.get_use_case(model)), (framework, model)
    for model in config.PYTORCH_TRAINING_MODELS:
        assert resolve_benchmark_path("pytorch", model, "training", config.get_use_case(model)), model


def test_families():
    assert config.get_model_family("resnet50") == "torchvision"
    assert config.get_model_family("efficientnet_v2_s") == "torchvision"
    assert config.get_model_family("Qwen/Qwen3-8B") == "llm"
    assert config.get_use_case("yolo11s") == "detection"
    assert config.get_use_case("flux_2_klein_4b") == "generation"


def test_vram_precheck():
    assert config.vram_skip_reason("resnet50", "fp32", 64, 1.0) is None  # small models never pre-skipped
    assert config.vram_skip_reason("flux_1_dev", "bf16", 1, 23.0)
    assert config.vram_skip_reason("Qwen/Qwen3-8B", "bf16", 1, 23.0) is None
    assert config.vram_skip_reason("Qwen/Qwen3-8B", "fp32", 1, 23.0)  # fp32 doubles weights
