#!/usr/bin/env python3
"""ONNX ResNet Detection Inference Benchmark (simplified detection model)"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from core.harness import OnnxHarness
from core.schema import BenchmarkMeta
from core.args import build_base_parser, add_onnx_args
from utils.download import get_sample_image_path

BENCHMARK_META = BenchmarkMeta(
    framework="onnx",
    model_family="resnet",
    supported_models=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="detection",
)


class SimpleDetectionModel(nn.Module):
    """Simplified detection model with ResNet backbone for ONNX benchmarking."""

    def __init__(self, backbone_name="resnet50"):
        super().__init__()
        backbone_channels = 2048
        if backbone_name in ("resnet18", "resnet34"):
            backbone_channels = 512

        backbone_map = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }
        loader = backbone_map.get(backbone_name, models.resnet50)
        full_backbone = loader(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(full_backbone.children())[:-2])

        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_channels, 1000),
            nn.ReLU(),
            nn.Linear(1000, 80 * 4),
        )

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        batch_size = x.shape[0]
        return detections.view(batch_size, 80, 4)


class OnnxResNetDetectionBenchmark(OnnxHarness):

    @property
    def use_case(self):
        return "detection"

    def get_onnx_model_path(self):
        name = (self.args.model or "resnet50").lower()
        precision = getattr(self.args, "precision", "fp32")
        return os.path.join(os.path.dirname(__file__),
                            f"simple_detection_{name}_{precision}.onnx")

    def export_to_onnx(self):
        name = (self.args.model or "resnet50").lower()
        precision = getattr(self.args, "precision", "fp32")
        onnx_path = self.get_onnx_model_path()

        print(f"Creating simplified ONNX detection model ({name}, {precision})...")

        model = SimpleDetectionModel(name)
        model.eval()

        dummy_input = torch.randn(1, 3, 480, 640, dtype=torch.float32)
        if precision == "fp16":
            model = model.half()
            dummy_input = dummy_input.half()

        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=["images"], output_names=["detections"],
            dynamic_axes={"images": {0: "batch_size"}, "detections": {0: "batch_size"}},
        )
        print(f"Model exported to {onnx_path}")
        return onnx_path

    def prepare_numpy_inputs(self):
        precision = getattr(self.args, "precision", "fp32")
        batch_size = getattr(self.args, "batch_size", 1)

        bench_input = np.random.randn(batch_size, 3, 480, 640).astype(np.float32)
        if precision == "fp16":
            bench_input = bench_input.astype(np.float16)

        return {"images": bench_input}


if __name__ == "__main__":
    parser = build_base_parser("ONNX ResNet Detection Inference Benchmark")
    add_onnx_args(parser)
    parser.set_defaults(model="resnet50")
    args = parser.parse_args()

    try:
        benchmark = OnnxResNetDetectionBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
