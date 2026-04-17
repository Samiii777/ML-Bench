#!/usr/bin/env python3
"""ONNX ResNet Segmentation Inference Benchmark (DeepLabV3)"""

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
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from PIL import Image

from core.harness import OnnxHarness
from core.schema import BenchmarkMeta, MetricEntry
from core.args import build_base_parser, add_onnx_args
from utils.download import get_sample_image_path

BENCHMARK_META = BenchmarkMeta(
    framework="onnx",
    model_family="resnet",
    supported_models=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="segmentation",
)


class SyntheticSegmentationModel(nn.Module):
    """Synthetic segmentation-like model for ONNX export fallback."""

    def __init__(self, backbone_name="resnet50"):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 21, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        return self.segmentation_head(self.backbone(x))


class OnnxResNetSegmentationBenchmark(OnnxHarness):

    @property
    def use_case(self):
        return "segmentation"

    def get_onnx_model_path(self):
        name = (self.args.model or "resnet50").lower()
        precision = getattr(self.args, "precision", "fp32")
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        return os.path.join(models_dir, f"{name}_segmentation_{precision}.onnx")

    def export_to_onnx(self):
        name = (self.args.model or "resnet50").lower()
        precision = getattr(self.args, "precision", "fp32")
        onnx_path = self.get_onnx_model_path()

        print(f"Converting {name} segmentation model to ONNX (precision: {precision})...")

        # Try real DeepLabV3 first, fall back to synthetic
        try:
            if "resnet101" in name:
                model = deeplabv3_resnet101(weights="DEFAULT", num_classes=21)
            else:
                model = deeplabv3_resnet50(weights="DEFAULT", num_classes=21)
            model.eval()

            if precision == "fp16":
                model = model.half()
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            dummy_input = torch.randn(1, 3, 520, 520, dtype=torch_dtype)

            torch.onnx.export(
                model, dummy_input, onnx_path,
                export_params=True, opset_version=11, do_constant_folding=True,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            print(f"Real DeepLabV3 exported to {onnx_path}")
            return onnx_path

        except Exception as e:
            print(f"Real model export failed ({e}), using synthetic model...")

        # Fallback to synthetic model
        model = SyntheticSegmentationModel(name)
        model.eval()

        if precision == "fp16":
            model = model.half()
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        dummy_input = torch.randn(1, 3, 520, 520, dtype=torch_dtype)

        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Synthetic segmentation model exported to {onnx_path}")
        return onnx_path

    def prepare_numpy_inputs(self):
        precision = getattr(self.args, "precision", "fp32")
        batch_size = getattr(self.args, "batch_size", 1)

        image_path = get_sample_image_path()
        transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(Image.open(image_path).convert("RGB"))
        arr = tensor.numpy()

        if precision == "fp16":
            arr = arr.astype(np.float16)

        batch = np.stack([arr for _ in range(batch_size)])
        return {"input": batch}

    def get_extra_metrics(self, session, inputs, outputs):
        seg_output = outputs[0]
        if len(seg_output.shape) == 4:
            preds = np.argmax(seg_output[0], axis=0)
        else:
            preds = seg_output[0]

        unique_classes = len(np.unique(preds))
        return [
            MetricEntry("detected_classes", float(unique_classes), "classes", "higher_is_better"),
        ]


if __name__ == "__main__":
    parser = build_base_parser("ONNX ResNet Segmentation Inference Benchmark")
    add_onnx_args(parser)
    parser.set_defaults(model="resnet50")
    args = parser.parse_args()

    try:
        benchmark = OnnxResNetSegmentationBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
