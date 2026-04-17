#!/usr/bin/env python3
"""ONNX ResNet Inference Classification Benchmark"""

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
import torchvision
from torchvision import transforms
from PIL import Image

from core.harness import OnnxHarness
from core.schema import BenchmarkMeta, MetricEntry
from core.args import build_base_parser, add_onnx_args
from utils.download import get_imagenet_classes_path, get_sample_image_path

BENCHMARK_META = BenchmarkMeta(
    framework="onnx",
    model_family="resnet",
    supported_models=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="classification",
)

MODEL_WEIGHTS = {
    "resnet18": "ResNet18_Weights.DEFAULT",
    "resnet34": "ResNet34_Weights.DEFAULT",
    "resnet50": "ResNet50_Weights.DEFAULT",
    "resnet101": "ResNet101_Weights.DEFAULT",
    "resnet152": "ResNet152_Weights.DEFAULT",
}


class OnnxResNetClassificationBenchmark(OnnxHarness):

    @property
    def use_case(self):
        return "classification"

    def get_onnx_model_path(self):
        name = (self.args.model or "resnet50").lower()
        precision = getattr(self.args, "precision", "fp32")
        return f"{name}_{precision}.onnx"

    def export_to_onnx(self):
        name = (self.args.model or "resnet50").lower()
        precision = getattr(self.args, "precision", "fp32")
        onnx_path = self.get_onnx_model_path()

        print(f"Converting {name} to ONNX format (precision: {precision})...")

        model_loaders = {
            "resnet18": lambda: torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT"),
            "resnet34": lambda: torchvision.models.resnet34(weights="ResNet34_Weights.DEFAULT"),
            "resnet50": lambda: torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT"),
            "resnet101": lambda: torchvision.models.resnet101(weights="ResNet101_Weights.DEFAULT"),
            "resnet152": lambda: torchvision.models.resnet152(weights="ResNet152_Weights.DEFAULT"),
        }

        if name not in model_loaders:
            raise ValueError(f"Unsupported model: {name}")

        model = model_loaders[name]()
        model.eval()

        if precision == "fp16":
            model = model.half()
            dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float16)
        else:
            dummy_input = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Model exported to {onnx_path}")
        return onnx_path

    def prepare_numpy_inputs(self):
        image_path = get_sample_image_path()
        precision = getattr(self.args, "precision", "fp32")
        batch_size = getattr(self.args, "batch_size", 1)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = preprocess(Image.open(image_path))
        arr = tensor.numpy()

        if batch_size > 1:
            batch = np.repeat(arr[np.newaxis, :], batch_size, axis=0)
        else:
            batch = arr[np.newaxis, :]

        if precision == "fp16":
            batch = batch.astype(np.float16)
        else:
            batch = batch.astype(np.float32)

        return {"input": batch}

    def get_extra_metrics(self, session, inputs, outputs):
        classes_file = get_imagenet_classes_path()
        with open(classes_file) as f:
            categories = [s.strip() for s in f.readlines()]

        output = outputs[0][0]
        output_max = np.max(output)
        exp_output = np.exp(output - output_max)
        probabilities = exp_output / np.sum(exp_output)

        top5_indices = np.argsort(probabilities)[-5:][::-1]

        print("\nTop 5 predictions:")
        for idx in top5_indices:
            print(f"{categories[idx]}: {probabilities[idx]:.4f}")

        return [
            MetricEntry("top1_prediction_confidence", float(probabilities[top5_indices[0]]),
                        "probability", "higher_is_better"),
        ]


if __name__ == "__main__":
    parser = build_base_parser("ONNX ResNet Inference Classification Benchmark")
    add_onnx_args(parser)
    parser.set_defaults(model="resnet50")
    args = parser.parse_args()

    try:
        benchmark = OnnxResNetClassificationBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
