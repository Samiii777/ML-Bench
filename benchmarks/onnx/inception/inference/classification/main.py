#!/usr/bin/env python3
"""ONNX InceptionV3 Classification Inference Benchmark"""

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
from torchvision import models, transforms
from PIL import Image

from core.harness import OnnxHarness
from core.schema import BenchmarkMeta, MetricEntry
from core.args import build_base_parser, add_onnx_args
from utils.download import get_imagenet_classes_path, get_sample_image_path

BENCHMARK_META = BenchmarkMeta(
    framework="onnx",
    model_family="inception",
    supported_models=["inceptionv3", "inception_v3"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="classification",
)


class OnnxInceptionClassificationBenchmark(OnnxHarness):

    @property
    def use_case(self):
        return "classification"

    def get_onnx_model_path(self):
        name = (self.args.model or "inceptionv3").lower()
        precision = getattr(self.args, "precision", "fp32")
        # Store in project-level models/onnx directory
        models_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "models" / "onnx"
        models_dir.mkdir(parents=True, exist_ok=True)
        return str(models_dir / f"{name}_{precision}.onnx")

    def export_to_onnx(self):
        name = (self.args.model or "inceptionv3").lower()
        precision = getattr(self.args, "precision", "fp32")
        onnx_path = self.get_onnx_model_path()

        print(f"Converting InceptionV3 to ONNX format (precision: {precision})...")

        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.eval()

        if precision == "fp16":
            model = model.half()
            dummy_input = torch.randn(1, 3, 299, 299, dtype=torch.float16)
        else:
            dummy_input = torch.randn(1, 3, 299, 299)

        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

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
            transforms.Resize(342),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = preprocess(Image.open(image_path))

        if batch_size > 1:
            batch = tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        else:
            batch = tensor.unsqueeze(0)

        arr = batch.numpy()
        if precision == "fp16":
            arr = arr.astype(np.float16)

        return {"input": arr}

    def get_extra_metrics(self, session, inputs, outputs):
        classes_file = get_imagenet_classes_path()
        with open(classes_file) as f:
            categories = [s.strip() for s in f.readlines()]

        output = outputs[0][0]
        probabilities = np.exp(output) / np.sum(np.exp(output))
        top5_indices = np.argsort(probabilities)[-5:][::-1]

        print("\nTop 5 predictions:")
        for i, idx in enumerate(top5_indices):
            print(f"{i+1}: {categories[idx]} ({probabilities[idx]*100:.2f}%)")

        return [
            MetricEntry("top1_prediction_confidence", float(probabilities[top5_indices[0]]),
                        "probability", "higher_is_better"),
        ]


if __name__ == "__main__":
    parser = build_base_parser("ONNX InceptionV3 Classification Inference Benchmark")
    add_onnx_args(parser)
    parser.set_defaults(model="inceptionv3")
    args = parser.parse_args()

    try:
        benchmark = OnnxInceptionClassificationBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
