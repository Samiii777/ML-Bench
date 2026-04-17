#!/usr/bin/env python3
"""PyTorch ResNet Inference Segmentation Benchmark (DeepLabV3-ResNet50/101)"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from PIL import Image

from core.harness import InferenceHarness
from core.schema import MetricEntry, BenchmarkMeta
from core.args import build_base_parser
from core.validation import ResultValidator
from utils.download import get_sample_image_path

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="resnet",
    supported_models=["resnet50", "resnet101"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="segmentation",
)


class ResNetSegmentationBenchmark(InferenceHarness):

    @property
    def use_case(self):
        return "segmentation"

    def load_model(self):
        name = (self.args.model or "resnet50").lower()

        if "resnet101" in name:
            print("Loading DeepLabV3-ResNet101 segmentation model")
            model = deeplabv3_resnet101(weights="DEFAULT")
            self.model_architecture = "DeepLabV3-ResNet101"
        elif "resnet50" in name:
            print("Loading DeepLabV3-ResNet50 segmentation model")
            model = deeplabv3_resnet50(weights="DEFAULT")
            self.model_architecture = "DeepLabV3-ResNet50"
        else:
            print(f"NOTE: Segmentation uses DeepLabV3-ResNet50 for '{name}' "
                  f"(only ResNet50/101 have pretrained DeepLabV3 weights)")
            model = deeplabv3_resnet50(weights="DEFAULT")
            self.model_architecture = "DeepLabV3-ResNet50"

        model.eval()
        model.to(self.device)

        if self.args.precision in ("fp16", "bf16"):
            model = model.half()
        return model

    def prepare_inputs(self):
        image_path = get_sample_image_path()
        preprocess = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = preprocess(Image.open(image_path).convert("RGB"))
        batch = tensor.unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1)
        batch = batch.to(self.device)

        if self.args.precision in ("fp16", "bf16"):
            batch = batch.half()
        return batch

    def run_step(self, model, inputs):
        return model(inputs)

    def get_extra_metrics(self, model, inputs, outputs):
        # DeepLabV3 returns a dict with 'out' key
        if isinstance(outputs, dict):
            seg_out = outputs["out"]
        else:
            seg_out = outputs

        predictions = torch.argmax(seg_out, dim=1)
        unique_classes = torch.unique(predictions).cpu().numpy()
        self._num_classes = len(unique_classes)

        print(f"\nSegmentation Results:")
        print(f"  Detected {self._num_classes} classes")
        print(f"  Class IDs: {unique_classes.tolist()}")

        return [
            MetricEntry("num_detected_classes", self._num_classes, "count", "higher_is_better"),
        ]

    def validate_result(self, model, inputs, outputs, validator: ResultValidator):
        validator.expect_greater_than("num_detected_classes", self._num_classes, 1)


if __name__ == "__main__":
    parser = build_base_parser("PyTorch ResNet Inference Segmentation Benchmark")
    parser.set_defaults(model="resnet50")
    args = parser.parse_args()

    try:
        benchmark = ResNetSegmentationBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
