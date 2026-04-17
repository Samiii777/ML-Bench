#!/usr/bin/env python3
"""PyTorch ResNet Inference Detection Benchmark (FCOS-ResNet50-FPN)"""

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
from torchvision.models.detection import fcos_resnet50_fpn
from PIL import Image

from core.harness import InferenceHarness
from core.schema import MetricEntry, BenchmarkMeta
from core.args import build_base_parser
from core.validation import ResultValidator
from utils.download import get_sample_image_path, get_coco_classes_path

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="resnet",
    supported_models=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="detection",
)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ResNetDetectionBenchmark(InferenceHarness):

    @property
    def use_case(self):
        return "detection"

    def load_model(self):
        name = (self.args.model or "resnet50").lower()
        if name != "resnet50":
            print(f"NOTE: Detection uses FCOS-ResNet50-FPN for all variants "
                  f"(requested '{name}', actual backbone: ResNet50)")
        else:
            print("Loading FCOS-ResNet50-FPN detection model")

        self.model_architecture = "FCOS-ResNet50-FPN"
        model = fcos_resnet50_fpn(pretrained=True)
        model.eval()
        model.to(self.device)

        if self.args.precision in ("fp16", "bf16"):
            model = model.half()
        return model

    def prepare_inputs(self):
        image_path = get_sample_image_path()
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(Image.open(image_path).convert("RGB"))

        if self.args.precision in ("fp16", "bf16"):
            input_tensor = input_tensor.half()

        # Detection models expect a list of tensors
        input_batch = [input_tensor.to(self.device)] * self.args.batch_size
        return input_batch

    def run_step(self, model, inputs):
        return model(inputs)

    def get_extra_metrics(self, model, inputs, outputs):
        if len(outputs) > 0 and len(outputs[0]["boxes"]) > 0:
            pred = outputs[0]
            scores = pred["scores"].cpu().numpy()
            boxes = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            high_conf = scores > 0.5
            if np.any(high_conf):
                print("\nDetection Results (confidence > 0.5):")
                for score, box, label in zip(scores[high_conf], boxes[high_conf], labels[high_conf]):
                    class_name = COCO_CLASSES[label - 1] if label - 1 < len(COCO_CLASSES) else f"class_{label}"
                    print(f"  {class_name}: {score:.3f} confidence")

            self._num_detections = int(np.sum(scores > 0.3))
            self._max_confidence = float(scores.max()) if len(scores) > 0 else 0.0
        else:
            self._num_detections = 0
            self._max_confidence = 0.0
            print("No objects detected")

        return [
            MetricEntry("num_detections_above_0.3", self._num_detections, "count", "higher_is_better"),
            MetricEntry("max_detection_confidence", self._max_confidence, "probability", "higher_is_better"),
        ]

    def validate_result(self, model, inputs, outputs, validator: ResultValidator):
        validator.expect_greater_than("num_detections", self._num_detections, 0)
        validator.expect_greater_than("max_confidence", self._max_confidence, 0.3)


if __name__ == "__main__":
    parser = build_base_parser("PyTorch ResNet Inference Detection Benchmark")
    parser.set_defaults(model="resnet50")
    args = parser.parse_args()

    try:
        benchmark = ResNetDetectionBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
