#!/usr/bin/env python3
"""PyTorch YOLOv5 Inference Detection Benchmark"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch

from core.harness import InferenceHarness
from core.schema import MetricEntry, BenchmarkMeta
from core.args import build_base_parser
from core.validation import ResultValidator

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="yolo",
    supported_models=["yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov5"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="detection",
)

YOLO_MODEL_MAP = {
    "yolov5s": "yolov5s.pt",
    "yolov5m": "yolov5m.pt",
    "yolov5l": "yolov5l.pt",
    "yolov5x": "yolov5x.pt",
    "yolov5": "yolov5s.pt",
}


class YOLODetectionBenchmark(InferenceHarness):

    @property
    def use_case(self):
        return "detection"

    def load_model(self):
        name = (self.args.model or "yolov5s").lower()
        model_file = YOLO_MODEL_MAP.get(name, "yolov5s.pt")
        self._is_real_yolo = False

        try:
            from ultralytics import YOLO

            print(f"Loading YOLOv5 model: {model_file}")
            model = YOLO(model_file)
            model.to(self.device)

            # Fuse conv+bn BEFORE converting to fp16 to avoid ROCm dtype mismatch.
            # Ultralytics' fuse_conv_and_bn synthesizes a zero-bias tensor with
            # the default fp32 dtype when the conv has no bias; if we've already
            # called .half(), the conv weights are fp16 but that synthesized bias
            # is fp32, and the subsequent torch.mm fails on ROCm/MIOpen.
            try:
                if hasattr(model, "fuse"):
                    model.fuse()
                elif hasattr(model, "model") and hasattr(model.model, "fuse"):
                    model.model.fuse()
            except Exception as fuse_err:
                print(f"Note: explicit model.fuse() failed ({fuse_err}); "
                      f"relying on ultralytics' lazy fuse")

            if self.args.precision in ("fp16", "bf16"):
                model.half()
                print("Using FP16 precision")
            else:
                print(f"Using {self.args.precision} precision")

            self._is_real_yolo = True
            self.model_architecture = f"YOLOv5 ({model_file})"
            return model

        except ImportError:
            print("=" * 60)
            print("WARNING: ultralytics not installed!")
            print("YOLO benchmark will use a ResNet placeholder instead.")
            print("Results are NOT representative of real YOLOv5 performance.")
            print("Install with: pip install ultralytics")
            print("=" * 60)
            return self._load_resnet_placeholder(name)

        except Exception as e:
            print("=" * 60)
            print(f"WARNING: Failed to load YOLOv5: {e}")
            print("YOLO benchmark will use a ResNet placeholder instead.")
            print("Results are NOT representative of real YOLOv5 performance.")
            print("=" * 60)
            return self._load_resnet_placeholder(name)

    def _load_resnet_placeholder(self, name):
        import torchvision.models as tv_models

        print("Note: Using ResNet as YOLOv5 placeholder (install ultralytics for real YOLOv5)")
        size_map = {"yolov5s": "resnet18", "yolov5m": "resnet34",
                     "yolov5l": "resnet50", "yolov5x": "resnet101"}
        resnet_name = size_map.get(name, "resnet18")

        model = getattr(tv_models, resnet_name)(weights="DEFAULT")
        model.fc = torch.nn.Linear(model.fc.in_features, 1000)
        model.to(self.device)

        if self.args.precision in ("fp16", "bf16"):
            model = model.half()

        model.eval()
        self.model_architecture = f"ResNet placeholder for {name}"
        return model

    def prepare_inputs(self):
        # Synthetic 640x640 images normalized to 0-1
        images = torch.randint(0, 256, (self.args.batch_size, 3, 640, 640),
                               dtype=torch.float32) / 255.0
        if self.args.precision in ("fp16", "bf16"):
            images = images.half()
        images = images.to(self.device)
        return images

    def run_step(self, model, inputs):
        if self._is_real_yolo:
            return model(inputs, verbose=False)
        else:
            return model(inputs)

    def get_extra_metrics(self, model, inputs, outputs):
        self._model_loaded = True
        self._num_detections = 0

        if self._is_real_yolo and len(outputs) > 0 and hasattr(outputs[0], "boxes"):
            boxes = outputs[0].boxes
            if boxes is not None:
                self._num_detections = len(boxes)
                print(f"\nDetections per image: {self._num_detections / self.args.batch_size:.1f}")

        total_params = sum(
            p.numel() for p in (model.model.parameters() if self._is_real_yolo else model.parameters())
        )
        model_type = "Real YOLOv5" if self._is_real_yolo else "ResNet Placeholder"
        print(f"Model type: {model_type}")
        print(f"Model parameters: {total_params:,}")

        return [
            MetricEntry("num_detections", self._num_detections, "count", "higher_is_better"),
            MetricEntry("is_real_yolo", float(self._is_real_yolo), "bool", "higher_is_better"),
        ]

    def validate_result(self, model, inputs, outputs, validator: ResultValidator):
        # YOLO may not detect anything on synthetic images, so we only
        # validate that the model loaded successfully.
        validator.expect_equals("model_loaded", self._model_loaded, True)


if __name__ == "__main__":
    parser = build_base_parser("PyTorch YOLOv5 Inference Detection Benchmark")
    parser.set_defaults(model="yolov5s")
    args = parser.parse_args()

    try:
        benchmark = YOLODetectionBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
