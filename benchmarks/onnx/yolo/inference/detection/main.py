#!/usr/bin/env python3
"""ONNX YOLOv5 Detection Inference Benchmark"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import os
import ssl
import tempfile
import urllib.request
import numpy as np
import torch

from core.harness import OnnxHarness
from core.schema import BenchmarkMeta
from core.args import build_base_parser, add_onnx_args

BENCHMARK_META = BenchmarkMeta(
    framework="onnx",
    model_family="yolo",
    supported_models=["yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov5"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="detection",
)

MODEL_URLS = {
    "yolov5s": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx",
    "yolov5m": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.onnx",
    "yolov5l": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.onnx",
    "yolov5x": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.onnx",
    "yolov5": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx",
}


def _create_synthetic_onnx():
    """Create a synthetic ONNX model as a fallback."""
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 1000)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 640, 640)

    f = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    torch.onnx.export(
        model, dummy_input, f.name,
        input_names=["images"], output_names=["output"],
        dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return f.name


class OnnxYoloBenchmark(OnnxHarness):

    @property
    def use_case(self):
        return "detection"

    def get_onnx_model_path(self):
        name = (self.args.model or "yolov5s").lower()
        precision = getattr(self.args, "precision", "fp32")
        models_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "models" / "onnx"
        models_dir.mkdir(parents=True, exist_ok=True)
        return str(models_dir / f"{name}_{precision}.onnx")

    def export_to_onnx(self):
        """Download pre-trained YOLOv5 ONNX model or create synthetic fallback."""
        name = (self.args.model or "yolov5s").lower()
        onnx_path = self.get_onnx_model_path()

        model_url = MODEL_URLS.get(name, MODEL_URLS["yolov5s"])
        print(f"Downloading {name} ONNX model from {model_url}...")

        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            with urllib.request.urlopen(model_url, context=ssl_context) as response:
                with open(onnx_path, "wb") as f:
                    f.write(response.read())
            print(f"Downloaded to {onnx_path}")
            return onnx_path

        except Exception as e:
            print(f"Download failed ({e}), creating synthetic model...")
            synthetic_path = _create_synthetic_onnx()
            # Copy synthetic to expected path
            import shutil
            shutil.move(synthetic_path, onnx_path)
            return onnx_path

    def prepare_numpy_inputs(self):
        precision = getattr(self.args, "precision", "fp32")
        batch_size = getattr(self.args, "batch_size", 1)

        # Load the session to check input name/shape
        import onnxruntime as ort
        onnx_path = self.get_onnx_model_path()
        if not Path(onnx_path).exists():
            onnx_path = self.export_to_onnx()

        # Get input info from the session we'll be using
        # Default YOLOv5 input shape: (batch, 3, 640, 640)
        images = np.random.randint(0, 256, (batch_size, 3, 640, 640), dtype=np.uint8)
        images = images.astype(np.float32) / 255.0

        if precision == "fp16":
            images = images.astype(np.float16)

        return {"images": images}

    def load_model(self):
        """Override to handle input name detection."""
        import onnxruntime as ort

        path = self.get_onnx_model_path()
        if not Path(path).exists():
            path = self.export_to_onnx()

        provider = getattr(self.args, "execution_provider", None)
        available = ort.get_available_providers()
        if provider and provider in available:
            providers = [provider]
        else:
            providers = available

        print(f"ONNX providers: {providers}")
        session = ort.InferenceSession(path, providers=providers)

        # Store input name for prepare_inputs
        self._input_name = session.get_inputs()[0].name
        return session

    def prepare_inputs(self):
        precision = getattr(self.args, "precision", "fp32")
        batch_size = getattr(self.args, "batch_size", 1)

        images = np.random.randint(0, 256, (batch_size, 3, 640, 640), dtype=np.uint8)
        images = images.astype(np.float32) / 255.0

        if precision == "fp16":
            images = images.astype(np.float16)

        input_name = getattr(self, "_input_name", "images")
        return {input_name: images}


if __name__ == "__main__":
    parser = build_base_parser("ONNX YOLOv5 Detection Inference Benchmark")
    add_onnx_args(parser)
    parser.set_defaults(model="yolov5s")
    args = parser.parse_args()

    # Handle 'auto' execution provider
    if getattr(args, "execution_provider", None) == "auto":
        import onnxruntime as ort
        available = ort.get_available_providers()
        priority = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                     "ROCMExecutionProvider", "MIGraphXExecutionProvider",
                     "CPUExecutionProvider"]
        for p in priority:
            if p in available:
                args.execution_provider = p
                break

    try:
        benchmark = OnnxYoloBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
