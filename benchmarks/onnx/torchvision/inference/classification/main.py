#!/usr/bin/env python3
"""ONNX Runtime image classification for torchvision models (exported on first use).

The exported graph has a dynamic batch dimension and is cached under
models/onnx/. Same input and top-1 check as the PyTorch benchmark, so the two
frameworks are directly comparable.
"""

import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision
from PIL import Image

from core.args import add_onnx_args, build_base_parser
from core.harness import OnnxHarness, run_main
from utils.download import get_imagenet_classes_path, get_sample_image_path

EXPECTED_TOP1 = "Samoyed"


class OnnxClassification(OnnxHarness):
    use_case = "classification"

    def get_onnx_model_path(self):
        name = self.args.model
        self._weights = torchvision.models.get_model_weights(name).DEFAULT
        path = ROOT / "models" / "onnx" / f"{name}_{self.precision}.onnx"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            dtype = torch.float16 if self.precision == "fp16" else torch.float32
            device = "cuda" if dtype == torch.float16 and torch.cuda.is_available() else "cpu"
            model = torchvision.models.get_model(name, weights=self._weights).eval().to(device, dtype)
            x = torch.randn(2, 3, 224, 224, dtype=dtype, device=device)
            print(f"Exporting {name} ({self.precision}) to {path}...")
            torch.onnx.export(model, (x,), str(path), input_names=["input"], output_names=["logits"],
                              dynamic_shapes={"x": {0: torch.export.Dim("batch")}}, dynamo=True)
        return str(path)

    def prepare_numpy_inputs(self):
        image = self._weights.transforms()(Image.open(get_sample_image_path()).convert("RGB")).numpy()
        batch = np.repeat(image[None], self.batch_size, axis=0)
        return {"input": batch.astype(np.float16 if self.precision == "fp16" else np.float32)}

    def validate_result(self, session, inputs, outputs, validator):
        categories = Path(get_imagenet_classes_path()).read_text().splitlines()
        top1 = [categories[i] for i in outputs[0].argmax(axis=-1)]
        print(f"Top-1: {top1[0]}")
        validator.expect_equals("top1_class", top1[0], EXPECTED_TOP1)
        validator.expect_equals("batch_consistent", len(set(top1)), 1)


if __name__ == "__main__":
    parser = build_base_parser("ONNX Runtime image classification")
    add_onnx_args(parser)
    parser.set_defaults(model="resnet50")
    run_main(OnnxClassification, parser)
