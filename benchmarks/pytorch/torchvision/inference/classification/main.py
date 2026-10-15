#!/usr/bin/env python3
"""Image classification inference for any torchvision model (resnet50, vit_b_16, convnext_tiny, ...).

Uses the model's own pretrained preprocessing and checks the top-1 prediction
on data/dog.jpg, so a numerically broken precision/backend shows up as FAIL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())))

import torch
import torchvision
from PIL import Image

from core.args import build_base_parser
from core.harness import InferenceHarness, run_main
from core.schema import MetricEntry
from utils.download import get_imagenet_classes_path, get_sample_image_path

EXPECTED_TOP1 = "Samoyed"


class TorchvisionClassification(InferenceHarness):
    use_case = "classification"

    def load_model(self):
        self._weights = torchvision.models.get_model_weights(self.args.model).DEFAULT
        model = torchvision.models.get_model(self.args.model, weights=self._weights)
        return model.eval().to(self.device, self.dtype)

    def prepare_inputs(self):
        image = self._weights.transforms()(Image.open(get_sample_image_path()).convert("RGB"))
        batch = image.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        return batch.to(self.device, self.dtype)

    def run_step(self, model, inputs):
        return model(inputs)

    def get_extra_metrics(self, model, inputs, outputs):
        categories = Path(get_imagenet_classes_path()).read_text().splitlines()
        probs = outputs.float().softmax(dim=-1)
        conf, idx = probs.max(dim=-1)
        self._top1 = [categories[i] for i in idx.tolist()]
        self._conf = conf[0].item()
        print(f"Top-1: {self._top1[0]} ({self._conf:.3f})")
        return [MetricEntry("top1_confidence", self._conf, "probability", "higher_is_better")]

    def validate_result(self, model, inputs, outputs, validator):
        validator.expect_equals("top1_class", self._top1[0], EXPECTED_TOP1)
        validator.expect_equals("batch_consistent", len(set(self._top1)), 1)
        validator.expect_greater_than("top1_confidence", self._conf, 0.1)


if __name__ == "__main__":
    parser = build_base_parser("Torchvision image classification inference")
    parser.set_defaults(model="resnet50")
    run_main(TorchvisionClassification, parser)
