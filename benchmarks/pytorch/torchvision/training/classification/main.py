#!/usr/bin/env python3
"""Image classification training throughput for any torchvision model.

Times full optimizer steps on a fixed device-resident synthetic batch (no
dataloader), so the number is GPU training throughput. Validation: the loss
on the fixed batch must decrease.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())))

import torch
import torchvision

from core.args import build_base_parser
from core.harness import TrainingHarness, run_main


class TorchvisionTraining(TrainingHarness):
    use_case = "classification"
    num_classes = 1000

    def load_model(self):
        model = torchvision.models.get_model(self.args.model, weights=None, num_classes=self.num_classes)
        return model.to(self.device)

    def prepare_inputs(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(self.batch_size, 3, 224, 224, generator=g)
        y = torch.randint(0, self.num_classes, (self.batch_size,), generator=g)
        return x.to(self.device), y.to(self.device)

    def compute_loss(self, model, inputs):
        x, y = inputs
        return torch.nn.functional.cross_entropy(model(x), y)

    def get_config(self):
        return {"input_shape": list(self._inputs[0].shape), "optimizer": "SGD(lr=0.01, momentum=0.9)"}


if __name__ == "__main__":
    parser = build_base_parser("Torchvision image classification training")
    parser.set_defaults(model="resnet50", batch_size=64)
    run_main(TorchvisionTraining, parser)
