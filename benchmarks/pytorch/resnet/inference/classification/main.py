#!/usr/bin/env python3
"""PyTorch ResNet Inference Classification Benchmark"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch
from torchvision import transforms
from PIL import Image

from core.harness import InferenceHarness
from core.schema import MetricEntry, BenchmarkMeta
from core.args import build_base_parser
from utils.download import get_imagenet_classes_path, get_sample_image_path

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
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


class ResNetClassificationBenchmark(InferenceHarness):

    @property
    def use_case(self):
        return "classification"

    def load_model(self):
        name = (self.args.model or "resnet50").lower()
        if name not in MODEL_WEIGHTS:
            raise ValueError(f"Unsupported model: {name}. Use one of {list(MODEL_WEIGHTS.keys())}")
        print(f"Loading model: {name}")
        model = torch.hub.load("pytorch/vision:v0.10.0", name, weights=MODEL_WEIGHTS[name])
        model.eval()
        model.to(self.device)
        if self.args.precision in ("fp16", "bf16"):
            model = model.half()
        return model

    def prepare_inputs(self):
        image_path = get_sample_image_path()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = preprocess(Image.open(image_path))
        batch = tensor.unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1)
        batch = batch.to(self.device)
        if self.args.precision in ("fp16", "bf16"):
            batch = batch.half()
        return batch

    def run_step(self, model, inputs):
        return model(inputs)

    def get_extra_metrics(self, model, inputs, outputs):
        classes_file = get_imagenet_classes_path()
        with open(classes_file) as f:
            categories = [s.strip() for s in f.readlines()]

        out = outputs.float()
        probs = torch.nn.functional.softmax(out[0], dim=0)
        top5_prob, top5_idx = torch.topk(probs, 5)

        print("\nTop 5 predictions:")
        for i in range(5):
            print(f"{categories[top5_idx[i]]}: {top5_prob[i].item():.4f}")

        return [
            MetricEntry("top1_prediction_confidence", top5_prob[0].item(), "probability", "higher_is_better"),
        ]


if __name__ == "__main__":
    parser = build_base_parser("PyTorch ResNet Inference Classification Benchmark")
    parser.set_defaults(model="resnet50")
    args = parser.parse_args()

    try:
        benchmark = ResNetClassificationBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
