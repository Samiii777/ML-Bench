#!/usr/bin/env python3
"""PyTorch InceptionV3 Inference Classification Benchmark"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch
from torchvision import transforms, models
from PIL import Image

from core.harness import InferenceHarness
from core.schema import MetricEntry, BenchmarkMeta
from core.args import build_base_parser
from core.validation import ResultValidator
from utils.download import get_imagenet_classes_path, get_sample_image_path

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="inception",
    supported_models=["inceptionv3", "inception_v3"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="classification",
)


class InceptionClassificationBenchmark(InferenceHarness):

    @property
    def use_case(self):
        return "classification"

    def load_model(self):
        name = (self.args.model or "inceptionv3").lower()
        if name not in ("inceptionv3", "inception_v3"):
            raise ValueError(f"Unsupported model: {name}. Use one of ['inceptionv3', 'inception_v3']")

        print(f"Loading model: {name}")
        self.model_architecture = "InceptionV3"
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.eval()
        model.to(self.device)

        if self.args.precision in ("fp16", "bf16"):
            model = model.half()
        return model

    def prepare_inputs(self):
        image_path = get_sample_image_path()
        preprocess = transforms.Compose([
            transforms.Resize(342),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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
            print(f"  {i+1}: {categories[top5_idx[i]]} ({top5_prob[i].item()*100:.2f}%)")

        self._top1_class = categories[top5_idx[0]]
        self._top1_confidence = top5_prob[0].item()

        return [
            MetricEntry("top1_prediction_confidence", self._top1_confidence, "probability", "higher_is_better"),
        ]

    def validate_result(self, model, inputs, outputs, validator: ResultValidator):
        validator.expect_equals("top1_class", self._top1_class, "Samoyed")
        validator.expect_greater_than("top1_confidence", self._top1_confidence, 0.1)


if __name__ == "__main__":
    parser = build_base_parser("PyTorch InceptionV3 Inference Classification Benchmark")
    parser.set_defaults(model="inceptionv3")
    args = parser.parse_args()

    try:
        benchmark = InceptionClassificationBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
