#!/usr/bin/env python3
"""PyTorch ResNet Training Classification Benchmark"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models

from core.harness import TrainingHarness
from core.schema import BenchmarkMeta
from core.args import build_base_parser, add_training_args


BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="resnet",
    supported_models=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="training",
    use_case="classification",
)


class SyntheticDataset(Dataset):
    """Synthetic dataset for training benchmarks to avoid download overhead"""

    def __init__(self, num_samples=1000, num_classes=1000, image_size=(224, 224), transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform

        torch.manual_seed(42)
        self.data = torch.randn(min(100, num_samples), 3, *image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < len(self.data):
            image = self.data[idx]
        else:
            torch.manual_seed(idx)
            image = torch.randn(3, *self.image_size)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ResNetTrainingBenchmark(TrainingHarness):

    @property
    def use_case(self):
        return "classification"

    def load_model(self):
        name = (self.args.model or "resnet50").lower()
        training_mode = getattr(self.args, "training_mode", "scratch")
        num_classes = 100

        model_map = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }

        if name not in model_map:
            raise ValueError(f"Unsupported model: {name}. Use one of {list(model_map.keys())}")

        if training_mode == "finetune":
            model = model_map[name](weights="DEFAULT")
        else:
            model = model_map[name](weights=None)

        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    def create_datasets(self):
        num_classes = 100

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_ds = SyntheticDataset(num_samples=500, num_classes=num_classes, transform=train_transform)
        val_ds = SyntheticDataset(num_samples=200, num_classes=num_classes, transform=val_transform)
        return train_ds, val_ds

    def create_optimizer(self, model):
        training_mode = getattr(self.args, "training_mode", "scratch")
        if training_mode == "finetune":
            lr = 0.001
        else:
            lr = 0.01

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return optimizer, scheduler

    def compute_loss(self, outputs, targets, criterion):
        return criterion(outputs, targets)

    def compute_accuracy(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        return correct / targets.size(0)


if __name__ == "__main__":
    parser = build_base_parser("PyTorch ResNet Training Classification Benchmark")
    add_training_args(parser)
    parser.set_defaults(model="resnet50", batch_size=32)
    args = parser.parse_args()

    try:
        benchmark = ResNetTrainingBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
