#!/usr/bin/env python3
"""PyTorch ResNet Segmentation Training Benchmark (DeepLabV3)"""

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
import torchvision.models.segmentation as segmentation_models

from core.harness import TrainingHarness
from core.schema import BenchmarkMeta
from core.args import build_base_parser, add_training_args


BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="resnet",
    supported_models=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="training",
    use_case="segmentation",
)


class SyntheticSegmentationDataset(Dataset):
    """Synthetic dataset for segmentation training benchmarks"""

    def __init__(self, num_samples=1000, num_classes=21, image_size=(512, 512), transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform

        torch.manual_seed(42)
        self.data = torch.randn(min(100, num_samples), 3, *image_size)

        self.masks = []
        for i in range(num_samples):
            mask = torch.randint(0, num_classes, image_size, dtype=torch.long)
            self.masks.append(mask)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < len(self.data):
            image = self.data[idx]
        else:
            torch.manual_seed(idx)
            image = torch.randn(3, *self.image_size)

        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)

        return image, mask


class SegmentationTrainingBenchmark(TrainingHarness):

    @property
    def use_case(self):
        return "segmentation"

    def load_model(self):
        name = (self.args.model or "resnet50").lower()
        training_mode = getattr(self.args, "training_mode", "scratch")
        num_classes = 21

        if training_mode == "finetune":
            if "resnet101" in name:
                model = segmentation_models.deeplabv3_resnet101(weights="DEFAULT")
            else:
                model = segmentation_models.deeplabv3_resnet50(weights="DEFAULT")
            model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        else:
            if "resnet101" in name:
                model = segmentation_models.deeplabv3_resnet101(weights=None, num_classes=num_classes)
            else:
                model = segmentation_models.deeplabv3_resnet50(weights=None, num_classes=num_classes)

        return model

    def create_datasets(self):
        num_classes = 21

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_ds = SyntheticSegmentationDataset(
            num_samples=300, num_classes=num_classes, transform=train_transform
        )
        val_ds = SyntheticSegmentationDataset(
            num_samples=100, num_classes=num_classes, transform=val_transform
        )
        return train_ds, val_ds

    def create_optimizer(self, model):
        training_mode = getattr(self.args, "training_mode", "scratch")
        if training_mode == "finetune":
            lr = 0.001
        else:
            lr = 0.01

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        return optimizer, scheduler

    def compute_loss(self, outputs, targets, criterion):
        # DeepLabV3 returns a dict with 'out' and 'aux' keys
        if isinstance(outputs, dict):
            main_loss = criterion(outputs["out"], targets)
            if "aux" in outputs:
                aux_loss = criterion(outputs["aux"], targets)
                return main_loss + 0.4 * aux_loss
            return main_loss
        return criterion(outputs, targets)

    def compute_accuracy(self, outputs, targets):
        if isinstance(outputs, dict):
            preds = outputs["out"]
        else:
            preds = outputs
        predicted = torch.argmax(preds, dim=1)
        correct = (predicted == targets).sum().item()
        return correct / targets.numel()


if __name__ == "__main__":
    parser = build_base_parser("PyTorch ResNet Segmentation Training Benchmark")
    add_training_args(parser)
    parser.set_defaults(model="resnet50", batch_size=4)
    args = parser.parse_args()

    try:
        benchmark = SegmentationTrainingBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
