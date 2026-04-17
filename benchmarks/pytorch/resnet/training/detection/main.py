#!/usr/bin/env python3
"""PyTorch ResNet Detection Training Benchmark (Faster R-CNN)"""

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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models

from core.harness import TrainingHarness
from core.schema import BenchmarkMeta
from core.args import build_base_parser, add_training_args


BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="resnet",
    supported_models=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="training",
    use_case="detection",
)


class SyntheticDetectionDataset(Dataset):
    """Synthetic dataset for detection training benchmarks"""

    def __init__(self, num_samples=1000, num_classes=80, image_size=(640, 480), transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform

        torch.manual_seed(42)
        self.data = torch.randn(min(100, num_samples), 3, *image_size)

        self.targets = []
        for i in range(num_samples):
            num_objects = torch.randint(1, 6, (1,)).item()
            boxes = []
            labels = []

            for _ in range(num_objects):
                x1 = torch.randint(0, image_size[1] // 2, (1,)).float()
                y1 = torch.randint(0, image_size[0] // 2, (1,)).float()
                x2 = x1 + torch.randint(50, image_size[1] // 2, (1,)).float()
                y2 = y1 + torch.randint(50, image_size[0] // 2, (1,)).float()

                x2 = torch.min(x2, torch.tensor(float(image_size[1])))
                y2 = torch.min(y2, torch.tensor(float(image_size[0])))

                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                labels.append(torch.randint(1, num_classes + 1, (1,)).item())

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
            self.targets.append(target)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < len(self.data):
            image = self.data[idx]
        else:
            torch.manual_seed(idx)
            image = torch.randn(3, *self.image_size)

        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


class DetectionTrainingBenchmark(TrainingHarness):

    @property
    def use_case(self):
        return "detection"

    def load_model(self):
        training_mode = getattr(self.args, "training_mode", "scratch")
        num_classes = 80

        if training_mode == "finetune":
            model = detection_models.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = detection_models.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes + 1
            )
        else:
            model = detection_models.fasterrcnn_resnet50_fpn(
                weights=None, num_classes=num_classes + 1
            )

        return model

    def create_datasets(self):
        num_classes = 80

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        train_ds = SyntheticDetectionDataset(
            num_samples=300, num_classes=num_classes, transform=train_transform
        )
        val_ds = SyntheticDetectionDataset(
            num_samples=100, num_classes=num_classes, transform=val_transform
        )
        return train_ds, val_ds

    def create_optimizer(self, model):
        training_mode = getattr(self.args, "training_mode", "scratch")
        if training_mode == "finetune":
            lr = 0.0005
        else:
            lr = 0.005

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        return optimizer, scheduler

    def compute_loss(self, outputs, targets, criterion):
        # Faster R-CNN returns a loss dict during training
        if isinstance(outputs, dict):
            return sum(loss for loss in outputs.values())
        return criterion(outputs, targets)

    def compute_accuracy(self, outputs, targets):
        # Detection models don't have a simple accuracy metric during training
        return 0.0

    def run(self):
        """Override run() to handle detection-specific data loading (collate_fn)."""
        import time as _time

        from utils.benchmark_utils import (
            measure_peak_memory, reset_memory_tracking, setup_torch_backends, compute_stats,
        )
        from core.schema import BenchmarkResult, MetricEntry
        from core.output import emit_result

        setup_torch_backends(cudnn_benchmark=True)
        self.print_device_info()

        precision = getattr(self.args, "precision", "fp32")
        model_name = getattr(self.args, "model", "unknown")
        batch_size = getattr(self.args, "batch_size", 8)
        num_epochs = getattr(self.args, "num_epochs", 1)
        use_amp = precision == "mixed" and self.device.type == "cuda"

        print("Loading model...")
        model = self.load_model()
        model.to(self.device)
        model.train()

        print("Creating datasets...")
        train_ds, val_ds = self.create_datasets()

        def collate_fn(batch):
            images, targets = zip(*batch)
            return list(images), list(targets)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device.type == "cuda"), collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(self.device.type == "cuda"), collate_fn=collate_fn,
        )

        optimizer, scheduler = self.create_optimizer(model)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        reset_memory_tracking(self.device)
        train_losses, val_losses = [], []
        train_throughput = 0.0
        val_throughput = 0.0

        for epoch in range(num_epochs):
            model.train()
            epoch_loss, train_samples = 0.0, 0
            t0 = _time.perf_counter()

            for batch_idx, (images, targets) in enumerate(train_loader):
                if batch_idx >= 15:
                    break
                images = [img.to(self.device, non_blocking=True) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.autocast(device_type="cuda"):
                        loss_dict = model(images, targets)
                        loss = sum(l for l in loss_dict.values())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = model(images, targets)
                    loss = sum(l for l in loss_dict.values())
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item() * len(images)
                train_samples += len(images)

            self.synchronize()
            train_time = _time.perf_counter() - t0
            train_throughput = train_samples / train_time if train_time > 0 else 0

            # Validation
            model.eval()
            val_loss_total, val_samples = 0.0, 0
            t0 = _time.perf_counter()

            with torch.inference_mode():
                for batch_idx, (images, targets) in enumerate(val_loader):
                    if batch_idx >= 8:
                        break
                    images = [img.to(self.device, non_blocking=True) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # Detection models need to be in train mode to compute loss
                    model.train()
                    loss_dict = model(images, targets)
                    loss = sum(l for l in loss_dict.values())
                    model.eval()

                    val_loss_total += loss.item() * len(images)
                    val_samples += len(images)

            self.synchronize()
            val_time = _time.perf_counter() - t0
            val_throughput = val_samples / val_time if val_time > 0 else 0
            avg_val_loss = val_loss_total / max(val_samples, 1)
            train_losses.append(epoch_loss / max(train_samples, 1))
            val_losses.append(avg_val_loss)

            if scheduler:
                scheduler.step()

            print(
                f"Epoch {epoch+1}/{num_epochs}: train_loss={train_losses[-1]:.4f} "
                f"val_loss={avg_val_loss:.4f} train={train_throughput:.1f} samples/sec"
            )

        peak_mem = measure_peak_memory(self.device)
        metrics = [
            MetricEntry("train_throughput", train_throughput, "samples/sec", "higher_is_better"),
            MetricEntry("val_throughput", val_throughput, "samples/sec", "higher_is_better"),
            MetricEntry("final_train_loss", train_losses[-1] if train_losses else 0, "loss", "lower_is_better"),
            MetricEntry("final_val_loss", val_losses[-1] if val_losses else 0, "loss", "lower_is_better"),
            MetricEntry("throughput", train_throughput, "samples/sec", "higher_is_better"),
        ]
        if peak_mem:
            metrics.append(MetricEntry("peak_memory_gb", peak_mem.get("peak_allocated_gb", 0), "GB", "lower_is_better"))

        result = BenchmarkResult(
            status="PASS", framework=self.framework, model=model_name,
            mode=self.mode, use_case=self.use_case, precision=precision,
            batch_size=batch_size, system_info=self._build_system_info(),
            metrics=metrics,
            latency_stats={"mean": (1000.0 / train_throughput) if train_throughput > 0 else 0},
        )
        emit_result(result)
        return result


if __name__ == "__main__":
    parser = build_base_parser("PyTorch ResNet Detection Training Benchmark")
    add_training_args(parser)
    parser.set_defaults(model="resnet50", batch_size=8)
    args = parser.parse_args()

    try:
        benchmark = DetectionTrainingBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
