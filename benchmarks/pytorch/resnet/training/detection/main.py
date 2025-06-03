import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torch.amp import autocast, GradScaler
import argparse
import time
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from utils.download import get_sample_image_path, get_coco_classes_path

class SyntheticDetectionDataset(Dataset):
    """Synthetic dataset for detection training benchmarks"""
    
    def __init__(self, num_samples=1000, num_classes=80, image_size=(640, 480), transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
        
        # Pre-generate some data
        torch.manual_seed(42)
        self.data = torch.randn(min(100, num_samples), 3, *image_size)
        
        # Generate random bounding boxes and labels
        self.targets = []
        for i in range(num_samples):
            num_objects = torch.randint(1, 6, (1,)).item()  # 1-5 objects per image
            boxes = []
            labels = []
            
            for _ in range(num_objects):
                # Generate random bounding box (x1, y1, x2, y2)
                x1 = torch.randint(0, image_size[1] // 2, (1,)).float()
                y1 = torch.randint(0, image_size[0] // 2, (1,)).float()
                x2 = x1 + torch.randint(50, image_size[1] // 2, (1,)).float()
                y2 = y1 + torch.randint(50, image_size[0] // 2, (1,)).float()
                
                # Ensure box is within image bounds
                x2 = torch.min(x2, torch.tensor(float(image_size[1])))
                y2 = torch.min(y2, torch.tensor(float(image_size[0])))
                
                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                labels.append(torch.randint(1, num_classes + 1, (1,)).item())  # 1-based class labels
            
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }
            self.targets.append(target)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Use cached data for first 100 samples, generate on-the-fly for others
        if idx < len(self.data):
            image = self.data[idx]
        else:
            torch.manual_seed(idx)
            image = torch.randn(3, *self.image_size)
        
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_detection_model(model_name, num_classes=80, training_mode="scratch"):
    """Create ResNet-based detection model for training"""
    
    if training_mode == "finetune":
        # Load pre-trained model
        if "resnet50" in model_name.lower():
            model = detection_models.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        else:
            # Default to ResNet-50 for other variants
            model = detection_models.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Replace the classifier head for new number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection_models.faster_rcnn.FastRCNNPredictor(in_features, num_classes + 1)  # +1 for background
    else:
        # Training from scratch - create model without pre-trained weights
        if "resnet50" in model_name.lower():
            model = detection_models.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes + 1)
        else:
            model = detection_models.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes + 1)
    
    return model

def get_transforms(training=True):
    """Get data transforms for training or validation"""
    if training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

def train_epoch(model, dataloader, optimizer, scaler, device, use_amp=False):
    """Train one epoch for detection"""
    model.train()
    running_loss = 0.0
    batch_times = []
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        start_time = time.time()
        
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
        
        running_loss += losses.item()
        
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        # Break early for benchmarking
        if batch_idx >= 15:  # Fewer batches for detection (more expensive)
            break
    
    avg_loss = running_loss / min(len(dataloader), 16)
    avg_batch_time = np.mean(batch_times)
    
    return avg_loss, avg_batch_time

def validate_epoch(model, dataloader, device, use_amp=False):
    """Validate one epoch for detection"""
    model.eval()
    running_loss = 0.0
    batch_times = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            start_time = time.time()
            
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            if use_amp:
                with autocast('cuda'):
                    # For validation, we need to switch to training mode temporarily to get loss
                    model.train()
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    model.eval()
            else:
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()
            
            running_loss += losses.item()
            
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Break early for benchmarking
            if batch_idx >= 8:
                break
    
    avg_loss = running_loss / min(len(dataloader), 9)
    avg_batch_time = np.mean(batch_times)
    
    return avg_loss, avg_batch_time

def benchmark_detection_training(model_name, training_mode, precision, batch_size, num_epochs=3):
    """Benchmark ResNet detection training performance"""
    
    print(f"Starting {model_name} detection training benchmark")
    print(f"Training mode: {training_mode}")
    print(f"Precision: {precision}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Enable mixed precision for FP16 or mixed precision
    use_amp = (precision in ["fp16", "mixed"])
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    # Create model
    num_classes = 80  # COCO classes
    model = create_detection_model(model_name, num_classes, training_mode)
    model = model.to(device)
    
    # Create datasets
    train_transform = get_transforms(training=True)
    val_transform = get_transforms(training=False)
    
    # Use smaller datasets for benchmarking
    train_dataset = SyntheticDetectionDataset(num_samples=300, num_classes=num_classes, transform=train_transform)
    val_dataset = SyntheticDetectionDataset(num_samples=100, num_classes=num_classes, transform=val_transform)
    
    # Custom collate function for detection
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True, collate_fn=collate_fn)
    
    # Create optimizer
    if training_mode == "finetune":
        # Lower learning rate for fine-tuning
        optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)
    else:
        # Higher learning rate for training from scratch
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    # Track memory before training
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    print("\nStarting training...")
    
    # Training loop
    total_train_time = 0
    total_val_time = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        start_time = time.time()
        train_loss, avg_train_batch_time = train_epoch(
            model, train_loader, optimizer, scaler, device, use_amp
        )
        train_time = time.time() - start_time
        total_train_time += train_time
        
        # Validation
        start_time = time.time()
        val_loss, avg_val_batch_time = validate_epoch(
            model, val_loader, device, use_amp
        )
        val_time = time.time() - start_time
        total_val_time += val_time
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Train Batch Time: {avg_train_batch_time*1000:.2f} ms")
        print(f"Val Batch Time: {avg_val_batch_time*1000:.2f} ms")
    
    # Calculate performance metrics
    avg_train_time_per_epoch = total_train_time / num_epochs
    avg_val_time_per_epoch = total_val_time / num_epochs
    
    # Calculate samples per second
    samples_per_train_batch = min(len(train_dataset), 16 * batch_size)
    samples_per_val_batch = min(len(val_dataset), 9 * batch_size)
    
    train_samples_per_sec = samples_per_train_batch / avg_train_time_per_epoch
    val_samples_per_sec = samples_per_val_batch / avg_val_time_per_epoch
    
    # Memory usage
    memory_used_gb = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used_gb = peak_memory / 1024**3
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== {model_name.upper()} DETECTION TRAINING BENCHMARK RESULTS ===")
    print(f"Training Mode: {training_mode}")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(f"GPU Memory Used: {memory_used_gb:.2f} GB")
    print()
    print("Performance Metrics:")
    print(f"Training Throughput: {train_samples_per_sec:.2f} samples/sec")
    print(f"Validation Throughput: {val_samples_per_sec:.2f} samples/sec")
    print(f"Average Training Time per Epoch: {avg_train_time_per_epoch:.2f} seconds")
    print(f"Average Validation Time per Epoch: {avg_val_time_per_epoch:.2f} seconds")
    print(f"Final Training Loss: {train_loss:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print("=" * 60)
    
    return {
        'model': model_name,
        'training_mode': training_mode,
        'device': str(device),
        'precision': precision,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'mixed_precision': use_amp,
        'memory_used_gb': memory_used_gb,
        'train_samples_per_sec': train_samples_per_sec,
        'val_samples_per_sec': val_samples_per_sec,
        'avg_train_time_per_epoch': avg_train_time_per_epoch,
        'avg_val_time_per_epoch': avg_val_time_per_epoch,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss
    }

def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Detection Training Benchmark')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet model to use as backbone for detection')
    parser.add_argument('--training_mode', type=str, default='scratch',
                       choices=['scratch', 'finetune'],
                       help='Training mode: scratch (random weights) or finetune (pre-trained weights)')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'mixed'],
                       help='Training precision')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training (detection uses smaller batches)')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs to train')
    
    args = parser.parse_args()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Run benchmark
    results = benchmark_detection_training(
        args.model, args.training_mode, args.precision, 
        args.batch_size, args.num_epochs
    )
    
    # Print final result in format expected by benchmark script
    print(f"\nFINAL RESULT: {results['train_samples_per_sec']:.2f} samples/sec")

if __name__ == "__main__":
    main() 