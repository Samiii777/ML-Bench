import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.amp import autocast, GradScaler
import argparse
import time
import sys
import os
import numpy as np
import subprocess
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from utils.download import get_sample_image_path

class SyntheticDataset(Dataset):
    """Synthetic dataset for training benchmarks to avoid download overhead"""
    
    def __init__(self, num_samples=1000, num_classes=1000, image_size=(224, 224), transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
        
        # Pre-generate some data to make it more realistic
        torch.manual_seed(42)  # For reproducibility
        self.data = torch.randn(min(100, num_samples), 3, *image_size)  # Cache first 100 samples
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Use cached data for first 100 samples, generate on-the-fly for others
        if idx < len(self.data):
            image = self.data[idx]
        else:
            # Generate deterministic data based on index
            torch.manual_seed(idx)
            image = torch.randn(3, *self.image_size)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_gpu_memory_usage():
    """Get GPU memory usage from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        memory_used_mb = int(result.stdout.strip())
        return memory_used_mb / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # Fallback to PyTorch memory tracking if nvidia-smi fails
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024**3
        return 0.0

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_model(model_name, num_classes=1000, training_mode="scratch"):
    """Create ResNet model for training"""
    
    if model_name == "resnet18":
        if training_mode == "finetune":
            model = models.resnet18(weights='DEFAULT')
        else:  # training_mode == "scratch"
            model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        if training_mode == "finetune":
            model = models.resnet34(weights='DEFAULT')
        else:
            model = models.resnet34(weights=None)
    elif model_name == "resnet50":
        if training_mode == "finetune":
            model = models.resnet50(weights='DEFAULT')
        else:
            model = models.resnet50(weights=None)
    elif model_name == "resnet101":
        if training_mode == "finetune":
            model = models.resnet101(weights='DEFAULT')
        else:
            model = models.resnet101(weights=None)
    elif model_name == "resnet152":
        if training_mode == "finetune":
            model = models.resnet152(weights='DEFAULT')
        else:
            model = models.resnet152(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Modify the final layer for the target number of classes
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def get_transforms(training=True):
    """Get data transforms for training or validation"""
    if training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp=False):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        start_time = time.time()
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        # Break early for benchmarking (don't need full epoch)
        if batch_idx >= 20:  # Just train 20 batches for benchmarking
            break
    
    avg_loss = running_loss / min(len(dataloader), 21)
    accuracy = 100.0 * correct / total
    avg_batch_time = np.mean(batch_times)
    
    return avg_loss, accuracy, avg_batch_time

def validate_epoch(model, dataloader, criterion, device, use_amp=False):
    """Validate one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            start_time = time.time()
            
            data, target = data.to(device), target.to(device)
            
            if use_amp:
                with autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Break early for benchmarking
            if batch_idx >= 10:  # Just validate 10 batches
                break
    
    avg_loss = running_loss / min(len(dataloader), 11)
    accuracy = 100.0 * correct / total
    avg_batch_time = np.mean(batch_times)
    
    return avg_loss, accuracy, avg_batch_time

def benchmark_training(model_name, training_mode, precision, batch_size, num_epochs=3):
    """Benchmark ResNet training performance"""
    
    print(f"Starting {model_name} training benchmark")
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
    num_classes = 100  # Use smaller number of classes for faster training
    model = create_model(model_name, num_classes, training_mode)
    model = model.to(device)
    
    # Create datasets
    train_transform = get_transforms(training=True)
    val_transform = get_transforms(training=False)
    
    # Use smaller datasets for benchmarking
    train_dataset = SyntheticDataset(num_samples=500, num_classes=num_classes, transform=train_transform)
    val_dataset = SyntheticDataset(num_samples=200, num_classes=num_classes, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    
    if training_mode == "finetune":
        # Lower learning rate for fine-tuning
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    else:
        # Higher learning rate for training from scratch
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate scheduler - use a more reasonable schedule for longer training
    # Reduce LR by 0.5 every 10 epochs instead of 0.1 every 2 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Track memory before training
    initial_memory_nvidia = 0.0
    peak_memory_nvidia = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()
        initial_memory_nvidia = get_gpu_memory_usage()
        peak_memory_nvidia = initial_memory_nvidia
        print(f"Initial GPU memory allocated (PyTorch): {initial_memory / 1024**3:.2f} GB")
        print(f"Initial GPU memory usage (nvidia-smi): {initial_memory_nvidia:.2f} GB")
    
    print("\nStarting training...")
    
    # Training loop
    total_train_time = 0
    total_val_time = 0
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        start_time = time.time()
        train_loss, train_acc, avg_train_batch_time = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        train_time = time.time() - start_time
        total_train_time += train_time
        
        # Validation
        start_time = time.time()
        val_loss, val_acc, avg_val_batch_time = validate_epoch(
            model, val_loader, criterion, device, use_amp
        )
        val_time = time.time() - start_time
        total_val_time += val_time
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Train Batch Time: {avg_train_batch_time*1000:.2f} ms")
        print(f"Val Batch Time: {avg_val_batch_time*1000:.2f} ms")
        
        # Report memory usage after each epoch
        if device.type == "cuda":
            current_memory_nvidia = get_gpu_memory_usage()
            peak_memory_nvidia = max(peak_memory_nvidia, current_memory_nvidia)
            print(f"GPU Memory - nvidia-smi Current: {current_memory_nvidia:.2f} GB, Peak: {peak_memory_nvidia:.2f} GB")
    
    # Calculate performance metrics
    avg_train_time_per_epoch = total_train_time / num_epochs
    avg_val_time_per_epoch = total_val_time / num_epochs
    
    # Calculate samples per second
    samples_per_train_batch = min(len(train_dataset), 21 * batch_size)
    samples_per_val_batch = min(len(val_dataset), 11 * batch_size)
    
    train_samples_per_sec = samples_per_train_batch / avg_train_time_per_epoch
    val_samples_per_sec = samples_per_val_batch / avg_val_time_per_epoch
    
    # Memory usage - use nvidia-smi for accurate measurement
    memory_used_gb = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory_nvidia = get_gpu_memory_usage()
        peak_memory_nvidia = max(peak_memory_nvidia, current_memory_nvidia)
        
        # Use nvidia-smi peak memory as the primary measurement
        memory_used_gb = peak_memory_nvidia
        
        print(f"\nFinal Memory Stats:")
        print(f"Peak GPU Memory (PyTorch): {peak_memory / 1024**3:.2f} GB")
        print(f"Peak GPU Memory (nvidia-smi): {peak_memory_nvidia:.2f} GB")
        print(f"Using nvidia-smi peak memory for benchmark: {memory_used_gb:.2f} GB")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== {model_name.upper()} TRAINING BENCHMARK RESULTS ===")
    print(f"Framework: PyTorch")
    print(f"Training Mode: {training_mode}")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(f"Total GPU Memory Used: {memory_used_gb:.2f} GB")
    print()
    print("Performance Metrics:")
    print(f"Training Throughput: {train_samples_per_sec:.2f} samples/sec")
    print(f"Validation Throughput: {val_samples_per_sec:.2f} samples/sec")
    print(f"Average Training Time per Epoch: {avg_train_time_per_epoch:.2f} seconds")
    print(f"Average Validation Time per Epoch: {avg_val_time_per_epoch:.2f} seconds")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
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
        'best_val_accuracy': best_val_acc,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss
    }

def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Training Benchmark')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet model to train')
    parser.add_argument('--training_mode', type=str, default='scratch',
                       choices=['scratch', 'finetune'],
                       help='Training mode: scratch (random weights) or finetune (pre-trained weights)')
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16', 'mixed'],
                       help='Training precision')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
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
    results = benchmark_training(
        args.model, args.training_mode, args.precision, 
        args.batch_size, args.num_epochs
    )
    
    # Print final result in format expected by benchmark script
    print(f"\nFINAL RESULT: {results['train_samples_per_sec']:.2f} samples/sec")

if __name__ == "__main__":
    main() 