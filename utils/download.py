"""
Download utilities for the benchmarking framework
Handles downloading and caching of common files like ImageNet classes and sample images
"""

import os
import urllib.request
from pathlib import Path
from .safe_print import safe_print, format_success_message

def get_data_dir():
    """Get the centralized data directory path"""
    # Find the root directory by looking for benchmark.py
    current_dir = Path(__file__).parent
    while current_dir.parent != current_dir:  # Not at filesystem root
        if (current_dir / "benchmark.py").exists():
            return current_dir / "data"
        current_dir = current_dir.parent
    
    # Fallback to current directory if benchmark.py not found
    return Path.cwd() / "data"

def download_file(url, filename, force_download=False):
    """Download a file if it doesn't exist or if forced"""
    data_dir = get_data_dir()
    data_dir.mkdir(exist_ok=True)
    
    filepath = data_dir / filename
    
    if not filepath.exists() or force_download:
        print(f"Downloading {filename} to {filepath}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            safe_print(format_success_message(f"{filename} downloaded successfully."))
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            raise
    else:
        safe_print(format_success_message(f"{filename} already exists at {filepath}"))
    
    return str(filepath)

def download_file_if_not_exists(url, filepath):
    """Download a file if it doesn't already exist"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if not filepath.exists():
        print(f"Downloading {filepath.name}...")
        try:
            urllib.request.urlretrieve(url, str(filepath))
            safe_print(format_success_message(f"{filepath.name} downloaded successfully."))
        except Exception as e:
            print(f"Error downloading {filepath.name}: {e}")
            raise
    else:
        safe_print(format_success_message(f"{filepath.name} already exists at {filepath}"))

def download_imagenet_classes(force_download=False):
    """Download ImageNet class labels"""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return download_file(url, "imagenet_classes.txt", force_download)

def download_sample_image(force_download=False):
    """Download a sample image for testing"""
    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    return download_file(url, "dog.jpg", force_download)

def download_coco_classes(force_download=False):
    """Download COCO class labels for object detection"""
    # We'll create a simple COCO classes file since the classes are hardcoded in our detection script
    data_dir = get_data_dir()
    data_dir.mkdir(exist_ok=True)
    
    filepath = data_dir / "coco_classes.txt"
    
    if not filepath.exists() or force_download:
        print(f"Creating COCO classes file at {filepath}...")
        
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        with open(filepath, 'w') as f:
            for class_name in coco_classes:
                f.write(class_name + '\n')
        
        safe_print(format_success_message("COCO classes file created successfully."))
    else:
        safe_print(format_success_message(f"COCO classes file already exists at {filepath}"))
    
    return str(filepath)

def download_all_common_files(force_download=False):
    """Download all commonly used files"""
    print("Downloading common benchmark files...")
    
    files = {}
    files['imagenet_classes'] = download_imagenet_classes(force_download)
    files['sample_image'] = download_sample_image(force_download)
    files['coco_classes'] = download_coco_classes(force_download)
    
    print(f"All files downloaded to: {get_data_dir()}")
    return files

def get_imagenet_classes_path():
    """Get the path to ImageNet classes file, download if needed"""
    return download_imagenet_classes()

def get_sample_image_path():
    """Get the path to sample image file, download if needed"""
    return download_sample_image()

def get_coco_classes_path():
    """Get the path to COCO classes file, download if needed"""
    return download_coco_classes()

def list_data_files():
    """List all files in the data directory"""
    data_dir = get_data_dir()
    if data_dir.exists():
        files = list(data_dir.iterdir())
        print(f"Files in {data_dir}:")
        for file in files:
            size = file.stat().st_size / 1024  # Size in KB
            print(f"  {file.name} ({size:.1f} KB)")
        return files
    else:
        print(f"Data directory {data_dir} does not exist")
        return []

def clean_data_dir():
    """Clean the data directory (remove all downloaded files)"""
    data_dir = get_data_dir()
    if data_dir.exists():
        files = list(data_dir.iterdir())
        for file in files:
            file.unlink()
            print(f"Removed {file.name}")
        print(f"Cleaned data directory: {data_dir}")
    else:
        print(f"Data directory {data_dir} does not exist") 