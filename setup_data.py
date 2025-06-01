#!/usr/bin/env python3
"""
Setup script to download common benchmark data files
This script downloads ImageNet classes and sample images to the data directory
"""

import sys
import os
from pathlib import Path

# Clean import of utils - no ugly relative paths!
import utils
from utils.download import download_all_common_files, list_data_files, get_data_dir

def main():
    print("ML Model Benchmarking Framework - Data Setup")
    print("=" * 50)
    
    print(f"Data directory: {get_data_dir()}")
    print()
    
    # Check if files already exist
    print("Checking existing files...")
    existing_files = list_data_files()
    
    if existing_files:
        print(f"\nFound {len(existing_files)} existing files.")
        choice = input("Do you want to re-download all files? (y/N): ").strip().lower()
        force_download = choice in ['y', 'yes']
    else:
        print("\nNo existing files found.")
        force_download = False
    
    print("\nDownloading common benchmark files...")
    try:
        files = download_all_common_files(force_download=force_download)
        
        print(f"\n✓ Setup completed successfully!")
        print(f"Downloaded files:")
        for name, path in files.items():
            print(f"  {name}: {path}")
        
        print(f"\nAll benchmark scripts will now use files from: {get_data_dir()}")
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 