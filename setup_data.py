#!/usr/bin/env python3
"""
Setup script to download and prepare common data files for benchmarking
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import utils
from utils.download import download_all_common_files, list_data_files
from utils.safe_print import safe_print, format_success_message

def main():
    print("ML Model Benchmarking Framework - Data Setup")
    print("=" * 50)
    
    print(f"Data directory: {utils.get_data_dir()}")
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
        
        print(f"\n" + "=" * 50)
        print("SETUP COMPLETE")
        print("=" * 50)
        
        safe_print(format_success_message("Setup completed successfully!"))
        print()
        list_data_files()
        
        print(f"\nAll benchmark scripts will now use files from: {utils.get_data_dir()}")
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 