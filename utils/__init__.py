"""
Utilities package for the ML benchmarking framework
"""

import os
import sys
from pathlib import Path

def find_project_root():
    """
    Find the project root directory by looking for benchmark.py
    This avoids ugly relative path manipulation with multiple '..'
    """
    current_path = Path(__file__).resolve()
    
    # Walk up the directory tree looking for benchmark.py
    for parent in current_path.parents:
        if (parent / "benchmark.py").exists():
            return parent
    
    # Fallback to current working directory if benchmark.py not found
    return Path.cwd()

def add_utils_to_path():
    """
    Add the project root to Python path cleanly
    Call this from any benchmark script to access utils
    """
    project_root = find_project_root()
    project_root_str = str(project_root)
    
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Automatically add project root to path when this module is imported
add_utils_to_path() 