"""
Shared device utilities for ML benchmarks
Contains common GPU memory measurement and device detection functionality
"""

import platform
import subprocess
import time

# Global cache for GPU memory readings
_gpu_memory_cache = {"data": None, "timestamp": 0, "ttl": 1.0}  # 1 second TTL

def get_gpu_memory_efficient():
    """
    Get GPU memory usage efficiently using nvidia-smi Python library or subprocess
    """
    # Method 1: Try nvidia-smi Python library (fastest)
    gpu_memory = _read_gpu_memory_nvml()
    if gpu_memory:
        return gpu_memory
    
    # Method 2: Try to read from Linux files (fast but limited)
    gpu_memory = _read_gpu_memory_from_files()
    if gpu_memory:
        return gpu_memory
    
    # Method 3: Use optimized nvidia-smi subprocess (fallback)
    gpu_memory = _read_gpu_memory_nvidia_smi()
    if gpu_memory:
        return gpu_memory
    
    # Method 4: Final fallback - return None to indicate failure
    return None

def _read_gpu_memory_nvml():
    """
    Try to read GPU memory using nvidia-smi Python library (NVML)
    This is the fastest method when available
    """
    global _gpu_memory_cache
    
    current_time = time.time()
    
    # Check if we have valid cached data
    if (_gpu_memory_cache["data"] is not None and 
        current_time - _gpu_memory_cache["timestamp"] < _gpu_memory_cache["ttl"]):
        return _gpu_memory_cache["data"]
    
    try:
        # Try importing nvidia-smi (nvidia-ml-py3 package)
        import nvidia_smi
        
        # Initialize NVML
        nvidia_smi.nvmlInit()
        
        # Get handle for first GPU (index 0)
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        
        # Get memory info
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        # Shutdown NVML
        nvidia_smi.nvmlShutdown()
        
        # Convert to GB and create result
        used_gb = info.used / 1024**3
        total_gb = info.total / 1024**3
        free_gb = info.free / 1024**3
        
        gpu_memory_data = {
            "total_gpu_used_gb": used_gb,
            "total_gpu_total_gb": total_gb,
            "total_gpu_free_gb": free_gb,
            "gpu_utilization_percent": (used_gb / total_gb) * 100,
            "method": "nvml"  # Indicate which method was used
        }
        
        # Cache the result
        _gpu_memory_cache["data"] = gpu_memory_data
        _gpu_memory_cache["timestamp"] = current_time
        
        return gpu_memory_data
        
    except ImportError:
        # nvidia-smi package not available
        return None
    except Exception as e:
        # NVML error (GPU not available, driver issues, etc.)
        return None

def _read_gpu_memory_from_files():
    """
    Try to read GPU memory directly from Linux files
    """
    try:
        # Try reading from NVIDIA proc files
        # Note: These files typically don't contain memory usage, but we check anyway
        nvidia_files = [
            "/proc/driver/nvidia/gpus/0000:01:00.0/information",
            "/sys/class/drm/card1/device/resource1",  # GPU memory BAR
        ]
        
        # For now, this is a placeholder - NVIDIA doesn't expose memory usage
        # in easily readable text files. The resource files are binary.
        # We could potentially read the resource size, but not current usage.
        
        return None  # No direct file method available for NVIDIA memory usage
        
    except Exception:
        return None

def _read_gpu_memory_nvidia_smi():
    """
    Use optimized nvidia-smi call with caching to get memory usage
    """
    global _gpu_memory_cache
    
    current_time = time.time()
    
    # Check if we have valid cached data
    if (_gpu_memory_cache["data"] is not None and 
        current_time - _gpu_memory_cache["timestamp"] < _gpu_memory_cache["ttl"]):
        return _gpu_memory_cache["data"]
    
    try:
        # Single optimized nvidia-smi call - much faster than full output parsing
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=1  # Very short timeout
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                # Parse first GPU (index 0)
                memory_values = lines[0].split(', ')
                if len(memory_values) >= 2:
                    used_mb = float(memory_values[0])
                    total_mb = float(memory_values[1])
                    
                    gpu_memory_data = {
                        "total_gpu_used_gb": used_mb / 1024,
                        "total_gpu_total_gb": total_mb / 1024,
                        "total_gpu_free_gb": (total_mb - used_mb) / 1024,
                        "gpu_utilization_percent": (used_mb / total_mb) * 100,
                        "method": "subprocess"  # Indicate which method was used
                    }
                    
                    # Cache the result
                    _gpu_memory_cache["data"] = gpu_memory_data
                    _gpu_memory_cache["timestamp"] = current_time
                    
                    return gpu_memory_data
    except Exception as e:
        # Return error info for debugging, but don't cache errors
        return {"nvidia_smi_error": str(e)}
    
    return None

def clear_gpu_memory_cache():
    """
    Clear the GPU memory cache to force fresh readings
    """
    global _gpu_memory_cache
    _gpu_memory_cache["data"] = None
    _gpu_memory_cache["timestamp"] = 0

def get_system_memory_usage():
    """
    Get system memory usage (for CPU workloads)
    """
    try:
        import psutil
        process = psutil.Process()
        return {
            "rss": process.memory_info().rss / 1024**3,
            "vms": process.memory_info().vms / 1024**3
        }
    except ImportError:
        return {"error": "psutil not available"}

def print_system_info():
    """
    Print basic system information
    """
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    # System info
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    
    # CPU info
    try:
        import psutil
        print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    except ImportError:
        print("CPU info: psutil not available")
    
    print("=" * 50) 