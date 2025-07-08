"""
Shared device utilities for ML benchmarks
Contains common GPU memory measurement and device detection functionality
"""

import platform
import subprocess
import time

# Global cache for GPU memory readings
_gpu_memory_cache = {"data": None, "timestamp": 0, "ttl": 1.0}  # 1 second TTL

def detect_gpu_vendor():
    """
    Detect GPU vendor (NVIDIA, AMD, or Intel)
    Returns: 'nvidia', 'amd', 'intel', or 'unknown'
    """
    try:
        # Try lspci first (most reliable on Linux)
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            output = result.stdout.lower()
            if 'nvidia' in output and 'vga' in output:
                return 'nvidia'
            elif 'amd' in output and ('vga' in output or 'radeon' in output):
                return 'amd'
            elif 'intel' in output and 'vga' in output:
                return 'intel'
    except:
        pass
    
    # Fallback: Try nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            return 'nvidia'
    except:
        pass
    
    # Fallback: Try ROCm detection for AMD
    try:
        result = subprocess.run(['rocm-smi', '--showid'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            return 'amd'
    except:
        pass
    
    return 'unknown'

def get_gpu_memory_efficient():
    """
    Get GPU memory usage efficiently using GPU-specific libraries or subprocess
    Automatically detects GPU vendor and uses appropriate method
    """
    # Detect GPU vendor to choose appropriate method
    gpu_vendor = detect_gpu_vendor()
    
    if gpu_vendor == 'nvidia':
        # Try NVIDIA methods in order of preference
        # Method 1: Try nvidia-smi Python library (fastest)
        gpu_memory = _read_gpu_memory_nvml()
        if gpu_memory:
            return gpu_memory
        
        # Method 2: Use optimized nvidia-smi subprocess (fallback)
        gpu_memory = _read_gpu_memory_nvidia_smi()
        if gpu_memory:
            return gpu_memory
    
    elif gpu_vendor == 'amd':
        # Try AMD methods in order of preference
        # Method 1: Try amdsmi Python library (fastest)
        gpu_memory = _read_gpu_memory_amdsmi()
        if gpu_memory:
            return gpu_memory
        
        # Method 2: Use rocm-smi subprocess (fallback)
        gpu_memory = _read_gpu_memory_rocm_smi()
        if gpu_memory:
            return gpu_memory
    
    # Fallback: Try to read from Linux files (generic)
    gpu_memory = _read_gpu_memory_from_files()
    if gpu_memory:
        return gpu_memory
    
    # Final fallback - return None to indicate failure
    return None

def _read_gpu_memory_amdsmi():
    """
    Try to read GPU memory using amdsmi Python library for AMD GPUs
    This is the fastest method when available
    """
    global _gpu_memory_cache
    
    current_time = time.time()
    
    # Check if we have valid cached data
    if (_gpu_memory_cache["data"] is not None and 
        current_time - _gpu_memory_cache["timestamp"] < _gpu_memory_cache["ttl"]):
        return _gpu_memory_cache["data"]
    
    try:
        import amdsmi
        
        # Initialize amdsmi
        amdsmi.amdsmi_init()
        
        # Get list of GPU devices
        devices = amdsmi.amdsmi_get_device_handles()
        
        if not devices:
            amdsmi.amdsmi_shut_down()
            return None
        
        # Use first GPU (index 0)
        device = devices[0]
        
        # Get memory info
        memory_info = amdsmi.amdsmi_get_gpu_memory_usage(device)
        
        # Shutdown amdsmi
        amdsmi.amdsmi_shut_down()
        
        # Parse memory info - amdsmi returns dict with memory_used and memory_total in bytes
        used_bytes = memory_info.get('memory_used', 0)
        total_bytes = memory_info.get('memory_total', 0)
        
        # Convert to GB
        used_gb = used_bytes / 1024**3
        total_gb = total_bytes / 1024**3
        free_gb = total_gb - used_gb
        
        gpu_memory_data = {
            "total_gpu_used_gb": used_gb,
            "total_gpu_total_gb": total_gb,
            "total_gpu_free_gb": free_gb,
            "gpu_utilization_percent": (used_gb / total_gb) * 100 if total_gb > 0 else 0,
            "method": "amdsmi",  # Indicate which method was used
            "vendor": "amd"
        }
        
        # Cache the result
        _gpu_memory_cache["data"] = gpu_memory_data
        _gpu_memory_cache["timestamp"] = current_time
        
        return gpu_memory_data
        
    except ImportError:
        # amdsmi package not available
        return None
    except Exception as e:
        # AMDSMI error (GPU not available, driver issues, etc.)
        return None

def _read_gpu_memory_rocm_smi():
    """
    Use rocm-smi call to get memory usage for AMD GPUs
    """
    global _gpu_memory_cache
    
    current_time = time.time()
    
    # Check if we have valid cached data
    if (_gpu_memory_cache["data"] is not None and 
        current_time - _gpu_memory_cache["timestamp"] < _gpu_memory_cache["ttl"]):
        return _gpu_memory_cache["data"]
    
    try:
        # Use rocm-smi to get memory info (total and used in bytes)
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--csv'],
            capture_output=True, text=True, timeout=2
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            
            # Parse CSV output - format: device,VRAM Total Memory (B),VRAM Total Used Memory (B)
            for line in lines:
                if 'device' in line.lower():
                    continue  # Skip header
                
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        device = parts[0].strip()
                        total_bytes = float(parts[1].strip())
                        used_bytes = float(parts[2].strip())
                        
                        # Convert bytes to GB
                        total_gb = total_bytes / 1024**3
                        used_gb = used_bytes / 1024**3
                        free_gb = total_gb - used_gb
                        
                        gpu_memory_data = {
                            "total_gpu_used_gb": used_gb,
                            "total_gpu_total_gb": total_gb,
                            "total_gpu_free_gb": free_gb,
                            "gpu_utilization_percent": (used_gb / total_gb) * 100 if total_gb > 0 else 0,
                            "method": "rocm-smi",  # Indicate which method was used
                            "vendor": "amd",
                            "device": device
                        }
                        
                        # Cache the result
                        _gpu_memory_cache["data"] = gpu_memory_data
                        _gpu_memory_cache["timestamp"] = current_time
                        
                        return gpu_memory_data
                        
                    except (ValueError, IndexError):
                        continue
                        
    except Exception as e:
        return None
    
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
            "method": "nvml",  # Indicate which method was used
            "vendor": "nvidia"
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
    Try to read GPU memory directly from Linux files (fallback method)
    """
    try:
        # For AMD GPUs, try reading from debugfs or sysfs
        amd_files = [
            "/sys/class/drm/card0/device/mem_info_vram_used",
            "/sys/class/drm/card0/device/mem_info_vram_total",
            "/sys/class/drm/card1/device/mem_info_vram_used",
            "/sys/class/drm/card1/device/mem_info_vram_total",
        ]
        
        used_mb = 0
        total_mb = 0
        
        # Try to read AMD memory info from sysfs
        for filepath in amd_files:
            try:
                with open(filepath, 'r') as f:
                    value = int(f.read().strip())
                    if 'used' in filepath:
                        used_mb = value / 1024 / 1024  # Convert from bytes to MB
                    elif 'total' in filepath:
                        total_mb = value / 1024 / 1024  # Convert from bytes to MB
            except:
                continue
        
        if used_mb > 0 and total_mb > 0:
            gpu_memory_data = {
                "total_gpu_used_gb": used_mb / 1024,
                "total_gpu_total_gb": total_mb / 1024,
                "total_gpu_free_gb": (total_mb - used_mb) / 1024,
                "gpu_utilization_percent": (used_mb / total_mb) * 100,
                "method": "sysfs",
                "vendor": "amd"
            }
            return gpu_memory_data
        
        # No direct file method available
        return None
        
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
                        "method": "subprocess",  # Indicate which method was used
                        "vendor": "nvidia"
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
    Print basic system information including GPU detection
    """
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    # System info
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    
    # GPU vendor detection
    gpu_vendor = detect_gpu_vendor()
    print(f"GPU vendor: {gpu_vendor}")
    
    # CPU info
    try:
        import psutil
        print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    except ImportError:
        print("CPU info: psutil not available")
    
    # GPU Memory info
    gpu_memory = get_gpu_memory_efficient()
    if gpu_memory:
        vendor = gpu_memory.get('vendor', 'unknown')
        method = gpu_memory.get('method', 'unknown')
        total_gb = gpu_memory.get('total_gpu_total_gb', 0)
        used_gb = gpu_memory.get('total_gpu_used_gb', 0)
        print(f"GPU Memory ({vendor}, {method}): {used_gb:.1f} GB / {total_gb:.1f} GB used")
    else:
        print("GPU Memory: Not available")
    
    print("=" * 50) 