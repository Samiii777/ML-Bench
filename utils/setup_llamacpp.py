#!/usr/bin/env python3
"""
llama.cpp Setup Script
Automates building llama.cpp with GPU support and optional model downloads.

Usage:
    python setup_llamacpp.py                           # Build for detected GPU
    python setup_llamacpp.py --gpu nvidia              # Build for NVIDIA
    python setup_llamacpp.py --gpu amd-rocm            # Build for AMD ROCm
    python setup_llamacpp.py --gpu amd-rock            # Build for AMD ROCK
    python setup_llamacpp.py --model llama-3.1-8b-q4   # Build + download model
    python setup_llamacpp.py --amdgpu-targets gfx1100,gfx1151  # Custom AMD targets
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download


def run_command(cmd, cwd=None, check=True, shell=True, env=None):
    """Run a shell command and handle errors."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=shell,
        cwd=cwd,
        check=False,  # Don't raise immediately, let us handle it
        capture_output=True,
        text=True,
        env=env
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Now check the return code and raise with better error info
    if check and result.returncode != 0:
        error_msg = f"Command failed with exit code {result.returncode}: {cmd}"
        if result.stderr:
            error_msg += f"\nError output:\n{result.stderr}"
        if result.stdout:
            error_msg += f"\nOutput:\n{result.stdout}"
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    
    return result


def check_cmake_installed():
    """Check if cmake is installed"""
    try:
        result = subprocess.run(['cmake', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ CMake is installed")
            print(result.stdout.split('\n')[0])
            return True
    except FileNotFoundError:
        pass
    
    print("❌ CMake is not installed")
    print("\nPlease install CMake:")
    print("  Ubuntu/Debian: sudo apt install cmake")
    print("  Fedora: sudo dnf install cmake")
    print("  Arch: sudo pacman -S cmake")
    return False


def check_curl_installed():
    """Check if libcurl development files are installed"""
    print("\n=== Checking for libcurl ===")
    # Check for curl-config or pkg-config
    try:
        result = subprocess.run(['pkg-config', '--exists', 'libcurl'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ libcurl is installed")
            return True
    except FileNotFoundError:
        pass
    
    # Try curl-config
    try:
        result = subprocess.run(['curl-config', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ libcurl is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠ libcurl development files not found")
    print("\nTo install libcurl:")
    print("  Ubuntu/Debian: sudo apt install libcurl4-openssl-dev")
    print("  Fedora: sudo dnf install libcurl-devel")
    print("  Arch: sudo pacman -S curl")
    print("\nWill attempt build with CURL support disabled...")
    return False


def detect_gpu():
    """Detect GPU type (nvidia, amd, or none)"""
    print("\n=== Detecting GPU ===")
    
    # Check for NVIDIA
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ Detected NVIDIA GPU")
            return 'nvidia'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check for AMD
    try:
        result = subprocess.run(['rocm-smi'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ Detected AMD GPU (ROCm)")
            return 'amd-rocm'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check for rocm-sdk (ROCK)
    try:
        result = subprocess.run(['rocm-sdk', 'path', '--root'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ Detected AMD GPU (ROCK)")
            return 'amd-rock'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("⚠ No GPU detected, will build CPU-only version")
    return 'cpu'


def clone_llamacpp(target_dir="llama.cpp"):
    """Clone the llama.cpp repository."""
    print("\n=== Cloning llama.cpp repository ===")
    if os.path.exists(target_dir):
        print(f"Directory '{target_dir}' already exists. Skipping clone.")
        return target_dir
    
    run_command(f"git clone https://github.com/ggml-org/llama.cpp {target_dir}")
    print(f"llama.cpp cloned to '{target_dir}'")
    return target_dir


def build_llamacpp_nvidia(llamacpp_dir, enable_curl=True):
    """Build llama.cpp for NVIDIA GPUs"""
    print("\n=== Building llama.cpp for NVIDIA GPU ===")
    
    # Build cmake command
    cmake_opts = "-DGGML_CUDA=ON"
    if not enable_curl:
        cmake_opts += " -DLLAMA_CURL=OFF"
    
    # Configure
    run_command(
        f"cmake -B build {cmake_opts}",
        cwd=llamacpp_dir
    )
    
    # Build
    run_command(
        "cmake --build build --config Release",
        cwd=llamacpp_dir
    )
    
    print("✓ llama.cpp built successfully for NVIDIA")


def build_llamacpp_amd_rocm(llamacpp_dir, amdgpu_targets, enable_curl=True):
    """Build llama.cpp for AMD GPUs using ROCm"""
    print("\n=== Building llama.cpp for AMD GPU (ROCm) ===")
    print(f"AMDGPU Targets: {amdgpu_targets}")
    
    # Get environment
    env = os.environ.copy()
    
    # Get hipconfig paths
    try:
        result = subprocess.run(['hipconfig', '-l'], 
                              capture_output=True, text=True, check=True)
        hipcxx_path = result.stdout.strip() + "/clang"
        env['HIPCXX'] = hipcxx_path
        
        result = subprocess.run(['hipconfig', '-R'], 
                              capture_output=True, text=True, check=True)
        hip_path = result.stdout.strip()
        env['HIP_PATH'] = hip_path
        
        print(f"HIPCXX: {hipcxx_path}")
        print(f"HIP_PATH: {hip_path}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"❌ Error: hipconfig not found. Is ROCm installed?")
        raise
    
    # Configure
    curl_opt = " -DLLAMA_CURL=OFF" if not enable_curl else ""
    cmd = (
        f"cmake -S . -B build "
        f"-DAMDGPU_TARGETS={amdgpu_targets} "
        f"-DGGML_HIP=ON "
        f"-DCMAKE_BUILD_TYPE=Release{curl_opt}"
    )
    run_command(cmd, cwd=llamacpp_dir, env=env)
    
    # Build
    run_command(
        "cmake --build build --config Release -- -j 16",
        cwd=llamacpp_dir,
        env=env
    )
    
    print("✓ llama.cpp built successfully for AMD (ROCm)")


def build_llamacpp_amd_rock(llamacpp_dir, amdgpu_targets, enable_curl=True):
    """Build llama.cpp for AMD GPUs using ROCK SDK"""
    print("\n=== Building llama.cpp for AMD GPU (ROCK) ===")
    print(f"AMDGPU Targets: {amdgpu_targets}")
    
    # Get environment
    env = os.environ.copy()
    
    # Get rocm-sdk paths
    try:
        result = subprocess.run(['rocm-sdk', 'path', '--root'], 
                              capture_output=True, text=True, check=True)
        rocm_root = result.stdout.strip()
        print(f"ROCm SDK Root: {rocm_root}")
        
        # Set environment variables
        hipcxx_path = f"{rocm_root}/llvm/bin/clang"
        env['HIPCXX'] = hipcxx_path
        env['HIP_PATH'] = rocm_root
        env['HIP_PLATFORM'] = 'amd'
        
        # Update CMAKE_PREFIX_PATH
        current_prefix = env.get('CMAKE_PREFIX_PATH', '')
        env['CMAKE_PREFIX_PATH'] = f"{rocm_root}:{current_prefix}" if current_prefix else rocm_root
        
        print(f"HIPCXX: {hipcxx_path}")
        print(f"HIP_PATH: {rocm_root}")
        
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"❌ Error: rocm-sdk not found. Is ROCK installed?")
        raise
    
    # Configure
    curl_opt = " -DLLAMA_CURL=OFF" if not enable_curl else ""
    cmd = (
        f"cmake --fresh -S . -B build "
        f"-DGGML_HIP=ON "
        f"-DAMDGPU_TARGETS={amdgpu_targets} "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_HIP_FLAGS:STRING=\"-I{rocm_root}/include\"{curl_opt}"
    )
    run_command(cmd, cwd=llamacpp_dir, env=env)
    
    # Build
    run_command(
        "cmake --build build --config Release -- -j 16",
        cwd=llamacpp_dir,
        env=env
    )
    
    print("✓ llama.cpp built successfully for AMD (ROCK)")


def build_llamacpp_cpu(llamacpp_dir, enable_curl=True):
    """Build llama.cpp for CPU only"""
    print("\n=== Building llama.cpp for CPU ===")
    
    # Build cmake command
    cmake_opts = "-DCMAKE_BUILD_TYPE=Release"
    if not enable_curl:
        cmake_opts += " -DLLAMA_CURL=OFF"
    
    # Configure
    run_command(
        f"cmake -B build {cmake_opts}",
        cwd=llamacpp_dir
    )
    
    # Build
    run_command(
        "cmake --build build --config Release",
        cwd=llamacpp_dir
    )
    
    print("✓ llama.cpp built successfully for CPU")


def download_model(llamacpp_dir, model_name):
    """Download model from HuggingFace"""
    print(f"\n=== Downloading model: {model_name} ===")
    
    # Model configurations
    models = {
        'llama-3.1-8b-q4': {
            'repo_id': 'ggml-org/Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF',
            'filename': 'meta-llama-3.1-8b-instruct-q4_0.gguf',
            'description': 'Meta Llama 3.1 8B Instruct Q4_0'
        },
        'llama-3.1-8b-q8': {
            'repo_id': 'ggml-org/Meta-Llama-3.1-8B-Instruct-Q8_0-GGUF',
            'filename': 'meta-llama-3.1-8b-instruct-q8_0.gguf',
            'description': 'Meta Llama 3.1 8B Instruct Q8_0'
        },
        'gpt-oss-20b': {
            'repo_id': 'ggml-org/gpt-oss-20b-GGUF',
            'filename': 'gpt-oss-20b-mxfp4.gguf',
            'description': 'GPT-OSS 20B mxFP4'
        }
    }
    
    if model_name not in models:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {', '.join(models.keys())}")
        return False
    
    model_info = models[model_name]
    models_dir = Path(llamacpp_dir) / "models"
    models_dir.mkdir(exist_ok=True)
    
    destination = models_dir / model_info['filename']
    
    # Check if already exists
    if destination.exists():
        print(f"✓ Model already exists: {destination}")
        return True
    
    print(f"Downloading {model_info['description']}...")
    print(f"  From: {model_info['repo_id']}/{model_info['filename']}")
    print(f"  To: {destination}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_info['repo_id'],
            filename=model_info['filename'],
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Successfully downloaded model to {destination}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}", file=sys.stderr)
        if destination.exists():
            destination.unlink()
        return False


def verify_build(llamacpp_dir):
    """Verify that llama.cpp was built successfully"""
    print("\n=== Verifying build ===")
    
    build_dir = Path(llamacpp_dir) / "build" / "bin"
    
    # Check for llama-bench
    llama_bench = build_dir / "llama-bench"
    if not llama_bench.exists():
        print(f"❌ llama-bench not found at {llama_bench}")
        return False
    
    print(f"✓ Found llama-bench at {llama_bench}")
    
    # Check for llama-cli (main executable)
    llama_cli = build_dir / "llama-cli"
    if not llama_cli.exists():
        print(f"⚠ llama-cli not found (optional)")
    else:
        print(f"✓ Found llama-cli at {llama_cli}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Automated llama.cpp setup script with GPU support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Auto-detect GPU and build
  %(prog)s --gpu nvidia                       # Build for NVIDIA
  %(prog)s --gpu amd-rocm                     # Build for AMD ROCm
  %(prog)s --gpu amd-rock                     # Build for AMD ROCK
  %(prog)s --model llama-3.1-8b-q4            # Build + download model
  %(prog)s --amdgpu-targets gfx1100,gfx1151   # Custom AMD targets
  %(prog)s --dir my-llama                     # Custom directory
        """
    )
    
    parser.add_argument(
        '--gpu',
        choices=['auto', 'nvidia', 'amd-rocm', 'amd-rock', 'cpu'],
        default='auto',
        help='GPU type to build for (default: auto-detect)'
    )
    
    parser.add_argument(
        '--model',
        choices=['llama-3.1-8b-q4', 'llama-3.1-8b-q8', 'gpt-oss-20b'],
        default=None,
        help='Model to download (default: none)'
    )
    
    parser.add_argument(
        '--amdgpu-targets',
        default='gfx1100,gfx1151,gfx1201',
        help='AMD GPU targets (default: gfx1100,gfx1151,gfx1201)'
    )
    
    parser.add_argument(
        '--dir',
        default='llama.cpp',
        help='Directory to install llama.cpp (default: llama.cpp)'
    )
    
    parser.add_argument(
        '--skip-clone',
        action='store_true',
        help='Skip cloning llama.cpp (useful if already cloned)'
    )
    
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip building llama.cpp'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("llama.cpp Automated Setup Script")
    print("=" * 60)
    
    try:
        # Check cmake
        if not args.skip_build:
            if not check_cmake_installed():
                print("\n❌ Setup failed: CMake is required")
                sys.exit(1)
        
        # Clone llama.cpp
        if not args.skip_clone:
            llamacpp_dir = clone_llamacpp(args.dir)
        else:
            llamacpp_dir = args.dir
            print(f"\nUsing existing directory: {llamacpp_dir}")
        
        # Build llama.cpp
        if not args.skip_build:
            # Check for curl
            enable_curl = check_curl_installed()
            
            # Detect or use specified GPU type
            gpu_type = args.gpu
            if gpu_type == 'auto':
                gpu_type = detect_gpu()
            
            print(f"\nBuilding for: {gpu_type}")
            
            if gpu_type == 'nvidia':
                build_llamacpp_nvidia(llamacpp_dir, enable_curl=enable_curl)
            elif gpu_type == 'amd-rocm':
                build_llamacpp_amd_rocm(llamacpp_dir, args.amdgpu_targets, enable_curl=enable_curl)
            elif gpu_type == 'amd-rock':
                build_llamacpp_amd_rock(llamacpp_dir, args.amdgpu_targets, enable_curl=enable_curl)
            elif gpu_type == 'cpu':
                build_llamacpp_cpu(llamacpp_dir, enable_curl=enable_curl)
            else:
                print(f"❌ Unknown GPU type: {gpu_type}")
                sys.exit(1)
            
            # Verify build
            if not verify_build(llamacpp_dir):
                print("\n❌ Build verification failed")
                sys.exit(1)
        else:
            print("\nSkipping build")
        
        # Download model if specified
        if args.model:
            if not download_model(llamacpp_dir, args.model):
                print("\n❌ Model download failed")
                sys.exit(1)
        else:
            print("\nNo model specified. llama.cpp setup complete without models.")
            print("To download models later, run with --model flag.")
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print(f"\nllama.cpp is installed in: {os.path.abspath(llamacpp_dir)}")
        print(f"\nTo run benchmark:")
        print(f"  cd {llamacpp_dir}")
        print(f"  ./build/bin/llama-bench -m models/model.gguf -ngl 999")
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during setup: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

