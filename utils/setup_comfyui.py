#!/usr/bin/env python3
"""
ComfyUI Setup Script
Automates the setup of ComfyUI with optional model downloads.

Usage:
    python setup_comfyui.py                    # Setup ComfyUI only, no models
    python setup_comfyui.py --model flux-dev   # Setup ComfyUI + FLUX.1-dev models
    python setup_comfyui.py --model flux-schnell  # Setup ComfyUI + FLUX.1-schnell models
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def run_command(cmd, cwd=None, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def clone_comfyui(target_dir="ComfyUI"):
    """Clone the ComfyUI repository."""
    print("\n=== Cloning ComfyUI repository ===")
    if os.path.exists(target_dir):
        print(f"Directory '{target_dir}' already exists. Skipping clone.")
        return target_dir
    
    run_command(f"git clone https://github.com/comfyanonymous/ComfyUI.git {target_dir}")
    print(f"ComfyUI cloned to '{target_dir}'")
    return target_dir


def modify_requirements(comfyui_dir):
    """Modify requirements.txt to comment out torch, torchaudio, and torchvision."""
    print("\n=== Modifying requirements.txt ===")
    requirements_path = Path(comfyui_dir) / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"Warning: {requirements_path} not found. Skipping modification.")
        return
    
    with open(requirements_path, 'r') as f:
        lines = f.readlines()
    
    # Packages to comment out (exact names only)
    packages_to_comment = {'torch', 'torchaudio', 'torchvision'}
    
    modified_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines and already commented lines
        if not stripped or stripped.startswith('#'):
            modified_lines.append(line)
            continue
        
        # Extract package name (before any version specifier like ==, >=, <, etc.)
        package_name = stripped.lower().split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].split('~=')[0].split('!=')[0].split('[')[0].strip()
        
        # Check if this is one of the exact packages we want to comment out
        if package_name in packages_to_comment:
            modified_lines.append(f"# {line}")
            print(f"Commented out: {line.strip()}")
        else:
            modified_lines.append(line)
    
    with open(requirements_path, 'w') as f:
        f.writelines(modified_lines)
    
    print("requirements.txt modified successfully")


def install_requirements(comfyui_dir):
    """Install requirements using pip."""
    print("\n=== Installing requirements ===")
    requirements_path = Path(comfyui_dir) / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"Warning: {requirements_path} not found. Skipping pip install.")
        return
    
    run_command(f"pip install -r {requirements_path}")
    print("Requirements installed successfully")


def download_hf_file(repo_id, filename, destination, description="file"):
    """Download a file from HuggingFace with authentication support."""
    print(f"Downloading {description}...")
    print(f"  From: {repo_id}/{filename}")
    print(f"  To: {destination}")
    
    # Create parent directory if it doesn't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if destination.exists():
        print(f"  File already exists, skipping download.")
        return
    
    try:
        # Download to a temporary location first
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=destination.parent,
            local_dir_use_symlinks=False
        )
        print(f"  Successfully downloaded {description}")
        
        # If the downloaded file is not at the expected location, move it
        if Path(downloaded_path) != destination:
            Path(downloaded_path).rename(destination)
            
    except Exception as e:
        print(f"\n  Error downloading {description}: {e}", file=sys.stderr)
        if destination.exists():
            destination.unlink()
        raise


def download_flux_common_files(comfyui_dir):
    """Download common FLUX files (text encoders and VAE)."""
    print("\n=== Downloading common FLUX files ===")
    comfyui_path = Path(comfyui_dir)
    
    # Text encoders
    download_hf_file(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="clip_l.safetensors",
        destination=comfyui_path / "models" / "text_encoders" / "clip_l.safetensors",
        description="clip_l.safetensors (text encoder)"
    )
    
    download_hf_file(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="t5xxl_fp16.safetensors",
        destination=comfyui_path / "models" / "text_encoders" / "t5xxl_fp16.safetensors",
        description="t5xxl_fp16.safetensors (text encoder)"
    )
    
    # VAE
    download_hf_file(
        repo_id="black-forest-labs/FLUX.1-schnell",
        filename="ae.safetensors",
        destination=comfyui_path / "models" / "vae" / "ae.safetensors",
        description="ae.safetensors (VAE)"
    )


def download_flux_dev(comfyui_dir):
    """Download FLUX.1-dev model."""
    print("\n=== Downloading FLUX.1-dev model ===")
    comfyui_path = Path(comfyui_dir)
    
    download_flux_common_files(comfyui_dir)
    
    # Diffusion model
    model_path = comfyui_path / "models" / "diffusion_models" / "flux1-dev.safetensors"
    download_hf_file(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
        destination=model_path,
        description="flux1-dev.safetensors (diffusion model)"
    )
    
    # Create symlink in checkpoints directory for ComfyUI compatibility
    checkpoint_link = comfyui_path / "models" / "checkpoints" / "flux1-dev.safetensors"
    if not checkpoint_link.exists():
        checkpoint_link.symlink_to(model_path.resolve())
        print(f"Created symlink: {checkpoint_link}")
    
    print("\nFLUX.1-dev setup complete!")


def download_flux_schnell(comfyui_dir):
    """Download FLUX.1-schnell model."""
    print("\n=== Downloading FLUX.1-schnell model ===")
    comfyui_path = Path(comfyui_dir)
    
    download_flux_common_files(comfyui_dir)
    
    # Diffusion model
    model_path = comfyui_path / "models" / "diffusion_models" / "flux1-schnell.safetensors"
    download_hf_file(
        repo_id="black-forest-labs/FLUX.1-schnell",
        filename="flux1-schnell.safetensors",
        destination=model_path,
        description="flux1-schnell.safetensors (diffusion model)"
    )
    
    # Create symlink in checkpoints directory for ComfyUI compatibility
    checkpoint_link = comfyui_path / "models" / "checkpoints" / "flux1-schnell.safetensors"
    if not checkpoint_link.exists():
        checkpoint_link.symlink_to(model_path.resolve())
        print(f"Created symlink: {checkpoint_link}")
    
    print("\nFLUX.1-schnell setup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Automated ComfyUI setup script with optional model downloads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Setup ComfyUI only
  %(prog)s --model flux-dev         # Setup ComfyUI + FLUX.1-dev
  %(prog)s --model flux-schnell     # Setup ComfyUI + FLUX.1-schnell
  %(prog)s --dir MyComfyUI          # Setup in custom directory
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['flux-dev', 'flux-schnell'],
        default=None,
        help='Model to download (default: none, only setup ComfyUI)'
    )
    
    parser.add_argument(
        '--dir',
        default='ComfyUI',
        help='Directory to install ComfyUI (default: ComfyUI)'
    )
    
    parser.add_argument(
        '--skip-clone',
        action='store_true',
        help='Skip cloning ComfyUI (useful if already cloned)'
    )
    
    parser.add_argument(
        '--skip-requirements',
        action='store_true',
        help='Skip modifying and installing requirements'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ComfyUI Automated Setup Script")
    print("=" * 60)
    
    try:
        # Clone ComfyUI
        if not args.skip_clone:
            comfyui_dir = clone_comfyui(args.dir)
        else:
            comfyui_dir = args.dir
            print(f"\nUsing existing directory: {comfyui_dir}")
        
        # Modify and install requirements
        if not args.skip_requirements:
            modify_requirements(comfyui_dir)
            install_requirements(comfyui_dir)
        else:
            print("\nSkipping requirements modification and installation")
        
        # Download models if specified
        if args.model == 'flux-dev':
            download_flux_dev(comfyui_dir)
        elif args.model == 'flux-schnell':
            download_flux_schnell(comfyui_dir)
        else:
            print("\nNo model specified. ComfyUI setup complete without models.")
            print("To download models later, run with --model flag.")
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print(f"\nComfyUI is installed in: {os.path.abspath(comfyui_dir)}")
        print(f"\nTo start ComfyUI, run:")
        print(f"  cd {comfyui_dir}")
        print(f"  python main.py")
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during setup: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
