#!/usr/bin/env python3
"""
FLUX.1 Inference Benchmark

This script benchmarks FLUX.1 models (Schnell and Dev) for image generation.
It uses the shared stable diffusion benchmark infrastructure but filters for FLUX models only.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

# Import the stable diffusion benchmark infrastructure
sd_script_path = Path(__file__).resolve().parents[3] / "stable_diffusion" / "inference" / "generation" / "main.py"

try:
    # Import as a module to avoid circular import issues
    import importlib.util
    spec = importlib.util.spec_from_file_location("sd_benchmark", sd_script_path)
    sd_benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd_benchmark)
    
    # Store the original function before any modifications
    original_get_model_configs = sd_benchmark.get_model_configs
    
    def get_flux_model_configs():
        """Get configurations for FLUX models only"""
        all_configs = original_get_model_configs()  # Use the original function
        return [config for config in all_configs if config['type'] in ['flux_schnell', 'flux_dev']]
    
    def main():
        """Main function for FLUX benchmark"""
        # Parse arguments with FLUX-specific defaults
        parser = argparse.ArgumentParser(description='FLUX.1 Inference Benchmark (Schnell and Dev)')
        
        # Model selection (FLUX models only)
        parser.add_argument('--model', type=str, default=None,
                            choices=['flux_1_schnell', 'flux1_schnell', 'flux_schnell', 'flux.1-schnell',
                                   'flux_1_dev', 'flux1_dev', 'flux_dev', 'flux.1-dev'],
                            help='FLUX model to benchmark (default: run all FLUX models)')
        
        # Precision settings
        parser.add_argument('--precision', type=str, default='fp16',
                            choices=['fp32', 'fp16', 'mixed'],
                            help='Precision mode (default: fp16)')
        
        # Generation parameters
        parser.add_argument('--batch_size', type=int, default=1,
                            help='Batch size for inference (default: 1)')
        parser.add_argument('--height', type=int, default=1024,
                            help='Image height (default: 1024 for FLUX)')
        parser.add_argument('--width', type=int, default=1024,
                            help='Image width (default: 1024 for FLUX)')
        parser.add_argument('--num-inference-steps', type=int, default=4,
                            help='Number of inference steps (default: auto - 4 for Schnell, 20 for Dev)')
        parser.add_argument('--guidance-scale', type=float, default=3.5,
                            help='Guidance scale (default: auto - 0.0 for Schnell, 3.5 for Dev)')
        
        # Benchmark settings
        parser.add_argument('--num-runs', type=int, default=5,
                            help='Number of benchmark runs (default: 5)')
        
        # Memory optimization
        parser.add_argument('--cpu-offload', action='store_true',
                            help='Enable CPU offload for FLUX (saves GPU memory)')
        
        # Device settings
        parser.add_argument('--device', type=str, default='auto',
                            help='Device to use for inference (cuda, cpu, or auto)')
        
        # Output settings (images are automatically saved to benchmark_results/images/)
        parser.add_argument('--save-images', action='store_true',
                            help='Legacy flag - images are now automatically saved to benchmark_results/images/')
        parser.add_argument('--output-dir', type=str, default=None,
                            help='Legacy option - images are automatically saved to benchmark_results/images/')
        parser.add_argument('--custom-prompt', type=str, default=None,
                            help='Custom prompt for generation (default: use test prompt)')
        parser.add_argument('--sdp-backend', type=str, default='auto',
                            choices=['auto', 'safe', 'math', 'mem_efficient', 'flash'],
                            help='Scaled-dot-product-attention backend (default: auto)')

        args = parser.parse_args()
        
        try:
            # Temporarily override the get_model_configs function to return only FLUX models
            sd_benchmark.get_model_configs = get_flux_model_configs
            
            # Run the benchmark using stable diffusion infrastructure
            results = sd_benchmark.run_inference(args)
            
            # Restore original function
            sd_benchmark.get_model_configs = original_get_model_configs
            
            # Check if any benchmarks failed
            failed_results = [r for r in results if r.get('status') == 'FAILED']
            if failed_results:
                print(f"\nFLUX benchmark failed! {len(failed_results)} out of {len(results)} benchmarks failed.")
                for failed in failed_results:
                    print(f"Failed: {failed['model']} - {failed['error']}")
                return 1
            else:
                print("\nFLUX benchmark completed successfully!")
                return 0
        except KeyboardInterrupt:
            print("\nFLUX benchmark interrupted by user")
            return 1
        except Exception as e:
            print(f"\nError during FLUX benchmark: {e}")
            import traceback
            traceback.print_exc()
            return 1

    if __name__ == "__main__":
        sys.exit(main())

except Exception as e:
    print(f"Error setting up FLUX benchmark: {e}")
    print("Please ensure the stable diffusion benchmark script is available.")
    import traceback
    traceback.print_exc()
    sys.exit(1) 