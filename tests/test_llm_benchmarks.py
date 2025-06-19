# tests/test_llm_benchmarks.py
import subprocess
import sys
import os
from pathlib import Path
import pytest

# Add project root to sys.path to allow importing benchmark.py for utility functions if needed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Path to the main benchmark script
BENCHMARK_SCRIPT_PATH = PROJECT_ROOT / "benchmark.py"

@pytest.mark.llm
def test_pytorch_gpt2_fp32_inference():
    """
    Tests the PyTorch LLM benchmark script with gpt2 model, fp32 precision.
    It checks for successful execution and presence of key output phrases.
    """
    if not BENCHMARK_SCRIPT_PATH.exists():
        pytest.fail(f"Main benchmark script not found at {BENCHMARK_SCRIPT_PATH}")

    # Command to run the benchmark
    # We target the specific LLM script via benchmark.py's model and framework arguments
    cmd = [
        sys.executable, str(BENCHMARK_SCRIPT_PATH),
        "--framework", "pytorch",
        "--model", "gpt2",       # Using a small, fast-downloading LLM
        "--mode", "inference",
        "--usecase", "generation", # Explicitly set usecase
        "--precision", "fp32",
        "--batch_size", "1",
        # We don't need to specify --comprehensive here as we target a single config
        # The LLM script itself has a default for max_new_tokens (50)
    ]

    try:
        # Set PYTHONPATH to include the project root, similar to how benchmark.py does for sub-scripts
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{str(PROJECT_ROOT)}{os.pathsep}{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(PROJECT_ROOT)

        print(f"Running command: {' '.join(cmd)}")
        print(f"Environment PYTHONPATH: {env.get('PYTHONPATH')}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # Increased timeout for model download and first run
            cwd=PROJECT_ROOT, # Run from project root
            env=env
        )

        print("stdout:")
        print(result.stdout)
        print("stderr:")
        print(result.stderr)

        assert result.returncode == 0, f"Benchmark script failed with error: {result.stderr}"

        # Check for key phrases in the output
        assert "LLM BENCHMARK RESULTS" in result.stdout, "LLM_BENCHMARK_RESULTS section missing in output."
        assert "Framework: PyTorch" in result.stdout, "Framework info missing or incorrect."
        assert "Model: gpt2" in result.stdout, "Model info missing or incorrect."
        assert "Throughput (tokens/sec):" in result.stdout, "Throughput (tokens/sec) metric missing."
        assert "PyTorch Inference Time =" in result.stdout, "PyTorch Inference Time metric missing."

    except subprocess.TimeoutExpired:
        pytest.fail("Benchmark script timed out.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during the test: {e}")

# To run this test:
# Ensure transformers, torch, etc. are installed.
# pytest tests/test_llm_benchmarks.py -s -v
# The -s flag shows print statements, which is useful for debugging.
# The -v flag provides more verbose output.
# You might need to mark it with pytest.mark.llm and run specific markers if you have many tests.
