# ML-Bench: Comprehensive Machine Learning Benchmarking Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.12+-green.svg)](https://onnxruntime.ai/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)

ML-Bench is a comprehensive benchmarking framework designed to evaluate machine learning models across multiple frameworks, precisions, and hardware configurations. It provides standardized performance measurements for deep learning inference workloads with support for PyTorch and ONNX Runtime.

## 🚀 Key Features

- **Multi-Framework Support**: PyTorch and ONNX Runtime with automatic optimization
- **Comprehensive Model Coverage**: Image classification, text-to-image generation, and GPU compute operations
- **Precision Testing**: FP32, FP16, and mixed precision benchmarking
- **Hardware Optimization**: CUDA, TensorRT, and CPU execution providers
- **Memory Monitoring**: Real-time GPU memory usage tracking with NVML
- **Automated Testing**: Comprehensive benchmarking across all valid configurations
- **Rich Output Formats**: JSON, CSV, and human-readable summary reports
- **Extensible Architecture**: Easy addition of new models and frameworks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Usage Examples](#usage-examples)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** (optional, for GPU benchmarks)
- **CUDA 11.0+** (for GPU acceleration)
- **8GB+ RAM** (16GB+ recommended for large models)
- **20GB+ disk space** (for model downloads)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ML-Bench.git
cd ML-Bench

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up data files (downloads models and datasets)
python setup_data.py
```

### Verify Installation

```bash
# Test basic functionality
python benchmark.py --framework pytorch --model resnet18 --precision fp32 --batch_size 1

# Check GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Run Comprehensive Benchmarks (Default)

```bash
# Run all models across all use cases (comprehensive)
python3 benchmark.py

# Run all models with specific precision
python3 benchmark.py --precision fp16

# Run all models with specific batch sizes
python3 benchmark.py --batch_size 1 4
```

### Run Specific Benchmarks

```bash
# Basic ResNet-50 benchmark
python benchmark.py --framework pytorch --model resnet50 --precision fp16 --batch_size 4

# Stable Diffusion generation benchmark
python benchmark.py --usecase generation --precision fp16

# GPU compute operations benchmark
python benchmark.py --usecase compute --model gemm_ops --precision fp16
```

### Run Framework-Specific Comprehensive Benchmarks

```bash
# Test all models with default settings
python benchmark.py --comprehensive

# Test specific framework comprehensively
python benchmark.py --framework pytorch --comprehensive

# Test specific use case comprehensively
python benchmark.py --usecase classification --comprehensive
```

### Quick Performance Test

```bash
# Fast performance overview
python benchmark.py --framework pytorch --model resnet18 --precision fp16 --batch_size 1 4 8
```

## 🎯 Supported Models

### Image Classification (PyTorch + ONNX)
- **ResNet Family**: ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- **Use Case**: `classification`
- **Precisions**: FP32, FP16, Mixed
- **Batch Sizes**: 1, 4, 8, 16, 32+

### Text-to-Image Generation (PyTorch)
- **Stable Diffusion 1.5**: High-speed image generation
- **Stable Diffusion 3 Medium**: Latest high-quality generation
- **Use Case**: `generation`
- **Precisions**: FP32, FP16, Mixed
- **Memory**: 4-22GB VRAM depending on model and precision

### GPU Compute Operations (PyTorch + ONNX)
- **GEMM Operations**: Matrix multiplication benchmarks
- **Convolution Operations**: 2D convolution performance
- **Memory Operations**: Memory bandwidth testing
- **Element-wise Operations**: Point-wise computations
- **Reduction Operations**: Sum, mean, max operations
- **Use Case**: `compute`

## 📊 Usage Examples

### Basic Usage

```bash
# Single model benchmark
python benchmark.py --framework pytorch --model resnet50 --precision fp16 --batch_size 4

# Multiple batch sizes
python benchmark.py --framework pytorch --model resnet50 --precision fp16 --batch_size 1 4 8 16

# Multiple precisions
python benchmark.py --framework pytorch --model resnet50 --precision fp32 fp16 mixed --batch_size 4
```

### Use Case-Based Benchmarking

```bash
# Image classification benchmarks
python benchmark.py --usecase classification --framework pytorch

# Text-to-image generation benchmarks
python benchmark.py --usecase generation --precision fp16

# GPU compute benchmarks
python benchmark.py --usecase compute --framework pytorch --precision fp16
```

### Framework Comparison

```bash
# Compare PyTorch vs ONNX for ResNet
python benchmark.py --framework pytorch onnx --model resnet50 --precision fp16 --batch_size 4

# Compare all frameworks for classification
python benchmark.py --usecase classification --framework pytorch onnx --comprehensive
```

### Advanced Configuration

```bash
# ONNX with specific execution provider
python benchmark.py --framework onnx --model resnet50 --execution_provider TensorrtExecutionProvider

# Comprehensive benchmarking with custom output
python benchmark.py --comprehensive --output_dir custom_results/

# Memory-optimized Stable Diffusion
python benchmark.py --usecase generation --model sd15 --precision fp16 --batch_size 1
```

## 🎯 Benchmark Results

### Output Formats

ML-Bench generates results in multiple formats:

1. **JSON Files**: Raw benchmark data with full metrics
2. **CSV Files**: Tabular data for analysis and plotting
3. **Summary Reports**: Human-readable performance summaries
4. **Console Output**: Real-time progress and results

### Sample Output

```
============================================================
BENCHMARK RESULTS SUMMARY
============================================================
Framework: PyTorch | Model: resnet50 | Precision: fp16 | Batch Size: 4

✅ PASS | Inference Time: 12.34 ms | Throughput: 324.2 samples/sec
   GPU Memory: 2.1 GB | Latency: 3.08 ms/sample

============================================================
STABLE DIFFUSION BENCHMARK SUMMARY  
============================================================
✅ Stable Diffusion 1.5: 2.34 images/sec, 3.9 GB VRAM
✅ Stable Diffusion 3 Medium: 0.81 images/sec, 14.2 GB VRAM
============================================================
```

### Results Directory Structure

```
benchmark_results/
├── benchmark_pytorch_resnet50_20241201_143022.json    # Raw results
├── benchmark_pytorch_resnet50_20241201_143022.csv     # Tabular data
├── benchmark_pytorch_comprehensive_summary.txt        # Human-readable summary
└── performance_comparison_20241201_143022.json        # Framework comparison
```

## 🏗️ Project Structure

```
ML-Bench/
├── benchmark.py                    # Main benchmarking script (1,104 lines)
├── requirements.txt                # Python dependencies
├── setup_data.py                   # Model and data setup
├── README.md                       # This file
├── LICENSE                         # Project license
│
├── utils/                          # Core utilities (194 lines total)
│   ├── config.py                   # Model and framework configuration
│   ├── logger.py                   # Colored logging system
│   ├── results.py                  # Results processing and output
│   ├── shared_device_utils.py      # GPU memory monitoring
│   └── download.py                 # Model download utilities
│
├── benchmarks/                     # Benchmark implementations
│   ├── pytorch/                    # PyTorch benchmarks
│   │   ├── resnet/                 # ResNet classification
│   │   ├── stable_diffusion/       # Stable Diffusion generation
│   │   └── gpu_ops/               # GPU compute operations
│   └── onnx/                      # ONNX Runtime benchmarks
│       ├── resnet/                # ResNet classification
│       └── gpu_ops/               # GPU compute operations
│
├── benchmark_results/              # Generated benchmark results
├── data/                          # Downloaded models and datasets
└── .venv/                         # Virtual environment (gitignored)
```

### Key Components

- **`benchmark.py`**: Main orchestrator handling test execution and result aggregation
- **`utils/config.py`**: Central configuration for models, frameworks, and simple VRAM checking
- **`utils/results.py`**: Comprehensive result processing and output generation
- **`benchmarks/{framework}/`**: Framework-specific benchmark implementations

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

### Adding New Models

1. Create benchmark script: `benchmarks/{framework}/{model_family}/{mode}/{usecase}/main.py`
2. Update configuration: Add model to `utils/config.py`
3. Follow existing patterns for argument parsing and output formatting
4. Test with the main framework: `python benchmark.py --model your_model`

## ⚡ Performance Tips

### For Best Performance

1. **Use FP16 precision** on CUDA GPUs for optimal speed/memory balance
2. **Enable TensorRT** for ONNX models: `--execution_provider TensorrtExecutionProvider`
3. **Optimize batch sizes** based on your GPU memory capacity
4. **Close other applications** to free GPU memory
5. **Use latest NVIDIA drivers** and CUDA toolkit

### Memory Optimization

```bash
# For large models like Stable Diffusion 3
python benchmark.py --usecase generation --model sd3 --precision fp16 --batch_size 1

# Enable CPU offload for SD3 if needed
python benchmarks/pytorch/stable_diffusion/inference/generation/main.py --model sd3 --cpu-offload
```

### Batch Size Guidelines

| Model Type | GPU Memory | Recommended Batch Size |
|------------|------------|----------------------|
| ResNet-50 FP16 | 8GB | 16-32 |
| ResNet-50 FP32 | 8GB | 8-16 |
| Stable Diffusion 1.5 | 8GB | 1-2 |
| Stable Diffusion 3 | 16GB+ | 1 |

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
python benchmark.py --model resnet50 --batch_size 1

# Use FP16 precision
python benchmark.py --model resnet50 --precision fp16
```

**Model Download Failures**
```bash
# Re-run setup with verbose output
python setup_data.py --verbose

# Check internet connection and disk space
df -h  # Check disk space
```

**TensorRT Compilation Errors**
```bash
# Fall back to CUDA provider
python benchmark.py --framework onnx --execution_provider CUDAExecutionProvider
```

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check virtual environment activation
which python  # Should point to .venv/bin/python
```

### Getting Help

1. Check existing [issues](https://github.com/your-username/ML-Bench/issues)
2. Review [troubleshooting documentation](docs/troubleshooting.md)
3. Open a new issue with:
   - System information (`nvidia-smi`, `python --version`)
   - Full error message and stack trace
   - Command that caused the issue

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **ONNX Runtime Team** for high-performance inference
- **Hugging Face** for model hosting and diffusers library
- **NVIDIA** for CUDA and TensorRT optimization tools

---

**Ready to benchmark?** Start with: `python3 benchmark.py` for comprehensive testing across all models and use cases, or `python3 benchmark.py --framework pytorch --model resnet18 --precision fp16 --batch_size 4` for specific model testing.

For more examples and advanced usage, see our [documentation](docs/) and [examples](examples/) directories.

## Memory Management

**Simple VRAM checking is enabled by default** to prevent out-of-memory crashes by skipping Stable Diffusion configurations that require more VRAM than available.

### How It Works

- **Large models** (Stable Diffusion): Checked against available VRAM and skipped if insufficient
- **Small models** (ResNet, GPU operations): Run without VRAM checking as they use minimal memory

### VRAM Requirements Table

| Model | FP32 | FP16 | Mixed |
|-------|------|------|-------|
| Stable Diffusion 1.5 | 12.0GB | 6.0GB | 9.0GB |
| Stable Diffusion 3 | >24GB | 20.0GB | >24GB |

*Requirements scale with batch size (each additional batch adds ~80% more VRAM)*

```bash
# SD3 FP32 will be skipped automatically
python3 benchmark.py --model sd3 --precision fp32 --batch_size 1
# ⚠️  SKIPPED - VRAM insufficient: Requires >24GB VRAM (available: 23.4GB)

# SD3 FP16 will run if you have enough VRAM
python3 benchmark.py --model sd3 --precision fp16 --batch_size 1
# ✓ 1.00 samples/sec
``` 