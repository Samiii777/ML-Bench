# Contributing to ML-Bench

Thank you for your interest in contributing to ML-Bench! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ML-Bench.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate the virtual environment: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Set up data files: `python setup_data.py`

## Development Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for GPU benchmarks)
- NVIDIA drivers and CUDA toolkit (for GPU support)

### Installing Development Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── benchmark.py              # Main benchmarking script
├── setup_data.py             # Data setup utility
├── requirements.txt          # Python dependencies
├── utils/                    # Utility modules
│   ├── config.py            # Configuration management
│   ├── logger.py            # Logging utilities
│   ├── results.py           # Results processing
│   ├── download.py          # File download utilities
│   └── shared_device_utils.py # GPU memory utilities
├── benchmarks/              # Benchmark implementations
│   ├── pytorch/             # PyTorch benchmarks
│   └── onnx/               # ONNX Runtime benchmarks
└── data/                   # Downloaded data files (gitignored)
```

## Adding New Benchmarks

### Adding a New Model

1. **Create the benchmark script**: Add a new directory under `benchmarks/{framework}/{model_family}/{mode}/{use_case}/`
2. **Implement the main.py**: Follow the existing pattern with proper argument parsing and output formatting
3. **Update configuration**: Add the model to `utils/config.py`
4. **Test the benchmark**: Ensure it works with the main framework

### Adding a New Framework

1. **Create framework directory**: `benchmarks/{new_framework}/`
2. **Implement benchmark scripts**: Follow the existing PyTorch/ONNX patterns
3. **Update configuration**: Add framework support to `utils/config.py`
4. **Update main script**: Add framework handling to `benchmark.py`

### Benchmark Script Requirements

Each benchmark script must:
- Accept standard command line arguments (`--model`, `--precision`, `--batch_size`)
- Output results in the expected format for parsing
- Handle errors gracefully
- Include proper device detection and memory measurement
- Follow the established output format for metrics

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

## Testing

Before submitting a pull request:

1. **Test your changes**:
   ```bash
   # Test individual benchmark
   python benchmark.py --framework pytorch --model resnet50 --precision fp32 --batch_size 1
   
   # Test comprehensive benchmarking
   python benchmark.py --framework pytorch --model resnet50
   ```

2. **Verify output formats**: Ensure results are properly parsed and displayed
3. **Check error handling**: Test with invalid inputs and edge cases
4. **Test on different hardware**: If possible, test on both CPU and GPU

## Submitting Changes

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Implement your feature or fix
3. **Test thoroughly**: Ensure all existing functionality still works
4. **Commit your changes**: Use clear, descriptive commit messages
5. **Push to your fork**: `git push origin feature/your-feature-name`
6. **Create a pull request**: Describe your changes and their purpose

## Pull Request Guidelines

- **Clear description**: Explain what your PR does and why
- **Test results**: Include benchmark results if adding new models/frameworks
- **Documentation**: Update README.md if adding new features
- **Small, focused changes**: Keep PRs focused on a single feature or fix
- **Follow existing patterns**: Maintain consistency with existing code

## Reporting Issues

When reporting bugs or requesting features:

1. **Check existing issues**: Avoid duplicates
2. **Provide details**: Include system info, error messages, and steps to reproduce
3. **Include benchmark results**: If relevant, include performance data
4. **Use clear titles**: Make it easy to understand the issue

## Performance Considerations

When adding new benchmarks:
- **Optimize for accuracy**: Ensure measurements are reliable
- **Handle memory efficiently**: Clean up GPU memory between tests
- **Support multiple precisions**: FP32, FP16, mixed precision where applicable
- **Include proper warmup**: Ensure stable performance measurements

## Documentation

- Update README.md for new features
- Add docstrings to new functions
- Update PROJECT_STRUCTURE.md if changing architecture
- Include usage examples for new functionality

## Questions?

If you have questions about contributing:
- Open an issue for discussion
- Check existing documentation
- Look at similar implementations in the codebase

Thank you for contributing to ML-Bench! 