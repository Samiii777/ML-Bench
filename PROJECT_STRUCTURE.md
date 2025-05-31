# ML Benchmarking Framework - Project Structure

## Current Directory Tree

```
sol/
├── benchmark.py                    # Main entry point (860 lines)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
├── setup_data.py                   # Data setup utility
├── demo_gpu_ops.py                 # GPU operations demo
├── FRAMEWORK_FLOWCHART.md          # This flowchart document
├── GPU_OPERATIONS_GUIDE.md         # GPU operations guide
├── ONNX_GPU_OPERATIONS_SUMMARY.md  # ONNX GPU operations summary
├── PERFORMANCE_METRICS_UPDATE.md   # Performance metrics documentation
│
├── utils/                          # Configuration and utilities
│   ├── __init__.py
│   ├── config.py                   # Model and framework configuration (125 lines)
│   ├── logger.py                   # Logging system (86 lines)
│   ├── results.py                  # Results processing and output (550 lines)
│   ├── shared_device_utils.py      # GPU memory utilities (202 lines)
│   └── download.py                 # Model download utilities (95 lines)
│
├── benchmarks/                     # Benchmark implementations
│   ├── pytorch/                    # PyTorch benchmarks
│   │   ├── resnet/
│   │   │   └── inference/
│   │   │       └── classification/
│   │   │           └── main.py     # ResNet classification benchmark
│   │   ├── stable_diffusion/
│   │   │   └── inference/
│   │   │       └── generation/
│   │   │           └── main.py     # Stable Diffusion generation benchmark
│   │   └── gpu_ops/
│   │       └── inference/
│   │           └── compute/
│   │               ├── gemm/
│   │               │   └── main.py # GEMM operations benchmark
│   │               ├── conv/
│   │               │   └── main.py # Convolution operations benchmark
│   │               ├── memory/
│   │               │   └── main.py # Memory operations benchmark
│   │               ├── elementwise/
│   │               │   └── main.py # Element-wise operations benchmark
│   │               └── reduction/
│   │                   └── main.py # Reduction operations benchmark
│   │
│   └── onnx/                       # ONNX benchmarks (similar structure)
│       ├── resnet/
│       │   └── inference/
│       │       └── classification/
│       │           └── main.py     # ONNX ResNet benchmark
│       └── gpu_ops/
│           └── inference/
│               └── compute/
│                   ├── gemm/
│                   │   └── main.py # ONNX GEMM benchmark
│                   ├── conv/
│                   │   └── main.py # ONNX Convolution benchmark
│                   ├── memory/
│                   │   └── main.py # ONNX Memory benchmark
│                   └── elementwise/
│                       └── main.py # ONNX Element-wise benchmark
│
├── benchmark_results/              # Output directory
│   ├── benchmark_pytorch_*.json    # Raw results in JSON format
│   ├── benchmark_pytorch_*.csv     # Results in CSV format
│   ├── benchmark_onnx_*.json       # ONNX results in JSON format
│   ├── benchmark_onnx_*.csv        # ONNX results in CSV format
│   └── *_summary.txt               # Human-readable summaries
│
├── data/                           # Model and dataset storage
│   └── (model files and datasets)
│

```

## Key File Descriptions

### Core Files

- **`benchmark.py`** (860 lines): Main orchestrator that handles argument parsing, test execution, and result aggregation
- **`utils/config.py`** (125 lines): Central configuration for models, frameworks, and execution providers
- **`utils/results.py`** (550 lines): Comprehensive result processing, CSV/JSON generation, and summary tables
- **`utils/shared_device_utils.py`** (202 lines): GPU memory measurement using NVML library
- **`utils/logger.py`** (86 lines): Colored logging system for progress tracking

### Benchmark Scripts

Each benchmark script follows the pattern:
```
benchmarks/{framework}/{model_family}/{mode}/{use_case}/main.py
```

For GPU operations:
```
benchmarks/{framework}/gpu_ops/{mode}/{use_case}/{operation}/main.py
```

### Output Files

Results are saved with timestamps:
```
benchmark_results/benchmark_{framework}_{model}_{timestamp}.{format}
```

## Data Flow Through Project Structure

```mermaid
flowchart TD
    A[benchmark.py] --> B[utils/config.py]
    A --> C[utils/logger.py]
    A --> D[benchmarks/{framework}/{model}/main.py]
    
    D --> E[utils/shared_device_utils.py]
    D --> F[Model Execution]
    
    F --> G[Performance Metrics]
    G --> H[utils/results.py]
    
    H --> I[benchmark_results/*.json]
    H --> J[benchmark_results/*.csv]
    H --> K[benchmark_results/*_summary.txt]
    
    B --> L[Model Configurations]
    B --> M[Framework Settings]
    B --> N[Execution Providers]
```

## Framework Extension Points

### Adding New Frameworks
1. Create `benchmarks/{new_framework}/` directory
2. Add framework to `utils/config.py`
3. Implement benchmark scripts following the directory pattern
4. Update `benchmark.py` to handle new framework

### Adding New Models
1. Add model definition to `utils/config.py`
2. Create directory: `benchmarks/{framework}/{model_family}/`
3. Implement `main.py` with standard argument parsing
4. Follow output format for metric extraction

### Adding New Operations
1. Create subdirectory in `benchmarks/{framework}/gpu_ops/inference/compute/`
2. Implement operation-specific benchmark
3. Use standard performance metric output format

## Configuration System

The `utils/config.py` file centralizes all configuration:

```python
# Model family mapping
MODEL_FAMILIES = {
    "resnet50": "resnet",
    "gemm_ops": "gpu_ops",
    "conv_ops": "gpu_ops",
    # ...
}

# Framework availability
FRAMEWORK_MODELS = {
    "pytorch": ["resnet50", "stable_diffusion", "gemm_ops", ...],
    "onnx": ["resnet50", "gemm_ops", "conv_ops", ...],
}

# Execution providers
ONNX_EXECUTION_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider", 
    "CPUExecutionProvider"
]
```

This structure enables:
- **Scalable Framework Support**: Easy addition of new ML frameworks
- **Flexible Model Organization**: Hierarchical model organization by family and use case
- **Consistent Interface**: All benchmarks follow the same argument and output patterns
- **Comprehensive Testing**: Automatic testing of all valid combinations
- **Rich Output**: Multiple output formats for different analysis needs 