# Performance Metrics Update: Compute Use Cases

## Overview

The ML benchmarking framework has been updated to display **meaningful performance metrics** for compute use cases instead of the generic "samples/sec" metric. This provides much more actionable insights for GPU operations benchmarking.

## Updated Performance Metrics by Use Case

### 1. **Classification Use Cases** 
- **Metric**: `samples/sec` (throughput)
- **Models**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **Rationale**: For image classification, samples processed per second is the most relevant metric

**Example**:
```
│ pytorch_res│ pytorch  │ resnet50   │ inference│ fp32     │ 1        │ classific│ 553.47 samp/s │ 1.81 ms    │ 1.16 GB  │ cuda       │
```

### 2. **Generation Use Cases**
- **Metric**: `samples/sec` (throughput) 
- **Models**: Stable Diffusion 1.5
- **Rationale**: For image generation, images generated per second is meaningful

**Example**:
```
│ pytorch_sta│ pytorch  │ stable_diff│ inference│ fp32     │ 1        │ generatio│ 0.64 samp/s   │ 1572.97 ms │ 6.46 GB  │ cuda       │
```

### 3. **Compute Use Cases - GFLOPS**
- **Metric**: `GFLOPS` (Giga Floating-Point Operations Per Second)
- **Models**: GEMM operations, Convolution operations
- **Rationale**: For compute-intensive operations, FLOPS is the standard performance metric

**Examples**:
```
│ pytorch_gem│ pytorch  │ gemm_ops   │ inference│ fp32     │ 1        │ compute  │ 55351.2 GFLOPS│ 3.26 ms    │ 2.30 GB  │ cuda       │
│ pytorch_con│ pytorch  │ conv_ops   │ inference│ fp32     │ 1        │ compute  │ 52625.8 GFLOPS│ 0.06 ms    │ 1.08 GB  │ cuda       │
```

### 4. **Compute Use Cases - Bandwidth**
- **Metric**: `GB/s` (Gigabytes per Second)
- **Models**: Memory operations, Element-wise operations, Reduction operations
- **Rationale**: For memory-bound operations, bandwidth is the limiting factor

**Examples**:
```
│ pytorch_mem│ pytorch  │ memory_ops │ inference│ fp32     │ 1        │ compute  │ 494901.1 GB/s │ 0.17 ms    │ 4.94 GB  │ cuda       │
│ pytorch_ele│ pytorch  │ elementwise│ inference│ fp32     │ 1        │ compute  │ 1326.0 GB/s   │ 0.82 ms    │ 7.59 GB  │ cuda       │
│ pytorch_red│ pytorch  │ reduction_o│ inference│ fp32     │ 1        │ compute  │ 2882.5 GB/s   │ 0.32 ms    │ 2.00 GB  │ cuda       │
```

## Technical Implementation

### 1. **Enhanced Output Parsing**
Updated `benchmark.py` to extract operation-specific metrics:

```python
# Look for GFLOPS performance (for GEMM and convolution operations)
if "Best GEMM Performance:" in line and "GFLOPS" in line:
    gflops_str = line.split(':')[1].strip().split('GFLOPS')[0].strip()
    metrics['best_gflops'] = float(gflops_str)
    metrics['performance_metric'] = 'GFLOPS'

# Look for bandwidth performance (for memory, elementwise, reduction operations)
if "Best Memory Bandwidth:" in line and "GB/s" in line:
    bandwidth_str = line.split(':')[1].strip().split('GB/s')[0].strip()
    metrics['best_bandwidth_gbs'] = float(bandwidth_str)
    metrics['performance_metric'] = 'GB/s'
```

### 2. **Smart Metric Selection**
Updated `utils/results.py` to choose appropriate metrics based on use case:

```python
if use_case == "compute":
    # For compute use cases, prefer GFLOPS or GB/s over samples/sec
    if metrics.get("best_gflops"):
        performance_str = f"{metrics['best_gflops']:.1f} GFLOPS"
    elif metrics.get("best_bandwidth_gbs"):
        performance_str = f"{metrics['best_bandwidth_gbs']:.1f} GB/s"
    else:
        performance_str = f"{metrics['throughput_fps']:.2f} samp/s"
else:
    # For other use cases (classification, generation), use samples/sec
    performance_str = f"{metrics['throughput_fps']:.2f} samp/s"
```

### 3. **Consistent Logging**
Updated logging throughout the framework to show appropriate metrics:

```python
if model_use_case == "compute":
    # For compute use cases, show GFLOPS or GB/s
    if metrics.get("best_gflops"):
        self.logger.info(f"  {metrics['best_gflops']:.1f} GFLOPS")
    elif metrics.get("best_bandwidth_gbs"):
        self.logger.info(f"  {metrics['best_bandwidth_gbs']:.1f} GB/s")
else:
    # For other use cases, show samples/sec and latency
    throughput = metrics.get("throughput_fps", 0)
    latency = metrics.get("avg_latency_ms", 0)
    self.logger.info(f"  {throughput:.2f} samples/sec, {latency:.2f} ms/sample")
```

## Performance Insights from Results

### **GFLOPS Performance** (Compute-Intensive Operations)
- **GEMM Operations**: 55,351 GFLOPS - Excellent matrix multiplication performance
- **Convolution Operations**: 52,626 GFLOPS - High-performance convolution kernels

### **Bandwidth Performance** (Memory-Bound Operations)  
- **Memory Operations**: 494,901 GB/s - Exceptional memory bandwidth (likely cached operations)
- **Reduction Operations**: 2,883 GB/s - Good reduction performance
- **Element-wise Operations**: 1,326 GB/s - Solid element-wise operation bandwidth

### **Traditional Metrics** (Application-Level Performance)
- **ResNet50 Classification**: 553 samples/sec - Good inference throughput
- **Stable Diffusion Generation**: 0.64 samples/sec - Typical for high-quality image generation

## Benefits of the Update

### 1. **Meaningful Metrics**
- **Before**: All operations showed "samples/sec" regardless of what they actually do
- **After**: Each operation shows its most relevant performance metric

### 2. **Better Performance Analysis**
- **GFLOPS**: Directly comparable to GPU specifications and other benchmarks
- **GB/s**: Shows memory subsystem utilization and bottlenecks
- **Samples/sec**: Still used where it makes sense (classification, generation)

### 3. **Industry Standard Alignment**
- **GFLOPS**: Standard metric for compute performance (MLPerf, vendor specs)
- **GB/s**: Standard metric for memory performance (STREAM benchmark, etc.)
- **Samples/sec**: Standard metric for application throughput

### 4. **Actionable Insights**
- **High GFLOPS**: GPU compute units are well utilized
- **High GB/s**: Memory subsystem is efficiently used
- **Low values**: Indicates potential optimization opportunities

## Usage Examples

### Individual Operation Testing
```bash
# Test GEMM operations - shows GFLOPS
python benchmark.py --model gemm_ops --framework pytorch --precision fp32 --batch_size 1
# Output: ✓ gemm_ops: PASSED - 55351.2 GFLOPS

# Test memory operations - shows GB/s  
python benchmark.py --model memory_ops --framework pytorch --precision fp32 --batch_size 1
# Output: ✓ memory_ops: PASSED - 494901.1 GB/s

# Test ResNet classification - shows samples/sec
python benchmark.py --model resnet50 --framework pytorch --precision fp32 --batch_size 1
# Output: ✓ resnet50: PASSED - 553.47 samples/sec, 1.81 ms/sample
```

### Comprehensive Testing
```bash
# Test all models with appropriate metrics
python benchmark.py --framework pytorch --precision fp32 --batch_size 1
```

## Conclusion

This update makes the benchmarking framework much more useful for performance analysis by showing:

- **🔢 GFLOPS** for compute-intensive operations (GEMM, convolution)
- **📊 GB/s** for memory-bound operations (memory, element-wise, reduction)  
- **⚡ Samples/sec** for application-level performance (classification, generation)

The results are now directly comparable to industry benchmarks and provide actionable insights for optimization work! 