# 📊 ML-Bench Visualization Guide

ML-Bench now includes powerful visualization capabilities to help you analyze and understand your benchmark results through interactive dashboards, static reports, and CLI tools.

## 🚀 Quick Start

### 1. Install Visualization Dependencies

```bash
# Install all visualization requirements
pip install -r requirements.txt

# Or install just visualization packages
pip install streamlit plotly matplotlib seaborn
```

### 2. Run Benchmarks with Auto-Visualization

```bash
# Run benchmarks and launch interactive dashboard
python benchmark.py --visualize

# Run benchmarks and create static HTML report
python benchmark.py --visualize --viz-mode static

# Run benchmarks with CLI visualization
python benchmark.py --visualize --viz-mode cli
```

### 3. Standalone Visualization

```bash
# Launch interactive dashboard
python visualize.py

# Create CLI summary
python visualize.py --mode cli

# Generate static HTML report
python visualize.py --mode static
```

---

## 📈 Visualization Modes

### 1. **Interactive Dashboard** (Recommended)
- **Launch**: `python visualize.py --mode dashboard`
- **Features**: Real-time filtering, interactive charts, data export
- **Access**: http://localhost:8501
- **Best for**: Detailed analysis, presentations, collaboration

### 2. **Static HTML Reports**
- **Launch**: `python visualize.py --mode static`
- **Features**: Self-contained HTML files, offline viewing
- **Output**: `visualization_output/index.html`
- **Best for**: Sharing results, archival, documentation

### 3. **CLI Visualization**
- **Launch**: `python visualize.py --mode cli`
- **Features**: Terminal-based summary, quick insights
- **Best for**: Quick analysis, SSH connections, automation

---

## 📊 Chart Types & Analysis

### **Performance Comparison**
- **What**: Throughput, latency, GFLOPS comparison across models/frameworks
- **Insights**: Identify best performing configurations
- **Filters**: Framework, precision, batch size

### **Memory Analysis**
- **What**: GPU/CPU memory usage vs batch size
- **Insights**: Memory scaling patterns, VRAM requirements
- **Use case**: Planning hardware requirements

### **Batch Size Scaling**
- **What**: Throughput scaling with increased batch sizes
- **Insights**: Optimal batch size for each model/framework
- **Use case**: Performance tuning

### **Precision Impact**
- **What**: Performance comparison between FP32, FP16, Mixed
- **Insights**: Speed vs accuracy tradeoffs
- **Use case**: Precision selection

### **Framework Heatmap**
- **What**: Performance matrix across models and frameworks
- **Insights**: Framework strengths by model type
- **Use case**: Framework selection

### **Model Radar Charts**
- **What**: Multi-dimensional model comparison
- **Insights**: Balanced view of model performance
- **Use case**: Model selection

---

## 🎯 Usage Examples

### Example 1: Comprehensive Analysis
```bash
# Run full benchmark suite with visualization
python benchmark.py --comprehensive --visualize
# Opens interactive dashboard at http://localhost:8501
```

### Example 2: Quick Model Comparison
```bash
# Compare ResNet models with CLI summary
python benchmark.py --model resnet --precision fp16 --visualize --viz-mode cli
```

### Example 3: Framework Comparison Report
```bash
# Compare PyTorch vs ONNX and generate HTML report
python benchmark.py --framework pytorch onnx --model resnet50 --visualize --viz-mode static
```

### Example 4: GPU Operations Analysis
```bash
# Analyze GPU compute operations
python benchmark.py --usecase compute --visualize
```

---

## 📋 Interactive Dashboard Features

### **Main Dashboard**
- **File Selection**: Choose from recent benchmark results
- **Live Filtering**: Filter by framework, model, precision
- **Summary Metrics**: Key performance indicators
- **Export Options**: Download filtered data as CSV

### **Chart Interactions**
- **Zoom**: Mouse wheel or click-drag
- **Pan**: Click and drag to move view
- **Hover**: Detailed information on data points
- **Legend**: Click to show/hide series

### **Data Export**
- **CSV Download**: Filtered benchmark data
- **Chart Export**: PNG, SVG, HTML formats
- **Report Generation**: Complete analysis reports

---

## 🛠️ Advanced Configuration

### Custom Port Configuration
```bash
# Use custom port for dashboard
python visualize.py --port 8502
```

### Specific Results File
```bash
# Visualize specific benchmark run
python visualize.py --results-file benchmark_results/benchmark_inference_20241201_143022.json
```

### Custom Output Directory
```bash
# Create reports in custom location
python visualize.py --mode static --output-dir my_reports/
```

---

## 📊 Understanding Your Data

### **Throughput (samples/sec)**
- Higher is better
- Measures inference speed
- Key metric for production deployment

### **Latency (ms)**
- Lower is better
- Measures response time
- Important for real-time applications

### **GFLOPS (Billion Floating Point Operations/sec)**
- Higher is better
- Measures computational efficiency
- Relevant for GPU compute benchmarks

### **Memory Usage (GB)**
- Lower is often better
- Critical for hardware planning
- Scales with batch size and precision

---

## 🔧 Troubleshooting

### **Dashboard Won't Start**
```bash
# Check if port is in use
netstat -an | grep 8501

# Try different port
python visualize.py --port 8502
```

### **Can't Stop Dashboard (Windows)**
On Windows, Ctrl+C may not stop the Streamlit dashboard properly. Use these solutions:

**Option 1: Use the dedicated launcher**
```bash
# Start with better stopping capability
python dashboard.py
```

**Option 2: Use the stop script**
```bash
# Run this in another terminal or double-click the file
stop_dashboard.bat
```

**Option 3: Manual stopping**
- Close the terminal window entirely
- Open Task Manager (Ctrl+Shift+Esc)
- Find "python.exe" processes and end them
- Or use: `taskkill /f /im python.exe`

### **Missing Dependencies**
```bash
# Install missing packages
pip install streamlit plotly matplotlib seaborn pandas
```

### **No Data Found**
```bash
# Ensure benchmark results exist
ls -la benchmark_results/*.json

# Run benchmarks first
python benchmark.py --model resnet18 --precision fp16 --batch_size 1
```

### **Memory Issues with Large Datasets**
```bash
# Use specific results file instead of loading all
python visualize.py --results-file benchmark_results/latest_results.json
```

---

## 🎨 Customization

### **Color Schemes**
The visualizer uses consistent color coding:
- **PyTorch**: Red (#EE4C2C)
- **ONNX**: Blue (#1F77B4)
- **TensorFlow**: Orange (#FF6F00)

### **Adding Custom Metrics**
To visualize custom metrics, ensure your benchmark scripts output metrics in the expected format:

```python
# In your benchmark script
metrics = {
    'custom_metric': 123.45,
    'throughput_fps': 250.0,
    'avg_latency_ms': 4.0
}
```

---

## 📈 Best Practices

### **Dashboard Usage**
1. **Start Broad**: Use default filters to see overall patterns
2. **Drill Down**: Apply filters to focus on specific comparisons
3. **Export Data**: Save filtered results for further analysis
4. **Share Reports**: Use static mode for presentations

### **Performance Analysis**
1. **Compare Apples to Apples**: Same batch size and precision
2. **Consider Use Case**: Throughput vs latency importance
3. **Memory Planning**: Check VRAM requirements early
4. **Framework Selection**: Consider both performance and ecosystem

### **Report Generation**
1. **Regular Snapshots**: Generate reports after major runs
2. **Version Control**: Track performance changes over time
3. **Document Findings**: Add context to your analysis
4. **Share Insights**: Use visualization for team communication

---

## 🔄 Integration with Existing Workflow

### **Automated Reporting**
```bash
#!/bin/bash
# run_and_visualize.sh
python benchmark.py --comprehensive
python visualize.py --mode static --output-dir reports/$(date +%Y%m%d)
```

### **CI/CD Integration**
```yaml
# .github/workflows/benchmark.yml
- name: Run Benchmarks
  run: python benchmark.py --model resnet50
  
- name: Generate Report
  run: python visualize.py --mode static
  
- name: Upload Reports
  uses: actions/upload-artifact@v2
  with:
    name: benchmark-report
    path: visualization_output/
```

---

## 📊 Sample Outputs

### **CLI Summary Example**
```
📊 ML-Bench Results Summary
==================================================
🚀 Performance by Framework:
              mean      max  count
framework                       
onnx        156.23   245.67      4
pytorch     234.56   445.12      6

⭐ Top 5 Configurations by Throughput:
  pytorch/resnet18 fp16 BS=8: 445.1 samples/sec
  pytorch/resnet50 fp16 BS=4: 312.4 samples/sec
  onnx/resnet18 fp32 BS=1: 245.7 samples/sec
  pytorch/resnet34 mixed BS=2: 198.3 samples/sec
  onnx/resnet50 fp16 BS=1: 187.9 samples/sec
```

### **Dashboard Features**
- Interactive charts with zoom/pan
- Real-time filtering and updates  
- Hover tooltips with detailed info
- Export capabilities for data and charts

---

Ready to visualize your ML benchmarks? Start with `python benchmark.py --visualize` and explore your performance data! 🚀 