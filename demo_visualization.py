#!/usr/bin/env python3
"""
ML-Bench Visualization Demo
Demonstrates the visualization capabilities with sample data
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import random

def create_sample_results():
    """Create sample benchmark results for demonstration"""
    
    models = ['resnet18', 'resnet50', 'resnet101', 'stable_diffusion_1_5']
    frameworks = ['pytorch', 'onnx']
    precisions = ['fp32', 'fp16', 'mixed']
    batch_sizes = [1, 4, 8, 16]
    usecases = ['classification', 'detection', 'generation']
    
    results = []
    
    for framework in frameworks:
        for model in models:
            # Skip certain combinations that don't make sense
            if model == 'stable_diffusion_1_5':
                test_usecases = ['generation']
                test_batch_sizes = [1, 2]
            else:
                test_usecases = ['classification', 'detection']
                test_batch_sizes = batch_sizes
            
            for usecase in test_usecases:
                for precision in precisions:
                    for batch_size in test_batch_sizes:
                        # Skip FP32 for some models to simulate real scenarios
                        if model == 'resnet101' and precision == 'fp32' and batch_size > 8:
                            continue
                        
                        # Generate realistic performance data
                        base_throughput = {
                            'resnet18': 400,
                            'resnet50': 250,
                            'resnet101': 120,
                            'stable_diffusion_1_5': 2.5
                        }.get(model, 200)
                        
                        # Adjust for framework
                        if framework == 'onnx':
                            base_throughput *= 0.85  # ONNX typically slightly slower
                        
                        # Adjust for precision
                        precision_multiplier = {
                            'fp32': 1.0,
                            'fp16': 1.8,  # FP16 is faster
                            'mixed': 1.4
                        }.get(precision, 1.0)
                        
                        # Adjust for batch size (diminishing returns)
                        batch_multiplier = min(batch_size * 0.9, batch_size * 0.6 + 2)
                        
                        throughput = base_throughput * precision_multiplier * batch_multiplier
                        throughput += random.uniform(-throughput*0.1, throughput*0.1)  # Add noise
                        
                        latency = (1000 / throughput) * batch_size
                        
                        # Memory usage
                        base_memory = {
                            'resnet18': 0.5,
                            'resnet50': 1.2,
                            'resnet101': 2.1,
                            'stable_diffusion_1_5': 6.0
                        }.get(model, 1.0)
                        
                        memory_multiplier = {
                            'fp32': 1.0,
                            'fp16': 0.6,
                            'mixed': 0.8
                        }.get(precision, 1.0)
                        
                        memory_usage = base_memory * memory_multiplier * (1 + batch_size * 0.3)
                        
                        result = {
                            "status": "PASS",
                            "execution_time": random.uniform(10, 60),
                            "metrics": {
                                "framework": framework.upper(),
                                "device": "cuda",
                                "avg_latency_ms": round(latency, 2),
                                "throughput_fps": round(throughput, 2),
                                "inference_time_ms": round(latency, 2),
                                "total_gpu_memory_used_gb": round(memory_usage, 2),
                                "gpu_memory_allocated_gb": round(memory_usage * 0.8, 2)
                            },
                            "framework": framework,
                            "model": model,
                            "mode": "inference",
                            "usecase": usecase,
                            "precision": precision,
                            "batch_size": batch_size,
                            "execution_provider": "CUDAExecutionProvider" if framework == "onnx" else None,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        results.append(result)
    
    # Add some GPU compute operations results
    gpu_ops = ['gemm_ops', 'conv_ops', 'memory_ops', 'elementwise_ops', 'reduction_ops']
    
    for framework in ['pytorch']:  # Only PyTorch for GPU ops
        for model in gpu_ops:
            for precision in ['fp32', 'fp16']:
                for batch_size in [1, 4, 16]:
                    # Generate GFLOPS or bandwidth data
                    if model in ['gemm_ops', 'conv_ops']:
                        performance = random.uniform(800, 2000)  # GFLOPS
                        metric_name = 'best_gflops'
                        metric_type = 'GFLOPS'
                    else:
                        performance = random.uniform(200, 800)  # GB/s
                        metric_name = 'best_bandwidth_gbs'
                        metric_type = 'GB/s'
                    
                    result = {
                        "status": "PASS",
                        "execution_time": random.uniform(5, 20),
                        "metrics": {
                            "framework": "PyTorch",
                            "device": "cuda",
                            metric_name: round(performance, 1),
                            "performance_metric": metric_type,
                            "throughput_fps": round(random.uniform(50, 200), 2)
                        },
                        "framework": framework,
                        "model": model,
                        "mode": "inference",
                        "usecase": "compute",
                        "precision": precision,
                        "batch_size": batch_size,
                        "execution_provider": None,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    results.append(result)
    
    return results

def save_demo_results(results):
    """Save demo results to file"""
    os.makedirs("benchmark_results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results/demo_results_{timestamp}.json"
    
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "demo": True,
            "description": "Demo benchmark results for visualization showcase"
        },
        "results": results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return filename

def run_demo():
    """Run the complete visualization demo"""
    print("🚀 ML-Bench Visualization Demo")
    print("=" * 50)
    
    print("📊 Generating sample benchmark results...")
    results = create_sample_results()
    results_file = save_demo_results(results)
    
    print(f"✅ Created demo results: {results_file}")
    print(f"📈 Generated {len(results)} benchmark configurations")
    
    # Show what we created
    frameworks = set(r['framework'] for r in results)
    models = set(r['model'] for r in results)
    usecases = set(r['usecase'] for r in results)
    
    print(f"\n📋 Demo Data Summary:")
    print(f"  Frameworks: {', '.join(frameworks)}")
    print(f"  Models: {', '.join(sorted(models))}")
    print(f"  Use Cases: {', '.join(usecases)}")
    
    print(f"\n🎯 Demo Options:")
    print(f"1. CLI Summary - Quick terminal analysis")
    print(f"2. Interactive Dashboard - Full web interface")
    print(f"3. Static Report - HTML report generation")
    print(f"4. All modes - Run all visualization modes")
    
    while True:
        choice = input("\nSelect demo mode (1-4) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("👋 Demo completed!")
            break
        elif choice == '1':
            print("\n📈 Generating CLI visualization...")
            subprocess.run([sys.executable, "visualize.py", "--mode", "cli", "--results-file", results_file])
        elif choice == '2':
            print("\n🌐 Launching interactive dashboard...")
            print("💡 The dashboard will open in your browser at http://localhost:8501")
            print("💡 Press Ctrl+C in the terminal to stop the server")
            subprocess.run([sys.executable, "visualize.py", "--mode", "dashboard", "--results-file", results_file])
        elif choice == '3':
            print("\n📊 Creating static HTML report...")
            subprocess.run([sys.executable, "visualize.py", "--mode", "static", "--results-file", results_file])
            print("✅ Report created in visualization_output/index.html")
        elif choice == '4':
            print("\n🚀 Running all visualization modes...")
            
            # CLI first
            print("\n1️⃣ CLI Summary:")
            subprocess.run([sys.executable, "visualize.py", "--mode", "cli", "--results-file", results_file])
            
            # Static report
            print("\n2️⃣ Creating static report...")
            subprocess.run([sys.executable, "visualize.py", "--mode", "static", "--results-file", results_file])
            print("✅ Static report: visualization_output/index.html")
            
            # Dashboard last (will block until user stops it)
            print("\n3️⃣ Launching dashboard...")
            print("💡 Dashboard: http://localhost:8501")
            print("💡 Press Ctrl+C to stop and return to menu")
            try:
                subprocess.run([sys.executable, "visualize.py", "--mode", "dashboard", "--results-file", results_file])
            except KeyboardInterrupt:
                print("\n📊 Dashboard stopped")
            
        else:
            print("❌ Invalid choice. Please select 1-4 or 'q'")

def check_requirements():
    """Check if visualization requirements are installed"""
    required = ['streamlit', 'plotly', 'matplotlib', 'seaborn', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("📦 Install with: pip install " + " ".join(missing))
        return False
    
    return True

if __name__ == "__main__":
    if not check_requirements():
        sys.exit(1)
    
    run_demo() 