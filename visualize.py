#!/usr/bin/env python3
"""
ML-Bench Visualization Tool
Launch interactive dashboard or create CLI visualizations of benchmark results
"""

import argparse
import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

def check_dependencies():
    """Check if required visualization dependencies are installed"""
    required_packages = [
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn')
    ]
    
    missing_packages = []
    
    for package, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("\n📦 Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def launch_dashboard(port: int = 8501, results_dir: str = "benchmark_results"):
    """Launch the Streamlit dashboard"""
    import subprocess
    import signal
    import threading
    import time
    
    print("🚀 Launching ML-Bench Visualization Dashboard...")
    print(f"📊 Dashboard will be available at: http://localhost:{port}")
    print("💡 Dashboard will open in your browser automatically")
    print("💡 To stop the server:")
    print("   - Close this terminal window, OR")
    print("   - Press Ctrl+C multiple times, OR") 
    print("   - Use Task Manager to end the process")
    
    try:
        # Launch streamlit with proper signal handling
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "utils/visualizer.py",
            "--server.port", str(port),
            "--server.headless", "false",  # Allow browser opening
            "--server.runOnSave", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Use Popen for better control
        process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        
        def signal_handler(signum, frame):
            print(f"\n⚠️  Received signal {signum}. Terminating dashboard...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("🔥 Force killing dashboard process...")
                process.kill()
            sys.exit(0)
        
        # Register signal handlers (though they may not work well on Windows)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("⏳ Starting dashboard server...")
        time.sleep(3)  # Give it time to start
        print("✅ Dashboard should be running!")
        print(f"🌐 Open: http://localhost:{port}")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        try:
            process.terminate()
        except:
            pass
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
    
    print("\n📊 Dashboard session ended.")

def create_cli_visualization(results_file: str = None):
    """Create CLI visualization"""
    from utils.visualizer import create_cli_charts
    
    print("📈 Creating CLI visualization...")
    create_cli_charts(results_file)

def create_static_report(results_file: str = None, output_dir: str = "visualization_output"):
    """Create static HTML report"""
    from utils.visualizer import BenchmarkVisualizer
    import plotly.io as pio
    import pandas as pd
    
    visualizer = BenchmarkVisualizer()
    
    try:
        df = visualizer.load_benchmark_results(results_file)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    if df.empty:
        print("❌ No successful benchmark results found")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"📊 Creating static HTML report in {output_path}...")
    
    # Generate all charts
    charts = {
        'performance_comparison': visualizer.create_performance_comparison(df),
        'memory_analysis': visualizer.create_memory_analysis(df),
        'batch_size_scaling': visualizer.create_batch_size_scaling(df),
        'precision_impact': visualizer.create_precision_impact(df),
        'framework_heatmap': visualizer.create_framework_heatmap(df),
        'model_radar': visualizer.create_model_radar_chart(df)
    }
    
    # Save individual charts
    for name, fig in charts.items():
        if fig.data:
            output_file = output_path / f"{name}.html"
            pio.write_html(fig, str(output_file))
            print(f"  ✅ Saved {name}.html")
    
    # Create combined report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML-Bench Visualization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 40px; }}
            .chart-container {{ margin: 20px 0; }}
            .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>🚀 ML-Bench Visualization Report</h1>
        
        <div class="summary">
            <h2>📊 Summary Statistics</h2>
            <p><strong>Total Configurations:</strong> {len(df)}</p>
            <p><strong>Frameworks:</strong> {', '.join(df['framework'].unique())}</p>
            <p><strong>Models:</strong> {', '.join(df['model'].unique())}</p>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="chart-container">
            <h2>📈 Performance Comparison</h2>
            <iframe src="performance_comparison.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>💾 Memory Analysis</h2>
            <iframe src="memory_analysis.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>📊 Batch Size Scaling</h2>
            <iframe src="batch_size_scaling.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>⚡ Precision Impact</h2>
            <iframe src="precision_impact.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>🔥 Framework Heatmap</h2>
            <iframe src="framework_heatmap.html" width="100%" height="600"></iframe>
        </div>
        
        <div class="chart-container">
            <h2>🎯 Model Radar Chart</h2>
            <iframe src="model_radar.html" width="100%" height="600"></iframe>
        </div>
        
    </body>
    </html>
    """
    
    # Save main report
    with open(output_path / "index.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Static report created: {output_path / 'index.html'}")
    print(f"🌐 Open in browser: file://{output_path.absolute() / 'index.html'}")

def main():
    parser = argparse.ArgumentParser(description="ML-Bench Visualization Tool")
    parser.add_argument("--mode", choices=["dashboard", "cli", "static"], default="dashboard",
                       help="Visualization mode (default: dashboard)")
    parser.add_argument("--port", type=int, default=8501,
                       help="Port for dashboard (default: 8501)")
    parser.add_argument("--results-file", type=str,
                       help="Specific results file to visualize (default: latest)")
    parser.add_argument("--output-dir", type=str, default="visualization_output",
                       help="Output directory for static reports")
    parser.add_argument("--results-dir", type=str, default="benchmark_results",
                       help="Directory containing benchmark results")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if results exist
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("💡 Run benchmarks first: python benchmark.py")
        sys.exit(1)
    
    json_files = list(results_dir.glob('*.json'))
    if not json_files:
        print(f"❌ No benchmark result files found in {results_dir}")
        print("💡 Run benchmarks first: python benchmark.py")
        sys.exit(1)
    
    print(f"📁 Found {len(json_files)} result files in {results_dir}")
    
    # Launch appropriate mode
    if args.mode == "dashboard":
        launch_dashboard(args.port, args.results_dir)
    elif args.mode == "cli":
        create_cli_visualization(args.results_file)
    elif args.mode == "static":
        create_static_report(args.results_file, args.output_dir)

if __name__ == "__main__":
    main() 