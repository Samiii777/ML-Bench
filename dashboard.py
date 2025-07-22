#!/usr/bin/env python3
"""
ML-Bench Dashboard Launcher
A standalone launcher for the visualization dashboard with better Windows support
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    print("🚀 ML-Bench Visualization Dashboard")
    print("=" * 50)
    
    # Check if utils/visualizer.py exists
    if not Path("utils/visualizer.py").exists():
        print("❌ Visualizer not found. Make sure you're in the ML-Bench directory.")
        sys.exit(1)
    
    # Check for results
    results_dir = Path("benchmark_results")
    if not results_dir.exists() or not list(results_dir.glob("*.json")):
        print("⚠️  No benchmark results found!")
        print("💡 Run benchmarks first: python benchmark.py")
        print("💡 Or try the demo: python demo_visualization.py")
        
        choice = input("\nContinue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print("👋 Exiting. Run benchmarks first!")
            sys.exit(0)
    
    port = 8501
    print(f"📊 Starting dashboard on port {port}...")
    print(f"🌐 Dashboard URL: http://localhost:{port}")
    print()
    print("💡 TO STOP THE DASHBOARD:")
    print("   1. Close this terminal window (recommended)")
    print("   2. Or use Task Manager to kill 'python.exe'")
    print("   3. On some systems: Press Ctrl+C multiple times")
    print()
    
    try:
        # Use streamlit directly
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "utils/visualizer.py",
            "--server.port", str(port),
            "--server.headless", "false",
            "--server.runOnSave", "false",
            "--browser.gatherUsageStats", "false",
            "--theme.backgroundColor", "#FFFFFF",
            "--theme.secondaryBackgroundColor", "#F0F2F6"
        ]
        
        print("⏳ Launching Streamlit...")
        print("✅ Dashboard starting... (this may take a few seconds)")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor the output
        startup_complete = False
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[Streamlit] {line.strip()}")
                
                if "Network URL:" in line or "Local URL:" in line:
                    if not startup_complete:
                        print("\n" + "=" * 50)
                        print("🎉 DASHBOARD IS READY!")
                        print("🌐 Open your browser and go to the URL above")
                        print("💡 To stop: Close this terminal window")
                        print("=" * 50)
                        startup_complete = True
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n⚠️  Received interrupt signal...")
        try:
            process.terminate()
            print("🔄 Attempting to stop dashboard...")
            process.wait(timeout=5)
            print("✅ Dashboard stopped successfully")
        except subprocess.TimeoutExpired:
            print("🔥 Force terminating dashboard...")
            process.kill()
        except Exception as e:
            print(f"⚠️  Error stopping dashboard: {e}")
    
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Make sure you're in the ML-Bench directory")
        print("  2. Check that the virtual environment is activated")
        print("  3. Try: pip install streamlit plotly")
    
    print("\n👋 Dashboard launcher exiting...")

if __name__ == "__main__":
    main() 