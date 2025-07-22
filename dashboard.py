#!/usr/bin/env python3
"""
ML-Bench Dashboard Launcher
A standalone launcher for the visualization dashboard with better Windows support
"""

import subprocess
import sys
import os
import time
import socket
from pathlib import Path

def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

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
    
    # Find an available port
    port = find_available_port()
    if port is None:
        print("❌ No available ports found in range 8501-8510")
        print("💡 Try stopping other Streamlit instances or use a different port range")
        sys.exit(1)
    
    if port != 8501:
        print(f"⚠️  Port 8501 in use, using port {port} instead")
    
    print(f"📊 Starting dashboard on port {port}...")
    print(f"🏠 Local access: http://localhost:{port}")
    print(f"🌐 Network access: http://<your-ip>:{port}")
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
            "--server.address", "0.0.0.0",
            "--server.headless", "false",
            "--server.runOnSave", "false",
            "--browser.gatherUsageStats", "false",
            "--global.showWarningOnDirectExecution", "false",
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
        
        # Monitor the output and capture URLs
        startup_complete = False
        local_url = None
        network_url = None
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[Streamlit] {line.strip()}")
                
                # Capture URLs
                if "Local URL:" in line:
                    local_url = line.split("Local URL:")[-1].strip()
                elif "Network URL:" in line:
                    network_url = line.split("Network URL:")[-1].strip()
                
                # Show summary when we have URLs
                if (local_url or network_url) and not startup_complete:
                    print("\n" + "=" * 60)
                    print("🎉 DASHBOARD IS READY!")
                    if local_url:
                        print(f"🏠 Local URL:   {local_url}")
                    if network_url:
                        print(f"🌐 Network URL: {network_url}")
                    print("💡 Use Network URL to access from other devices")
                    print("💡 To stop: Close this terminal window")
                    print("=" * 60)
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