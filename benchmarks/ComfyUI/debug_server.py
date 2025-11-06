#!/usr/bin/env python3
"""
ComfyUI Server Debug Script
Tests server startup and shutdown to diagnose issues
"""

import sys
import os
import time
import subprocess
import signal
import requests
from pathlib import Path

class ComfyUIServerDebug:
    """Debug version of ComfyUI server manager with verbose logging"""
    
    def __init__(self, comfyui_dir, port=8188):
        self.comfyui_dir = Path(comfyui_dir)
        self.port = port
        self.server_address = f"127.0.0.1:{port}"
        self.process = None
        
    def start(self, timeout=60):
        """Start ComfyUI server with verbose logging"""
        print("=" * 60)
        print("STARTING COMFYUI SERVER")
        print("=" * 60)
        print(f"Directory: {self.comfyui_dir}")
        print(f"Port: {self.port}")
        print(f"Address: {self.server_address}")
        
        # Check if directory exists
        if not self.comfyui_dir.exists():
            print(f"❌ ERROR: Directory does not exist: {self.comfyui_dir}")
            return False
        
        main_py = self.comfyui_dir / "main.py"
        if not main_py.exists():
            print(f"❌ ERROR: main.py not found at: {main_py}")
            return False
        
        print(f"✓ Found main.py at: {main_py}")
        
        # Kill any existing process on this port first
        print(f"\nChecking for existing processes on port {self.port}...")
        try:
            result = subprocess.run(['fuser', '-k', f'{self.port}/tcp'], 
                                   capture_output=True, timeout=2)
            if result.returncode == 0:
                print(f"✓ Killed existing process on port {self.port}")
                time.sleep(2)
            else:
                print(f"✓ No existing process on port {self.port}")
        except FileNotFoundError:
            print("ℹ fuser command not available, skipping port check")
        except Exception as e:
            print(f"⚠ Warning during port check: {e}")
        
        # Get python from current environment
        python_exe = sys.executable
        print(f"\nPython executable: {python_exe}")
        
        # Start server process
        server_cmd = [
            python_exe,
            str(main_py),
            "--listen", "127.0.0.1",
            "--port", str(self.port)
        ]
        
        print(f"\nStarting server with command:")
        print(f"  {' '.join(server_cmd)}")
        print(f"  Working directory: {self.comfyui_dir}")
        print()
        
        try:
            self.process = subprocess.Popen(
                server_cmd,
                cwd=str(self.comfyui_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
                text=True,
                bufsize=1
            )
            print(f"✓ Process started with PID: {self.process.pid}")
        except Exception as e:
            print(f"❌ ERROR: Failed to start process: {e}")
            return False
        
        # Wait for server to be ready
        print(f"\nWaiting for server to be ready (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            
            # Check if process died
            poll_result = self.process.poll()
            if poll_result is not None:
                print(f"\n❌ ERROR: Server process died with exit code {poll_result}")
                print("\nServer output:")
                print("-" * 60)
                try:
                    output, _ = self.process.communicate(timeout=1)
                    print(output)
                except:
                    pass
                print("-" * 60)
                return False
            
            # Try to connect
            try:
                response = requests.get(f"http://{self.server_address}/queue", timeout=1)
                if response.status_code == 200:
                    print(f"✓ Server ready after {elapsed:.1f}s")
                    print(f"✓ Server responding at http://{self.server_address}")
                    return True
            except requests.exceptions.ConnectionError:
                # Still starting up
                pass
            except Exception as e:
                print(f"⚠ Connection attempt error: {e}")
            
            # Show progress
            if int(elapsed) % 5 == 0 and elapsed > 0:
                print(f"  Still waiting... ({elapsed:.0f}s elapsed)")
            
            time.sleep(0.5)
        
        print(f"\n❌ ERROR: Server did not become ready within {timeout}s")
        print("\nAttempting to read server output...")
        try:
            # Try to read some output
            if self.process.stdout:
                import select
                if hasattr(select, 'select'):
                    ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                    if ready:
                        output = self.process.stdout.read(4096)
                        print("-" * 60)
                        print(output)
                        print("-" * 60)
        except:
            pass
        
        return False
    
    def stop(self):
        """Stop ComfyUI server with verbose logging"""
        print("\n" + "=" * 60)
        print("STOPPING COMFYUI SERVER")
        print("=" * 60)
        
        if not self.process:
            print("ℹ No process to stop")
            return True
        
        print(f"Process PID: {self.process.pid}")
        
        try:
            # Check if process is still running
            if self.process.poll() is not None:
                print(f"ℹ Process already terminated with code {self.process.poll()}")
                self.process = None
                return True
            
            print("Sending SIGTERM to process group...")
            
            # Try to kill process group if possible (cleaner)
            if hasattr(os, 'killpg'):
                try:
                    pgid = os.getpgid(self.process.pid)
                    print(f"Process group ID: {pgid}")
                    os.killpg(pgid, signal.SIGTERM)
                    print("✓ Sent SIGTERM to process group")
                except ProcessLookupError:
                    print("⚠ Process group not found, trying direct termination")
                    self.process.terminate()
                except Exception as e:
                    print(f"⚠ Error killing process group: {e}")
                    print("Trying direct termination...")
                    self.process.terminate()
            else:
                print("killpg not available, using terminate()")
                self.process.terminate()
            
            # Wait for process to terminate
            print("Waiting for process to terminate (5s timeout)...")
            try:
                self.process.wait(timeout=5)
                exit_code = self.process.returncode
                print(f"✓ Process terminated with exit code: {exit_code}")
            except subprocess.TimeoutExpired:
                print("⚠ Process did not terminate, sending SIGKILL...")
                self.process.kill()
                self.process.wait(timeout=2)
                print("✓ Process killed forcefully")
            
            self.process = None
            
            # Wait for port to be released
            print("Waiting for port to be released...")
            time.sleep(2)
            print("✓ Server stopped successfully")
            return True
            
        except Exception as e:
            print(f"❌ ERROR during shutdown: {e}")
            import traceback
            traceback.print_exc()
            
            # Last resort - kill forcefully
            if self.process:
                try:
                    print("\nAttempting forceful kill...")
                    self.process.kill()
                    self.process.wait(timeout=2)
                    print("✓ Forcefully killed process")
                except:
                    print("❌ Failed to kill process")
            
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug ComfyUI server startup/shutdown')
    parser.add_argument('--dir', type=str, default='ComfyUI',
                       help='ComfyUI directory (default: ComfyUI)')
    parser.add_argument('--port', type=int, default=8188,
                       help='Port to run server on (default: 8188)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Startup timeout in seconds (default: 60)')
    parser.add_argument('--keep-running', action='store_true',
                       help='Keep server running (don\'t stop it)')
    
    args = parser.parse_args()
    
    # Resolve ComfyUI directory
    benchmark_dir = Path(__file__).parent
    comfyui_path = benchmark_dir / args.dir
    
    print("ComfyUI Server Debug Script")
    print("=" * 60)
    print(f"ComfyUI path: {comfyui_path}")
    print(f"Port: {args.port}")
    print(f"Timeout: {args.timeout}s")
    print()
    
    # Check if ComfyUI exists
    if not comfyui_path.exists():
        print(f"❌ ERROR: ComfyUI not found at {comfyui_path}")
        print("\nPlease run the setup first:")
        print(f"  python utils/setup_comfyui.py --dir {comfyui_path}")
        sys.exit(1)
    
    server = ComfyUIServerDebug(comfyui_path, port=args.port)
    
    try:
        # Start server
        success = server.start(timeout=args.timeout)
        
        if not success:
            print("\n" + "=" * 60)
            print("SERVER STARTUP FAILED")
            print("=" * 60)
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("SERVER STARTUP SUCCESSFUL")
        print("=" * 60)
        
        if args.keep_running:
            print("\nServer is running. Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nReceived interrupt signal...")
        else:
            print("\nServer started successfully!")
            print("Waiting 3 seconds before stopping...")
            time.sleep(3)
        
        # Stop server
        success = server.stop()
        
        if not success:
            print("\n" + "=" * 60)
            print("SERVER SHUTDOWN HAD ISSUES")
            print("=" * 60)
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("SERVER SHUTDOWN SUCCESSFUL")
        print("=" * 60)
        print("\n✓ All tests passed!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user, stopping server...")
        server.stop()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        server.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()

