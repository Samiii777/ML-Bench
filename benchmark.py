#!/usr/bin/env python3
"""
Benchmarking Framework for ML Models
Supports PyTorch inference for image classification
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, List, Any
import torch

# Clean import of utils - no ugly relative paths!
import utils
from utils.logger import BenchmarkLogger
from utils.config import get_model_family, get_available_models, DEFAULT_PRECISIONS, DEFAULT_BATCH_SIZES, get_onnx_execution_providers, get_default_frameworks, get_default_use_case_for_model, get_available_frameworks_for_model, get_unique_models, get_models_for_use_case, get_available_frameworks_for_use_case, get_vram_requirement, should_skip_for_vram, get_default_use_cases
from utils.results import BenchmarkResults
from utils.shared_device_utils import get_gpu_memory_efficient

class BenchmarkRunner:
    def __init__(self):
        self.logger = BenchmarkLogger()
        self.results = BenchmarkResults()
        
    def get_available_models(self, framework: str, model_prefix: str = None) -> List[str]:
        """Get list of available models for a framework"""
        if framework in ["pytorch", "onnx"]:
            available_models = get_available_models(framework)
            if model_prefix == "resnet":
                return [m for m in available_models if m.startswith("resnet")]
            elif model_prefix:
                return [model_prefix]
            else:
                return available_models
        return []
    
    def get_benchmark_script_path(self, framework: str, model: str, mode: str, use_case: str) -> str:
        """Get the path to the benchmark script"""
        # Get the actual directory name (model family)
        directory_name = get_model_family(model)
        
        # Handle GPU operations with subdirectories
        if directory_name == "gpu_ops":
            # Map model names to their subdirectories
            gpu_ops_mapping = {
                "gemm_ops": "gemm",
                "conv_ops": "conv", 
                "memory_ops": "memory",
                "elementwise_ops": "elementwise",
                "reduction_ops": "reduction"
            }
            
            if model in gpu_ops_mapping:
                subdirectory = gpu_ops_mapping[model]
                base_path = Path("benchmarks") / framework / directory_name / mode / use_case / subdirectory
            else:
                # Fallback for unknown GPU ops
                base_path = Path("benchmarks") / framework / directory_name / mode / use_case
        else:
            # Standard path for other models
            base_path = Path("benchmarks") / framework / directory_name / mode / use_case
        
        script_path = base_path / "main.py"
        return str(script_path)
    
    def run_single_benchmark(self, framework: str, model: str, mode: str, use_case: str, 
                           precision: str, batch_size: int, execution_provider: str = None) -> Dict[str, Any]:
        """Run a single benchmark and return results"""
        script_path = self.get_benchmark_script_path(framework, model, mode, use_case)
        
        if not os.path.exists(script_path):
            return {
                "status": "FAIL",
                "error": f"Benchmark script not found: {script_path}",
                "execution_time": 0,
                "metrics": {}
            }
        
        # Simple VRAM check for GPU benchmarks (only for large models like Stable Diffusion)
        if framework == "pytorch" and torch.cuda.is_available():
            gpu_info = get_gpu_memory_efficient()
            if gpu_info and gpu_info.get("total_gpu_free_gb"):
                available_vram = gpu_info["total_gpu_free_gb"]
                should_skip, reason = should_skip_for_vram(model, precision, batch_size, available_vram)
                
                if should_skip:
                    vram_req = get_vram_requirement(model, precision, batch_size)
                    return {
                        "status": "SKIP",
                        "error": f"VRAM insufficient: {reason}",
                        "execution_time": 0,
                        "metrics": {
                            "vram_requirement": vram_req,
                            "skip_reason": "vram_insufficient"
                        }
                    }
                else:
                    self.logger.debug(f"VRAM check passed for {model} {precision} BS={batch_size}: {reason}")
            else:
                self.logger.debug(f"Could not check VRAM - running {model} {precision} BS={batch_size} without pre-checks")
        
        # Prepare command
        cmd = [
            sys.executable, "main.py",
            "--model", model,
            "--precision", precision,
            "--batch_size", str(batch_size)
        ]
        
        # Add execution provider for ONNX
        if framework == "onnx" and execution_provider:
            cmd.extend(["--execution_provider", execution_provider])
        
        start_time = time.time()
        try:
            # Use longer timeout for TensorRT due to compilation overhead
            timeout_seconds = 900 if execution_provider == "TensorrtExecutionProvider" else 300  # 15 min vs 5 min
            
            # Set environment variables to include the root directory in Python path
            env = os.environ.copy()
            root_dir = os.path.abspath(os.path.dirname(__file__))  # The directory containing benchmark.py
            current_pythonpath = env.get('PYTHONPATH', '')
            if current_pythonpath:
                env['PYTHONPATH'] = f"{root_dir}{os.pathsep}{current_pythonpath}"
            else:
                env['PYTHONPATH'] = root_dir
            
            # Run the benchmark script
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout_seconds,
                cwd=os.path.dirname(script_path),
                env=env
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse output for metrics (inference time, accuracy, etc.)
                metrics = self._parse_benchmark_output(result.stdout)
                # Add execution provider to metrics if available
                if execution_provider:
                    metrics['execution_provider'] = execution_provider
                
                return {
                    "status": "PASS",
                    "execution_time": execution_time,
                    "metrics": metrics,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                # Check if this was likely an OOM error
                error_output = result.stderr.lower()
                if "out of memory" in error_output or "cuda out of memory" in error_output:
                    # Provide helpful OOM guidance
                    error_msg = f"CUDA Out of Memory. Try: reduce batch size, use fp16 precision, or enable CPU offload"
                else:
                    error_msg = f"Script failed with return code {result.returncode}"
                
                return {
                    "status": "FAIL",
                    "error": error_msg,
                    "execution_time": execution_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "metrics": {}
                }
                
        except subprocess.TimeoutExpired:
            timeout_minutes = 15 if execution_provider == "TensorrtExecutionProvider" else 5
            return {
                "status": "FAIL",
                "error": f"Benchmark timed out after {timeout_minutes} minutes",
                "execution_time": time.time() - start_time,
                "metrics": {}
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "error": f"Unexpected error: {str(e)}",
                "execution_time": time.time() - start_time,
                "metrics": {}
            }
    
    def _parse_benchmark_output(self, output: str) -> Dict[str, Any]:
        """Parse benchmark output to extract metrics"""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            # Look for inference time
            if "Inference Time" in line and "ms" in line and "=" in line:
                try:
                    # Extract number from line like "PyTorch Inference Time = 123.45 ms"
                    time_str = line.split('=')[1].strip().replace('ms', '').strip()
                    metrics['inference_time_ms'] = float(time_str)
                except:
                    pass
            
            # Look for per-sample latency
            if "Per-sample Latency:" in line:
                try:
                    # Extract number from line like "Per-sample Latency: 1.23 ms/sample"
                    time_str = line.split(':')[1].strip().replace('ms/sample', '').strip()
                    metrics['avg_latency_ms'] = float(time_str)
                except:
                    pass
            
            # Look for throughput
            if "Throughput:" in line and "samples/sec" in line:
                try:
                    # Extract number from line like "Throughput: 123.45 samples/sec"
                    throughput_str = line.split(':')[1].strip().replace('samples/sec', '').strip()
                    metrics['throughput_fps'] = float(throughput_str)
                except:
                    pass
            
            # Look for GFLOPS performance (for GEMM and convolution operations)
            if "Best GEMM Performance:" in line and "GFLOPS" in line:
                try:
                    # Extract number from line like "Best GEMM Performance: 2847.32 GFLOPS (4096x4096x4096)"
                    gflops_str = line.split(':')[1].strip().split('GFLOPS')[0].strip()
                    metrics['best_gflops'] = float(gflops_str)
                    metrics['performance_metric'] = 'GFLOPS'
                except:
                    pass
            
            if "Best Conv Performance:" in line and "GFLOPS" in line:
                try:
                    # Extract number from line like "Best Conv Performance: 1847.21 GFLOPS (ResNet Mid Layer)"
                    gflops_str = line.split(':')[1].strip().split('GFLOPS')[0].strip()
                    metrics['best_gflops'] = float(gflops_str)
                    metrics['performance_metric'] = 'GFLOPS'
                except:
                    pass
            
            # Look for bandwidth performance (for memory, elementwise, reduction operations)
            if "Best Memory Bandwidth:" in line and "GB/s" in line:
                try:
                    # Extract number from line like "Best Memory Bandwidth: 847.3 GB/s (Copy Large)"
                    bandwidth_str = line.split(':')[1].strip().split('GB/s')[0].strip()
                    metrics['best_bandwidth_gbs'] = float(bandwidth_str)
                    metrics['performance_metric'] = 'GB/s'
                except:
                    pass
            
            if "Best Element-wise Bandwidth:" in line and "GB/s" in line:
                try:
                    # Extract number from line like "Best Element-wise Bandwidth: 1247.8 GB/s (add Large)"
                    bandwidth_str = line.split(':')[1].strip().split('GB/s')[0].strip()
                    metrics['best_bandwidth_gbs'] = float(bandwidth_str)
                    metrics['performance_metric'] = 'GB/s'
                except:
                    pass
            
            if "Best Reduction Bandwidth:" in line and "GB/s" in line:
                try:
                    # Extract number from line like "Best Reduction Bandwidth: 892.4 GB/s (sum_dim1 Large)"
                    bandwidth_str = line.split(':')[1].strip().split('GB/s')[0].strip()
                    metrics['best_bandwidth_gbs'] = float(bandwidth_str)
                    metrics['performance_metric'] = 'GB/s'
                except:
                    pass
            
            if "Best Convolution Performance:" in line and "GFLOPS" in line:
                try:
                    # Extract number from line like "Best Convolution Performance: 1847.21 GFLOPS (ResNet Mid Layer)"
                    gflops_str = line.split(':')[1].strip().split('GFLOPS')[0].strip()
                    metrics['best_gflops'] = float(gflops_str)
                    metrics['performance_metric'] = 'GFLOPS'
                except:
                    pass
            
            # Look for device information
            if "Device:" in line:
                try:
                    device_str = line.split(':')[1].strip()
                    metrics['device'] = device_str
                except:
                    pass
            
            # Look for framework information
            if "Framework:" in line:
                try:
                    framework_str = line.split(':')[1].strip()
                    metrics['framework'] = framework_str
                except:
                    pass
            
            # Look for memory information
            if "GPU Memory Allocated:" in line:
                try:
                    memory_str = line.split(':')[1].strip().replace('GB', '').strip()
                    metrics['gpu_memory_allocated_gb'] = float(memory_str)
                except:
                    pass
            
            if "GPU Memory Cached:" in line:
                try:
                    memory_str = line.split(':')[1].strip().replace('GB', '').strip()
                    metrics['gpu_memory_cached_gb'] = float(memory_str)
                except:
                    pass
            
            if "Total GPU Memory Used:" in line:
                try:
                    memory_str = line.split(':')[1].strip().replace('GB', '').strip()
                    metrics['total_gpu_memory_used_gb'] = float(memory_str)
                except:
                    pass
            
            if "System Memory RSS:" in line:
                try:
                    memory_str = line.split(':')[1].strip().replace('GB', '').strip()
                    metrics['system_memory_rss_gb'] = float(memory_str)
                except:
                    pass
            
            # Look for accuracy metrics
            if "Accuracy" in line or "Top-1" in line or "Top-5" in line:
                try:
                    # Extract accuracy values
                    if ":" in line:
                        key, value = line.split(':', 1)
                        metrics[key.strip().lower().replace(' ', '_')] = float(value.strip().replace('%', ''))
                except:
                    pass
        
        return metrics
    
    def run_comprehensive_benchmarks(self, args) -> None:
        """Run comprehensive benchmarks across all configurations"""
        print("Starting comprehensive benchmark run")
        
        # Determine frameworks to test
        if args.framework:
            if isinstance(args.framework, list):
                frameworks = args.framework
            else:
                frameworks = [args.framework]
        else:
            # Use configured default frameworks (both pytorch and onnx)
            frameworks = get_default_frameworks()
        
        print(f"Frameworks: {frameworks}")
        
        # Determine use case
        if args.usecase:
            use_case = args.usecase
        elif args.model:
            # Handle both single model and list of models
            model_name = args.model[0] if isinstance(args.model, list) else args.model
            use_case = get_default_use_case_for_model(model_name)
        else:
            use_case = "all (comprehensive)"
        
        print(f"Use case: {use_case}")
        
        # Calculate total tests
        total_tests = self._calculate_total_tests(args)
        print(f"Total tests to run: {total_tests}")
        print()
        
        # Run tests for each framework
        all_results = []
        test_counter = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for framework in frameworks:
            print(f"{'='*100}")
            print(f"TESTING FRAMEWORK: {framework.upper()}")
            print(f"{'='*100}")
            
            # Create a copy of args with the current framework set
            framework_args = type('Args', (), vars(args).copy())()
            framework_args.framework = framework
            
            framework_results = self._run_framework_comprehensive_benchmarks(framework_args, test_counter, total_tests)
            all_results.extend(framework_results["results"])
            test_counter = framework_results["test_counter"]
            total_passed += framework_results["passed"]
            total_failed += framework_results["failed"]
            total_skipped += framework_results["skipped"]
        
        # Save and display results
        self.results.save_results(all_results, args)
        
        # Print overall summary
        print(f"\n{'='*100}")
        print("OVERALL COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*100}")
        print(f"Frameworks tested: {frameworks}")
        
        total_tests_run = len(all_results)
        attempted_tests = total_passed + total_failed
        
        print(f"Total combinations: {total_tests_run}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Skipped: {total_skipped}")
        
        if attempted_tests > 0:
            success_rate = (total_passed / attempted_tests) * 100
            print(f"Success rate: {success_rate:.1f}% (of attempted tests)")
        
        if total_skipped > 0:
            skip_rate = (total_skipped / total_tests_run) * 100
            print(f"Skip rate: {skip_rate:.1f}% (intelligent memory management)")
        
        print()
        
        # Print detailed results table
        self.results.print_summary_table(all_results)
    
    def _calculate_total_tests(self, args) -> int:
        """Calculate total number of tests for a framework"""
        # Determine frameworks to test
        if args.framework:
            if isinstance(args.framework, list):
                frameworks = args.framework
            else:
                frameworks = [args.framework]
        else:
            # Use configured default frameworks (both pytorch and onnx)
            frameworks = get_default_frameworks()
        
        total = 0
        for framework in frameworks:
            # Determine which models to test - prioritize --model over --usecase
            if args.model is not None:
                if isinstance(args.model, list):
                    models = args.model
                elif args.model in ["resnet", "resnet*"]:
                    models = self.get_available_models(framework, "resnet")
                elif args.model == "gpu_ops":
                    models = ["gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"]
                else:
                    models = [args.model]
            elif args.usecase:
                # Filter models by usecase only if no specific model was requested
                models = get_models_for_use_case(args.usecase, framework)
            else:
                # Test unique models only (no aliases) to avoid duplicates
                models = get_unique_models(framework)
            
            # Determine which precisions to test
            if args.precision is None:
                precisions = DEFAULT_PRECISIONS
            elif isinstance(args.precision, list):
                precisions = args.precision
            else:
                precisions = [args.precision]
            
            # Determine which batch sizes to test
            if args.batch_size is None:
                batch_sizes = DEFAULT_BATCH_SIZES
            elif isinstance(args.batch_size, list):
                batch_sizes = args.batch_size
            else:
                batch_sizes = [args.batch_size]
            
            # For ONNX, also test different execution providers
            execution_providers = []
            if framework == "onnx":
                execution_providers = get_onnx_execution_providers()
            else:
                execution_providers = [None]
            
            # Calculate total combinations, accounting for skipped tests
            for model in models:
                # Determine which use cases to test for this model
                if args.usecase:
                    # If user specified a specific use case, only test that one
                    use_cases_to_test = [args.usecase]
                else:
                    # Test all applicable use cases for this model
                    model_family = get_model_family(model)
                    if model_family == "resnet":
                        # For ResNet, test all use cases that ResNet supports
                        use_cases_to_test = ["classification", "detection", "segmentation"]
                    elif model_family == "stable_diffusion":
                        use_cases_to_test = ["generation"]
                    elif model_family == "gpu_ops":
                        use_cases_to_test = ["compute"]
                    else:
                        # For unknown models, use the default use case
                        use_cases_to_test = [get_default_use_case_for_model(model)]
                
                for use_case in use_cases_to_test:
                    # Check if this use case is available for this framework
                    available_frameworks = get_available_frameworks_for_use_case(use_case)
                    if framework not in available_frameworks:
                        continue  # Skip this use case for this framework
                    
                    for precision in precisions:
                        for batch_size in batch_sizes:
                            for execution_provider in execution_providers:
                                # Skip FP16 on CPU
                                if precision == "fp16" and not torch.cuda.is_available():
                                    continue
                                # Skip FP16 for CPU execution provider in ONNX
                                if framework == "onnx" and precision == "fp16" and execution_provider == "CPUExecutionProvider":
                                    continue
                                total += 1
        
        return total

    def _run_framework_comprehensive_benchmarks(self, args, start_test_num: int = 0, total_tests: int = 0) -> dict:
        """Run comprehensive benchmarks for a specific framework"""
        
        # Determine which framework we're testing (take first if multiple)
        if isinstance(args.framework, list):
            framework = args.framework[0]  # This method handles one framework at a time
        else:
            framework = args.framework
        
        # Determine which models to test - prioritize --model over --usecase
        if args.model is not None:
            if isinstance(args.model, list):
                models = args.model
            elif args.model in ["resnet", "resnet*"]:
                models = self.get_available_models(framework, "resnet")
            elif args.model == "gpu_ops":
                # Expand gpu_ops to all GPU operation models
                models = ["gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"]
            else:
                models = [args.model]
        elif args.usecase:
            # Filter models by usecase only if no specific model was requested
            models = get_models_for_use_case(args.usecase, framework)
        else:
            # Test unique models only (no aliases) to avoid duplicates
            models = get_unique_models(framework)
        
        # Determine which precisions to test
        if args.precision is None:
            precisions = DEFAULT_PRECISIONS  # ["fp32", "fp16"]
        elif isinstance(args.precision, list):
            precisions = args.precision
        else:
            precisions = [args.precision]
        
        # Determine which batch sizes to test
        if args.batch_size is None:
            batch_sizes = DEFAULT_BATCH_SIZES  # [1, 4, 8, 16]
        elif isinstance(args.batch_size, list):
            batch_sizes = args.batch_size
        else:
            batch_sizes = [args.batch_size]
        
        # For ONNX, also test different execution providers
        execution_providers = []
        if framework == "onnx":
            execution_providers = get_onnx_execution_providers()
        else:
            execution_providers = [None]  # No execution provider for other frameworks
        
        passed = 0
        failed = 0
        skipped = 0
        framework_results = []
        
        current_test = start_test_num
        
        for model in models:
            # Determine which use cases to test for this model
            if args.usecase:
                # If user specified a specific use case, only test that one
                use_cases_to_test = [args.usecase]
            else:
                # Test all applicable use cases for this model
                model_family = get_model_family(model)
                if model_family == "resnet":
                    # For ResNet, test all use cases that ResNet supports
                    use_cases_to_test = ["classification", "detection", "segmentation"]
                elif model_family == "stable_diffusion":
                    use_cases_to_test = ["generation"]
                elif model_family == "gpu_ops":
                    use_cases_to_test = ["compute"]
                else:
                    # For unknown models, use the default use case
                    use_cases_to_test = [get_default_use_case_for_model(model)]
            
            for use_case in use_cases_to_test:
                # Check if this use case is available for this framework
                available_frameworks = get_available_frameworks_for_use_case(use_case)
                if framework not in available_frameworks:
                    continue  # Skip this use case for this framework
                
                for precision in precisions:
                    for batch_size in batch_sizes:
                        for execution_provider in execution_providers:
                            # Skip FP16 on CPU
                            if precision == "fp16" and not torch.cuda.is_available():
                                continue
                            
                            # Skip FP16 for CPU execution provider in ONNX
                            if framework == "onnx" and precision == "fp16" and execution_provider == "CPUExecutionProvider":
                                continue
                            
                            current_test += 1
                            
                            # Create test description
                            provider_info = f" ({execution_provider})" if execution_provider else ""
                            test_desc = f"{framework}/{model}[{use_case}] {precision} BS={batch_size}{provider_info}"
                            
                            # Show single line test status
                            print(f"[{current_test:3d}/{total_tests}] {test_desc:<60} ", end="", flush=True)
                            
                            result = self.run_single_benchmark(
                                framework, model, args.mode, use_case,
                                precision, batch_size, execution_provider
                            )
                            
                            # Add metadata to result
                            result.update({
                                "framework": framework,
                                "model": model,
                                "mode": args.mode,
                                "usecase": use_case,
                                "precision": precision,
                                "batch_size": batch_size,
                                "execution_provider": execution_provider,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            framework_results.append(result)
                            
                            if result["status"] == "PASS":
                                passed += 1
                                # Show performance metric on the same line
                                metrics = result["metrics"]
                                if use_case == "compute":
                                    if metrics.get("best_gflops"):
                                        print(f"✓ {metrics['best_gflops']:.1f} GFLOPS")
                                    elif metrics.get("best_bandwidth_gbs"):
                                        print(f"✓ {metrics['best_bandwidth_gbs']:.1f} GB/s")
                                    else:
                                        throughput = metrics.get("throughput_fps", 0)
                                        print(f"✓ {throughput:.2f} samples/sec")
                                else:
                                    throughput = metrics.get("throughput_fps", 0)
                                    print(f"✓ {throughput:.2f} samples/sec")
                            elif result["status"] == "SKIP":
                                skipped += 1
                                skip_reason = result["metrics"].get("skip_reason", "unknown")
                                required_memory = result["metrics"].get("required_memory_gb", 0)
                                if skip_reason == "insufficient_vram":
                                    print(f"⚠ SKIPPED - Insufficient VRAM ({required_memory:.1f}GB required)")
                                else:
                                    print(f"⚠ SKIPPED - {result.get('error', 'Unknown reason')}")
                            else:
                                failed += 1
                                print(f"✗ FAILED - {result.get('error', 'Unknown error')}")
        
        # Print framework summary with memory management stats
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"{framework.upper()} FRAMEWORK SUMMARY")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Combinations tested: {len(framework_results)}")
        self.logger.success(f"Passed: {passed}")
        self.logger.warning(f"Skipped: {skipped}")
        self.logger.error(f"Failed: {failed}")
        
        # Calculate success rate based on attempted tests (excluding skipped)
        attempted = passed + failed
        if attempted > 0:
            success_rate = (passed / attempted) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}% (of attempted tests)")
        
        return {
            'results': framework_results,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'test_counter': current_test
        }
    
    def _print_performance_analysis(self, results: List[Dict[str, Any]]) -> None:
        """Print performance analysis across different configurations"""
        if not results:
            return
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("PERFORMANCE ANALYSIS")
        self.logger.info(f"{'='*80}")
        
        # Group results by model
        models_data = {}
        for result in results:
            if result["status"] == "PASS":
                model = result["model"]
                if model not in models_data:
                    models_data[model] = []
                models_data[model].append(result)
        
        # Analyze each model
        for model, model_results in models_data.items():
            self.logger.info(f"\n{model.upper()} Performance Analysis:")
            self.logger.info("-" * 50)
            
            # Find best throughput
            best_throughput = max(model_results, key=lambda x: x["metrics"].get("throughput_fps", 0))
            best_latency = min(model_results, key=lambda x: x["metrics"].get("avg_latency_ms", float('inf')))
            
            self.logger.info(f"Best Throughput: {best_throughput['metrics'].get('throughput_fps', 0):.2f} samples/sec")
            self.logger.info(f"  Configuration: {best_throughput['precision']}, BS={best_throughput['batch_size']}")
            
            self.logger.info(f"Best Latency: {best_latency['metrics'].get('avg_latency_ms', 0):.2f} ms/sample")
            self.logger.info(f"  Configuration: {best_latency['precision']}, BS={best_latency['batch_size']}")
            
            # Precision comparison (for batch size 1)
            bs1_results = [r for r in model_results if r["batch_size"] == 1]
            if len(bs1_results) > 1:
                self.logger.info("Precision Comparison (BS=1):")
                for result in bs1_results:
                    precision = result["precision"]
                    throughput = result["metrics"].get("throughput_fps", 0)
                    self.logger.info(f"  {precision}: {throughput:.2f} samples/sec")
            
            # Batch size scaling (for fp32)
            fp32_results = [r for r in model_results if r["precision"] == "fp32"]
            if len(fp32_results) > 1:
                self.logger.info("Batch Size Scaling (FP32):")
                for result in sorted(fp32_results, key=lambda x: x["batch_size"]):
                    bs = result["batch_size"]
                    throughput = result["metrics"].get("throughput_fps", 0)
                    self.logger.info(f"  BS={bs}: {throughput:.2f} samples/sec")
    
    def _print_framework_comparison(self, results: List[Dict[str, Any]], frameworks: List[str]) -> None:
        """Print performance comparison between frameworks"""
        if len(frameworks) < 2:
            return
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("FRAMEWORK COMPARISON")
        self.logger.info(f"{'='*80}")
        
        # Group results by framework and model
        framework_data = {}
        for result in results:
            if result["status"] == "PASS":
                framework = result["framework"]
                model = result["model"]
                precision = result["precision"]
                batch_size = result["batch_size"]
                
                key = f"{model}_{precision}_bs{batch_size}"
                
                if framework not in framework_data:
                    framework_data[framework] = {}
                
                if key not in framework_data[framework]:
                    framework_data[framework][key] = []
                
                framework_data[framework][key].append(result)
        
        # Find common configurations across frameworks
        common_configs = set()
        if len(framework_data) >= 2:
            framework_keys = list(framework_data.keys())
            common_configs = set(framework_data[framework_keys[0]].keys())
            for framework in framework_keys[1:]:
                common_configs = common_configs.intersection(set(framework_data[framework].keys()))
        
        if not common_configs:
            self.logger.info("No common configurations found between frameworks")
            return
        
        self.logger.info(f"Comparing {len(common_configs)} common configurations:")
        self.logger.info(f"{'Configuration':<25}{'Framework':<12}{'Best Performance':<18}{'Best Latency':<15}{'Provider':<15}")
        self.logger.info("-" * 85)
        
        for config in sorted(common_configs):
            # Split from the right to handle model names with underscores
            parts = config.rsplit('_', 2)  # Split into at most 3 parts from the right
            if len(parts) == 3:
                model, precision, batch_info = parts
                batch_size = batch_info.replace('bs', '')
            else:
                # Fallback for unexpected format
                continue
            
            config_display = f"{model} {precision} BS={batch_size}"
            
            for framework in frameworks:
                if framework in framework_data and config in framework_data[framework]:
                    results_for_config = framework_data[framework][config]
                    
                    # Find best result for this framework/config using appropriate metric
                    # For compute operations, prioritize GFLOPS, then bandwidth, then throughput
                    def get_performance_score(result):
                        metrics = result["metrics"]
                        use_case = result.get("usecase", "")
                        
                        if use_case == "compute":
                            if metrics.get("best_gflops"):
                                return metrics["best_gflops"]
                            elif metrics.get("best_bandwidth_gbs"):
                                return metrics["best_bandwidth_gbs"]
                            else:
                                return metrics.get("throughput_fps", 0)
                        else:
                            return metrics.get("throughput_fps", 0)
                    
                    best_result = max(results_for_config, key=get_performance_score)
                    
                    # Get the appropriate performance metric and unit
                    metrics = best_result["metrics"]
                    use_case = best_result.get("usecase", "")
                    
                    if use_case == "compute":
                        if metrics.get("best_gflops"):
                            performance_value = metrics["best_gflops"]
                            performance_unit = "GFLOPS"
                        elif metrics.get("best_bandwidth_gbs"):
                            performance_value = metrics["best_bandwidth_gbs"]
                            performance_unit = "GB/s"
                        else:
                            performance_value = metrics.get("throughput_fps", 0)
                            performance_unit = "samp/s"
                    else:
                        performance_value = metrics.get("throughput_fps", 0)
                        performance_unit = "samp/s"
                    
                    latency = metrics.get("avg_latency_ms", 0)
                    provider = best_result.get("execution_provider", "N/A")
                    
                    if provider and "ExecutionProvider" in provider:
                        provider = provider.replace("ExecutionProvider", "")
                    elif provider is None:
                        provider = "Default"
                    
                    performance_display = f"{performance_value:.1f} {performance_unit}"
                    self.logger.info(f"{config_display:<25}{framework:<12}{performance_display:<18}{latency:<15.2f}{provider:<15}")
                    config_display = ""  # Only show config name once
            
            self.logger.info("")  # Empty line between configurations
        
        # Overall framework performance summary
        self.logger.info("FRAMEWORK PERFORMANCE SUMMARY:")
        self.logger.info("-" * 50)
        
        for framework in frameworks:
            if framework in framework_data:
                all_framework_results = []
                for config_results in framework_data[framework].values():
                    all_framework_results.extend(config_results)
                
                if all_framework_results:
                    # Calculate performance using appropriate metrics
                    def get_performance_score(result):
                        metrics = result["metrics"]
                        use_case = result.get("usecase", "")
                        
                        if use_case == "compute":
                            if metrics.get("best_gflops"):
                                return metrics["best_gflops"]
                            elif metrics.get("best_bandwidth_gbs"):
                                return metrics["best_bandwidth_gbs"]
                            else:
                                return metrics.get("throughput_fps", 0)
                        else:
                            return metrics.get("throughput_fps", 0)
                    
                    performance_scores = [get_performance_score(r) for r in all_framework_results]
                    avg_performance = sum(performance_scores) / len(performance_scores)
                    max_performance = max(performance_scores)
                    best_result = max(all_framework_results, key=get_performance_score)
                    
                    # Determine the unit for this framework's best result
                    best_metrics = best_result["metrics"]
                    best_use_case = best_result.get("usecase", "")
                    
                    if best_use_case == "compute":
                        if best_metrics.get("best_gflops"):
                            unit = "GFLOPS"
                        elif best_metrics.get("best_bandwidth_gbs"):
                            unit = "GB/s"
                        else:
                            unit = "samples/sec"
                    else:
                        unit = "samples/sec"
                    
                    self.logger.info(f"{framework.upper()}:")
                    self.logger.info(f"  Average Performance: {avg_performance:.2f} {unit}")
                    self.logger.info(f"  Peak Performance: {max_performance:.2f} {unit}")
                    self.logger.info(f"  Best Configuration: {best_result['model']} {best_result['precision']} BS={best_result['batch_size']}")
                    if best_result.get("execution_provider"):
                        provider = best_result["execution_provider"].replace("ExecutionProvider", "")
                        self.logger.info(f"  Best Provider: {provider}")
                    self.logger.info("")
    
    def run_benchmarks(self, args) -> None:
        """Run benchmarks based on arguments (backward compatibility)"""
        # If specific parameters are provided AND a specific use case is specified, use the original behavior
        # Otherwise, run comprehensive benchmarks to test all use cases for the model
        if args.framework and all([args.model, args.precision is not None, args.batch_size is not None, args.usecase is not None]):
            self._run_specific_benchmarks(args)
        else:
            # Run comprehensive benchmarks (possibly multi-framework, multi-use-case)
            self.run_comprehensive_benchmarks(args)
    
    def _run_specific_benchmarks(self, args) -> None:
        """Run specific benchmarks when all parameters are provided"""
        
        # Extract single values from lists if needed (due to nargs='*')
        framework = args.framework[0] if isinstance(args.framework, list) else args.framework
        model = args.model[0] if isinstance(args.model, list) else args.model
        precision = args.precision[0] if isinstance(args.precision, list) else args.precision
        batch_size = args.batch_size[0] if isinstance(args.batch_size, list) else args.batch_size
        
        self.logger.info("Starting benchmark run")
        self.logger.info(f"Framework: {framework}")
        self.logger.info(f"Model: {model}")
        self.logger.info(f"Mode: {args.mode}")
        self.logger.info(f"Use case: {args.usecase}")
        self.logger.info(f"Precision: {precision}")
        self.logger.info(f"Batch size: {batch_size}")
        
        # Determine which models to run
        if model in ["resnet", "resnet*"]:
            models = self.get_available_models(framework, "resnet")
        elif model == "gpu_ops":
            # Expand gpu_ops to all GPU operation models
            models = ["gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"]
        else:
            models = [model]
        
        # For ONNX, test all execution providers
        execution_providers = []
        if framework == "onnx":
            execution_providers = get_onnx_execution_providers()
        else:
            execution_providers = [None]  # No execution provider for other frameworks
        
        total_benchmarks = len(models) * len(execution_providers)
        passed = 0
        failed = 0
        skipped = 0
        current_test = 0
        
        all_results = []
        
        for model_name in models:
            for execution_provider in execution_providers:
                current_test += 1
                
                # Create test description
                provider_info = f" ({execution_provider})" if execution_provider else ""
                test_desc = f"{framework}/{model_name} {precision} BS={batch_size}{provider_info}"
                
                # Show single line test status
                print(f"[{current_test:3d}/{total_benchmarks}] {test_desc:<60} ", end="", flush=True)
                
                result = self.run_single_benchmark(
                    framework, model_name, args.mode, args.usecase,
                    precision, batch_size, execution_provider
                )
                
                # Add metadata to result
                result.update({
                    "framework": framework,
                    "model": model_name,
                    "mode": args.mode,
                    "usecase": args.usecase,
                    "precision": precision,
                    "batch_size": batch_size,
                    "execution_provider": execution_provider,
                    "timestamp": datetime.now().isoformat()
                })
                
                all_results.append(result)
                
                if result["status"] == "PASS":
                    passed += 1
                    # Show performance metric on the same line
                    metrics = result["metrics"]
                    if args.usecase == "compute":
                        if metrics.get("best_gflops"):
                            print(f"✓ {metrics['best_gflops']:.1f} GFLOPS")
                        elif metrics.get("best_bandwidth_gbs"):
                            print(f"✓ {metrics['best_bandwidth_gbs']:.1f} GB/s")
                        else:
                            throughput = metrics.get("throughput_fps", 0)
                            print(f"✓ {throughput:.2f} samples/sec")
                    else:
                        throughput = metrics.get("throughput_fps", 0)
                        print(f"✓ {throughput:.2f} samples/sec")
                elif result["status"] == "SKIP":
                    skipped += 1
                    skip_reason = result["metrics"].get("skip_reason", "unknown")
                    required_memory = result["metrics"].get("required_memory_gb", 0)
                    if skip_reason == "insufficient_vram":
                        print(f"⚠ SKIPPED - Insufficient VRAM ({required_memory:.1f}GB required)")
                    else:
                        print(f"⚠ SKIPPED - {result.get('error', 'Unknown reason')}")
                else:
                    failed += 1
                    print(f"✗ FAILED - {result.get('error', 'Unknown error')}")
        
        # Save results
        self.results.save_results(all_results, args)
        
        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("BENCHMARK SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total benchmarks: {total_benchmarks}")
        self.logger.success(f"Passed: {passed}")
        self.logger.warning(f"Skipped: {skipped}")
        self.logger.error(f"Failed: {failed}")
        
        # Calculate success rate based on attempted tests (excluding skipped)
        attempted = passed + failed
        if attempted > 0:
            success_rate = (passed / attempted) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}% (of attempted tests)")
        
        # Print individual results
        if framework == "onnx" and len(execution_providers) > 1:
            self.logger.info("\nIndividual Results:")
            self.logger.info("=" * 80)
            for result in all_results:
                if result["status"] == "PASS":
                    provider = result.get("execution_provider", "N/A")
                    if provider and "ExecutionProvider" in provider:
                        provider = provider.replace("ExecutionProvider", "")
                    metrics = result["metrics"]
                    # Show appropriate performance metric based on use case
                    if args.usecase == "compute":
                        # For compute use cases, show GFLOPS or GB/s
                        if metrics.get("best_gflops"):
                            self.logger.success(f"✓ {result['model']} - {framework.upper()} ({provider}) ({result['precision']}, BS={result['batch_size']}): {metrics['best_gflops']:.1f} GFLOPS")
                        elif metrics.get("best_bandwidth_gbs"):
                            self.logger.success(f"✓ {result['model']} - {framework.upper()} ({provider}) ({result['precision']}, BS={result['batch_size']}): {metrics['best_bandwidth_gbs']:.1f} GB/s")
                        else:
                            throughput = metrics.get("throughput_fps", 0)
                            latency = metrics.get("avg_latency_ms", 0)
                            self.logger.success(f"✓ {result['model']} - {framework.upper()} ({provider}) ({result['precision']}, BS={result['batch_size']}): {throughput:.2f} samples/sec, {latency:.2f} ms/sample")
                    else:
                        # For other use cases, show samples/sec and latency
                        throughput = metrics.get("throughput_fps", 0)
                        latency = metrics.get("avg_latency_ms", 0)
                        self.logger.success(f"✓ {result['model']} - {framework.upper()} ({provider}) ({result['precision']}, BS={result['batch_size']}): {throughput:.2f} samples/sec, {latency:.2f} ms/sample")
        
        # Print detailed results table
        self.results.print_summary_table(all_results)

    def check_memory_requirements(self, args) -> None:
        """Check memory requirements for planned benchmarks"""
        print("=" * 60)
        print("MEMORY REQUIREMENTS CHECK")
        print("=" * 60)
        
        # Get GPU memory info
        gpu_memory_info = get_gpu_memory_efficient()
        if not gpu_memory_info or not gpu_memory_info.get("total_gpu_total_gb"):
            print("❌ Could not detect GPU memory. Skipping memory check.")
            return
        
        available_memory = gpu_memory_info["total_gpu_free_gb"]
        total_memory = gpu_memory_info["total_gpu_total_gb"]
        used_memory = gpu_memory_info["total_gpu_used_gb"]
        
        print(f"GPU Memory Status:")
        print(f"  Total: {total_memory:.1f} GB")
        print(f"  Used: {used_memory:.1f} GB")
        print(f"  Available: {available_memory:.1f} GB")
        print()
        
        # Show VRAM requirements for Stable Diffusion models only
        print("VRAM Requirements (Large Models Only):")
        print("-" * 60)
        
        from utils.config import VRAM_REQUIREMENTS
        
        for model, requirements in VRAM_REQUIREMENTS.items():
            print(f"\n{model.upper()}:")
            for precision, vram_req in requirements.items():
                if isinstance(vram_req, str):
                    status = "❌ EXCEEDS GPU" if vram_req == ">24GB" else "✅ FITS"
                else:
                    status = "✅ FITS" if vram_req <= available_memory * 0.9 else "❌ INSUFFICIENT"
                print(f"  {precision}: {vram_req} ({status})")
        
        print(f"\n" + "=" * 60)
        print("NOTE: ResNet and GPU compute models are small and not checked")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="ML Model Benchmarking Framework")
    parser.add_argument("--framework", type=str, nargs='*',
                       choices=["pytorch", "onnx", "tensorflow"],
                       help="ML framework to benchmark (default: all available)")
    parser.add_argument("--model", type=str, nargs='*',
                       help="Model name (e.g., resnet50, resnet for all resnet models, default: all available)")
    parser.add_argument("--mode", type=str,
                       choices=["inference", "training"],
                       help="Benchmark mode (default: inference)")
    parser.add_argument("--usecase", type=str,
                       choices=["classification", "detection", "segmentation", "generation", "compute"],
                       help="Use case for the benchmark (default: comprehensive - all use cases)")
    parser.add_argument("--precision", type=str, nargs='*',
                       choices=["fp32", "fp16", "mixed", "int8"],
                       help="Precision for inference (default: all available)")
    parser.add_argument("--batch_size", type=int, nargs='*',
                       help="Batch size for inference (default: test multiple sizes)")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive benchmarks across all configurations")
    parser.add_argument("--execution_provider", type=str,
                       choices=get_onnx_execution_providers(),
                       help="ONNX execution provider (default: auto-detect best)")
    parser.add_argument("--check-memory", action="store_true",
                       help="Check memory requirements for planned benchmarks without running them")
    
    args = parser.parse_args()
    
    # Set defaults if not specified (but don't set framework default to allow multi-framework testing)
    if args.mode is None:
        args.mode = "inference"
    # Don't automatically set usecase - let it be None to enable comprehensive benchmarks with all use cases
    # Only use default use case for backward compatibility when all specific parameters are provided
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create runner
    runner = BenchmarkRunner()
    
    # Check memory requirements if requested
    if getattr(args, 'check_memory', False):
        runner.check_memory_requirements(args)
        return
    
    # Run benchmarks
    runner.run_benchmarks(args)

if __name__ == "__main__":
    main()
