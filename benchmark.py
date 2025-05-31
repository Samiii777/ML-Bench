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

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.logger import BenchmarkLogger
from utils.config import get_model_family, get_available_models, DEFAULT_PRECISIONS, DEFAULT_BATCH_SIZES, get_onnx_execution_providers, get_default_frameworks, get_default_use_case_for_model, get_available_frameworks_for_model, get_unique_models, get_models_for_use_case, get_available_frameworks_for_use_case
from utils.results import BenchmarkResults

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
            
            # Run the benchmark script
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout_seconds,
                cwd=os.path.dirname(script_path)
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
                return {
                    "status": "FAIL",
                    "error": f"Script failed with return code {result.returncode}",
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
        """Run comprehensive benchmarks testing all combinations when parameters not specified"""
        
        # Determine which frameworks to test
        if hasattr(args, 'framework') and args.framework:
            frameworks = [args.framework]
        elif args.model:
            # If a specific model is provided, use model-aware framework selection
            frameworks = get_available_frameworks_for_model(args.model)
        elif args.use_case:
            # If a specific use case is provided, use use case-aware framework selection
            frameworks = get_available_frameworks_for_use_case(args.use_case)
        else:
            frameworks = get_default_frameworks()
        
        # Calculate total number of tests across all frameworks
        total_tests = 0
        for framework in frameworks:
            framework_args = type('Args', (), vars(args).copy())()
            framework_args.framework = framework
            total_tests += self._calculate_total_tests(framework_args)
        
        self.logger.info("Starting comprehensive benchmark run")
        self.logger.info(f"Frameworks: {frameworks}")
        if args.use_case:
            self.logger.info(f"Use case: {args.use_case}")
        self.logger.info(f"Total tests to run: {total_tests}")
        
        total_passed = 0
        total_failed = 0
        all_results = []
        current_test = 0
        
        for framework in frameworks:
            self.logger.info(f"\n{'='*100}")
            self.logger.info(f"TESTING FRAMEWORK: {framework.upper()}")
            self.logger.info(f"{'='*100}")
            
            # Create a copy of args with the current framework
            framework_args = type('Args', (), vars(args).copy())()
            framework_args.framework = framework
            
            # Run framework-specific comprehensive benchmarks
            framework_results = self._run_framework_comprehensive_benchmarks(framework_args, current_test, total_tests)
            
            # Collect results
            all_results.extend(framework_results['results'])
            total_passed += framework_results['passed']
            total_failed += framework_results['failed']
            current_test += len(framework_results['results'])
        
        # Save combined results
        self.results.save_results(all_results, args)
        
        # Print overall summary
        self.logger.info(f"\n{'='*100}")
        self.logger.info("OVERALL COMPREHENSIVE BENCHMARK SUMMARY")
        self.logger.info(f"{'='*100}")
        self.logger.info(f"Frameworks tested: {frameworks}")
        self.logger.info(f"Total combinations tested: {len(all_results)}")
        self.logger.success(f"Total passed: {total_passed}")
        self.logger.error(f"Total failed: {total_failed}")
        self.logger.info(f"Overall success rate: {(total_passed/(total_passed + total_failed))*100:.1f}%")
        
        # Print detailed results table
        self.results.print_summary_table(all_results)
        
        # Print performance analysis
        self._print_performance_analysis(all_results)
        
        # Print framework comparison
        self._print_framework_comparison(all_results, frameworks)
    
    def _calculate_total_tests(self, args) -> int:
        """Calculate total number of tests for a framework"""
        # Determine which models to test
        if args.model is None:
            if args.use_case:
                # Filter models by use case
                models = get_models_for_use_case(args.use_case, args.framework)
            else:
                models = get_unique_models(args.framework)
        elif args.model in ["resnet", "resnet*"]:
            models = self.get_available_models(args.framework, "resnet")
        elif args.model == "gpu_ops":
            models = ["gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"]
        else:
            models = [args.model]
        
        # Determine which precisions to test
        if args.precision is None:
            precisions = DEFAULT_PRECISIONS
        else:
            precisions = [args.precision]
        
        # Determine which batch sizes to test
        if args.batch_size is None:
            batch_sizes = DEFAULT_BATCH_SIZES
        else:
            batch_sizes = [args.batch_size]
        
        # For ONNX, also test different execution providers
        execution_providers = []
        if args.framework == "onnx":
            execution_providers = get_onnx_execution_providers()
        else:
            execution_providers = [None]
        
        # Calculate total combinations, accounting for skipped tests
        total = 0
        for model in models:
            for precision in precisions:
                for batch_size in batch_sizes:
                    for execution_provider in execution_providers:
                        # Skip FP16 on CPU
                        if precision == "fp16" and not torch.cuda.is_available():
                            continue
                        # Skip FP16 for CPU execution provider in ONNX
                        if args.framework == "onnx" and precision == "fp16" and execution_provider == "CPUExecutionProvider":
                            continue
                        total += 1
        
        return total

    def _run_framework_comprehensive_benchmarks(self, args, start_test_num: int = 0, total_tests: int = 0) -> dict:
        """Run comprehensive benchmarks for a specific framework"""
        
        # Determine which models to test
        if args.model is None:
            if args.use_case:
                # Filter models by use case
                models = get_models_for_use_case(args.use_case, args.framework)
            else:
                # Test unique models only (no aliases) to avoid duplicates
                models = get_unique_models(args.framework)
        elif args.model in ["resnet", "resnet*"]:
            models = self.get_available_models(args.framework, "resnet")
        elif args.model == "gpu_ops":
            # Expand gpu_ops to all GPU operation models
            models = ["gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"]
        else:
            models = [args.model]
        
        # Determine which precisions to test
        if args.precision is None:
            precisions = DEFAULT_PRECISIONS  # ["fp32", "fp16"]
        else:
            precisions = [args.precision]
        
        # Determine which batch sizes to test
        if args.batch_size is None:
            batch_sizes = DEFAULT_BATCH_SIZES  # [1, 4, 8, 16]
        else:
            batch_sizes = [args.batch_size]
        
        # For ONNX, also test different execution providers
        execution_providers = []
        if args.framework == "onnx":
            execution_providers = get_onnx_execution_providers()
        else:
            execution_providers = [None]  # No execution provider for other frameworks
        
        passed = 0
        failed = 0
        framework_results = []
        
        current_test = start_test_num
        
        for model in models:
            for precision in precisions:
                for batch_size in batch_sizes:
                    for execution_provider in execution_providers:
                        # Skip FP16 on CPU
                        if precision == "fp16" and not torch.cuda.is_available():
                            continue
                        
                        # Skip FP16 for CPU execution provider in ONNX
                        if args.framework == "onnx" and precision == "fp16" and execution_provider == "CPUExecutionProvider":
                            continue
                        
                        current_test += 1
                        
                        # Determine the appropriate use case for this specific model
                        # If user specified a use case, use that; otherwise use model's default
                        if args.use_case:
                            model_use_case = args.use_case
                        else:
                            model_use_case = get_default_use_case_for_model(model)
                        
                        # Create test description
                        provider_info = f" ({execution_provider})" if execution_provider else ""
                        test_desc = f"{args.framework}/{model} {precision} BS={batch_size}{provider_info}"
                        
                        # Show single line test status
                        print(f"[{current_test:3d}/{total_tests}] {test_desc:<60} ", end="", flush=True)
                        
                        result = self.run_single_benchmark(
                            args.framework, model, args.mode, model_use_case,
                            precision, batch_size, execution_provider
                        )
                        
                        # Add metadata to result
                        result.update({
                            "framework": args.framework,
                            "model": model,
                            "mode": args.mode,
                            "use_case": model_use_case,
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
                            if model_use_case == "compute":
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
                        else:
                            failed += 1
                            print(f"✗ FAILED - {result.get('error', 'Unknown error')}")
        
        # Print framework summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"{args.framework.upper()} FRAMEWORK SUMMARY")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Combinations tested: {len(framework_results)}")
        self.logger.success(f"Passed: {passed}")
        self.logger.error(f"Failed: {failed}")
        self.logger.info(f"Success rate: {(passed/len(framework_results))*100:.1f}%")
        
        return {
            'results': framework_results,
            'passed': passed,
            'failed': failed
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
                        use_case = result.get("use_case", "")
                        
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
                    use_case = best_result.get("use_case", "")
                    
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
                        use_case = result.get("use_case", "")
                        
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
                    best_use_case = best_result.get("use_case", "")
                    
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
        # If specific parameters are provided AND framework is specified, use the original behavior
        if args.framework and all([args.model, args.precision is not None, args.batch_size is not None]):
            self._run_specific_benchmarks(args)
        else:
            # Otherwise, run comprehensive benchmarks (possibly multi-framework)
            self.run_comprehensive_benchmarks(args)
    
    def _run_specific_benchmarks(self, args) -> None:
        """Run specific benchmarks when all parameters are provided"""
        self.logger.info("Starting benchmark run")
        self.logger.info(f"Framework: {args.framework}")
        self.logger.info(f"Model: {args.model}")
        self.logger.info(f"Mode: {args.mode}")
        self.logger.info(f"Use case: {args.use_case}")
        self.logger.info(f"Precision: {args.precision}")
        self.logger.info(f"Batch size: {args.batch_size}")
        
        # Determine which models to run
        if args.model in ["resnet", "resnet*"]:
            models = self.get_available_models(args.framework, "resnet")
        elif args.model == "gpu_ops":
            # Expand gpu_ops to all GPU operation models
            models = ["gemm_ops", "conv_ops", "memory_ops", "elementwise_ops", "reduction_ops"]
        else:
            models = [args.model]
        
        # For ONNX, test all execution providers
        execution_providers = []
        if args.framework == "onnx":
            execution_providers = get_onnx_execution_providers()
        else:
            execution_providers = [None]  # No execution provider for other frameworks
        
        total_benchmarks = len(models) * len(execution_providers)
        passed = 0
        failed = 0
        current_test = 0
        
        all_results = []
        
        for model in models:
            for execution_provider in execution_providers:
                current_test += 1
                
                # Create test description
                provider_info = f" ({execution_provider})" if execution_provider else ""
                test_desc = f"{args.framework}/{model} {args.precision} BS={args.batch_size}{provider_info}"
                
                # Show single line test status
                print(f"[{current_test:3d}/{total_benchmarks}] {test_desc:<60} ", end="", flush=True)
                
                result = self.run_single_benchmark(
                    args.framework, model, args.mode, args.use_case,
                    args.precision, args.batch_size, execution_provider
                )
                
                # Add metadata to result
                result.update({
                    "framework": args.framework,
                    "model": model,
                    "mode": args.mode,
                    "use_case": args.use_case,
                    "precision": args.precision,
                    "batch_size": args.batch_size,
                    "execution_provider": execution_provider,
                    "timestamp": datetime.now().isoformat()
                })
                
                all_results.append(result)
                
                if result["status"] == "PASS":
                    passed += 1
                    # Show performance metric on the same line
                    metrics = result["metrics"]
                    if args.use_case == "compute":
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
        self.logger.error(f"Failed: {failed}")
        self.logger.info(f"Success rate: {(passed/total_benchmarks)*100:.1f}%")
        
        # Print individual results
        if args.framework == "onnx" and len(execution_providers) > 1:
            self.logger.info("\nIndividual Results:")
            self.logger.info("=" * 80)
            for result in all_results:
                if result["status"] == "PASS":
                    provider = result.get("execution_provider", "N/A")
                    if provider and "ExecutionProvider" in provider:
                        provider = provider.replace("ExecutionProvider", "")
                    metrics = result["metrics"]
                    # Show appropriate performance metric based on use case
                    if args.use_case == "compute":
                        # For compute use cases, show GFLOPS or GB/s
                        if metrics.get("best_gflops"):
                            self.logger.success(f"✓ {result['model']} - {args.framework.upper()} ({provider}) ({result['precision']}, BS={result['batch_size']}): {metrics['best_gflops']:.1f} GFLOPS")
                        elif metrics.get("best_bandwidth_gbs"):
                            self.logger.success(f"✓ {result['model']} - {args.framework.upper()} ({provider}) ({result['precision']}, BS={result['batch_size']}): {metrics['best_bandwidth_gbs']:.1f} GB/s")
                        else:
                            throughput = metrics.get("throughput_fps", 0)
                            latency = metrics.get("avg_latency_ms", 0)
                            self.logger.success(f"✓ {result['model']} - {args.framework.upper()} ({provider}) ({result['precision']}, BS={result['batch_size']}): {throughput:.2f} samples/sec, {latency:.2f} ms/sample")
                    else:
                        # For other use cases, show samples/sec and latency
                        throughput = metrics.get("throughput_fps", 0)
                        latency = metrics.get("avg_latency_ms", 0)
                        self.logger.success(f"✓ {result['model']} - {args.framework.upper()} ({provider}) ({result['precision']}, BS={result['batch_size']}): {throughput:.2f} samples/sec, {latency:.2f} ms/sample")
        
        # Print detailed results table
        self.results.print_summary_table(all_results)

def main():
    parser = argparse.ArgumentParser(description="ML Model Benchmarking Framework")
    parser.add_argument("--framework", type=str, 
                       choices=["pytorch", "onnx", "tensorflow"],
                       help="ML framework to benchmark (default: all available)")
    parser.add_argument("--model", type=str,
                       help="Model name (e.g., resnet50, resnet for all resnet models, default: all available)")
    parser.add_argument("--mode", type=str,
                       choices=["inference", "training"],
                       help="Benchmark mode (default: inference)")
    parser.add_argument("--use_case", type=str,
                       choices=["classification", "detection", "segmentation", "generation", "compute"],
                       help="Use case for the benchmark (default: classification)")
    parser.add_argument("--precision", type=str,
                       choices=["fp32", "fp16", "mixed", "int8"],
                       help="Precision for inference (default: all available)")
    parser.add_argument("--batch_size", type=int,
                       help="Batch size for inference (default: test multiple sizes)")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    # Set defaults if not specified (but don't set framework default to allow multi-framework testing)
    if args.mode is None:
        args.mode = "inference"
    if args.use_case is None:
        # Intelligently determine use case based on model if specified
        if args.model:
            args.use_case = get_default_use_case_for_model(args.model)
        else:
            args.use_case = "classification"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks
    runner = BenchmarkRunner()
    runner.run_benchmarks(args)

if __name__ == "__main__":
    main()
