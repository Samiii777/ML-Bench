"""
Results management utilities for the benchmarking framework
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd

class BenchmarkResults:
    """Manage benchmark results - saving, loading, and displaying"""
    
    def __init__(self):
        self.results = []
    
    def save_results(self, results: List[Dict[str, Any]], args) -> None:
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"benchmark_{args.framework}_{args.model}_{timestamp}"
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        json_file = output_dir / f"{base_filename}.json"
        self._save_json(results, json_file, args)
        
        # Save as CSV
        csv_file = output_dir / f"{base_filename}.csv"
        self._save_csv(results, csv_file)
        
        # Save summary
        summary_file = output_dir / f"{base_filename}_summary.txt"
        self._save_summary(results, summary_file, args)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        print(f"  Summary: {summary_file}")
    
    def _save_json(self, results: List[Dict[str, Any]], filepath: Path, args) -> None:
        """Save results as JSON"""
        output_data = {
            "benchmark_info": {
                "framework": args.framework,
                "model": args.model,
                "mode": args.mode,
                "use_case": args.use_case,
                "precision": args.precision,
                "batch_size": args.batch_size,
                "timestamp": datetime.now().isoformat(),
                "total_benchmarks": len(results),
                "passed": sum(1 for r in results if r["status"] == "PASS"),
                "failed": sum(1 for r in results if r["status"] == "FAIL")
            },
            "results": results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def _save_csv(self, results: List[Dict[str, Any]], filepath: Path) -> None:
        """Save results as CSV"""
        if not results:
            return
        
        # Flatten the results for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                "framework": result.get("framework", ""),
                "model": result.get("model", ""),
                "mode": result.get("mode", ""),
                "use_case": result.get("use_case", ""),
                "precision": result.get("precision", ""),
                "batch_size": result.get("batch_size", ""),
                "status": result.get("status", ""),
                "execution_time": result.get("execution_time", 0),
                "timestamp": result.get("timestamp", "")
            }
            
            # Add metrics
            metrics = result.get("metrics", {})
            for key, value in metrics.items():
                flat_result[f"metric_{key}"] = value
            
            # Add error if failed
            if result.get("error"):
                flat_result["error"] = result["error"]
            
            flattened_results.append(flat_result)
        
        # Write CSV
        if flattened_results:
            df = pd.DataFrame(flattened_results)
            df.to_csv(filepath, index=False)
    
    def _save_summary(self, results: List[Dict[str, Any]], filepath: Path, args) -> None:
        """Save a human-readable summary"""
        with open(filepath, 'w') as f:
            f.write("BENCHMARK SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write(f"Framework: {args.framework}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Use Case: {args.use_case}\n")
            f.write(f"Precision: {args.precision}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall results
            total = len(results)
            passed = sum(1 for r in results if r["status"] == "PASS")
            failed = total - passed
            
            f.write(f"Total Benchmarks: {total}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success Rate: {(passed/total)*100:.1f}%\n\n")
            
            # Add benchmark results table
            f.write("BENCHMARK RESULTS TABLE\n")
            f.write("=" * 50 + "\n\n")
            self._write_table_to_file(f, results)
            
            # Add performance analysis
            f.write("\nPERFORMANCE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            self._write_performance_analysis_to_file(f, results)
            
            # Detailed results
            f.write("\nDETAILED RESULTS\n")
            f.write("-" * 30 + "\n\n")
            
            for result in results:
                f.write(f"Model: {result.get('model', 'Unknown')}\n")
                f.write(f"Status: {result.get('status', 'Unknown')}\n")
                f.write(f"Execution Time: {result.get('execution_time', 0):.2f}s\n")
                
                if result.get("metrics"):
                    f.write("Metrics:\n")
                    for key, value in result["metrics"].items():
                        f.write(f"  {key}: {value}\n")
                
                if result.get("error"):
                    f.write(f"Error: {result['error']}\n")
                
                f.write("\n")
    
    def _print_simple_table(self, headers: List[str], table_data: List[List[str]]) -> None:
        """Print a simple ASCII table when rich is not available"""
        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in table_data:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(min(max_width + 2, 15))  # Cap at 15 chars
        
        # Print header
        header_row = "| "
        for i, header in enumerate(headers):
            header_row += f"{header:<{col_widths[i]-1}}| "
        print(header_row)
        
        # Print separator
        separator = "|-"
        for width in col_widths:
            separator += "-" * (width-1) + "|-"
        print(separator)
        
        # Print data rows
        for row in table_data:
            data_row = "| "
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell)[:col_widths[i]-2]  # Truncate if too long
                    data_row += f"{cell_str:<{col_widths[i]-1}}| "
            print(data_row)
    
    def print_individual_results(self, results: List[Dict[str, Any]]) -> None:
        """Print individual benchmark results"""
        passed_results = [r for r in results if r.get("status") == "PASS"]
        
        if not passed_results:
            return
        
        print("\nIndividual Results:")
        print("=" * 80)
        
        for result in passed_results:
            framework = result.get("framework", "unknown").upper()
            model = result.get("model", "unknown")
            precision = result.get("precision", "")
            batch_size = result.get("batch_size", "")
            use_case = result.get("use_case", "")
            execution_provider = result.get("execution_provider")
            metrics = result.get("metrics", {})
            
            # Format execution provider
            provider_suffix = ""
            if execution_provider and "ExecutionProvider" in execution_provider:
                provider = execution_provider.replace("ExecutionProvider", "")
                provider_suffix = f" ({provider})"
            
            # Get performance metrics based on use case
            if use_case == "compute":
                # For compute use cases, show GFLOPS or GB/s
                if metrics.get("best_gflops"):
                    performance_str = f"{metrics['best_gflops']:.1f} GFLOPS"
                elif metrics.get("best_bandwidth_gbs"):
                    performance_str = f"{metrics['best_bandwidth_gbs']:.1f} GB/s"
                else:
                    throughput = metrics.get("throughput_fps", 0)
                    latency = metrics.get("avg_latency_ms", 0)
                    performance_str = f"{throughput:.2f} samples/sec, {latency:.2f} ms/sample"
            else:
                # For other use cases, show samples/sec and latency
                throughput = metrics.get("throughput_fps", 0)
                latency = metrics.get("avg_latency_ms", 0)
                performance_str = f"{throughput:.2f} samples/sec, {latency:.2f} ms/sample"
            
            print(f"✓ {model} - {framework}{provider_suffix} ({precision}, BS={batch_size}): {performance_str}")
    
    def print_summary_table(self, results: List[Dict[str, Any]]) -> None:
        """Print a summary table of all benchmark results"""
        if not results:
            print("No results to display")
            return
        
        # Filter out failed results for the table
        passed_results = [r for r in results if r.get("status") == "PASS"]
        
        if not passed_results:
            print("No successful results to display")
            return
        
        print(f"\n{'='*150}")
        print(" " * 60 + "Benchmark Results")
        print(f"\n{'='*150}")
        
        # Create table data
        table_data = []
        headers = ["Test Name", "Framework", "Model Name", "Mode", "Precision", "Batch Size", "Use Case", "Performance", "Latency", "Memory", "Device"]
        
        for result in passed_results:
            # Create test name
            framework = result.get("framework", "unknown")
            model = result.get("model", "unknown")
            test_name = f"{framework}_{model[:3]}"
            
            # Get metrics
            metrics = result.get("metrics", {})
            
            # Determine performance metric based on use case and available metrics
            use_case = result.get("use_case", "")
            performance_str = "N/A"
            
            if use_case == "compute":
                # For compute use cases, prefer GFLOPS or GB/s over samples/sec
                if metrics.get("best_gflops"):
                    performance_str = f"{metrics['best_gflops']:.1f} GFLOPS"
                elif metrics.get("best_bandwidth_gbs"):
                    performance_str = f"{metrics['best_bandwidth_gbs']:.1f} GB/s"
                elif metrics.get("throughput_fps"):
                    performance_str = f"{metrics['throughput_fps']:.2f} samp/s"
                else:
                    performance_str = "N/A"
            else:
                # For other use cases (classification, generation), use samples/sec
                if metrics.get("throughput_fps"):
                    performance_str = f"{metrics['throughput_fps']:.2f} samp/s"
                else:
                    performance_str = "N/A"
            
            # Get latency
            latency = metrics.get("avg_latency_ms", metrics.get("inference_time_ms", 0))
            latency_str = f"{latency:.2f} ms" if latency else "N/A"
            
            # Get memory usage (prioritize total GPU memory, then GPU allocated, then system memory)
            memory_gb = (
                metrics.get("total_gpu_memory_used_gb") or 
                metrics.get("gpu_memory_allocated_gb") or 
                metrics.get("system_memory_rss_gb") or 
                self._estimate_memory_usage(result.get("model", ""), result.get("precision", "fp32"))
            )
            memory_str = f"{memory_gb:.2f} GB" if memory_gb else "N/A"
            
            # Get device
            device = metrics.get("device", "unknown")
            if device.startswith("cuda"):
                device = "cuda"
            
            # Add execution provider for ONNX
            execution_provider = result.get("execution_provider")
            if execution_provider and "ExecutionProvider" in execution_provider:
                provider_short = execution_provider.replace("ExecutionProvider", "")
                device = f"{device} ({provider_short})"
            
            table_data.append([
                test_name,
                framework,
                model,
                result.get("mode", ""),
                result.get("precision", ""),
                str(result.get("batch_size", "")),
                use_case,
                performance_str,
                latency_str,
                memory_str,
                device
            ])
        
        # Print table using rich
        try:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            table = Table(show_header=True, header_style="bold magenta")
            
            for header in headers:
                table.add_column(header)
            
            for row in table_data:
                table.add_row(*row)
            
            console.print(table)
            
        except ImportError:
            # Fallback to simple table if rich is not available
            self._print_simple_table(headers, table_data)
    
    def _estimate_memory_usage(self, model: str, precision: str) -> float:
        """Estimate memory usage based on model and precision"""
        base_memory = {
            "resnet18": 0.05,
            "resnet34": 0.08,
            "resnet50": 0.11,
            "resnet101": 0.17,
            "resnet152": 0.23,
        }
        
        memory = base_memory.get(model, 0.1)  # Default 0.1 GB
        
        # Adjust for precision
        if precision == "fp16":
            memory *= 0.6  # Roughly 60% of fp32
        elif precision == "int8":
            memory *= 0.3  # Roughly 30% of fp32
        
        return memory
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load results from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results from {filepath}: {e}")
            return {}
    
    def _write_table_to_file(self, f, results: List[Dict[str, Any]]) -> None:
        """Write the benchmark results table to file"""
        # Filter out failed results for the table
        passed_results = [r for r in results if r.get("status") == "PASS"]
        
        if not passed_results:
            f.write("No successful results to display\n")
            return
        
        # Create table data
        headers = ["Test Name", "Framework", "Model Name", "Mode", "Precision", "Batch Size", "Use Case", "Performance", "Latency", "Memory", "Device"]
        table_data = []
        
        for result in passed_results:
            # Create test name
            framework = result.get("framework", "unknown")
            model = result.get("model", "unknown")
            test_name = f"{framework}_{model[:3]}"
            
            # Get metrics
            metrics = result.get("metrics", {})
            
            # Determine performance metric based on use case and available metrics
            use_case = result.get("use_case", "")
            performance_str = "N/A"
            
            if use_case == "compute":
                # For compute use cases, prefer GFLOPS or GB/s over samples/sec
                if metrics.get("best_gflops"):
                    performance_str = f"{metrics['best_gflops']:.1f} GFLOPS"
                elif metrics.get("best_bandwidth_gbs"):
                    performance_str = f"{metrics['best_bandwidth_gbs']:.1f} GB/s"
                elif metrics.get("throughput_fps"):
                    performance_str = f"{metrics['throughput_fps']:.2f} samp/s"
                else:
                    performance_str = "N/A"
            else:
                # For other use cases (classification, generation), use samples/sec
                if metrics.get("throughput_fps"):
                    performance_str = f"{metrics['throughput_fps']:.2f} samp/s"
                else:
                    performance_str = "N/A"
            
            # Get latency
            latency = metrics.get("avg_latency_ms", metrics.get("inference_time_ms", 0))
            latency_str = f"{latency:.2f} ms" if latency else "N/A"
            
            # Get memory usage (prioritize total GPU memory, then GPU allocated, then system memory)
            memory_gb = (
                metrics.get("total_gpu_memory_used_gb") or 
                metrics.get("gpu_memory_allocated_gb") or 
                metrics.get("system_memory_rss_gb") or 
                self._estimate_memory_usage(result.get("model", ""), result.get("precision", "fp32"))
            )
            memory_str = f"{memory_gb:.2f} GB" if memory_gb else "N/A"
            
            # Get device
            device = metrics.get("device", "unknown")
            if device.startswith("cuda"):
                device = "cuda"
            
            # Add execution provider for ONNX
            execution_provider = result.get("execution_provider")
            if execution_provider and "ExecutionProvider" in execution_provider:
                provider_short = execution_provider.replace("ExecutionProvider", "")
                device = f"{device} ({provider_short})"
            
            table_data.append([
                test_name,
                framework,
                model,
                result.get("mode", ""),
                result.get("precision", ""),
                str(result.get("batch_size", "")),
                use_case,
                performance_str,
                latency_str,
                memory_str,
                device
            ])
        
        # Write table to file using simple ASCII format
        self._write_simple_table_to_file(f, headers, table_data)
    
    def _write_simple_table_to_file(self, f, headers: List[str], table_data: List[List[str]]) -> None:
        """Write a simple ASCII table to file"""
        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in table_data:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(min(max_width + 2, 20))  # Cap at 20 chars for file
        
        # Write header
        header_row = "| "
        for i, header in enumerate(headers):
            header_row += f"{header:<{col_widths[i]-1}}| "
        f.write(header_row + "\n")
        
        # Write separator
        separator = "|-"
        for width in col_widths:
            separator += "-" * (width-1) + "|-"
        f.write(separator + "\n")
        
        # Write data rows
        for row in table_data:
            data_row = "| "
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell)[:col_widths[i]-2]  # Truncate if too long
                    data_row += f"{cell_str:<{col_widths[i]-1}}| "
            f.write(data_row + "\n")
    
    def _write_performance_analysis_to_file(self, f, results: List[Dict[str, Any]]) -> None:
        """Write performance analysis to file"""
        if not results:
            f.write("No results to analyze\n")
            return
        
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
            f.write(f"{model.upper()} Performance Analysis:\n")
            f.write("-" * 50 + "\n")
            
            # Find best throughput and latency
            best_throughput = max(model_results, key=lambda x: x["metrics"].get("throughput_fps", 0))
            best_latency = min(model_results, key=lambda x: x["metrics"].get("avg_latency_ms", float('inf')))
            
            # Check if this is a compute model (GPU ops)
            use_case = model_results[0].get("use_case", "")
            if use_case == "compute":
                # For compute models, show GFLOPS or GB/s
                best_gflops = max(model_results, key=lambda x: x["metrics"].get("best_gflops", 0))
                best_bandwidth = max(model_results, key=lambda x: x["metrics"].get("best_bandwidth_gbs", 0))
                
                if best_gflops["metrics"].get("best_gflops", 0) > 0:
                    f.write(f"Best GFLOPS: {best_gflops['metrics']['best_gflops']:.1f} GFLOPS\n")
                    f.write(f"  Configuration: {best_gflops['precision']}, BS={best_gflops['batch_size']}\n")
                
                if best_bandwidth["metrics"].get("best_bandwidth_gbs", 0) > 0:
                    f.write(f"Best Bandwidth: {best_bandwidth['metrics']['best_bandwidth_gbs']:.1f} GB/s\n")
                    f.write(f"  Configuration: {best_bandwidth['precision']}, BS={best_bandwidth['batch_size']}\n")
            else:
                # For other models, show throughput and latency
                f.write(f"Best Throughput: {best_throughput['metrics'].get('throughput_fps', 0):.2f} samples/sec\n")
                f.write(f"  Configuration: {best_throughput['precision']}, BS={best_throughput['batch_size']}\n")
                
                f.write(f"Best Latency: {best_latency['metrics'].get('avg_latency_ms', 0):.2f} ms/sample\n")
                f.write(f"  Configuration: {best_latency['precision']}, BS={best_latency['batch_size']}\n")
            
            # Precision comparison (for batch size 1)
            bs1_results = [r for r in model_results if r["batch_size"] == 1]
            if len(bs1_results) > 1:
                f.write("Precision Comparison (BS=1):\n")
                for result in bs1_results:
                    precision = result["precision"]
                    if use_case == "compute":
                        if result["metrics"].get("best_gflops"):
                            f.write(f"  {precision}: {result['metrics']['best_gflops']:.1f} GFLOPS\n")
                        elif result["metrics"].get("best_bandwidth_gbs"):
                            f.write(f"  {precision}: {result['metrics']['best_bandwidth_gbs']:.1f} GB/s\n")
                        else:
                            throughput = result["metrics"].get("throughput_fps", 0)
                            f.write(f"  {precision}: {throughput:.2f} samples/sec\n")
                    else:
                        throughput = result["metrics"].get("throughput_fps", 0)
                        f.write(f"  {precision}: {throughput:.2f} samples/sec\n")
            
            f.write("\n") 