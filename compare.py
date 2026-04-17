#!/usr/bin/env python3
"""
Benchmark Results Comparison Tool

Compare performance between two benchmark result CSV files.
Shows percentage differences for matching test configurations.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def load_csv_file(file_path):
    """Load and validate CSV file"""
    try:
        if not Path(file_path).exists():
            print(f"Error: File '{file_path}' not found")
            return None

        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_json_results(file_path):
    """Load benchmark results from a JSON file and convert to DataFrame."""
    import json
    try:
        if not Path(file_path).exists():
            print(f"Error: File '{file_path}' not found")
            return None

        with open(file_path) as f:
            data = json.load(f)

        results = data.get("results", data if isinstance(data, list) else [data])
        rows = []
        for r in results:
            row = {
                "framework": r.get("framework", ""),
                "model": r.get("model", ""),
                "mode": r.get("mode", ""),
                "usecase": r.get("usecase", r.get("use_case", "")),
                "precision": r.get("precision", ""),
                "batch_size": r.get("batch_size", 1),
                "status": r.get("status", ""),
            }
            metrics = r.get("metrics", {})
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    row[f"metric_{k}"] = v
            elif isinstance(metrics, list):
                for m in metrics:
                    if isinstance(m, dict) and "name" in m:
                        row[f"metric_{m['name']}"] = m.get("value", 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        print(f"Loaded {len(df)} results from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading JSON {file_path}: {e}")
        return None


def load_result_file(file_path):
    """Auto-detect file format and load results."""
    p = Path(file_path)
    if p.suffix == ".json":
        return load_json_results(file_path)
    return load_csv_file(file_path)

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change from old to new value"""
    if pd.isna(old_value) or pd.isna(new_value):
        return None
    if old_value == 0:
        return float('inf') if new_value > 0 else 0
    return ((new_value - old_value) / old_value) * 100

def format_percentage(pct):
    """Format percentage for display"""
    if pct is None:
        return "N/A"
    if pct == float('inf'):
        return "∞%"
    if pct == float('-inf'):
        return "-∞%"
    
    # Color coding for significant changes
    if abs(pct) >= 20:
        color = "🔴" if pct < 0 else "🟢"  # Red for worse, Green for better
    elif abs(pct) >= 5:
        color = "🟡"  # Yellow for moderate changes
    else:
        color = ""  # No color for small changes
    
    return f"{color}{pct:+.1f}%"

def identify_key_columns(df):
    """Identify the key columns used for matching test configurations"""
    # Common column name variations
    possible_columns = {
        'test_name': ['test_name', 'benchmark_name', 'name'],
        'framework': ['framework', 'fw'],
        'model': ['model', 'model_name'],
        'mode': ['mode', 'test_mode'],
        'precision': ['precision', 'dtype'],
        'batch_size': ['batch_size', 'batch'],
        'use_case': ['use_case', 'task', 'workload'],
        'execution_provider': ['execution_provider', 'provider', 'ep']
    }
    
    key_columns = []
    available_columns = df.columns.tolist()
    
    for key, variations in possible_columns.items():
        for variation in variations:
            if variation in available_columns:
                key_columns.append(variation)
                break
    
    return key_columns

def identify_performance_columns(df):
    """Identify performance metric columns"""
    performance_keywords = [
        'latency', 'time', 'throughput', 'fps', 'tokens_per_second', 
        'memory', 'vram', 'utilization', 'speed', 'bandwidth'
    ]
    
    performance_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in performance_keywords):
            # Skip if it's clearly not a numeric performance metric
            if df[col].dtype in ['object', 'string'] and not col_lower.endswith('_gb'):
                continue
            performance_columns.append(col)
    
    return performance_columns

def compare_results(df1, df2, file1_name, file2_name, name1="baseline", name2="comparison",
                   baseline_cols=None, comparison_cols=None, custom_col_names=None):
    """Compare results between two dataframes with custom names and column mappings"""
    
    # Use specific key columns for matching as requested by user
    required_key_columns = ['framework', 'model', 'mode', 'usecase', 'precision', 'batch_size', 'status']
    key_columns = []
    
    # Check which of the required columns exist in both dataframes
    for col in required_key_columns:
        if col in df1.columns and col in df2.columns:
            key_columns.append(col)
        else:
            print(f"Warning: Column '{col}' not found in both files. Skipping from matching criteria.")
    
    if not key_columns:
        print("Error: None of the required key columns found in both files.")
        print(f"Required columns: {required_key_columns}")
        print(f"File 1 columns: {list(df1.columns)}")
        print(f"File 2 columns: {list(df2.columns)}")
        return
    
    print(f"Matching on columns: {', '.join(key_columns)}")
    
    # Identify performance columns
    if baseline_cols and comparison_cols:
        # Use user-specified columns
        perf_columns = baseline_cols
        print(f"Using specified columns for comparison: {', '.join(perf_columns)}")
        
        # Validate that specified columns exist
        missing_in_df1 = [col for col in baseline_cols if col not in df1.columns]
        missing_in_df2 = [col for col in comparison_cols if col not in df2.columns]
        
        if missing_in_df1:
            print(f"Error: Columns missing in baseline file: {missing_in_df1}")
            return
        if missing_in_df2:
            print(f"Error: Columns missing in comparison file: {missing_in_df2}")
            return
    else:
        # Auto-detect performance columns
        perf_columns = identify_performance_columns(df1)
        if not perf_columns:
            print("Warning: Could not identify performance columns. Using all numeric columns.")
            perf_columns = [col for col in df1.columns if df1[col].dtype in ['float64', 'int64']]
        print(f"Comparing performance metrics: {', '.join(perf_columns)}")
    
    # Ensure key columns exist in both dataframes
    missing_in_df2 = [col for col in key_columns if col not in df2.columns]
    if missing_in_df2:
        print(f"Warning: Key columns missing in second file: {missing_in_df2}")
        key_columns = [col for col in key_columns if col in df2.columns]
        print(f"Updated matching columns: {', '.join(key_columns)}")
    
    if not key_columns:
        print("Error: No common key columns found between the two files.")
        return
    
    # Handle custom column mapping
    column_mappings = {}
    if baseline_cols and comparison_cols:
        # Create a temporary dataframe with renamed columns for easier merging
        df1_renamed = df1.copy()
        df2_renamed = df2.copy()
        
        # Create mappings for performance columns
        for i, (base_col, comp_col) in enumerate(zip(baseline_cols, comparison_cols)):
            if custom_col_names:
                # Use custom name if provided
                unified_name = custom_col_names[i]
            else:
                # Use baseline column name as unified name
                unified_name = base_col
            
            column_mappings[base_col] = unified_name
            column_mappings[comp_col] = unified_name
            
            # Rename columns to unified names for comparison
            df1_renamed = df1_renamed.rename(columns={base_col: unified_name})
            df2_renamed = df2_renamed.rename(columns={comp_col: unified_name})
        
        # Update perf_columns to use unified names
        if custom_col_names:
            perf_columns = custom_col_names
        else:
            perf_columns = baseline_cols
        
        print(f"Column mappings: {column_mappings}")
        
        # Use renamed dataframes for merging
        df1, df2 = df1_renamed, df2_renamed
    
    # Merge dataframes on key columns with custom suffixes
    suffix1 = f"_{name1}"
    suffix2 = f"_{name2}"
    
    try:
        merged = pd.merge(df1, df2, on=key_columns, how='outer', suffixes=(suffix1, suffix2))
        print(f"Merged dataframe has {len(merged)} rows and {len(merged.columns)} columns")
    except Exception as e:
        print(f"Error during merge: {e}")
        return
    
    # Check which performance columns actually exist after merge
    available_name1_cols = []
    available_name2_cols = []
    available_perf_columns = []
    
    print(f"Debug: Available columns after merge: {list(merged.columns)}")
    
    for perf_col in perf_columns:
        name1_col = perf_col + suffix1
        name2_col = perf_col + suffix2
        
        # Check if both suffixed columns exist
        if name1_col in merged.columns and name2_col in merged.columns:
            available_name1_cols.append(name1_col)
            available_name2_cols.append(name2_col)
            available_perf_columns.append(perf_col)
            print(f"Debug: Found comparable column pair: {name1_col} & {name2_col}")
        # Check if the original column exists (might not have been split due to no conflicts)
        elif perf_col in merged.columns:
            # This means the column values were identical in both dataframes
            # We can still use it by creating our own custom-named columns
            print(f"Info: Column '{perf_col}' appears identical in both files - creating artificial comparison")
            
            # For identical columns, we'll mark both as the same value
            merged[name1_col] = merged[perf_col]
            merged[name2_col] = merged[perf_col]
            available_name1_cols.append(name1_col)
            available_name2_cols.append(name2_col)
            available_perf_columns.append(perf_col)
        else:
            print(f"Debug: Column '{perf_col}' not found in expected forms")
    
    if not available_perf_columns:
        print("Error: No comparable performance columns found after merge.")
        print("Available columns after merge:", list(merged.columns))
        return
    
    print(f"Found comparable metrics: {', '.join(available_perf_columns)}")
    
    # Filter to rows that have data in both datasets
    subset_cols = available_name1_cols + available_name2_cols
    
    # Instead of requiring ALL metrics to be present, just require at least one metric pair
    # Create a mask for rows that have at least one valid comparison
    valid_rows_mask = pd.Series([False] * len(merged), index=merged.index)
    
    for i, perf_col in enumerate(available_perf_columns):
        name1_col = available_name1_cols[i]
        name2_col = available_name2_cols[i]
        
        # Mark rows as valid if they have both values for this metric
        pair_valid = merged[name1_col].notna() & merged[name2_col].notna()
        valid_rows_mask |= pair_valid
    
    matched_rows = merged[valid_rows_mask]
    
    print(f"Debug: Total rows after merge: {len(merged)}")
    print(f"Debug: Rows with at least one comparable metric: {len(matched_rows)}")
    
    # Always save a comparison file, even if no matches
    if column_mappings:
        output_file = f"comparison_{name1}_vs_{name2}_custom_columns_{Path(file1_name).stem}_{Path(file2_name).stem}.csv"
    else:
        output_file = f"comparison_{name1}_vs_{name2}_{Path(file1_name).stem}_{Path(file2_name).stem}.csv"
    
    if len(matched_rows) == 0:
        print("⚠️  No matching test configurations found between the two files.")
        print("\n📊 DIAGNOSTIC INFORMATION:")
        
        # Show sample from merged data
        print("\nSample configurations from merged data:")
        display_cols = [col for col in key_columns + available_perf_columns[:3] if col in merged.columns]
        if display_cols:
            print(merged[display_cols].head(10))
        else:
            print("No comparable columns found")
        
        # Analyze why no matches were found
        print(f"\n🔍 ANALYSIS:")
        print(f"Total rows after merge: {len(merged)}")
        
        # Check which key columns have mismatches
        only_in_dataset1 = merged[merged[available_name2_cols].isna().all(axis=1) & 
                                 merged[available_name1_cols].notna().any(axis=1)]
        only_in_dataset2 = merged[merged[available_name1_cols].isna().all(axis=1) & 
                                 merged[available_name2_cols].notna().any(axis=1)]
        
        print(f"Configurations only in {name1}: {len(only_in_dataset1)}")
        print(f"Configurations only in {name2}: {len(only_in_dataset2)}")
        
        # Show unique values for key columns
        for col in key_columns[:3]:  # Show first 3 key columns
            if col in merged.columns:
                unique_vals = merged[col].dropna().unique()
                if len(unique_vals) <= 10:
                    print(f"{col}: {list(unique_vals)}")
                else:
                    print(f"{col}: {len(unique_vals)} unique values (too many to display)")
        
        # Create a summary dataframe for saving
        summary_data = []
        
        if len(only_in_dataset1) > 0:
            for _, row in only_in_dataset1.head(20).iterrows():  # Limit to 20 rows
                result = {'dataset': name1}
                for col in key_columns:
                    if col in row and pd.notna(row[col]):
                        result[col] = row[col]
                for perf_col in available_perf_columns[:5]:  # Limit to 5 perf columns
                    name1_col = perf_col + suffix1
                    if name1_col in row and pd.notna(row[name1_col]):
                        result[f"{perf_col}_{name1}"] = row[name1_col]
                summary_data.append(result)
        
        if len(only_in_dataset2) > 0:
            for _, row in only_in_dataset2.head(20).iterrows():  # Limit to 20 rows
                result = {'dataset': name2}
                for col in key_columns:
                    if col in row and pd.notna(row[col]):
                        result[col] = row[col]
                for perf_col in available_perf_columns[:5]:  # Limit to 5 perf columns
                    name2_col = perf_col + suffix2
                    if name2_col in row and pd.notna(row[name2_col]):
                        result[f"{perf_col}_{name2}"] = row[name2_col]
                summary_data.append(result)
        
        if summary_data:
            results_df = pd.DataFrame(summary_data)
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 Summary of non-matching configurations saved to: {output_file}")
        else:
            # Save the merged data as-is for inspection
            merged.to_csv(output_file, index=False)
            print(f"\n💾 Raw merged data saved for inspection to: {output_file}")
        
        print(f"\n💡 SUGGESTIONS:")
        print(f"1. Check if the files have different model names, batch sizes, or other key identifiers")
        print(f"2. Use specific column mapping if column names differ between files")
        print(f"3. Verify that both files contain the same type of benchmark data")
        
        return
    
    print(f"\n✅ Found {len(matched_rows)} matching test configurations")
    
    # Calculate percentage changes
    comparison_results = []
    
    for _, row in matched_rows.iterrows():
        result = {}
        
        # Add key identifying information
        for col in key_columns:
            result[col] = row[col]
        
        # Calculate percentage changes for performance metrics
        for perf_col in available_perf_columns:
            name1_col = perf_col + suffix1
            name2_col = perf_col + suffix2
            
            if name1_col in row and name2_col in row:
                name1_val = row[name1_col]
                name2_val = row[name2_col]
                
                # Only calculate percentage if both values are not null/NaN
                if pd.notna(name1_val) and pd.notna(name2_val):
                    pct_change = calculate_percentage_change(name1_val, name2_val)
                    
                    result[f"{perf_col}_{name1}"] = name1_val
                    result[f"{perf_col}_{name2}"] = name2_val
                    result[f"{perf_col}_change%"] = pct_change
        
        comparison_results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(comparison_results)
    
    # Display results
    print(f"\n{'='*80}")
    if column_mappings:
        print(f"PERFORMANCE COMPARISON: {name1.upper()} vs {name2.upper()} (Custom Column Mapping)")
    else:
        print(f"PERFORMANCE COMPARISON: {name1.upper()} vs {name2.upper()}")
    print(f"Files: {file1_name} vs {file2_name}")
    if column_mappings:
        print(f"Column mappings: {column_mappings}")
    print(f"{'='*80}")
    
    # Show summary statistics
    print("\nSUMMARY:")
    change_columns = [col for col in results_df.columns if col.endswith('_change%')]
    
    for col in change_columns:
        metric_name = col.replace('_change%', '')
        changes = results_df[col].dropna()  # Drop NaN values for statistics
        if len(changes) > 0:
            avg_change = changes.mean()
            significant_improvements = (changes > 5).sum()
            significant_regressions = (changes < -5).sum()
            
            print(f"{metric_name:25} | Avg: {avg_change:+6.1f}% | "
                  f"Improvements: {significant_improvements} | "
                  f"Regressions: {significant_regressions} | "
                  f"Valid comparisons: {len(changes)}")
        else:
            print(f"{metric_name:25} | No valid comparisons found")
    
    # Show detailed results
    print(f"\nDETAILED COMPARISON:")
    print(f"Legend: 🟢 = >20% better, 🟡 = 5-20% change, 🔴 = >20% worse")
    print(f"Format: {name1} → {name2}")
    print("-" * 120)
    
    for _, row in results_df.iterrows():
        # Show test configuration
        config_parts = []
        for col in key_columns:
            if pd.notna(row[col]):
                config_parts.append(f"{col}={row[col]}")
        
        print(f"\nTest: {' | '.join(config_parts)}")
        
        # Show performance changes
        for col in change_columns:
            if pd.notna(row[col]):
                metric_name = col.replace('_change%', '')
                name1_col = metric_name + f'_{name1}'
                name2_col = metric_name + f'_{name2}'
                
                name1_val = row.get(name1_col, 'N/A')
                name2_val = row.get(name2_col, 'N/A')
                pct_change = row[col]
                
                # Only show metrics that have valid data
                if pd.notna(name1_val) and pd.notna(name2_val):
                    print(f"  {metric_name:20} | {name1_val:>10} → {name2_val:>10} | {format_percentage(pct_change)}")
    
    # Show configurations that only exist in one dataset
    only_in_dataset1 = merged[merged[available_name2_cols].isna().all(axis=1) & 
                             merged[available_name1_cols].notna().any(axis=1)]
    only_in_dataset2 = merged[merged[available_name1_cols].isna().all(axis=1) & 
                             merged[available_name2_cols].notna().any(axis=1)]
    
    if len(only_in_dataset1) > 0:
        print(f"\nConfigurations only in {name1.upper()}:")
        for _, row in only_in_dataset1.iterrows():
            config_parts = [f"{col}={row[col]}" for col in key_columns if pd.notna(row[col])]
            print(f"  {' | '.join(config_parts)}")
    
    if len(only_in_dataset2) > 0:
        print(f"\nConfigurations only in {name2.upper()}:")
        for _, row in only_in_dataset2.iterrows():
            config_parts = [f"{col}={row[col]}" for col in key_columns if pd.notna(row[col])]
            print(f"  {' | '.join(config_parts)}")
    
    # Save detailed results to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results between two CSV files")
    parser.add_argument("baseline_file", help="Baseline CSV file (first file)")
    parser.add_argument("comparison_file", help="Comparison CSV file (second file)")
    parser.add_argument("--baseline-name", "-b", default="baseline", 
                       help="Custom name for the baseline dataset (default: baseline)")
    parser.add_argument("--comparison-name", "-c", default="comparison",
                       help="Custom name for the comparison dataset (default: comparison)")
    parser.add_argument("--baseline-columns", 
                       help="Comma-separated list of specific columns to use from baseline file")
    parser.add_argument("--comparison-columns",
                       help="Comma-separated list of specific columns to use from comparison file (must match baseline-columns order)")
    parser.add_argument("--column-names",
                       help="Comma-separated list of custom names for the comparison columns in output")
    parser.add_argument("--output", "-o", help="Output CSV file name (optional)")
    
    args = parser.parse_args()
    
    # Parse column specifications
    baseline_cols = None
    comparison_cols = None
    custom_col_names = None
    
    if args.baseline_columns:
        baseline_cols = [col.strip() for col in args.baseline_columns.split(',')]
    
    if args.comparison_columns:
        comparison_cols = [col.strip() for col in args.comparison_columns.split(',')]
        
    if args.column_names:
        custom_col_names = [name.strip() for name in args.column_names.split(',')]
    
    # Validate column specifications
    if baseline_cols and comparison_cols:
        if len(baseline_cols) != len(comparison_cols):
            print("Error: baseline-columns and comparison-columns must have the same number of columns")
            sys.exit(1)
            
    if custom_col_names:
        if not baseline_cols:
            print("Error: column-names requires baseline-columns to be specified")
            sys.exit(1)
        if len(custom_col_names) != len(baseline_cols):
            print("Error: column-names must have the same number of names as baseline-columns")
            sys.exit(1)
    
    # Load result files (auto-detect CSV or JSON)
    df1 = load_result_file(args.baseline_file)
    df2 = load_result_file(args.comparison_file)
    
    if df1 is None or df2 is None:
        sys.exit(1)
    
    # Compare results with custom names and column mappings
    compare_results(df1, df2, args.baseline_file, args.comparison_file, 
                   args.baseline_name, args.comparison_name,
                   baseline_cols, comparison_cols, custom_col_names)

if __name__ == "__main__":
    main() 