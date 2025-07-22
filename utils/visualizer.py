"""
Visualization utilities for ML-Bench benchmarking framework
Provides both web dashboard and CLI visualization capabilities
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import glob
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class BenchmarkVisualizer:
    """Main visualizer class for ML-Bench results"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.color_palette = px.colors.qualitative.Set3
        self.framework_colors = {
            'pytorch': '#EE4C2C',
            'onnx': '#1F77B4',
            'tensorflow': '#FF6F00'
        }
        
    def load_benchmark_results(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load benchmark results from JSON files"""
        if filepath:
            # Load specific file
            with open(filepath, 'r') as f:
                data = json.load(f)
            results = data.get('results', [])
        else:
            # Load latest results
            json_files = list(self.results_dir.glob('*.json'))
            if not json_files:
                raise FileNotFoundError("No benchmark result files found")
            
            # Get most recent file
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                data = json.load(f)
            results = data.get('results', [])
        
        # Convert to DataFrame
        df_data = []
        for result in results:
            if result.get('status') != 'PASS':
                continue
                
            metrics = result.get('metrics', {})
            
            # Extract main fields
            row = {
                'framework': result.get('framework', ''),
                'model': result.get('model', ''),
                'usecase': result.get('usecase', ''),
                'precision': result.get('precision', ''),
                'batch_size': result.get('batch_size', 1),
                'status': result.get('status', ''),
                'execution_time': result.get('execution_time', 0),
                'timestamp': result.get('timestamp', ''),
                'execution_provider': result.get('execution_provider', 'Default')
            }
            
            # Add all metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[f'metric_{key}'] = value
                else:
                    row[f'info_{key}'] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return df
    
    def create_performance_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create performance comparison chart"""
        # Determine primary performance metric based on use case
        performance_cols = ['metric_throughput_fps', 'metric_best_gflops', 'metric_best_bandwidth_gbs']
        perf_col = None
        
        for col in performance_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                perf_col = col
                break
        
        if not perf_col:
            st.error("No performance metrics found in data")
            return go.Figure()
        
        # Create comparison chart
        fig = px.bar(
            df, 
            x='model', 
            y=perf_col,
            color='framework',
            facet_col='precision',
            title='Performance Comparison Across Models and Frameworks',
            color_discrete_map=self.framework_colors,
            hover_data=['batch_size', 'usecase']
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            showlegend=True
        )
        
        return fig
    
    def create_memory_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create memory usage analysis"""
        memory_cols = [
            'metric_total_gpu_memory_used_gb',
            'metric_gpu_memory_allocated_gb', 
            'metric_system_memory_rss_gb'
        ]
        
        memory_col = None
        for col in memory_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                memory_col = col
                break
        
        if not memory_col:
            st.warning("No memory usage data available")
            return go.Figure()
        
        fig = px.scatter(
            df,
            x='batch_size',
            y=memory_col,
            color='model',
            size='metric_throughput_fps' if 'metric_throughput_fps' in df.columns else None,
            title='Memory Usage vs Batch Size',
            hover_data=['framework', 'precision']
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_batch_size_scaling(self, df: pd.DataFrame) -> go.Figure:
        """Create batch size scaling analysis"""
        if 'metric_throughput_fps' not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        for framework in df['framework'].unique():
            for model in df[df['framework'] == framework]['model'].unique():
                subset = df[(df['framework'] == framework) & (df['model'] == model)]
                if len(subset) < 2:
                    continue
                
                subset = subset.sort_values('batch_size')
                
                fig.add_trace(go.Scatter(
                    x=subset['batch_size'],
                    y=subset['metric_throughput_fps'],
                    mode='lines+markers',
                    name=f'{framework}-{model}',
                    line=dict(color=self.framework_colors.get(framework, '#000000'))
                ))
        
        fig.update_layout(
            title='Throughput Scaling vs Batch Size',
            xaxis_title='Batch Size',
            yaxis_title='Throughput (samples/sec)',
            height=500
        )
        
        return fig
    
    def create_precision_impact(self, df: pd.DataFrame) -> go.Figure:
        """Create precision impact analysis"""
        if 'metric_throughput_fps' not in df.columns:
            return go.Figure()
        
        # Group by model and precision
        precision_data = df.groupby(['model', 'precision'])['metric_throughput_fps'].mean().reset_index()
        
        fig = px.bar(
            precision_data,
            x='model',
            y='metric_throughput_fps',
            color='precision',
            title='Precision Impact on Performance',
            barmode='group'
        )
        
        fig.update_layout(height=500, xaxis_tickangle=-45)
        return fig
    
    def create_framework_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create framework comparison heatmap"""
        if 'metric_throughput_fps' not in df.columns:
            return go.Figure()
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values='metric_throughput_fps',
            index='model',
            columns='framework',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale='Viridis',
            title='Framework Performance Heatmap (Throughput)',
            text_auto='.1f'
        )
        
        return fig
    
    def create_model_radar_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create model performance radar chart"""
        metrics = ['metric_throughput_fps', 'metric_avg_latency_ms']
        
        if not all(col in df.columns for col in metrics if col in df.columns):
            return go.Figure()
        
        fig = go.Figure()
        
        for model in df['model'].unique()[:5]:  # Limit to 5 models for readability
            model_data = df[df['model'] == model].iloc[0]
            
            values = []
            labels = []
            for metric in metrics:
                if metric in df.columns and pd.notna(model_data.get(metric)):
                    values.append(model_data[metric])
                    labels.append(metric.replace('metric_', '').replace('_', ' ').title())
            
            if values:
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=model
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        return fig

class DashboardApp:
    """Streamlit dashboard application"""
    
    def __init__(self):
        self.visualizer = BenchmarkVisualizer()
        
    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="ML-Bench Visualization Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("🚀 ML-Bench Visualization Dashboard")
        st.markdown("Interactive analysis of machine learning benchmark results")
        
        # Sidebar for controls
        st.sidebar.header("Controls")
        
        # File selection
        result_files = list(self.visualizer.results_dir.glob('*.json'))
        if not result_files:
            st.error("No benchmark result files found!")
            return
        
        file_options = ["Latest Results"] + [f.name for f in result_files]
        selected_file = st.sidebar.selectbox("Select Results File", file_options)
        
        # Load data
        try:
            if selected_file == "Latest Results":
                df = self.visualizer.load_benchmark_results()
            else:
                filepath = self.visualizer.results_dir / selected_file
                df = self.visualizer.load_benchmark_results(str(filepath))
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
        
        if df.empty:
            st.error("No successful benchmark results found in the selected file")
            return
        
        # Filters
        st.sidebar.subheader("Filters")
        
        frameworks = st.sidebar.multiselect(
            "Frameworks", 
            df['framework'].unique(), 
            default=df['framework'].unique()
        )
        
        models = st.sidebar.multiselect(
            "Models",
            df['model'].unique(),
            default=df['model'].unique()
        )
        
        precisions = st.sidebar.multiselect(
            "Precisions",
            df['precision'].unique(),
            default=df['precision'].unique()
        )
        
        # Apply filters
        filtered_df = df[
            (df['framework'].isin(frameworks)) &
            (df['model'].isin(models)) &
            (df['precision'].isin(precisions))
        ]
        
        # Main content
        if filtered_df.empty:
            st.warning("No data matches the selected filters")
            return
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Configurations", len(filtered_df))
        
        with col2:
            if 'metric_throughput_fps' in filtered_df.columns:
                avg_throughput = filtered_df['metric_throughput_fps'].mean()
                st.metric("Avg Throughput", f"{avg_throughput:.1f} samples/sec")
        
        with col3:
            if 'metric_avg_latency_ms' in filtered_df.columns:
                avg_latency = filtered_df['metric_avg_latency_ms'].mean()
                st.metric("Avg Latency", f"{avg_latency:.1f} ms")
        
        with col4:
            unique_models = len(filtered_df['model'].unique())
            st.metric("Models Tested", unique_models)
        
        # Charts
        st.subheader("📈 Performance Analysis")
        
        # Performance comparison
        perf_fig = self.visualizer.create_performance_comparison(filtered_df)
        if perf_fig.data:
            st.plotly_chart(perf_fig, use_container_width=True)
        
        # Two-column layout for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Memory Analysis")
            memory_fig = self.visualizer.create_memory_analysis(filtered_df)
            if memory_fig.data:
                st.plotly_chart(memory_fig, use_container_width=True)
            
            st.subheader("⚡ Precision Impact")
            precision_fig = self.visualizer.create_precision_impact(filtered_df)
            if precision_fig.data:
                st.plotly_chart(precision_fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Batch Size Scaling")
            scaling_fig = self.visualizer.create_batch_size_scaling(filtered_df)
            if scaling_fig.data:
                st.plotly_chart(scaling_fig, use_container_width=True)
            
            st.subheader("🎯 Model Radar Chart")
            radar_fig = self.visualizer.create_model_radar_chart(filtered_df)
            if radar_fig.data:
                st.plotly_chart(radar_fig, use_container_width=True)
        
        # Framework heatmap
        st.subheader("🔥 Framework Performance Heatmap")
        heatmap_fig = self.visualizer.create_framework_heatmap(filtered_df)
        if heatmap_fig.data:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Raw data table
        with st.expander("📋 Raw Data"):
            st.dataframe(filtered_df)
        
        # Export options
        st.sidebar.subheader("Export")
        if st.sidebar.button("Download CSV"):
            csv = filtered_df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def create_cli_charts(results_file: str = None):
    """Create CLI-friendly ASCII charts"""
    visualizer = BenchmarkVisualizer()
    
    try:
        df = visualizer.load_benchmark_results(results_file)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    if df.empty:
        print("No successful benchmark results found")
        return
    
    print("📊 ML-Bench Results Summary")
    print("=" * 50)
    
    # Performance summary
    if 'metric_throughput_fps' in df.columns:
        perf_by_framework = df.groupby('framework')['metric_throughput_fps'].agg(['mean', 'max', 'count'])
        print("\n🚀 Performance by Framework:")
        print(perf_by_framework.round(2))
    
    # Top performers
    if 'metric_throughput_fps' in df.columns:
        top_configs = df.nlargest(5, 'metric_throughput_fps')
        print(f"\n⭐ Top 5 Configurations by Throughput:")
        for _, row in top_configs.iterrows():
            print(f"  {row['framework']}/{row['model']} {row['precision']} BS={row['batch_size']}: {row['metric_throughput_fps']:.1f} samples/sec")
    
    # Memory usage
    memory_cols = ['metric_total_gpu_memory_used_gb', 'metric_gpu_memory_allocated_gb']
    for col in memory_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            print(f"\n💾 Memory Usage ({col}):")
            memory_summary = df.groupby(['framework', 'model'])[col].mean()
            print(memory_summary.round(2))
            break

# Main dashboard launcher
def launch_dashboard():
    """Launch the Streamlit dashboard"""
    app = DashboardApp()
    app.run()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        results_file = sys.argv[2] if len(sys.argv) > 2 else None
        create_cli_charts(results_file)
    else:
        launch_dashboard() 