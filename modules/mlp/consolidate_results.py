#!/usr/bin/env python3
"""
Script to consolidate results from all architectures and versions into a single LaTeX table.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse

# Define the model types and versions to compare
MODEL_TYPES = ['mobilenet_v3', 'resnet18', 'resnet50', 'efficientnet_v2_s']
VERSIONS = ['normalized', 'rgb']

# Define the metrics to include in the comparison
METRICS = ['F1', 'Accuracy']

# Define the performance metrics to include from summary.json
PERFORMANCE_METRICS = ['param_count', 'avg_train_time_per_epoch', 'avg_inference_time_per_batch']

def read_metrics_csv(base_dir, version, model_type):
    """
    Read metrics from CSV file.
    
    Args:
        base_dir: Base directory for results
        version: Data version (normalized, rgb)
        model_type: Model type
        
    Returns:
        DataFrame with metrics or None if file doesn't exist
    """
    metrics_path = Path(base_dir) / version / model_type / f"{model_type}_metrics.csv"
    
    if not metrics_path.exists():
        print(f"Warning: Metrics file not found at {metrics_path}")
        return None
    
    try:
        df = pd.read_csv(metrics_path)
        # Filter to only include Mean and Std rows
        df = df[df['Fold'].isin(['Mean', 'Std'])]
        return df
    except Exception as e:
        print(f"Error reading metrics from {metrics_path}: {e}")
        return None

def read_summary_json(base_dir, version, model_type):
    """
    Read summary JSON file to get AUC values and performance metrics.
    
    Args:
        base_dir: Base directory for results
        version: Data version (normalized, rgb)
        model_type: Model type
        
    Returns:
        Dictionary with metrics or None if file doesn't exist
    """
    summary_path = Path(base_dir) / version / model_type / "cross_validation_summary.json"
    
    if not summary_path.exists():
        print(f"Warning: Summary file not found at {summary_path}")
        return None
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        result = {
            'AUC_mean': summary.get('mean_auc', 0),
            'AUC_std': summary.get('std_auc', 0)
        }
        
        # Add performance metrics
        for metric in PERFORMANCE_METRICS:
            result[f"{metric}"] = summary.get(metric, 0)
        
        return result
    except Exception as e:
        print(f"Error reading summary from {summary_path}: {e}")
        return None

def generate_consolidated_table(base_dir, output_path):
    """
    Generate a consolidated LaTeX table comparing all architectures across versions.
    
    Args:
        base_dir: Base directory for results
        output_path: Path to save the LaTeX table
    """
    # Create a DataFrame to store the consolidated results
    columns = ['Model', 'Version'] + [f"{m}_mean" for m in METRICS + ['AUC']] + [f"{m}_std" for m in METRICS + ['AUC']] + PERFORMANCE_METRICS
    consolidated_df = pd.DataFrame(columns=columns)
    
    # Collect metrics for each model type and version
    for version in VERSIONS:
        for model_type in MODEL_TYPES:
            # Read metrics from CSV
            metrics_df = read_metrics_csv(base_dir, version, model_type)
            
            # Read AUC and performance metrics from summary JSON
            summary_dict = read_summary_json(base_dir, version, model_type)
            
            if metrics_df is not None and summary_dict is not None:
                # Extract mean and std values
                mean_row = metrics_df[metrics_df['Fold'] == 'Mean'].iloc[0]
                std_row = metrics_df[metrics_df['Fold'] == 'Std'].iloc[0]
                
                # Create a new row for the consolidated DataFrame
                new_row = {
                    'Model': model_type,
                    'Version': version
                }
                
                # Add mean values
                for metric in METRICS:
                    new_row[f"{metric}_mean"] = mean_row[metric]
                    new_row[f"{metric}_std"] = std_row[metric]
                
                # Add AUC values
                new_row['AUC_mean'] = summary_dict['AUC_mean']
                new_row['AUC_std'] = summary_dict['AUC_std']
                
                # Add performance metrics
                for metric in PERFORMANCE_METRICS:
                    new_row[metric] = summary_dict[metric]
                
                # Add the row to the consolidated DataFrame
                consolidated_df = consolidated_df._append(new_row, ignore_index=True)
    
    # Sort by version and then by model type
    consolidated_df = consolidated_df.sort_values(['Version', 'Model'])
    
    # Save as CSV for reference
    consolidated_df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    # Generate LaTeX table
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{CNN Performance on HSIL Classification}\n")
        f.write("\\label{tab:cnn_performance}\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\columnwidth}{!}{%\n")  # Make table fit page width
        f.write("\\begin{tabular}{lc|cc|ccc}\n")
        f.write("\\toprule\n")
        
        # Header row with multicolumns for grouping
        f.write("\\multirow{2}{*}{\\textbf{Model}} & \\multirow{2}{*}{\\textbf{Version}} & ")
        f.write("\\multicolumn{2}{c|}{\\textbf{Performance Metrics (\%)}} & ")
        f.write("\\multicolumn{3}{c}{\\textbf{Efficiency Metrics}} \\\\\n")
        
        f.write("\\cmidrule(lr){3-4} \\cmidrule(lr){5-7}\n")
        f.write("& & \\textbf{F1} & \\textbf{Acc.} & \\textbf{Params} & \\textbf{Train} & \\textbf{Infer.} \\\\\n")
        f.write("\\midrule\n")
        
        # Format model names more nicely
        model_display_names = {
            'mobilenet_v3': 'MobileNet-V3',
            'resnet18': 'ResNet-18',
            'resnet50': 'ResNet-50',
            'efficientnet_v2_s': 'EfficientNet-V2'
        }
        
        # Format versions more nicely
        version_display = {
            'normalized': 'Norm.',
            'rgb': 'RGB'
        }
        
        # Data rows
        current_model = None
        # Sort by model first, then by version
        sorted_df = consolidated_df.sort_values(['Model', 'Version'])
        
        for _, row in sorted_df.iterrows():
            model = row['Model']
            version = row['Version']
            
            # Format model name nicely
            display_model = model_display_names.get(model, model)
            display_version = version_display.get(version, version)
            
            # Add model name with bold formatting
            if model != current_model:
                f.write(f"\\textbf{{{display_model}}} & ")
                current_model = model
            else:
                f.write("& ")
            
            # Add version
            f.write(f"{display_version} & ")
            
            # Add F1 and Accuracy with mean ± std
            for metric in METRICS:
                mean = row[f"{metric}_mean"] * 100  # Convert to percentage
                std = row[f"{metric}_std"] * 100    # Convert to percentage
                f.write(f"${mean:.1f} \\pm {std:.1f}$ & ")
            
            # Add AUC with mean ± std
            auc_mean = row['AUC_mean']
            auc_std = row['AUC_std']
            
            # Add parameter count (formatted with commas for readability)
            param_count = row['param_count']
            f.write(f"{param_count:,} & ")
            
            # Add training time per epoch (in seconds)
            train_time = row['avg_train_time_per_epoch']
            f.write(f"{train_time:.2f}s & ")
            
            # Add inference time per batch (in milliseconds)
            infer_time = row['avg_inference_time_per_batch'] * 1000  # Convert to ms
            f.write(f"{infer_time:.1f}ms")
            
            f.write(" \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")  # End resizebox
        
        f.write("\\end{table}\n")
    
    print(f"Consolidated results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Consolidate results from all architectures and versions")
    parser.add_argument("--results-dir", type=str, default="../../results/mlp_training_output",
                        help="Base directory for results")
    parser.add_argument("--output", type=str, default="../../results/consolidated_results_table.tex",
                        help="Path to save the LaTeX table")
    
    args = parser.parse_args()
    
    base_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    generate_consolidated_table(base_dir, output_path)

if __name__ == "__main__":
    main()
