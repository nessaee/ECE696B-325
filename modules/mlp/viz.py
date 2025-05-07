#!/usr/bin/env python3
"""
fold_visualizer.py

A utility script for visualizing training histories across multiple folds.
Loads training history files and generates comprehensive visualizations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import os


def load_fold_histories(base_dir):
    """
    Load all training history JSON files from fold directories.
    
    Args:
        base_dir (str): The base output directory containing fold subdirectories
        
    Returns:
        dict: Dictionary mapping fold names to their training histories
    """
    base_path = Path(base_dir)
    fold_histories = {}
    
    # Find all fold directories
    fold_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('fold')]
    
    if not fold_dirs:
        print(f"No fold directories found in {base_dir}")
        return {}
    
    # Load each fold's training history
    for fold_dir in sorted(fold_dirs):
        history_path = fold_dir / "training_history.json"
        
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                fold_histories[fold_dir.name] = history
                print(f"Loaded training history for {fold_dir.name}")
            except Exception as e:
                print(f"Error loading {history_path}: {e}")
        else:
            print(f"No training history found at {history_path}")
    
    return fold_histories


def visualize_fold_histories(fold_histories, output_dir, metrics=None, styles=None):
    """
    Generate comprehensive visualizations of training histories across folds.
    
    Args:
        fold_histories (dict): Dictionary of fold histories
        output_dir (str): Directory to save visualizations
        metrics (list): List of metrics to visualize, defaults to auto-detection or ['loss', 'acc']
        styles (dict): Dictionary of visualization styles
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if not fold_histories:
        print("No fold histories to visualize")
        return
    
    # Auto-detect metrics if not provided
    if metrics is None:
        # First check what metrics are available in the histories
        available_metrics = set()
        for history in fold_histories.values():
            for key in history.keys():
                if key.startswith('train_') or key.startswith('val_'):
                    # Extract the metric name without train_/val_ prefix and without trailing s
                    metric = key.split('_')[1]
                    if metric.endswith('s'):
                        metric = metric[:-1]  # Remove trailing 's'
                    available_metrics.add(metric)
        
        if available_metrics:
            metrics = list(available_metrics)
            print(f"Found metrics in training histories: {', '.join(metrics)}")
        else:
            # Fallback to common metric names
            metrics = ['loss', 'acc']
            print(f"No metrics found in training histories, using defaults: {', '.join(metrics)}")
    
    if styles is None:
        styles = {
            'figure.figsize': (12, 8),
            'lines.linewidth': 2,
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 10,
            'figure.dpi': 300
        }
    
    # Set plotting style
    plt.rcParams.update(styles)
    
    # Check for each metric which key format is used in the histories
    metric_keys = {}
    for metric in metrics:
        # Try variations of the key format
        possible_train_keys = [f'train_{metric}s', f'train_{metric}', f'train_{metric}_history']
        possible_val_keys = [f'val_{metric}s', f'val_{metric}', f'val_{metric}_history']
        
        # Find which key format is actually used
        train_key = None
        val_key = None
        
        # Check first history to determine key format
        if fold_histories:
            first_history = next(iter(fold_histories.values()))
            for key in possible_train_keys:
                if key in first_history:
                    train_key = key
                    break
            
            for key in possible_val_keys:
                if key in first_history:
                    val_key = key
                    break
        
        # If not found, use default format
        if train_key is None:
            train_key = possible_train_keys[0]
        if val_key is None:
            val_key = possible_val_keys[0]
        
        metric_keys[metric] = {
            'train': train_key,
            'val': val_key
        }
        
        print(f"Using keys for {metric}: train='{train_key}', val='{val_key}'")
    
    # ================ #
    # Combined Metrics #
    # ================ #
    
    # 1. Combined Train/Val for each metric in a single figure
    print("Generating combined metric visualizations...")
    
    for metric in metrics:
        plt.figure(figsize=styles['figure.figsize'])
        
        # Get the appropriate data keys based on metric
        train_key = metric_keys[metric]['train']
        val_key = metric_keys[metric]['val']
        
        # Check if we have any valid histories with these keys
        valid_histories = {name: history for name, history in fold_histories.items()
                          if train_key in history and val_key in history}
        
        if not valid_histories:
            print(f"Warning: No valid histories found with keys {train_key} and {val_key}")
            plt.close()
            continue
        
        # Colors for different folds
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_histories)))
        
        # Plot train/val for each fold with consistent colors
        for i, (fold_name, history) in enumerate(valid_histories.items()):
            color = colors[i]
            
            epochs = range(1, len(history[train_key]) + 1)
            
            # Plot training data with solid line
            plt.plot(epochs, history[train_key], color=color, linestyle='-', 
                     label=f'{fold_name} - Train')
            
            # Plot validation data with dashed line
            plt.plot(epochs, history[val_key], color=color, linestyle='--', 
                     label=f'{fold_name} - Val')
        
        # Calculate and plot mean across folds
        calculate_and_plot_mean(valid_histories, train_key, val_key)
        
        plt.title(f'{metric.capitalize()} During Training Across All Folds')
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()}')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', frameon=True, framealpha=0.8, ncol=2)
        
        # Add min/max annotations only if we have < 5 folds to avoid cluttering
        if len(fold_histories) < 5:
            annotate_min_max_values(fold_histories, train_key, val_key)
        
        plt.tight_layout()
        plt.savefig(output_path / f'combined_{metric}_history.png')
        plt.close()
    
    # ============== #
    # Individual Metrics #
    # ============== #
    
    # 2. Separate plots for train and validation metrics
    print("Generating individual train/val visualizations...")
    
    for metric in metrics:
        train_key = f'train_{metric}s'
        val_key = f'val_{metric}s'
        
        # 2.1 Training metrics across folds
        plt.figure(figsize=styles['figure.figsize'])
        
        # Counter to check if we plotted anything
        plot_count = 0
        
        for i, (fold_name, history) in enumerate(fold_histories.items()):
            if train_key in history and len(history[train_key]) > 0:
                epochs = range(1, len(history[train_key]) + 1)
                plt.plot(epochs, history[train_key], label=f'{fold_name}')
                plot_count += 1
        
        # Only add legend and save if we plotted at least one line
        if plot_count > 0:
            plt.title(f'Training {metric.capitalize()} Across All Folds')
            plt.xlabel('Epoch')
            plt.ylabel(f'Training {metric.capitalize()}')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(output_path / f'train_{metric}_comparison.png')
        else:
            print(f"Warning: No valid training {metric} data found to plot")
        
        plt.close()
        
        # 2.2 Validation metrics across folds
        plt.figure(figsize=styles['figure.figsize'])
        
        # Reset counter
        plot_count = 0
        
        for i, (fold_name, history) in enumerate(fold_histories.items()):
            if val_key in history and len(history[val_key]) > 0:
                epochs = range(1, len(history[val_key]) + 1)
                plt.plot(epochs, history[val_key], label=f'{fold_name}')
                plot_count += 1
        
        # Only add legend and save if we plotted at least one line
        if plot_count > 0:
            plt.title(f'Validation {metric.capitalize()} Across All Folds')
            plt.xlabel('Epoch')
            plt.ylabel(f'Validation {metric.capitalize()}')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(output_path / f'val_{metric}_comparison.png')
        else:
            print(f"Warning: No valid validation {metric} data found to plot")
        
        plt.close()
    
    # ===================== #
    # Multi-panel Dashboard #
    # ===================== #
    
    # 3. Create a comprehensive dashboard view
    print("Generating comprehensive dashboard visualization...")
    
    # Create a subplot grid based on number of metrics
    fig, axs = plt.subplots(len(metrics), 2, figsize=(20, 8 * len(metrics)))
    
    # Handle the case of a single metric (makes axs a 1D array)
    if len(metrics) == 1:
        axs = np.array([axs])  # Convert to 2D array with one row
    
    for i, metric in enumerate(metrics):
        train_key = f'train_{metric}s'
        val_key = f'val_{metric}s'
        
        # Counter to check if we plotted anything
        train_plot_count = 0
        val_plot_count = 0
        
        # Left panel: Training metric
        for fold_name, history in fold_histories.items():
            if train_key in history and len(history[train_key]) > 0:
                epochs = range(1, len(history[train_key]) + 1)
                axs[i, 0].plot(epochs, history[train_key], label=fold_name)
                train_plot_count += 1
        
        axs[i, 0].set_title(f'Training {metric.capitalize()}')
        axs[i, 0].set_xlabel('Epoch')
        axs[i, 0].set_ylabel(f'Training {metric.capitalize()}')
        axs[i, 0].grid(True, alpha=0.3)
        
        # Only add legend if we plotted at least one line
        if train_plot_count > 0:
            axs[i, 0].legend(loc='best')
        else:
            axs[i, 0].text(0.5, 0.5, f'No training {metric} data available', 
                         ha='center', va='center', transform=axs[i, 0].transAxes)
        
        # Right panel: Validation metric
        for fold_name, history in fold_histories.items():
            if val_key in history and len(history[val_key]) > 0:
                epochs = range(1, len(history[val_key]) + 1)
                axs[i, 1].plot(epochs, history[val_key], label=fold_name)
                val_plot_count += 1
        
        axs[i, 1].set_title(f'Validation {metric.capitalize()}')
        axs[i, 1].set_xlabel('Epoch')
        axs[i, 1].set_ylabel(f'Validation {metric.capitalize()}')
        axs[i, 1].grid(True, alpha=0.3)
        
        # Only add legend if we plotted at least one line
        if val_plot_count > 0:
            axs[i, 1].legend(loc='best')
        else:
            axs[i, 1].text(0.5, 0.5, f'No validation {metric} data available', 
                         ha='center', va='center', transform=axs[i, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_dashboard.png')
    plt.close()
    
    # ====================== #
    # Convergence Analysis #
    # ====================== #
    
    # 4. Analyze convergence
    print("Generating convergence analysis...")
    
    plt.figure(figsize=styles['figure.figsize'])
    
    # Store best epoch and best metric for each fold
    best_epochs = {}
    
    # Check if we have any val_accs to analyze
    has_valid_data = False
    
    for fold_name, history in fold_histories.items():
        val_acc_key = 'val_accs'
        if val_acc_key in history and len(history[val_acc_key]) > 0:
            has_valid_data = True
            val_accs = history[val_acc_key]
            best_acc_idx = np.argmax(val_accs)
            best_epochs[fold_name] = best_acc_idx + 1  # +1 because epochs are 1-indexed
            
            # Plot vertical line at best epoch
            plt.axvline(x=best_epochs[fold_name], color='gray', linestyle='--', alpha=0.5)
            
            # Annotate best epoch
            plt.text(best_epochs[fold_name], 0.5, f"{fold_name}: {best_epochs[fold_name]}", 
                     rotation=90, verticalalignment='center')
    
    if has_valid_data:
        plt.title('Convergence Analysis: Best Epoch by Fold')
        plt.xlabel('Epoch')
        plt.ylabel('Best Epoch Marker')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max([len(h['val_accs']) for h in fold_histories.values() if 'val_accs' in h]) + 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_path / 'convergence_analysis.png')
        
        # Calculate mean and std of best epochs
        if best_epochs:
            mean_best_epoch = np.mean(list(best_epochs.values()))
            std_best_epoch = np.std(list(best_epochs.values()))
            print(f"Mean best epoch: {mean_best_epoch:.1f} ± {std_best_epoch:.1f}")
    else:
        print("Warning: No validation accuracy data found for convergence analysis")
    
    plt.close()
    
    print(f"All visualizations saved to {output_path}")


def calculate_and_plot_mean(fold_histories, train_key, val_key):
    """Calculate and plot mean performance across folds"""
    
    # Check if we have valid histories with the required keys
    valid_histories = [history for history in fold_histories.values() 
                       if train_key in history and val_key in history]
    
    if not valid_histories:
        print(f"Warning: No valid histories found with keys {train_key} and {val_key}")
        return
    
    # Find the shortest history length among valid histories
    min_length = min([len(history[train_key]) for history in valid_histories])
    
    if min_length == 0:
        print(f"Warning: Found empty training history for {train_key}")
        return
    
    # Count valid histories
    valid_count = len(valid_histories)
    
    # Initialize arrays to hold the values
    train_values = np.zeros((valid_count, min_length))
    val_values = np.zeros((valid_count, min_length))
    
    # Fill in the arrays
    for i, history in enumerate(valid_histories):
        train_values[i, :] = history[train_key][:min_length]
        val_values[i, :] = history[val_key][:min_length]
    
    # Calculate mean and std
    train_mean = np.mean(train_values, axis=0)
    train_std = np.std(train_values, axis=0)
    val_mean = np.mean(val_values, axis=0)
    val_std = np.std(val_values, axis=0)
    
    # Plot mean with shaded std region
    epochs = range(1, min_length + 1)
    
    # Plot training mean
    plt.plot(epochs, train_mean, 'b-', linewidth=2, label='Mean Train')
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')
    
    # Plot validation mean
    plt.plot(epochs, val_mean, 'r-', linewidth=2, label='Mean Val')
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')


def annotate_min_max_values(fold_histories, train_key, val_key):
    """Annotate min and max values on the plot"""
    
    # Find global min/max for validation metric
    all_val_values = []
    for history in fold_histories.values():
        if val_key in history:
            all_val_values.extend(history[val_key])
    
    if not all_val_values:
        return
    
    global_min = min(all_val_values)
    global_max = max(all_val_values)
    
    # Add annotations for each fold's best value
    for fold_name, history in fold_histories.items():
        if val_key in history:
            best_idx = np.argmax(history[val_key])
            best_val = history[val_key][best_idx]
            best_epoch = best_idx + 1
            
            # Only annotate if it's close to the global max
            if best_val > global_max * 0.95:
                plt.annotate(f'{best_val:.4f}',
                            xy=(best_epoch, best_val),
                            xytext=(best_epoch + 1, best_val),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                            fontsize=8)


def generate_latex_tables(fold_histories, output_dir):
    """Generate LaTeX tables summarizing training history"""
    output_path = Path(output_dir)
    
    # Table 1: Convergence metrics (epochs to best validation performance)
    with open(output_path / 'convergence_table.tex', 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Convergence Analysis by Fold}\n")
        f.write("\\begin{tabular}{|l|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Fold} & \\textbf{Best Epoch} & \\textbf{Best Val Acc} & "
                "\\textbf{Final Train Acc} & \\textbf{Train-Val Gap} \\\\ \\hline\n")
        
        best_epochs = []
        best_val_accs = []
        
        for fold_name, history in fold_histories.items():
            if 'val_accs' in history and 'train_accs' in history:
                val_accs = history['val_accs']
                train_accs = history['train_accs']
                
                best_val_idx = np.argmax(val_accs)
                best_epoch = best_val_idx + 1
                best_val_acc = val_accs[best_val_idx]
                
                # Get the corresponding training accuracy
                if best_val_idx < len(train_accs):
                    train_acc_at_best = train_accs[best_val_idx]
                else:
                    train_acc_at_best = train_accs[-1]
                
                # Calculate gap between train and validation
                gap = train_acc_at_best - best_val_acc
                
                f.write(f"{fold_name} & {best_epoch} & {best_val_acc:.4f} & "
                        f"{train_acc_at_best:.4f} & {gap:.4f} \\\\ \\hline\n")
                
                best_epochs.append(best_epoch)
                best_val_accs.append(best_val_acc)
        
        # Add summary row
        if best_epochs:
            mean_epoch = np.mean(best_epochs)
            std_epoch = np.std(best_epochs)
            mean_val_acc = np.mean(best_val_accs)
            std_val_acc = np.std(best_val_accs)
            
            f.write(f"\\textbf{{Mean}} & {mean_epoch:.1f} $\\pm$ {std_epoch:.1f} & "
                    f"{mean_val_acc:.4f} $\\pm$ {std_val_acc:.4f} & - & - \\\\ \\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:convergence}\n")
        f.write("\\end{table}\n")


def main():
    """Main entry point of the script"""
    parser = argparse.ArgumentParser(description='Visualize training histories across folds.')
    parser.add_argument('--base_dir', type=str, default='data/mlp_training_output',
                        help='Base directory containing fold subdirectories')
    parser.add_argument('--output_dir', type=str, default='data/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--generate_latex', action='store_true',
                        help='Generate LaTeX tables with training statistics')
    args = parser.parse_args()
    
    print(f"Loading training histories from {args.base_dir}")
    fold_histories = load_fold_histories(args.base_dir)
    
    if fold_histories:
        print(f"Found {len(fold_histories)} fold histories")
        visualize_fold_histories(fold_histories, args.output_dir)
        
        if args.generate_latex:
            generate_latex_tables(fold_histories, args.output_dir)
    else:
        print("No training histories found. Please check the base directory.")


if __name__ == "__main__":
    main()