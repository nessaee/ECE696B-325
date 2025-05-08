#!/usr/bin/env python3
"""
Monte Carlo-based 5-fold cross-validation for SLSM dataset.

This script creates balanced folds for cross-validation while ensuring:
1. All images from a study are in the same fold (to prevent data leakage)
2. Class distribution (HSIL, LSIL, NSA) is balanced across folds
3. Number of images is balanced across folds
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import time

# Path to the new annotation format files
ANNOTATIONS_FILE = '../../data/dataset/labels/mlp/annotations.csv'
METADATA_FILE = '../../data/dataset/labels/mlp/metadata.csv'

# Constants
NUM_FOLDS = 5
NUM_MONTE_CARLO_ITERATIONS = 10000
OUTPUT_DIR = Path(__file__).parent / 'output'

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
def load_filtered_data():
    """
    Load the filtered dataset with the new annotation format
    """
    print(f"Loading annotations from: {ANNOTATIONS_FILE}")
    annotations_df = pd.read_csv(ANNOTATIONS_FILE)
    
    print(f"Loading metadata from: {METADATA_FILE}")
    metadata_df = pd.read_csv(METADATA_FILE)
    
    # Merge annotations with metadata
    df = pd.merge(annotations_df, metadata_df, on='image_path', how='inner')
    
    print(f"Loaded {len(df)} samples from {df['study_id'].nunique()} studies")
    return df


def get_study_statistics(df):
    """
    Compute statistics for each study to be used for splitting
    
    Returns:
        pandas.DataFrame: DataFrame with study-level statistics
    """
    # Create a DataFrame to hold study-level statistics
    study_stats = pd.DataFrame()
    
    # Basic counts
    study_stats['image_count'] = df.groupby('study_id').size()
    study_stats['biopsy_count'] = df.groupby('study_id')['biopsy_num'].nunique()
    
    # Count images by class (exact match since new format has clean labels)
    hsil_counts = df[df['label'] == 'HSIL'].groupby('study_id').size()
    study_stats['hsil_count'] = hsil_counts.reindex(study_stats.index, fill_value=0)
    
    lsil_counts = df[df['label'] == 'LSIL'].groupby('study_id').size()
    study_stats['lsil_count'] = lsil_counts.reindex(study_stats.index, fill_value=0)
    
    nsa_counts = df[df['label'] == 'NSA'].groupby('study_id').size()
    study_stats['nsa_count'] = nsa_counts.reindex(study_stats.index, fill_value=0)
    
    # Calculate class proportions for each study
    total = study_stats['image_count']
    study_stats['hsil_prop'] = study_stats['hsil_count'] / total
    study_stats['lsil_prop'] = study_stats['lsil_count'] / total
    study_stats['nsa_prop'] = study_stats['nsa_count'] / total
    
    # Add exposure time statistics
    expo_time_mean = df.groupby('study_id')['expo_time'].mean()
    study_stats['expo_time_mean'] = expo_time_mean.reindex(study_stats.index, fill_value=0)
    
    return study_stats


def evaluate_fold_balance(folds, study_stats, weight_image=1.0, weight_hsil=2.0, weight_lsil=2.0, weight_nsa=2.0):
    """
    Evaluate how balanced the folds are in terms of class distribution and sample count.
    Lower score is better (more balanced).
    
    Args:
        folds (list): List of lists, where each inner list contains study IDs for a fold
        study_stats (pandas.DataFrame): DataFrame with study-level statistics
        weight_image (float): Weight for image count balance (default: 1.0)
        weight_hsil (float): Weight for HSIL class balance (default: 2.0)
        weight_lsil (float): Weight for LSIL class balance (default: 2.0)
        weight_nsa (float): Weight for NSA class balance (default: 2.0)
    
    Returns:
        float: Imbalance score (lower is better)
    """
    # Initialize arrays to hold counts for each fold
    fold_stats = np.zeros((NUM_FOLDS, 4))  # [image_count, hsil_count, lsil_count, nsa_count]
    
    # Calculate statistics for each fold
    for i, fold in enumerate(folds):
        fold_data = study_stats.loc[fold]
        fold_stats[i, 0] = fold_data['image_count'].sum()  # Total images
        fold_stats[i, 1] = fold_data['hsil_count'].sum()   # HSIL images
        fold_stats[i, 2] = fold_data['lsil_count'].sum()   # LSIL images
        fold_stats[i, 3] = fold_data['nsa_count'].sum()    # NSA images
    
    # Calculate coefficient of variation (std/mean) for each metric
    # This gives us a normalized measure of dispersion
    cv_image_count = np.std(fold_stats[:, 0]) / np.mean(fold_stats[:, 0]) if np.mean(fold_stats[:, 0]) > 0 else 0
    cv_hsil = np.std(fold_stats[:, 1]) / np.mean(fold_stats[:, 1]) if np.mean(fold_stats[:, 1]) > 0 else 0
    cv_lsil = np.std(fold_stats[:, 2]) / np.mean(fold_stats[:, 2]) if np.mean(fold_stats[:, 2]) > 0 else 0
    cv_nsa = np.std(fold_stats[:, 3]) / np.mean(fold_stats[:, 3]) if np.mean(fold_stats[:, 3]) > 0 else 0
    
    # Weight the metrics based on importance using configurable weights
    imbalance_score = (weight_image * cv_image_count + 
                       weight_hsil * cv_hsil + 
                       weight_lsil * cv_lsil + 
                       weight_nsa * cv_nsa)
    
    # Create a dictionary with detailed scores for analysis and logging
    detailed_scores = {
        'cv_image_count': cv_image_count,
        'cv_hsil': cv_hsil,
        'cv_lsil': cv_lsil,
        'cv_nsa': cv_nsa,
        'total_score': imbalance_score
    }
    
    return imbalance_score, detailed_scores


def create_monte_carlo_split(study_stats, num_iterations=NUM_MONTE_CARLO_ITERATIONS, weight_image=1.0, weight_hsil=2.0, weight_lsil=2.0, weight_nsa=2.0):
    """
    Create 5-fold CV split using Monte Carlo sampling to optimize balance.
    
    Args:
        study_stats (pandas.DataFrame): DataFrame with study-level statistics
        num_iterations (int): Number of Monte Carlo iterations
        weight_image (float): Weight for image count balance
        weight_hsil (float): Weight for HSIL class balance
        weight_lsil (float): Weight for LSIL class balance
        weight_nsa (float): Weight for NSA class balance
    
    Returns:
        tuple: (folds, metrics_data)
            - folds: List of lists, where each inner list contains study IDs for a fold
            - metrics_data: DataFrame with metrics progression (iteration, score, detailed_scores)
    """
    study_ids = study_stats.index.tolist()
    
    # Initialize metrics tracking
    metrics_data = []
    detailed_metrics_history = []
    
    # Initial split: random assignment of studies to folds
    best_folds = [[] for _ in range(NUM_FOLDS)]
    random.shuffle(study_ids)
    for i, study_id in enumerate(study_ids):
        fold_idx = i % NUM_FOLDS
        best_folds[fold_idx].append(study_id)
    
    best_score, detailed_scores = evaluate_fold_balance(
        best_folds, study_stats, 
        weight_image=weight_image, 
        weight_hsil=weight_hsil, 
        weight_lsil=weight_lsil, 
        weight_nsa=weight_nsa
    )
    print(f"Initial random split score: {best_score:.4f} (weights: image={weight_image}, hsil={weight_hsil}, lsil={weight_lsil}, nsa={weight_nsa})")
    print(f"Detailed initial scores: {detailed_scores}")
    
    # Add initial score to metrics
    metrics_data.append({'iteration': 0, 'score': best_score})
    detailed_metrics_history.append({'iteration': 0, **detailed_scores})
    
    # Monte Carlo optimization
    start_time = time.time()
    improvements = 0
    last_improved_iteration = 0
    
    for iteration in range(1, num_iterations + 1):
        # Progress update every 1000 iterations
        if iteration % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {iteration}/{num_iterations}, Best score: {best_score:.4f}, "  
                  f"Improvements: {improvements}, Time elapsed: {elapsed:.2f}s")
        
        # Create a new candidate split by switching two random studies between folds
        candidate_folds = [fold.copy() for fold in best_folds]
        
        # Pick two random folds
        fold1, fold2 = random.sample(range(NUM_FOLDS), 2)
        
        # Only try to swap if both folds have at least one study
        if candidate_folds[fold1] and candidate_folds[fold2]:
            # Pick one random study from each fold
            study1 = random.choice(candidate_folds[fold1])
            study2 = random.choice(candidate_folds[fold2])
            
            # Swap the studies
            candidate_folds[fold1].remove(study1)
            candidate_folds[fold1].append(study2)
            candidate_folds[fold2].remove(study2)
            candidate_folds[fold2].append(study1)
            
            # Evaluate the new split
            candidate_score, candidate_detailed = evaluate_fold_balance(
                candidate_folds, study_stats,
                weight_image=weight_image, 
                weight_hsil=weight_hsil, 
                weight_lsil=weight_lsil, 
                weight_nsa=weight_nsa
            )
            
            # If it's better, keep it
            if candidate_score < best_score:
                best_folds = candidate_folds
                best_score = candidate_score
                detailed_scores = candidate_detailed
                improvements += 1
                last_improved_iteration = iteration
                
                # Add to metrics when score improves
                metrics_data.append({'iteration': iteration, 'score': best_score})
                detailed_metrics_history.append({'iteration': iteration, **detailed_scores})
        
        # Periodically add data points even if no improvement
        # This helps visualize plateaus in optimization
        if iteration % 500 == 0 and iteration != last_improved_iteration:
            metrics_data.append({'iteration': iteration, 'score': best_score})
            detailed_metrics_history.append({'iteration': iteration, **detailed_scores})
    
    elapsed = time.time() - start_time
    print(f"\nOptimization completed:")
    print(f"Final score: {best_score:.4f}")
    print(f"Final detailed scores: {detailed_scores}")
    print(f"Total improvements: {improvements}")
    print(f"Total time: {elapsed:.2f}s")
    
    # Convert metrics data to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    detailed_metrics_df = pd.DataFrame(detailed_metrics_history)
    
    return best_folds, metrics_df, detailed_metrics_df


def visualize_optimization_progress(metrics_df, detailed_metrics_df, output_dir, weights=None):
    """
    Visualize the progression of optimization metrics over iterations.
    
    Args:
        metrics_df (pandas.DataFrame): DataFrame with metrics progression data
        detailed_metrics_df (pandas.DataFrame): DataFrame with detailed metrics
        output_dir (Path): Directory to save visualization files
        weights (dict): Dictionary containing the weights used for optimization
    """
    # Set up matplotlib to use a serif font (DejaVu Serif is commonly available on Linux)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    
    # Create weight information string if weights are provided
    weight_info = ''
    if weights:
        # Using LaTeX notation for subscripts in matplotlib
        weight_info = f'$(n_{{img}}, n_{{hsil}}, n_{{lsil}}, n_{{nsa}}) = ({weights["weight_image"]}, {weights["weight_hsil"]}, '\
                    f'{weights["weight_lsil"]}, {weights["weight_nsa"]})$'
    
    # Combined plot with dual y-axes for overall score and component metrics
    if len(detailed_metrics_df) > 0 and 'cv_image_count' in detailed_metrics_df.columns:
        fig, ax1 = plt.subplots(figsize=(6,4))
        
        # Primary y-axis: Overall imbalance score (right side)
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        # ax1.set_xscale('log')  # Use logarithmic scale for x-axis
        ax1.set_ylabel('Imbalance Score', color=color)
        main_line = ax1.plot(metrics_df['iteration'], metrics_df['score'], 
                           color=color, marker='o', linestyle='-', markersize=4, 
                           linewidth=2.5, label='Overall Score')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Add first and last points for reference
        # first_point = metrics_df.iloc[0]
        # last_point = metrics_df.iloc[-1]
        # ax1.text(0, first_point['score'], f'{first_point["score"]:.4f}', 
        #         ha='left', va='bottom', color=color)
        # ax1.text(last_point['iteration'], last_point['score'], f'{last_point["score"]:.4f}', 
        #         ha='right', va='top', color=color)
        
        # Secondary y-axis: Component metrics (left side)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Component CV Values')
        
        # Plot each component with transparency
        line1 = ax2.plot(detailed_metrics_df['iteration'], detailed_metrics_df['cv_image_count'], 
                       marker='o', linestyle='-', markersize=3, alpha=0.2, 
                       label='Image Count CV', color='tab:gray')
        line2 = ax2.plot(detailed_metrics_df['iteration'], detailed_metrics_df['cv_hsil'], 
                       marker='s', linestyle='-', markersize=3, alpha=0.2, 
                       label='HSIL CV', color='tab:red')
        line3 = ax2.plot(detailed_metrics_df['iteration'], detailed_metrics_df['cv_lsil'], 
                       marker='^', linestyle='-', markersize=3, alpha=0.2, 
                       label='LSIL CV', color='tab:orange')
        line4 = ax2.plot(detailed_metrics_df['iteration'], detailed_metrics_df['cv_nsa'], 
                       marker='D', linestyle='-', markersize=3, alpha=0.2, 
                       label='NSA CV', color='tab:green')
        
        # Add a grid that's linked to the secondary axis
        # ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Combine all lines for the legend
        lines = main_line + line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        
        # Add title with weight information if available
        if weight_info:
            plt.title(f'Optimization Progress and Component Metrics\n{weight_info}', fontsize=11)
        else:
            plt.title('Optimization Progress and Component Metrics')
        
        # Add legend below the plot
        plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=True, shadow=True)
        
        # Instead of tight_layout, manually set margins
        fig.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.9)
        plt.savefig(output_dir / 'combined_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # Also save the individual plots for reference
    plt.figure(figsize=(6,4))
    plt.plot(metrics_df['iteration'], metrics_df['score'], marker='o', linestyle='-', markersize=4)
    plt.xlabel('Iteration')
    plt.xscale('log')  # Use logarithmic scale for x-axis
    plt.ylabel('Imbalance Score')
    
    # Include weight information in title if available
    if weight_info:
        plt.title(f'Optimization Progress\n{weight_info}', fontsize=10)
    else:
        plt.title('Optimization Progress')
        
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add first and last points for reference
    first_point = metrics_df.iloc[0]
    last_point = metrics_df.iloc[-1]
    plt.text(0, first_point['score'], f'{first_point["score"]:.4f}', ha='left', va='bottom')
    plt.text(last_point['iteration'], last_point['score'], f'{last_point["score"]:.4f}', ha='right', va='top')
    
    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    plt.savefig(output_dir / 'optimization_progress.png', dpi=300)
    plt.close()
    
    # Save metrics data to CSV for further analysis
    metrics_df.to_csv(output_dir / 'optimization_metrics.csv', index=False)
    if len(detailed_metrics_df) > 0:
        detailed_metrics_df.to_csv(output_dir / 'detailed_metrics.csv', index=False)


def visualize_splits(folds, study_stats, output_dir):
    """
    Create visualizations of the fold distributions.
    
    Args:
        folds (list): List of lists, where each inner list contains study IDs for a fold
        study_stats (pandas.DataFrame): DataFrame with study-level statistics
        output_dir (Path): Directory to save visualization files
    """
    ensure_dir(output_dir)
    
    # Calculate statistics for each fold
    fold_data = []
    for i, fold in enumerate(folds):
        fold_stats = study_stats.loc[fold]
        fold_data.append({
            'fold': i+1,
            'studies': len(fold),
            'images': fold_stats['image_count'].sum(),
            'hsil': fold_stats['hsil_count'].sum(),
            'lsil': fold_stats['lsil_count'].sum(),
            'nsa': fold_stats['nsa_count'].sum()
        })
    
    fold_df = pd.DataFrame(fold_data)
    fold_df['hsil_pct'] = fold_df['hsil'] / fold_df['images'] * 100
    fold_df['lsil_pct'] = fold_df['lsil'] / fold_df['images'] * 100
    fold_df['nsa_pct'] = fold_df['nsa'] / fold_df['images'] * 100
    
    # Plot 1: Sample counts per fold
    plt.figure(figsize=(6,4))
    bars = plt.bar(fold_df['fold'], fold_df['images'])
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() -15, 
                str(int(fold_df.iloc[i]['images'])), 
                ha='center', va='bottom')
    plt.xlabel('Fold')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Fold')
    plt.xticks(fold_df['fold'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'images_per_fold.png', dpi=300)
    plt.close()
    
    # Plot 2: Label distribution per fold
    plt.figure(figsize=(6,4))
    bar_width = 0.25
    x = np.arange(len(fold_df))
    
    # Create the bars and store their references
    hsil_bars = plt.bar(x - bar_width, fold_df['hsil'], width=bar_width, label='HSIL', color='#ff7f0e')
    lsil_bars = plt.bar(x, fold_df['lsil'], width=bar_width, label='LSIL', color='#2ca02c')
    nsa_bars = plt.bar(x + bar_width, fold_df['nsa'], width=bar_width, label='NSA', color='#1f77b4')
    
    # Add count labels on top of each bar
    for i, bar in enumerate(hsil_bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()-4, 
                str(int(fold_df.iloc[i]['hsil'])), 
                ha='center', va='bottom', fontsize=9)
    
    for i, bar in enumerate(lsil_bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()-4, 
                str(int(fold_df.iloc[i]['lsil'])), 
                ha='center', va='bottom', fontsize=9)
    
    for i, bar in enumerate(nsa_bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()-4, 
                str(int(fold_df.iloc[i]['nsa'])), 
                ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Fold')
    plt.ylabel('Number of Images')
    plt.title('Label Distribution Across Folds')
    plt.xticks(x, fold_df['fold'])
    plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'label_distribution.png', dpi=300)
    plt.close()
    
    # Plot 3: Proportional label distribution
    plt.figure(figsize=(6,4))
    x = np.arange(len(fold_df))
    
    plt.bar(x - bar_width, fold_df['hsil_pct'], width=bar_width, label='HSIL', color='#ff7f0e')
    plt.bar(x, fold_df['lsil_pct'], width=bar_width, label='LSIL', color='#2ca02c')
    plt.bar(x + bar_width, fold_df['nsa_pct'], width=bar_width, label='NSA', color='#1f77b4')
    
    plt.xlabel('Fold')
    plt.ylabel('Percentage of Images (%)')
    plt.title('Label Percentage Distribution Across Folds')
    plt.xticks(x, fold_df['fold'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'label_percentage.png', dpi=300)
    plt.close()
    
    # Save fold statistics to CSV
    fold_df.to_csv(output_dir / 'fold_statistics.csv', index=False)
    print(f"Saved fold statistics to {output_dir / 'fold_statistics.csv'}")
    
    # Print fold statistics
    print("\nFold Statistics:")
    print(fold_df[['fold', 'studies', 'images', 'hsil', 'lsil', 'nsa']].to_string(index=False))
    print("\nPercentage Distribution:")
    print(fold_df[['fold', 'hsil_pct', 'lsil_pct', 'nsa_pct']].round(1).to_string(index=False))


def save_splits(folds, df, output_dir):
    """
    Save the 5-fold splits to disk.
    
    Args:
        folds (list): List of lists, where each inner list contains study IDs for a fold
        df (pandas.DataFrame): Original dataframe with all samples
        output_dir (Path): Directory to save split files
    """
    ensure_dir(output_dir)
    
    # Add a 'fold' column to the dataframe
    df['fold'] = -1  # Initialize with -1 (no fold assigned)
    
    # Assign fold numbers based on study_id
    for fold_idx, studies in enumerate(folds):
        df.loc[df['study_id'].isin(studies), 'fold'] = fold_idx
    
    # Verify all samples have a fold assigned
    assert (df['fold'] >= 0).all(), "Error: Some samples do not have a fold assigned"
    
    # Save the full dataset with fold assignments
    df.to_csv(output_dir / 'all_folds.csv', index=False)
    print(f"Saved full dataset with fold assignments to {output_dir / 'all_folds.csv'}")
    
    # Save individual fold files
    # for fold_idx in range(NUM_FOLDS):
    #     # Create test set (current fold)
    #     test_fold = df[df['fold'] == fold_idx]
    #     test_fold.to_csv(output_dir / f'fold{fold_idx+1}_test.csv', index=False)
        
    #     # Create training set (all other folds)
    #     train_fold = df[df['fold'] != fold_idx]
    #     train_fold.to_csv(output_dir / f'fold{fold_idx+1}_train.csv', index=False)
    
    print(f"Saved individual fold files to {output_dir}")
    
    # Save the study-to-fold mapping as JSON for reference
    fold_dict = {}
    for fold_idx, studies in enumerate(folds):
        fold_dict[f"fold{fold_idx+1}"] = sorted(studies)
    
    with open(output_dir / 'study_fold_assignments.json', 'w') as f:
        json.dump(fold_dict, f, indent=2)
    
    print(f"Saved study fold assignments to {output_dir / 'study_fold_assignments.json'}")
    
    # Save image-based fold assignments in the same format as the reference file
    image_fold_dict = {}
    for fold_idx in range(NUM_FOLDS):
        # Get all images for this fold
        fold_images = df[df['fold'] == fold_idx]['image_path'].tolist()
        image_fold_dict[f"fold{fold_idx+1}"] = sorted(fold_images)
    
    with open(output_dir / 'fold_assignments.json', 'w') as f:
        json.dump(image_fold_dict, f, indent=2)
    
    print(f"Saved image fold assignments to {output_dir / 'fold_assignments.json'}")
    
    # Also save a copy to the data directory for easy access
    data_dir = Path('../../data/dataset/labels/mlp')
    with open(data_dir / 'fold_assignments.json', 'w') as f:
        json.dump(image_fold_dict, f, indent=2)
    
    print(f"Saved image fold assignments to {data_dir / 'fold_assignments.json'}")


def main():
    """
    Main function to create 5-fold cross-validation splits.
    """
    parser = argparse.ArgumentParser(description='Create 5-fold CV splits for SLSM dataset')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Output directory (default: ./output)')
    parser.add_argument('--iterations', type=int, default=NUM_MONTE_CARLO_ITERATIONS,
                        help=f'Number of Monte Carlo iterations (default: {NUM_MONTE_CARLO_ITERATIONS})')
    parser.add_argument('--weight-image', type=float, default=1.0,
                        help='Weight for image count balance (default: 1.0)')
    parser.add_argument('--weight-hsil', type=float, default=2.0,
                        help='Weight for HSIL class balance (default: 2.0)')
    parser.add_argument('--weight-lsil', type=float, default=2.0,
                        help='Weight for LSIL class balance (default: 2.0)')
    parser.add_argument('--weight-nsa', type=float, default=2.0,
                        help='Weight for NSA class balance (default: 2.0)')
    
    args = parser.parse_args()
    
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = OUTPUT_DIR
    
    ensure_dir(base_output_dir)
    
    # Log the weights being used
    print(f"Using balance weights: image={args.weight_image}, hsil={args.weight_hsil}, lsil={args.weight_lsil}, nsa={args.weight_nsa}")
    
    # Create a subfolder based on the weights used
    weights_folder_name = f"weights_{args.weight_image}_{args.weight_hsil}_{args.weight_lsil}_{args.weight_nsa}"
    output_dir = base_output_dir / weights_folder_name
    ensure_dir(output_dir)
    print(f"Saving results to: {output_dir}")
    
    # Load and process data
    df = load_filtered_data()
    study_stats = get_study_statistics(df)
    print(f"\nCalculated statistics for {len(study_stats)} studies")
    
    # Step 3: Create optimized splits using Monte Carlo sampling with metrics tracking
    folds, metrics_df, detailed_metrics_df = create_monte_carlo_split(
        study_stats, 
        num_iterations=args.iterations,
        weight_image=args.weight_image,
        weight_hsil=args.weight_hsil,
        weight_lsil=args.weight_lsil,
        weight_nsa=args.weight_nsa
    )
    
    # Step 4: Visualize the optimization progress
    weights = {
        'weight_image': args.weight_image,
        'weight_hsil': args.weight_hsil,
        'weight_lsil': args.weight_lsil,
        'weight_nsa': args.weight_nsa
    }
    visualize_optimization_progress(metrics_df, detailed_metrics_df, output_dir, weights=weights)
    
    # Step 5: Visualize the splits
    visualize_splits(folds, study_stats, output_dir)
    
    # Step 6: Save the splits to disk
    save_splits(folds, df, output_dir)
    
    # Save weight configuration for reference
    with open(output_dir / 'weight_config.json', 'w') as f:
        json.dump({
            'weight_image': args.weight_image,
            'weight_hsil': args.weight_hsil,
            'weight_lsil': args.weight_lsil,
            'weight_nsa': args.weight_nsa,
            'annotations_file': ANNOTATIONS_FILE,
            'metadata_file': METADATA_FILE
        }, f, indent=2)
    
    print("\nSplit creation complete!")


if __name__ == "__main__":
    main()