"""
Evaluation utilities for the UA-SLSM MLP training pipeline.
Provides functions for model evaluation, metrics computation, and results visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from pathlib import Path
import json
import os
import pandas as pd

def evaluate(model, val_loader, criterion, device, use_amp=False, is_binary=False):
    """
    Evaluate a model on validation data and measure inference time.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader with validation data
        criterion: Loss function
        device: Device to run evaluation on
        use_amp: Whether to use automatic mixed precision
        is_binary: Whether the task is binary classification
        
    Returns:
        Tuple of (loss, accuracy, inference_time_per_batch)
    """
    import time
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_inference_time = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            batch_count += 1
            inference_start_time = time.time()
            features, labels = features.to(device), labels.to(device)
            labels_for_loss = labels.float().view(-1, 1) if is_binary else labels
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(features)
                loss = criterion(outputs, labels_for_loss)
            
            running_loss += loss.item() * features.size(0)
            
            predicted = (torch.sigmoid(outputs) > 0.5).long().view(-1) if is_binary else outputs.max(1)[1]
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Measure inference time for this batch
            inference_time = time.time() - inference_start_time
            total_inference_time += inference_time
    
    # Calculate average inference time per batch
    avg_inference_time_per_batch = total_inference_time / batch_count if batch_count > 0 else 0
    
    return running_loss / total, correct / total, avg_inference_time_per_batch

def compute_roc_data(model, data_loader, device, use_amp=False, is_binary=False):
    """
    Compute ROC curve data for a model.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on
        use_amp: Whether to use automatic mixed precision
        is_binary: Whether the task is binary classification
        
    Returns:
        Tuple of (labels, probabilities, predictions)
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(features)
            
            # Get probabilities and predictions
            if is_binary:
                probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
                preds = (torch.sigmoid(outputs) > 0.5).long().view(-1).cpu().numpy()
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_labels.append(labels.numpy())
            all_probs.append(probs)
            all_preds.append(preds)
    
    # Concatenate all batches
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    
    return labels, probs, preds

def generate_metrics_table(fold_matrices, output_dir, is_binary=True, model_type=""):
    """
    Generate a table of metrics (F1, sensitivity, specificity) across folds.
    
    Args:
        fold_matrices: Dictionary mapping fold names to their confusion matrices
        output_dir: Directory to save the table
        is_binary: Whether the task is binary classification
        model_type: The type of model used (for table title)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Calculate metrics for each fold
    fold_metrics = {}
    for fold_name, cm in fold_matrices.items():
        fold_metrics[fold_name] = calculate_metrics_from_confusion_matrix(cm, is_binary)
    
    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame(columns=['Fold', 'F1', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy'])
    
    for fold_name, metrics in fold_metrics.items():
        metrics_df = metrics_df._append({
            'Fold': fold_name,
            'F1': metrics['f1'],
            'Sensitivity': metrics['sensitivity'],
            'Specificity': metrics['specificity'],
            'Precision': metrics['precision'],
            'Accuracy': metrics['accuracy']
        }, ignore_index=True)
    
    # Calculate mean and standard deviation
    mean_metrics = metrics_df[['F1', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy']].mean()
    std_metrics = metrics_df[['F1', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy']].std()
    
    # Add mean and std to the DataFrame
    metrics_df = metrics_df._append({
        'Fold': 'Mean',
        'F1': mean_metrics['F1'],
        'Sensitivity': mean_metrics['Sensitivity'],
        'Specificity': mean_metrics['Specificity'],
        'Precision': mean_metrics['Precision'],
        'Accuracy': mean_metrics['Accuracy']
    }, ignore_index=True)
    
    metrics_df = metrics_df._append({
        'Fold': 'Std',
        'F1': std_metrics['F1'],
        'Sensitivity': std_metrics['Sensitivity'],
        'Specificity': std_metrics['Specificity'],
        'Precision': std_metrics['Precision'],
        'Accuracy': std_metrics['Accuracy']
    }, ignore_index=True)
    
    # Save as CSV
    metrics_df.to_csv(output_path / f'{model_type}_metrics.csv', index=False)
    
    # Save as LaTeX table
    with open(output_path / f'{model_type}_metrics_table.tex', 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Performance Metrics for {model_type} Across Folds}}\n")
        f.write("\\begin{tabular}{|l|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Fold} & \\textbf{F1} & \\textbf{Sensitivity} & \\textbf{Specificity} & \\textbf{Precision} & \\textbf{Accuracy} \\\\ \\hline\n")
        
        for _, row in metrics_df.iterrows():
            fold = row['Fold']
            f1 = row['F1']
            sens = row['Sensitivity']
            spec = row['Specificity']
            prec = row['Precision']
            acc = row['Accuracy']
            
            if fold in ['Mean', 'Std']:
                f.write(f"\\textbf{{{fold}}} & {f1:.4f} & {sens:.4f} & {spec:.4f} & {prec:.4f} & {acc:.4f} \\\\ \\hline\n")
            else:
                f.write(f"{fold} & {f1:.4f} & {sens:.4f} & {spec:.4f} & {prec:.4f} & {acc:.4f} \\\\ \\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:metrics_" + model_type.replace("_", "") + "}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved metrics table for {model_type} to {output_path}")
    
    return mean_metrics, std_metrics


def generate_tables(fold_results, output_dir):
    """
    Generate LaTeX tables summarizing results.
    
    Args:
        fold_results: Dictionary mapping fold names to their results
        output_dir: Directory to save the tables
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Table 1: Performance metrics by fold
    with open(output_path / 'performance_table.tex', 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Metrics by Fold}\n")
        f.write("\\begin{tabular}{|l|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Fold} & \\textbf{Train Acc} & \\textbf{Val Acc} & \\textbf{AUC} \\\\ \\hline\n")
        
        val_accs = []
        aucs = []
        
        for fold_name, results in fold_results.items():
            train_acc = results['train_acc']
            val_acc = results['val_acc']
            val_auc = results['val_auc']
            
            f.write(f"{fold_name} & {train_acc:.4f} & {val_acc:.4f} & {val_auc:.4f} \\\\ \\hline\n")
            
            val_accs.append(val_acc)
            aucs.append(val_auc)
        
        # Add summary row
        mean_val_acc = np.mean(val_accs)
        std_val_acc = np.std(val_accs)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        f.write(f"\\textbf{{Mean}} & - & {mean_val_acc:.4f} $\\pm$ {std_val_acc:.4f} & "
                f"{mean_auc:.4f} $\\pm$ {std_auc:.4f} \\\\ \\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:performance}\n")
        f.write("\\end{table}\n")
    
    print(f"Generated LaTeX tables in {output_path}")

def plot_roc_curves(folds_data, output_dir):
    """
    Plot ROC curves for all folds.
    
    Args:
        folds_data: Dictionary mapping fold names to their ROC data
        output_dir: Directory to save the plots
        
    Returns:
        Tuple of (mean_auc, std_auc)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(10, 8))
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold_name, data in folds_data.items():
        labels = data['labels']
        probs = data['probs']
        
        # For multi-class, we plot ROC for each class (one-vs-rest)
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            n_classes = probs.shape[1]
            
            # Convert labels to one-hot encoding
            labels_one_hot = np.eye(n_classes)[labels]
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(labels_one_hot[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                         label=f'{fold_name} Class {i} (AUC = {roc_auc:.3f})')
                
                # Interpolate to common FPR grid
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
        else:
            # Binary classification
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label=f'{fold_name} (AUC = {roc_auc:.3f})')
            
            # Interpolate to common FPR grid
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
    
    # Plot mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, 'b-',
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
             lw=2, alpha=0.8)
    
    # Plot standard deviation
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                     label=f'± 1 std. dev.')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    
    plt.savefig(output_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'roc_curves.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROC curves to {output_path}")
    return mean_auc, std_auc

def plot_precision_recall_curves(folds_data, output_dir):
    """
    Plot precision-recall curves for all folds.
    
    Args:
        folds_data: Dictionary mapping fold names to their evaluation data
        output_dir: Directory to save the plots
        
    Returns:
        Tuple of (mean_ap, std_ap)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(10, 8))
    
    precisions = []
    aps = []
    mean_recall = np.linspace(0, 1, 100)
    
    for fold_name, data in folds_data.items():
        labels = data['labels']
        probs = data['probs']
        
        # For multi-class, we plot PR curves for each class
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            n_classes = probs.shape[1]
            
            # Convert labels to one-hot encoding
            labels_one_hot = np.eye(n_classes)[labels]
            
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(labels_one_hot[:, i], probs[:, i])
                ap = average_precision_score(labels_one_hot[:, i], probs[:, i])
                
                plt.plot(recall, precision, lw=1, alpha=0.3,
                         label=f'{fold_name} Class {i} (AP = {ap:.3f})')
                
                # Interpolate to common recall grid
                interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                precisions.append(interp_precision)
                aps.append(ap)
        else:
            # Binary classification
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = average_precision_score(labels, probs)
            
            plt.plot(recall, precision, lw=1, alpha=0.3,
                     label=f'{fold_name} (AP = {ap:.3f})')
            
            # Interpolate to common recall grid
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(interp_precision)
            aps.append(ap)
    
    # Plot mean precision-recall curve
    mean_precision = np.mean(precisions, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    
    plt.plot(mean_recall, mean_precision, 'b-',
             label=f'Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})',
             lw=2, alpha=0.8)
    
    # Plot standard deviation
    std_precision = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=0.2,
                     label=f'± 1 std. dev.')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(alpha=0.3)
    
    plt.savefig(output_path / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'precision_recall_curves.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved precision-recall curves to {output_path}")
    return mean_ap, std_ap


def plot_training_curves(fold_name, history, output_dir):
    """
    Plot training and validation loss/accuracy curves for a single fold.
    
    Args:
        fold_name: Name of the fold
        history: Dictionary containing training history with keys:
                 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        output_dir: Directory to save the plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with two subplots (loss and accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title(f'{fold_name} - Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title(f'{fold_name} - Accuracy Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Mark the best epoch
    best_epoch = history.get('best_epoch', 0)
    if best_epoch > 0:
        ax1.axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
        ax2.axvline(x=best_epoch+1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
        ax1.legend()
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / f'{fold_name}_training_curves.png', dpi=300, bbox_inches='tight')
    # plt.savefig(output_path / f'{fold_name}_training_curves.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves for {fold_name} to {output_path}")


def plot_confusion_matrix(labels, predictions, fold_name, output_dir, class_names=None):
    """
    Generate and plot confusion matrix for a fold.
    
    Args:
        labels: Ground truth labels
        predictions: Model predictions
        fold_name: Name of the fold
        output_dir: Directory to save the plot
        class_names: Optional list of class names for the labels
        
    Returns:
        The confusion matrix
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot the raw counts (separate figure)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'{fold_name} - Confusion Matrix (Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path / f'{fold_name}_confusion_matrix_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot the normalized values (separate figure)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.title(f'{fold_name} - Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path / f'{fold_name}_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrices for {fold_name} to {output_path}")
    
    return cm


def calculate_metrics_from_confusion_matrix(cm, is_binary=True):
    """
    Calculate various metrics from a confusion matrix.
    
    Args:
        cm: Confusion matrix
        is_binary: Whether the task is binary classification
        
    Returns:
        Dictionary of metrics including F1, sensitivity, specificity, etc.
    """
    metrics = {}
    
    if is_binary:
        # For binary classification
        tn, fp, fn, tp = cm.ravel()
        
        # Sensitivity (Recall, True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['sensitivity'] = sensitivity
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = specificity
        
        # Precision (Positive Predictive Value)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['precision'] = precision
        
        # F1 Score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        metrics['f1'] = f1
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics['accuracy'] = accuracy
    else:
        # For multi-class, calculate macro-averaged metrics
        n_classes = cm.shape[0]
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []
        
        for i in range(n_classes):
            # True positives for class i
            tp = cm[i, i]
            
            # False negatives for class i
            fn = np.sum(cm[i, :]) - tp
            
            # False positives for class i
            fp = np.sum(cm[:, i]) - tp
            
            # True negatives for class i
            tn = np.sum(cm) - tp - fp - fn
            
            # Sensitivity for class i
            sensitivity_i = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivities.append(sensitivity_i)
            
            # Specificity for class i
            specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity_i)
            
            # Precision for class i
            precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision_i)
            
            # F1 score for class i
            f1_i = 2 * (precision_i * sensitivity_i) / (precision_i + sensitivity_i) if (precision_i + sensitivity_i) > 0 else 0
            f1_scores.append(f1_i)
        
        # Macro-averaged metrics
        metrics['sensitivity'] = np.mean(sensitivities)
        metrics['specificity'] = np.mean(specificities)
        metrics['precision'] = np.mean(precisions)
        metrics['f1'] = np.mean(f1_scores)
        metrics['accuracy'] = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0
        
        # Store per-class metrics as well
        metrics['per_class'] = {
            'sensitivity': sensitivities,
            'specificity': specificities,
            'precision': precisions,
            'f1': f1_scores
        }
    
    return metrics


def plot_combined_confusion_matrix(fold_matrices, output_dir, class_names=None):
    """
    Generate and plot a combined confusion matrix from all folds.
    
    Args:
        fold_matrices: Dictionary mapping fold names to their confusion matrices
        output_dir: Directory to save the plot
        class_names: Optional list of class names for the labels
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Combine all matrices
    combined_cm = sum(fold_matrices.values())
    
    # Normalize the combined confusion matrix
    combined_cm_normalized = combined_cm.astype('float') / combined_cm.sum(axis=1)[:, np.newaxis]
    
    # Plot the raw counts (separate figure)
    plt.figure(figsize=(6, 4))
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Combined Confusion Matrix (Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path / 'combined_confusion_matrix_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot the normalized values (separate figure)
    plt.figure(figsize=(6, 4))
    sns.heatmap(combined_cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.title('Combined Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path / 'combined_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined confusion matrices to {output_path}")
    
    return combined_cm


def plot_combined_training_curves(fold_histories, output_dir):
    """
    Plot combined training and validation loss/accuracy curves for all folds.
    
    Args:
        fold_histories: Dictionary mapping fold names to their training histories
        output_dir: Directory to save the plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create figure with two subplots (loss and accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors for different folds
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    # Track mean values across folds
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    max_epochs = 0
    
    for i, (fold_name, history) in enumerate(fold_histories.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(history['train_loss']) + 1)
        max_epochs = max(max_epochs, len(epochs))
        
        # Pad histories to ensure they're all the same length
        train_losses = history['train_loss']
        val_losses = history['val_loss']
        train_accs = history['train_acc']
        val_accs = history['val_acc']
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)
        
        # Plot individual fold curves with low alpha
        ax1.plot(epochs, train_losses, color=color, linestyle='-', alpha=0.3)
        ax1.plot(epochs, val_losses, color=color, linestyle='--', alpha=0.3)
        
        ax2.plot(epochs, train_accs, color=color, linestyle='-', alpha=0.3)
        ax2.plot(epochs, val_accs, color=color, linestyle='--', alpha=0.3)
    
    # Compute means (pad shorter sequences with NaN)
    def pad_and_stack(sequences, max_len):
        padded = []
        for seq in sequences:
            padded_seq = seq + [float('nan')] * (max_len - len(seq))
            padded.append(padded_seq)
        return np.array(padded)
    
    # Pad sequences and compute means
    if fold_histories:
        train_losses_padded = pad_and_stack(all_train_losses, max_epochs)
        val_losses_padded = pad_and_stack(all_val_losses, max_epochs)
        train_accs_padded = pad_and_stack(all_train_accs, max_epochs)
        val_accs_padded = pad_and_stack(all_val_accs, max_epochs)
        
        # Compute means ignoring NaN values
        mean_train_losses = np.nanmean(train_losses_padded, axis=0)
        mean_val_losses = np.nanmean(val_losses_padded, axis=0)
        mean_train_accs = np.nanmean(train_accs_padded, axis=0)
        mean_val_accs = np.nanmean(val_accs_padded, axis=0)
        
        # Plot mean curves
        epochs = range(1, max_epochs + 1)
        ax1.plot(epochs, mean_train_losses, 'b-', linewidth=2, label='Mean Train Loss')
        ax1.plot(epochs, mean_val_losses, 'r-', linewidth=2, label='Mean Val Loss')
        
        ax2.plot(epochs, mean_train_accs, 'b-', linewidth=2, label='Mean Train Acc')
        ax2.plot(epochs, mean_val_accs, 'r-', linewidth=2, label='Mean Val Acc')
    
    # Set titles and labels
    ax1.set_title('Loss Curves (All Folds)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_title('Accuracy Curves (All Folds)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'combined_training_curves.png', dpi=300, bbox_inches='tight')
    # plt.savefig(output_path / 'combined_training_curves.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined training curves to {output_path}")
