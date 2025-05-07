"""
Evaluation utilities for the UA-SLSM MLP training pipeline.
Provides functions for model evaluation, metrics computation, and results visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path
import json

def evaluate(model, val_loader, criterion, device, use_amp=False, is_binary=False):
    """
    Evaluate a model on validation data.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader with validation data
        criterion: Loss function
        device: Device to run evaluation on
        use_amp: Whether to use automatic mixed precision
        is_binary: Whether the task is binary classification
        
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            labels_for_loss = labels.float().view(-1, 1) if is_binary else labels
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(features)
                loss = criterion(outputs, labels_for_loss)
            
            running_loss += loss.item() * features.size(0)
            
            predicted = (torch.sigmoid(outputs) > 0.5).long().view(-1) if is_binary else outputs.max(1)[1]
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, correct / total

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
        Tuple of (labels, probabilities)
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(features)
            
            # Get probabilities
            if is_binary:
                probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_labels.append(labels.numpy())
            all_probs.append(probs)
    
    # Concatenate all batches
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    
    return labels, probs

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
