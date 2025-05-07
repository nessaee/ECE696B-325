"""
Main training script for the UA-SLSM MLP training pipeline.
Handles model training, evaluation, and hyperparameter optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import json
import logging
import optuna
from sklearn.metrics import roc_curve, auc, roc_auc_score

from config import (
    BASE_DATA_DIR, FEATURES_DIR, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, MLP_HIDDEN_DIMS, DROPOUT_RATE, NUM_EPOCHS, SEED, USE_AMP,
    NUM_WORKERS, PIN_MEMORY, PATIENCE, MIN_DELTA
)
from utils import set_seed, get_device, save_json
from model import MLPClassifier
from evaluation import evaluate, compute_roc_data, generate_tables, plot_roc_curves


class FeatureDataset(torch.utils.data.Dataset):
    """
    Dataset for precomputed features extracted from images.
    
    Args:
        features: Tensor of precomputed features [N, feature_dim]
        labels: Tensor of corresponding labels [N]
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# MLPClassifier has been moved to model.py


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, 
                   epoch, num_epochs, use_amp=False, scaler=None, is_binary=False, grad_clip=None):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        labels_for_loss = labels.float().view(-1, 1) if is_binary else labels
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(features)
            loss = criterion(outputs, labels_for_loss)
        
        optimizer.zero_grad()
        
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item() * features.size(0)
        
        predicted = (torch.sigmoid(outputs) > 0.5).long().view(-1) if is_binary else outputs.max(1)[1]
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                 f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
    
    return running_loss / total, correct / total


def train_fold(fold_name, features_dir=None, output_dir=None, hidden_dims=None, dropout_rate=None, 
               batch_size=None, num_epochs=None, learning_rate=None, weight_decay=None, 
               device=None, use_amp=None, data_version=None):
    """
    Train a model for one fold.
    
    Args:
        fold_name: Name of the fold to train on
        features_dir: Directory containing precomputed features
        output_dir: Directory to save outputs
        hidden_dims: List of hidden dimensions for the MLP
        dropout_rate: Dropout rate for the MLP
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Tuple of (model, train_accuracy, validation_accuracy, validation_auc)
    """
    # Get data version from parameter or environment variable
    import os
    data_version = data_version or os.environ.get("DATA_VERSION", "normalized")
    
    # Use default values from config if not provided
    if features_dir is None:
        features_dir = Path(FEATURES_DIR) / data_version
    else:
        features_dir = Path(features_dir)
    
    # Create version-specific output directory
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR) / data_version / fold_name
    else:
        output_dir = Path(output_dir)
    hidden_dims = MLP_HIDDEN_DIMS if hidden_dims is None else hidden_dims
    dropout_rate = DROPOUT_RATE if dropout_rate is None else dropout_rate
    batch_size = BATCH_SIZE["train"] if batch_size is None else batch_size
    num_epochs = NUM_EPOCHS if num_epochs is None else num_epochs
    learning_rate = LEARNING_RATE if learning_rate is None else learning_rate
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    device = get_device() if device is None else device
    use_amp = USE_AMP if use_amp is None else use_amp
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load features
    train_features_path = features_dir / f"{fold_name}_train_features.pt"
    val_features_path = features_dir / f"{fold_name}_val_features.pt"
    
    if not train_features_path.exists():
        raise FileNotFoundError(f"Training features not found at {train_features_path}")
    
    if not val_features_path.exists():
        logging.warning(f"Validation features not found at {val_features_path}. Using training data for validation.")
        val_features_path = train_features_path
    
    # Load data
    train_data = torch.load(train_features_path)
    train_features = train_data['features']
    train_labels = train_data['labels']
    feature_dim = train_features.shape[1]
    num_classes = train_data['num_classes']
    
    # Create datasets
    train_dataset = FeatureDataset(train_features, train_labels)
    
    val_data = torch.load(val_features_path)
    val_features = val_data['features']
    val_labels = val_data['labels']
    val_dataset = FeatureDataset(val_features, val_labels)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0
    )
    
    # Initialize model
    model = MLPClassifier(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(device)
    
    # Loss function based on number of classes
    is_binary = (num_classes == 2)
    
    # Print class distribution
    class_counts = torch.bincount(train_labels)
    print(f"Class distribution in training set: {class_counts.tolist()}")
    
    # Loss function with class weights
    if is_binary:
        pos_weight = torch.tensor([2.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        fixed_weights = torch.ones(num_classes) * 1.8
        fixed_weights[0] = 0.9
        criterion = nn.CrossEntropyLoss(weight=fixed_weights.to(device))
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0002,
        betas=(0.9, 0.999),
        eps=1.0e-08,
        weight_decay=1.0e-05
    )
    
    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
        anneal_strategy='cos'
    )
    
    # Setup AMP
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    grad_clip = 0.5
    
    # Training history
    history = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Starting training for {fold_name}:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Model: MLP with hidden dims {hidden_dims}, dropout {dropout_rate}")
    print(f"  - Training: {num_epochs} epochs, batch size {batch_size}, lr {learning_rate}, weight decay {weight_decay}")
    print(f"  - Device: {device}, AMP: {use_amp}")
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, 
            epoch, num_epochs, use_amp, scaler, is_binary
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp, is_binary)
        
        # Update history
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            # Final evaluation on validation set
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp, is_binary)
            best_val_acc = val_acc  # Update the best validation accuracy
            best_epoch = epoch
            best_model_state = model.state_dict().copy()  # Save the current model state
            history['best_epoch'] = epoch
            history['best_val_acc'] = val_acc
            print(f"New best model with validation accuracy: {val_acc:.4f}")
    
    # Save training history
    save_json(history, output_dir / 'training_history.json')
    
    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, output_dir / 'best_model.pt')
        model.load_state_dict(best_model_state)  # Load best model for final evaluation
    
    # Compute ROC curve data
    try:
        labels, probs = compute_roc_data(model, val_loader, device, use_amp, is_binary)
        
        # For binary classification
        if is_binary or (isinstance(probs, np.ndarray) and len(probs.shape) == 1):
            # Check if we have both positive and negative samples
            if len(np.unique(labels)) > 1:
                fpr, tpr, _ = roc_curve(labels, probs)
                roc_auc = auc(fpr, tpr)
            else:
                logging.warning(f"Cannot compute ROC curve: only one class present in validation set")
                roc_auc = 0.0
        else:
            # For multi-class, compute micro-average ROC
            try:
                fpr, tpr, _ = roc_curve(labels.ravel(), probs.ravel())
                roc_auc = auc(fpr, tpr)
            except Exception as e:
                logging.warning(f"Error computing multi-class ROC: {e}")
                roc_auc = 0.0
    except Exception as e:
        logging.warning(f"Error computing ROC curve: {e}")
        roc_auc = 0.0
    
    # Get the final training accuracy from history
    final_train_acc = history['train_accs'][-1] if history['train_accs'] else 0.0
    
    print(f"\nTraining completed for {fold_name}")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return model, final_train_acc, best_val_acc, roc_auc


def objective(trial, fold_name, features_dir, output_dir, device, use_amp, num_epochs=100, seed=42):
    """Objective function for Optuna hyperparameter optimization"""
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Sample hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Sample network architecture
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dims = [trial.suggest_categorical(f"hidden_dim_{i}", [64, 128, 256, 512]) for i in range(n_layers)]
    
    # Create trial-specific output directory
    trial_output_dir = Path(output_dir) / f"trial_{trial.number}"
    trial_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Train model with these hyperparameters
    import os
    data_version = os.environ.get("DATA_VERSION", "normalized")
    
    # If features_dir is provided, use it directly, otherwise construct from FEATURES_DIR and data_version
    if features_dir is None:
        features_dir = Path(FEATURES_DIR) / data_version
    else:
        features_dir = Path(features_dir)
    
    output_dir = Path(output_dir or OUTPUT_DIR)
    hidden_dims = hidden_dims or MLP_HIDDEN_DIMS
    dropout_rate = dropout_rate or DROPOUT_RATE
    batch_size = batch_size or BATCH_SIZE["train"]
    num_epochs = num_epochs or NUM_EPOCHS
    learning_rate = learning_rate or LEARNING_RATE
    weight_decay = weight_decay or WEIGHT_DECAY
    device = device or get_device()
    use_amp = use_amp if use_amp is not None else USE_AMP

    _, _, val_acc, val_auc = train_fold(
        fold_name=fold_name,
        features_dir=features_dir,
        output_dir=trial_output_dir,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        use_amp=use_amp
    )
    
    return val_acc


def main():
    """
    Main entry point for the training script.
    Handles cross-validation training across all folds.
    """
    # Get the data version from environment variable or use default
    import os
    data_version = os.environ.get("DATA_VERSION", "normalized")
    
    # Create version-specific output directory
    output_dir = Path(OUTPUT_DIR) / data_version
    output_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving outputs to {output_dir}")
    
    # Find available folds in the version-specific directory
    features_dir = Path(FEATURES_DIR) / data_version
    logging.info(f"Looking for fold data in {features_dir}")
    
    # Make sure the directory exists
    if not features_dir.exists():
        features_dir.mkdir(exist_ok=True, parents=True)
        logging.warning(f"Features directory {features_dir} did not exist and was created")
    
    # Find fold files
    fold_files = list(features_dir.glob("fold*_train_features.pt"))
    
    if not fold_files:
        # If no files found, check if we have files from features.py
        logging.warning(f"No fold files found in {features_dir}. Checking if features were extracted...")
        
        # Verify if we have the fold1 file at least
        if not (features_dir / "fold1_train_features.pt").exists():
            raise FileNotFoundError(f"No fold data found in {features_dir}. Run feature extraction first with: ./run.sh --version {data_version} --stage features")
    
    fold_names = sorted([f.name.split("_")[0] for f in fold_files])
    
    if not fold_names:
        raise FileNotFoundError(f"No fold data found in {features_dir}. Run precompute_features.py first.")
    
    print(f"Found {len(fold_names)} folds: {', '.join(fold_names)}")
    
    # Set random seeds for reproducibility
    set_seed(SEED)
    
    # Get device
    device = get_device()
    
    # Store results
    fold_results = {}
    folds_roc_data = {}
    
    # Train on each fold
    all_results = []
    
    for fold_name in fold_names:
        print("\n" + "=" * 50)
        print(f"Training on {fold_name}")
        print("=" * 50)
        
        model, train_acc, val_acc, val_auc = train_fold(fold_name=fold_name, data_version=data_version)
        
        # Load validation data for ROC analysis
        val_features_path = features_dir / f"{fold_name}_val_features.pt"
        val_data = torch.load(val_features_path)
        val_dataset = FeatureDataset(val_data['features'], val_data['labels'])
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE["val"], shuffle=False, pin_memory=True
        )
        
        # Compute ROC data
        num_classes = val_data['num_classes'] if 'num_classes' in val_data else len(torch.unique(val_data['labels']))
        is_binary = (num_classes == 2)
        labels, probs = compute_roc_data(model, val_loader, device, USE_AMP, is_binary)
        folds_roc_data[fold_name] = {'labels': labels, 'probs': probs}
        
        # Store results
        fold_results[fold_name] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_auc': val_auc
        }
    
    # Generate LaTeX tables for results
    generate_tables(fold_results, output_dir)
    
    # Plot combined ROC curves
    mean_auc, std_auc = plot_roc_curves(folds_roc_data, output_dir)
    
    # Create summary
    summary = {
        'fold_results': fold_results,
        'mean_val_acc': np.mean([r['val_acc'] for r in fold_results.values()]),
        'std_val_acc': np.std([r['val_acc'] for r in fold_results.values()]),
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'num_folds': len(fold_names),
        'fold_names': fold_names,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_json(summary, output_dir / 'cross_validation_summary.json')
    
    print("\n" + "="*60)
    print("Cross-validation complete!")
    print(f"Mean validation accuracy: {summary['mean_val_acc']:.4f} ± {summary['std_val_acc']:.4f}")
    print(f"Mean ROC AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print("="*60)
    

if __name__ == "__main__":
    import time
    import argparse
    import os
    from utils import setup_logging
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train MLP models on precomputed features")
    parser.add_argument("--version", type=str, default="normalized", 
                       help="Dataset version to use (e.g., 'normalized', 'rgb')")
    args = parser.parse_args()
    
    # Set environment variable for the data version
    os.environ["DATA_VERSION"] = args.version
    
    # Set up logging
    log_dir = Path(OUTPUT_DIR)
    log_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(log_file=log_dir / f"training_{args.version}.log")
    
    print(f"Training on features from version: {args.version}")
    
    try:
        start_time = time.time()
        main()
        duration = time.time() - start_time
        print(f"\nTotal training time: {duration/60:.2f} minutes")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}", exc_info=True)
        print(f"\nError: {e}")
        print(f"Make sure to run feature extraction first with: ./run.sh --version {args.version} --stage features")
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        print(f"\nError during training: {e}")
        print("See log file for details.")
        import traceback
        traceback.print_exc()