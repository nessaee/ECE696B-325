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
    NUM_WORKERS, PIN_MEMORY, PATIENCE, MIN_DELTA, GRAD_CLIP, FEATURE_DIMS
)
from utils import set_seed, get_device, save_json
from model import MLPClassifier, count_parameters
from evaluation import (evaluate, compute_roc_data, generate_tables, plot_roc_curves, 
                      plot_training_curves, plot_combined_training_curves, plot_confusion_matrix,
                      plot_combined_confusion_matrix, generate_metrics_table, calculate_metrics_from_confusion_matrix)


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
    """Train the model for one epoch and measure training time"""
    import time
    start_time = time.time()
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
    
    # Calculate training time for this epoch
    epoch_time = time.time() - start_time
    
    return running_loss / total, correct / total, epoch_time


def train_fold(fold_name, features_dir=None, output_dir=None, hidden_dims=None, dropout_rate=None, 
               batch_size=None, num_epochs=None, learning_rate=None, weight_decay=None, 
               device=None, use_amp=None, data_version=None, model_type=None, binary=True):
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
        data_version: Dataset version to use
        model_type: Model type used for feature extraction
        binary: Whether to use binary classification mode (default: True)
        
    Returns:
        Tuple of (model, train_accuracy, validation_accuracy, validation_auc)
    """
    # Get data version and model type from parameter or environment variable
    import os
    data_version = data_version or os.environ.get("DATA_VERSION", "normalized")
    model_type = model_type or os.environ.get("MODEL_TYPE", "mobilenet_v3")
    
    # Use default values from config if not provided
    if features_dir is None:
        features_dir = Path(FEATURES_DIR) / data_version / model_type
    else:
        features_dir = Path(features_dir)
    
    # Create version-specific output directory
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR) / data_version / model_type / fold_name
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
    
    train_data = torch.load(train_features_path)
    
    # Get feature dimension from the data
    feature_dim = train_data['features'].shape[1]
    
    # Create datasets
    train_dataset = FeatureDataset(train_data['features'], train_data['labels'])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE["train"], shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    
    # Load validation data
    val_features_path = features_dir / f"{fold_name}_val_features.pt"
    val_data = torch.load(val_features_path)
    val_dataset = FeatureDataset(val_data['features'], val_data['labels'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE["val"], shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    
    # Determine number of classes
    num_classes = train_data['num_classes'] if 'num_classes' in train_data else len(torch.unique(train_data['labels']))
    if binary and num_classes > 2:
        print(f"Warning: Binary mode requested but found {num_classes} classes. Forcing binary mode.")
        num_classes = 2
    
    # Get model-specific hyperparameters
    hidden_dims = MLP_HIDDEN_DIMS[model_type]
    dropout_rate = DROPOUT_RATE[model_type]
    learning_rate = LEARNING_RATE[model_type]
    weight_decay = WEIGHT_DECAY[model_type]
    
    print(f"Using model-specific hyperparameters for {model_type}:")
    print(f"  - Hidden dims: {hidden_dims}")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    
    # Initialize model
    model = MLPClassifier(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        model_type=model_type
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Loss function based on number of classes
    is_binary = (num_classes == 2)
    
    # Print class distribution
    class_counts = torch.bincount(train_data['labels'])
    print(f"Class distribution in training set: {class_counts.tolist()}")
    
    # Loss function with class weights
    if is_binary:
        pos_weight = torch.tensor([2.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        fixed_weights = torch.ones(num_classes) * 1.8
        fixed_weights[0] = 0.9
        criterion = nn.CrossEntropyLoss(weight=fixed_weights.to(device))
    
    # Use model-specific optimizer parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,  # Use model-specific learning rate
        betas=(0.9, 0.999),
        eps=1.0e-08,
        weight_decay=weight_decay  # Use model-specific weight decay
    )
    
    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 5,  # Scale max_lr based on base learning rate
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
        anneal_strategy='cos'
    )
    
    # Setup AMP
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    grad_clip = GRAD_CLIP if GRAD_CLIP is not None else 0.5
    
    # Count model parameters
    param_count = count_parameters(model)
    print(f"Number of trainable parameters: {param_count:,}")
    
    # Train the model
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_time_per_epoch': [], 'inference_time_per_batch': []
    }
    
    # Set up early stopping
    patience = PATIENCE
    min_delta = MIN_DELTA
    counter = 0
    best_val_loss = float('inf')
    
    # Set up AMP scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"  - Model: MLP with hidden dims {hidden_dims}, dropout {dropout_rate}")
    print(f"  - Training: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE['train']}, lr {learning_rate}, weight decay {weight_decay}")
    print(f"  - Device: {device}, AMP: {USE_AMP}")
    print(f"  - Feature extractor: {model_type} (dimension: {feature_dim})")
    print(f"  - Number of classes: {num_classes}, Binary mode: {is_binary}")

    
    for epoch in range(NUM_EPOCHS):
        # Train one epoch
        train_loss, train_acc, epoch_train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, 
            epoch, NUM_EPOCHS, USE_AMP, scaler, is_binary
        )
        
        # Evaluate
        val_loss, val_acc, inference_time = evaluate(model, val_loader, criterion, device, USE_AMP, is_binary)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_time_per_epoch'].append(epoch_train_time)
        history['inference_time_per_batch'].append(inference_time)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model and check for early stopping
        if val_acc > best_val_acc + MIN_DELTA:
            # Final evaluation on validation set
            val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device, use_amp, is_binary)
            best_val_acc = val_acc  # Update the best validation accuracy
            best_epoch = epoch
            best_model_state = model.state_dict().copy()  # Save the current model state
            history['best_epoch'] = epoch
            history['best_val_acc'] = val_acc
            patience_counter = 0  # Reset patience counter
            print(f"New best model with validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy. Patience: {patience_counter}/{PATIENCE}")
            
            # Check if we should stop early
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                history['early_stopped'] = True
                history['patience_counter'] = patience_counter
                break
    
    # Save training history
    save_json(history, output_dir / 'training_history.json')
    
    # Plot training curves
    plot_training_curves(fold_name, history, output_dir)
    
    # Save best model and model architecture information
    if best_model_state is not None:
        # Save model state
        torch.save(best_model_state, output_dir / 'best_model.pt')
        print(f"Saved best model to {output_dir / 'best_model.pt'}")
        
        # Save model architecture information
        model_info = {
            'input_dim': feature_dim,
            'num_classes': num_classes,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'model_type': model_type,
            'param_count': param_count,
            'avg_train_time_per_epoch': np.mean(history['train_time_per_epoch']),
            'avg_inference_time_per_batch': np.mean(history['inference_time_per_batch'])
        }
        with open(output_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Saved model architecture information to {output_dir / 'model_info.json'}")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)  # Load best model for final evaluation
    
    # Compute ROC curve data and predictions for confusion matrix
    try:
        labels, probs, predictions = compute_roc_data(model, val_loader, device, USE_AMP, is_binary)
        
        # Generate confusion matrix
        class_names = ["Negative", "Positive"] if is_binary else [str(i) for i in range(num_classes)]
        confusion_mat = plot_confusion_matrix(labels, predictions, fold_name, output_dir, class_names=class_names)
        
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
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0.0
    
    print(f"\nTraining completed for {fold_name}")
    if history.get('early_stopped', False):
        print(f"Training stopped early due to no improvement for {PATIENCE} epochs")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return model, final_train_acc, best_val_acc, roc_auc, confusion_mat


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

    _, _, val_acc, val_auc, _ = train_fold(
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


def main(binary=True):
    """
    Main entry point for the training script.
    Handles cross-validation training across all folds.
    
    Args:
        binary: If True, use binary classification mode
    """
    # Get data version and model type from environment variables
    import os
    data_version = os.environ.get("DATA_VERSION", "normalized")
    model_type = os.environ.get("MODEL_TYPE", "mobilenet_v3")
    
    # Determine feature directory based on data version and model type
    features_dir = Path(FEATURES_DIR) / data_version / model_type
    output_dir = Path(OUTPUT_DIR) / data_version / model_type
    output_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving outputs to {output_dir}")
    
    # Find available folds in the version and model-specific directory
    features_dir = Path(FEATURES_DIR) / data_version / model_type
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
    fold_histories = {}
    fold_confusion_matrices = {}  # Store confusion matrices for each fold
    
    # Get parameter count for the model type
    # Create a dummy model to count parameters
    dummy_model = MLPClassifier(
        input_dim=FEATURE_DIMS.get(model_type, 960),
        num_classes=2 if binary else 3,  # Default to 10 classes for multi-class
        model_type=model_type
    )
    param_count = count_parameters(dummy_model)
    
    # Train on each fold
    all_results = []
    
    for fold_name in fold_names:
        print("\n" + "=" * 50)
        print(f"Training on {fold_name}")
        print("=" * 50)
        
        model, train_acc, val_acc, val_auc, confusion_mat = train_fold(fold_name=fold_name, data_version=data_version, model_type=model_type, binary=binary)
        
        # Store the confusion matrix
        fold_confusion_matrices[fold_name] = confusion_mat
        
        # Get the model type from the training directory
        best_model_path = Path(OUTPUT_DIR) / data_version / model_type / fold_name / 'best_model.pt'
        model_info_path = Path(OUTPUT_DIR) / data_version / model_type / fold_name / 'model_info.json'
        
        if best_model_path.exists() and model_info_path.exists():
            print(f"Loading best model from {best_model_path}")
            
            # Load model info to get the correct feature dimensions and model type
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                feature_dim = model_info.get('input_dim')
                saved_model_type = model_info.get('model_type', model_type)
                print(f"Found feature dimension {feature_dim} from model info")
                print(f"Model was trained with model type: {saved_model_type}")
            
            # Make sure to load validation data from the correct features directory
            correct_features_dir = Path(FEATURES_DIR) / data_version / saved_model_type
            val_features_path = correct_features_dir / f"{fold_name}_val_features.pt"
            
            if val_features_path.exists():
                print(f"Loading validation data from {val_features_path}")
                val_data = torch.load(val_features_path)
                val_dataset = FeatureDataset(val_data['features'], val_data['labels'])
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=BATCH_SIZE["val"], shuffle=False, pin_memory=True
                )
                
                # Verify feature dimensions match
                actual_feature_dim = val_data['features'].shape[1]
                if actual_feature_dim != feature_dim:
                    print(f"Warning: Feature dimension mismatch! Model expects {feature_dim} but data has {actual_feature_dim}")
                    print(f"Skipping ROC computation for {fold_name} due to dimension mismatch")
                    continue
                
                # Compute ROC data
                num_classes = val_data['num_classes'] if 'num_classes' in val_data else len(torch.unique(val_data['labels']))
                is_binary = (num_classes == 2)
                
                # Create a model with the same architecture as the saved one
                loaded_model = MLPClassifier(
                    input_dim=feature_dim,
                    num_classes=num_classes,
                    hidden_dims=MLP_HIDDEN_DIMS[saved_model_type] if isinstance(MLP_HIDDEN_DIMS, dict) else MLP_HIDDEN_DIMS,
                    dropout_rate=DROPOUT_RATE[saved_model_type] if isinstance(DROPOUT_RATE, dict) else DROPOUT_RATE,
                    model_type=saved_model_type
                ).to(device)
                
                # Load the saved state
                loaded_model.load_state_dict(torch.load(best_model_path))
                loaded_model.eval()
                
                # Compute ROC data and predictions for confusion matrix
                labels, probs, predictions = compute_roc_data(loaded_model, val_loader, device, USE_AMP, is_binary)
                
                # Generate confusion matrix
                class_names = ["Negative", "Positive"] if is_binary else [str(i) for i in range(num_classes)]
                confusion_mat = plot_confusion_matrix(labels, predictions, fold_name, output_dir, class_names=class_names)
                
                # Store the confusion matrix
                fold_confusion_matrices[fold_name] = confusion_mat
            else:
                print(f"Warning: Validation features not found at {val_features_path}")
                print(f"Skipping ROC computation for {fold_name}")
                continue
        else:
            print(f"Warning: Best model not found at {best_model_path}. Skipping ROC computation for {fold_name}.")
            continue
        folds_roc_data[fold_name] = {'labels': labels, 'probs': probs}
        
        # Load training history
        history_path = Path(OUTPUT_DIR) / data_version / model_type / fold_name / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                fold_histories[fold_name] = history
        
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
    
    # Plot combined training curves
    if fold_histories:
        plot_combined_training_curves(fold_histories, output_dir)
        
    # Generate combined confusion matrix from all folds
    if fold_confusion_matrices:
        # Determine class names based on binary mode
        class_names = ["Negative", "Positive"] if binary else [str(i) for i in range(len(fold_confusion_matrices[list(fold_confusion_matrices.keys())[0]]))]
        plot_combined_confusion_matrix(fold_confusion_matrices, output_dir, class_names=class_names)
        
        # Generate metrics table (F1, sensitivity, specificity) across folds
        mean_metrics, std_metrics = generate_metrics_table(fold_confusion_matrices, output_dir, is_binary=binary, model_type=model_type)
    
    # Create summary
    summary = {
        'fold_results': fold_results,
        'mean_val_acc': np.mean([r['val_acc'] for r in fold_results.values()]),
        'std_val_acc': np.std([r['val_acc'] for r in fold_results.values()]),
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'num_folds': len(fold_names),
        'fold_names': fold_names,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'param_count': param_count,
        'avg_train_time_per_epoch': np.mean([np.mean(history['train_time_per_epoch']) for history in fold_histories.values()]) if fold_histories else 0,
        'avg_inference_time_per_batch': np.mean([np.mean(history['inference_time_per_batch']) for history in fold_histories.values()]) if fold_histories else 0
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
    from model import ModelType
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train MLP models on precomputed features")
    parser.add_argument("--version", type=str, default="normalized", 
                       help="Dataset version to use (e.g., 'normalized', 'rgb')")
    parser.add_argument("--model", type=str, default="mobilenet_v3",
                       choices=[m.value for m in ModelType],
                       help="Model type used for feature extraction")
    parser.add_argument("--binary", action="store_true", default=True,
                       help="Use binary classification mode")
    args = parser.parse_args()
    
    # Set environment variable for the data version
    os.environ["DATA_VERSION"] = args.version
    os.environ["MODEL_TYPE"] = args.model
    
    # Set up logging
    log_dir = Path(OUTPUT_DIR)
    log_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(log_file=log_dir / f"training_{args.version}_{args.model}.log")
    
    print(f"Training on features from version: {args.version}, model: {args.model}")
    
    try:
        start_time = time.time()
        main(binary=args.binary)
        duration = time.time() - start_time
        print(f"\nTotal training time: {duration/60:.2f} minutes")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}", exc_info=True)
        print(f"\nError: {e}")
        print(f"Make sure to run feature extraction first with: ./run.sh --version {args.version} --model {args.model} --stage features")
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        print(f"\nError during training: {e}")
        print("See log file for details.")
        import traceback
        traceback.print_exc()