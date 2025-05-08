"""
Feature extraction utilities for the UA-SLSM MLP training pipeline.
Provides functions for precomputing and extracting features from various pretrained models.
"""

import torch
import time
import gc
import json
import logging
import os
from pathlib import Path

from config import (
    BASE_DATA_DIR, DATASET_DIR, FEATURES_DIR, ANNOTATIONS_FILE, 
    FOLD_ASSIGNMENTS_FILE, BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE,
    NORMALIZATION_PARAMS
)
from utils import get_device
from dataset import create_train_val_datasets, get_dataloader
from model import get_model, get_model_feature_extractor, ModelType

def precompute_features(version="normalized", binary_classification=True, device=None, model_type=ModelType.MOBILENET_V3):
    """
    Precompute features from a pretrained model backbone for all images in the dataset
    and save them to disk for faster training of just the classifier head.
    
    Args:
        version (str): Dataset version to use (e.g., "normalized", "rgb")
        binary_classification (bool): Whether to use binary classification mode
        device (torch.device): Device to use for computation (if None, uses CUDA if available)
        model_type (ModelType): Type of model to use for feature extraction
    """
    # Handle string input for model_type
    if isinstance(model_type, str):
        model_type = ModelType.from_string(model_type)
    
    print(f"Starting feature precomputation for version: {version} using {model_type.value}...")
    
    # Configuration
    image_dir = DATASET_DIR / version
    model_subdir = model_type.value
    features_dir = FEATURES_DIR / version / model_subdir
    
    # Create features directory if it doesn't exist
    features_dir.mkdir(exist_ok=True, parents=True)
    
    # Get version-specific normalization parameters
    if version in NORMALIZATION_PARAMS:
        mean = NORMALIZATION_PARAMS[version]["MEAN"]
        std = NORMALIZATION_PARAMS[version]["STD"]
        logging.info(f"Using {version}-specific normalization: mean={mean}, std={std}")
    else:
        mean = NORMALIZATION_PARAMS["normalized"]["MEAN"]
        std = NORMALIZATION_PARAMS["normalized"]["STD"]
        logging.warning(f"No normalization parameters found for version '{version}', using default")
    
    # Get all fold names from fold assignments file
    try:
        with open(FOLD_ASSIGNMENTS_FILE, 'r') as f:
            fold_assignments = json.load(f)
        fold_names = list(fold_assignments.keys())
    except FileNotFoundError:
        # Create a default fold structure if file doesn't exist
        logging.warning(f"Fold assignments file not found at {FOLD_ASSIGNMENTS_FILE}")
        logging.warning("Creating a default fold structure for demonstration purposes")
        
        # Create the parent directory if it doesn't exist
        FOLD_ASSIGNMENTS_FILE.parent.mkdir(exist_ok=True, parents=True)
        
        # Create a default fold structure with a single fold
        fold_assignments = {"fold1": []}
        
        # Save the default fold structure
        with open(FOLD_ASSIGNMENTS_FILE, 'w') as f:
            json.dump(fold_assignments, f, indent=4)
        
        fold_names = list(fold_assignments.keys())
        
        print(f"Created default fold structure at {FOLD_ASSIGNMENTS_FILE}")
        print("This is for demonstration purposes only. Please provide real fold assignments for production use.")
    batch_size = BATCH_SIZE["feature_extraction"]
    
    # Use provided device or default
    if device is None:
        device = get_device()
    
    print(f"Using device: {device}")
    print(f"Processing folds: {fold_names}")
    print(f"Binary classification mode: {binary_classification}")
    
    # Load the model backbone only (without classifier head)
    num_classes = 2 if binary_classification else 3
    model, feature_dim = get_model(model_type=model_type, num_classes=num_classes, freeze_backbone=True)
    feature_extractor = get_model_feature_extractor(model, model_type=model_type)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # Get feature shape by passing a dummy input
    dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    with torch.no_grad():
        dummy_output = feature_extractor(dummy_input)
    feature_shape = dummy_output.shape[1]
    print(f"Feature shape: {feature_shape}")
    
    total_start_time = time.time()
    
    # Process each fold
    for fold_name in fold_names:
        fold_start_time = time.time()
        print(f"\nProcessing fold: {fold_name}")
        
        # Check if annotations file exists
        if not ANNOTATIONS_FILE.exists():
            # Create a default annotations file if it doesn't exist
            logging.warning(f"Annotations file not found at {ANNOTATIONS_FILE}")
            logging.warning("Creating a default annotations file for demonstration purposes")
            
            # Create the parent directory if it doesn't exist
            ANNOTATIONS_FILE.parent.mkdir(exist_ok=True, parents=True)
            
            # Create a default annotations dataframe with a few dummy entries
            import pandas as pd
            annotations = pd.DataFrame({
                'image_path': ['dummy1.jpg', 'dummy2.jpg', 'dummy3.jpg'],
                'label': ['HSIL', 'non-HSIL', 'HSIL'] if binary_classification else ['HSIL', 'LSIL', 'NSA']
            })
            
            # Save the default annotations
            annotations.to_csv(ANNOTATIONS_FILE, index=False)
            
            print(f"Created default annotations at {ANNOTATIONS_FILE}")
            print("This is for demonstration purposes only. Please provide real annotations for production use.")
            
            # Update the fold assignments to include the dummy images
            fold_assignments[fold_name] = ['dummy3.jpg']  # One image for validation
            with open(FOLD_ASSIGNMENTS_FILE, 'w') as f:
                json.dump(fold_assignments, f, indent=4)
        
        # Create the image directory if it doesn't exist
        if not image_dir.exists():
            image_dir.mkdir(exist_ok=True, parents=True)
            logging.warning(f"Image directory not found at {image_dir}. Created directory.")
            
            # Create dummy images
            for img_name in ['dummy1.jpg', 'dummy2.jpg', 'dummy3.jpg']:
                from PIL import Image
                import numpy as np
                
                # Create a random image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(image_dir / img_name)
                
            print(f"Created dummy images in {image_dir}")
            print("This is for demonstration purposes only. Please provide real images for production use.")
        
        # Create datasets for this fold
        try:
            train_ds, val_ds = create_train_val_datasets(
                annotations_path=ANNOTATIONS_FILE,
                fold_assignments_path=FOLD_ASSIGNMENTS_FILE,
                fold_name=fold_name,
                image_dir=image_dir,
                binary_mode=binary_classification,
                train_mean=mean,
                train_std=std,
                val_mean=mean,
                val_std=std
            )
        except Exception as e:
            logging.error(f"Error creating datasets: {e}")
            raise
        
        # Create dataloaders
        train_loader = get_dataloader(train_ds, batch_size, shuffle=False, 
                                     num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = get_dataloader(val_ds, batch_size, shuffle=False, 
                                   num_workers=NUM_WORKERS, pin_memory=True)
        
        # Extract and save features for training set
        train_features, train_labels = extract_features(feature_extractor, train_loader, device)
        train_feature_path = features_dir / f"{fold_name}_train_features.pt"
        torch.save({
            'features': train_features,
            'labels': train_labels,
            'shape': feature_shape,
            'binary_mode': binary_classification,
            'num_classes': train_ds.num_classes,
            'class_names': train_ds.classes,
            'model_type': model_type.value
        }, train_feature_path)
        print(f"Saved training features to {train_feature_path}")
        
        # Extract and save features for validation set if exists
        if len(val_ds) > 0:
            val_features, val_labels = extract_features(feature_extractor, val_loader, device)
            val_feature_path = features_dir / f"{fold_name}_val_features.pt"
            torch.save({
                'features': val_features,
                'labels': val_labels,
                'shape': feature_shape,
                'binary_mode': binary_classification,
                'num_classes': val_ds.num_classes,
                'class_names': val_ds.classes,
                'model_type': model_type.value
            }, val_feature_path)
            print(f"Saved validation features to {val_feature_path}")
        else:
            logging.warning(f"No validation data for fold {fold_name}")
        
        fold_duration = time.time() - fold_start_time
        print(f"Completed fold {fold_name} in {fold_duration:.2f} seconds")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_duration = time.time() - total_start_time
    print(f"\nFeature precomputation completed in {total_duration/60:.2f} minutes")

def extract_features(feature_extractor, dataloader, device):
    """
    Extract features from the backbone for all images in the dataloader.
    
    Args:
        feature_extractor: Model backbone to extract features
        dataloader: DataLoader containing images
        device: Device to run extraction on
        
    Returns:
        features: Tensor of extracted features [N, feature_dim]
        labels: Tensor of corresponding labels [N]
    """
    feature_extractor.eval()
    
    # Initialize lists to store features and labels
    all_features = []
    all_labels = []
    
    # Setup progress tracking
    total_batches = len(dataloader)
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move data to device
            images = images.to(device, non_blocking=True)
            
            # Extract features
            features = feature_extractor(images)
            
            # Move to CPU and store
            all_features.append(features.cpu())
            all_labels.append(labels)
            
            # Print progress at intervals
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                elapsed = time.time() - start_time
                print(f"Progress: {batch_idx+1}/{total_batches} batches ({progress:.1f}%) - {elapsed:.1f}s elapsed")
    
    # Concatenate all features and labels
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    print(f"Extracted features: {features_tensor.shape}, Labels: {labels_tensor.shape}")
    return features_tensor, labels_tensor

if __name__ == "__main__":
    import argparse
    import logging
    from utils import setup_logging
    
    # Set up logging
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Precompute features for MLP training")
    parser.add_argument("--version", type=str, default="normalized", 
                        help="Dataset version to use (e.g., 'normalized', 'rgb')")
    parser.add_argument("--binary", action="store_true", default=True,
                        help="Use binary classification mode (default: True)")
    parser.add_argument("--multi", action="store_true", 
                        help="Use multi-class classification mode (overrides --binary)")
    parser.add_argument("--model", type=str, default="mobilenet_v3",
                        choices=[m.value for m in ModelType],
                        help="Model type to use for feature extraction")
    args = parser.parse_args()
    
    # If --multi is specified, override binary mode
    binary_mode = not args.multi if args.multi else args.binary
    
    # Convert model type string to enum
    model_type = ModelType.from_string(args.model)
    
    print(f"Starting feature extraction with settings:")
    print(f"  - Version: {args.version}")
    print(f"  - Model: {model_type.value}")
    print(f"  - Classification mode: {'binary' if binary_mode else 'multi-class'}")
    
    try:
        # Call the feature precomputation function with the specified parameters
        precompute_features(version=args.version, binary_classification=binary_mode, model_type=model_type)
    except KeyboardInterrupt:
        print("\nFeature extraction interrupted by user.")
    except Exception as e:
        logging.error(f"Error during feature extraction: {e}", exc_info=True)
