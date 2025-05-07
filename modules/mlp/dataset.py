"""
Dataset and DataLoader components for the UA-SLSM MLP training pipeline.
Provides classes and functions for loading and processing LSM image data.
"""

import json
import logging
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import MEAN, STD, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, NORMALIZATION_PARAMS

class LSMDataset(Dataset):
    """
    Dataset for loading LSM images and labels from annotations dataframe.
    
    Handles both binary (HSIL vs non-HSIL) and multi-class (NSA, LSIL, HSIL) classification.
    """
    def __init__(self, annotations_df: pd.DataFrame, image_dir: Path, 
                 transform: transforms.Compose, output_size: tuple = IMAGE_SIZE, 
                 binary_mode: bool = False):
        self.annotations = annotations_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.output_size = output_size
        self.binary_mode = binary_mode

        if binary_mode:
            self.classes = ['non-HSIL', 'HSIL']
            self.num_classes = 2
        else:
            self.classes = ['NSA', 'LSIL', 'HSIL']
            self.num_classes = 3
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if not self.annotations.empty:
            first_img_path = self.image_dir / self.annotations.iloc[0]['image_path']
            if not first_img_path.exists():
                logging.warning(f"First image {first_img_path} not found in {self.image_dir}. Check paths.")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image and its label at the specified index.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        row = self.annotations.iloc[idx]
        img_relative_path = row['image_path']
        label_str = row['label']
        
        full_img_path = self.image_dir / img_relative_path

        # Convert label string to index
        if self.binary_mode:
            label_idx = self.class_to_idx['HSIL'] if label_str == 'HSIL' else self.class_to_idx['non-HSIL']
        else:
            label_idx = self.class_to_idx.get(label_str)
            if label_idx is None:
                logging.warning(f"Unknown label '{label_str}' for {img_relative_path}. Defaulting to 0.")
                label_idx = 0 
                
        label = torch.tensor(label_idx, dtype=torch.long)
        
        # Load image with error handling
        try:
            image = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            logging.warning(f"Failed to load image {full_img_path}: {e}. Using black image.")
            image = Image.new('RGB', self.output_size, color='black')
            
        image_tensor = self.transform(image)
        return image_tensor, label

def get_transform(resize_shape: tuple = IMAGE_SIZE, augment: bool = False, 
                  mean: list = MEAN, std: list = STD) -> transforms.Compose:
    """
    Create a transform pipeline for image preprocessing.
    
    Args:
        resize_shape: Target image size (height, width)
        augment: Whether to apply data augmentation
        mean: Channel-wise mean for normalization
        std: Channel-wise standard deviation for normalization
        
    Returns:
        A torchvision.transforms.Compose object with the transformation pipeline
    """
    transform_list = []
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    transform_list.extend([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms.Compose(transform_list)

def create_train_val_datasets(
    annotations_path: Path, 
    fold_assignments_path: Path, 
    fold_name: str, 
    image_dir: Path,
    train_resize_shape: tuple = IMAGE_SIZE,
    val_resize_shape: tuple = IMAGE_SIZE,
    binary_mode: bool = False,
    train_mean: list = MEAN, train_std: list = STD,
    val_mean: list = MEAN, val_std: list = STD
) -> tuple[LSMDataset, LSMDataset]:
    """
    Create training and validation datasets for a specific fold.
    
    Args:
        annotations_path: Path to the annotations CSV file
        fold_assignments_path: Path to the fold assignments JSON file
        fold_name: Name of the fold to use
        image_dir: Directory containing the images
        train_resize_shape: Image size for training
        val_resize_shape: Image size for validation
        binary_mode: Whether to use binary classification
        train_mean: Channel-wise mean for training data normalization
        train_std: Channel-wise std for training data normalization
        val_mean: Channel-wise mean for validation data normalization
        val_std: Channel-wise std for validation data normalization
        
    Returns:
        Tuple of (train_dataset, val_dataset)
        
    Raises:
        FileNotFoundError: If annotations or fold assignments files don't exist
        ValueError: If fold name is not found or required columns are missing
    """
    try:
        annotations_df = pd.read_csv(annotations_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Annotations CSV not found at: {annotations_path}")
    
    try:
        with open(fold_assignments_path, 'r') as f:
            fold_assignments = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Fold assignments JSON not found at: {fold_assignments_path}")
    
    if fold_name not in fold_assignments:
        raise ValueError(f"Fold '{fold_name}' not found in {fold_assignments_path}")
    if 'image_path' not in annotations_df.columns:
        raise ValueError("'image_path' column not found in annotations CSV.")
    if 'label' not in annotations_df.columns:
        raise ValueError("'label' column not found in annotations CSV.")

    # Split data into train and validation sets
    val_image_paths = set(fold_assignments[fold_name])
    val_mask = annotations_df['image_path'].isin(val_image_paths)
    val_annotations_df = annotations_df[val_mask]
    train_annotations_df = annotations_df[~val_mask]
    
    if train_annotations_df.empty:
        logging.warning(f"Training set for fold '{fold_name}' is empty.")
    if val_annotations_df.empty:
        logging.warning(f"Validation set for fold '{fold_name}' is empty.")

    # Create transforms
    train_transform = get_transform(resize_shape=train_resize_shape, augment=True, 
                                   mean=train_mean, std=train_std)
    val_transform = get_transform(resize_shape=val_resize_shape, augment=False, 
                                 mean=val_mean, std=val_std)
    
    # Create datasets
    train_dataset = LSMDataset(
        annotations_df=train_annotations_df, image_dir=image_dir, 
        transform=train_transform, output_size=train_resize_shape, binary_mode=binary_mode
    )
    val_dataset = LSMDataset(
        annotations_df=val_annotations_df, image_dir=image_dir, 
        transform=val_transform, output_size=val_resize_shape, binary_mode=binary_mode
    )
    
    print(f"Created datasets for fold '{fold_name}': {len(train_dataset)} train, {len(val_dataset)} val samples. Num classes: {train_dataset.num_classes}")
    return train_dataset, val_dataset

def get_dataloader(dataset: Dataset, batch_size: int = None, shuffle: bool = True, 
                   num_workers: int = NUM_WORKERS, pin_memory: bool = PIN_MEMORY) -> DataLoader:
    """
    Create a DataLoader for a dataset with optimized settings.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size (if None, uses config values based on shuffle parameter)
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        A configured DataLoader
    """
    if batch_size is None:
        batch_size = BATCH_SIZE["train"] if shuffle else BATCH_SIZE["val"]
        
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True if shuffle else False,
        prefetch_factor=4,
    )






