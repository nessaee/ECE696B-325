"""
Centralized configuration for the UA-SLSM MLP training pipeline.
Contains shared parameters, paths, and constants used across the codebase.
"""

from pathlib import Path

# Base directories
BASE_DATA_DIR = Path("../../data")
BASE_RESULTS_DIR = Path("../../results")
DATASET_DIR = BASE_DATA_DIR / "dataset"
FEATURES_DIR = BASE_DATA_DIR / "features"  # Will be appended with version in code
OUTPUT_DIR = BASE_RESULTS_DIR / "mlp_training_output"  # Will be appended with version in code
VISUALIZATIONS_DIR = BASE_RESULTS_DIR / "visualizations"

# Dataset parameters
ANNOTATIONS_FILE = DATASET_DIR / "annotations.csv"
FOLD_ASSIGNMENTS_FILE = DATASET_DIR / "fold_assignments.json"

# Image parameters
IMAGE_SIZE = (224, 224)

# Pre-calculated channel-wise mean and std for LSM dataset (values scaled to [0,1])
# Different normalization parameters for each version
NORMALIZATION_PARAMS = {
    # "normalized": {
    #     "MEAN": [55.774969 / 255.0, 55.774969 / 255.0, 55.774969 / 255.0],
    #     "STD": [34.109242 / 255.0, 34.109242 / 255.0, 34.109242 / 255.0]

    # },
    # "rgb" : {
    #     "MEAN": [55.774969 / 255.0, 22.307015 / 255.0, 128.993943 / 255.0],
    #     "STD": [34.109242 / 255.0, 35.034817 / 255.0, 73.475657 / 255.0]
    # }

    "rgb": {
        "MEAN": [0.485, 0.456, 0.406],  # ImageNet mean
        "STD": [0.229, 0.224, 0.225]   # ImageNet std
    },
    "normalized": {
        "MEAN": [0.485, 0.456, 0.406],  # ImageNet mean
        "STD": [0.229, 0.224, 0.225]   # ImageNet std
    },  
    # Add more versions as needed
}

# Default values (for backward compatibility)
MEAN = NORMALIZATION_PARAMS["normalized"]["MEAN"]
STD = NORMALIZATION_PARAMS["normalized"]["STD"]

# DataLoader parameters
BATCH_SIZE = {
    "train": 8,
    "val": 64,
    "feature_extraction": 64
}
NUM_WORKERS = 8
PIN_MEMORY = True

# Model parameters
MOBILENET_FEATURE_DIM = 960  # Output dimension of MobileNetV3 Large backbone
MLP_HIDDEN_DIMS = [512, 256]
DROPOUT_RATE = 0.5

# Training parameters
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
USE_AMP = True  # Automatic Mixed Precision
GRAD_CLIP = None  # Set to a value (e.g., 1.0) to enable gradient clipping

# Early stopping parameters
PATIENCE = 10
MIN_DELTA = 0.001

# Random seed for reproducibility
SEED = 42
