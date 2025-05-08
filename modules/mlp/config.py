"""
Centralized configuration for the UA-SLSM MLP training pipeline.
Contains shared parameters, paths, and constants used across the codebase.
"""

from pathlib import Path

# Base directories
BASE_DATA_DIR = Path("../../data/dataset")
BASE_RESULTS_DIR = Path("../../results")
DATASET_DIR = BASE_DATA_DIR / "processed"
FEATURES_DIR = BASE_DATA_DIR / "features"  # Will be appended with version in code
OUTPUT_DIR = BASE_RESULTS_DIR / "mlp_training_output"  # Will be appended with version in code
VISUALIZATIONS_DIR = BASE_RESULTS_DIR / "visualizations"

# Dataset parameters
ANNOTATIONS_FILE = BASE_DATA_DIR / "labels/mlp/annotations.csv"
FOLD_ASSIGNMENTS_FILE = BASE_DATA_DIR / "labels/mlp/fold_assignments.json"

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
# Feature dimensions for different model types
FEATURE_DIMS = {
    "mobilenet_v3": 960,
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_v2_s": 1280,
    "vit_b_16": 768
}

# Model-specific MLP hidden dimensions
MLP_HIDDEN_DIMS = {
    "mobilenet_v3": [512, 256],
    "resnet18": [512, 256],
    "resnet50": [1024, 512],
    "efficientnet_v2_s": [768, 384],
    "vit_b_16": [512, 256]
}

# Default for backward compatibility
MOBILENET_FEATURE_DIM = FEATURE_DIMS["mobilenet_v3"]

# Model-specific dropout rates
DROPOUT_RATE = {
    "mobilenet_v3": 0.5,
    "resnet18": 0.5,
    "resnet50": 0.6,
    "efficientnet_v2_s": 0.4,
    "vit_b_16": 0.3
}

# Model-specific learning rates
LEARNING_RATE = {
    "mobilenet_v3": 2e-4,
    "resnet18": 1e-4,
    "resnet50": 5e-5,
    "efficientnet_v2_s": 1e-4,
    "vit_b_16": 5e-5
}

# Model-specific weight decay
WEIGHT_DECAY = {
    "mobilenet_v3": 1e-5,
    "resnet18": 1e-5,
    "resnet50": 1e-4,
    "efficientnet_v2_s": 1e-5,
    "vit_b_16": 1e-4
}
NUM_EPOCHS = 100
USE_AMP = True  # Automatic Mixed Precision
GRAD_CLIP = None  # Set to a value (e.g., 1.0) to enable gradient clipping

# Early stopping parameters
PATIENCE = 25
MIN_DELTA = 0.001

# Random seed for reproducibility
SEED = 42
