# Configuration Guide

This document explains how to configure the UA-SLSM MLP Training Pipeline to suit your specific needs.

## Configuration File

The main configuration for the pipeline is located in `modules/mlp/config.py`. This file contains all the parameters and settings used throughout the pipeline.

### Directory Paths

```python
# Base directories
BASE_DATA_DIR = Path("../../data/dataset")
BASE_RESULTS_DIR = Path("../../results")
DATASET_DIR = BASE_DATA_DIR / "processed"
FEATURES_DIR = BASE_DATA_DIR / "features"
OUTPUT_DIR = BASE_RESULTS_DIR / "mlp_training_output"
```

You can modify these paths to point to your own data directories.

### Dataset Parameters

```python
# Dataset parameters
ANNOTATIONS_FILE = BASE_DATA_DIR / "labels/mlp/annotations.csv"
FOLD_ASSIGNMENTS_FILE = BASE_DATA_DIR / "labels/mlp/fold_assignments.json"

# Image parameters
IMAGE_SIZE = (224, 224)
```

### Normalization Parameters

The pipeline supports different normalization strategies:

```python
NORMALIZATION_PARAMS = {
    "rgb": {
        "MEAN": [0.485, 0.456, 0.406],  # ImageNet mean
        "STD": [0.229, 0.224, 0.225]    # ImageNet std
    },
    "normalized": {
        "MEAN": [0.485, 0.456, 0.406],  # ImageNet mean
        "STD": [0.229, 0.224, 0.225]    # ImageNet std
    },
}
```

You can add custom normalization parameters for your own data versions.

### DataLoader Parameters

```python
# DataLoader parameters
BATCH_SIZE = {
    "train": 8,
    "val": 64,
    "feature_extraction": 64
}
NUM_WORKERS = 8
PIN_MEMORY = True
```

Adjust these parameters based on your hardware capabilities:
- Increase `BATCH_SIZE` if you have more GPU memory
- Increase `NUM_WORKERS` if you have more CPU cores

### Model Parameters

The pipeline uses model-specific hyperparameters for optimal performance:

#### Feature Dimensions

```python
FEATURE_DIMS = {
    "mobilenet_v3": 960,
    "resnet18": 512,
    "resnet50": 2048,
    "efficientnet_v2_s": 1280,
    "vit_b_16": 768
}
```

#### MLP Hidden Dimensions

```python
MLP_HIDDEN_DIMS = {
    "mobilenet_v3": [512, 256],
    "resnet18": [512, 256],
    "resnet50": [1024, 512],
    "efficientnet_v2_s": [768, 384],
    "vit_b_16": [512, 256]
}
```

#### Dropout Rates

```python
DROPOUT_RATE = {
    "mobilenet_v3": 0.5,
    "resnet18": 0.5,
    "resnet50": 0.6,
    "efficientnet_v2_s": 0.4,
    "vit_b_16": 0.3
}
```

#### Learning Rates

```python
LEARNING_RATE = {
    "mobilenet_v3": 2e-4,
    "resnet18": 1e-4,
    "resnet50": 5e-5,
    "efficientnet_v2_s": 1e-4,
    "vit_b_16": 5e-5
}
```

#### Weight Decay

```python
WEIGHT_DECAY = {
    "mobilenet_v3": 1e-5,
    "resnet18": 1e-5,
    "resnet50": 1e-4,
    "efficientnet_v2_s": 1e-5,
    "vit_b_16": 1e-4
}
```

### Training Parameters

```python
NUM_EPOCHS = 100
USE_AMP = True  # Automatic Mixed Precision
GRAD_CLIP = None  # Set to a value (e.g., 1.0) to enable gradient clipping
```

### Early Stopping Parameters

```python
PATIENCE = 25
MIN_DELTA = 0.001
```

### Random Seed

```python
SEED = 42
```

## Adding Custom Models

To add a custom model to the pipeline:

1. Add the model's feature dimensions to `FEATURE_DIMS`
2. Add appropriate hyperparameters to all model-specific dictionaries
3. Update the feature extraction code in `features.py` to support your model
4. Update the command-line argument parsing in `run.sh` to include your model

Example:

```python
# In config.py
FEATURE_DIMS["custom_model"] = 1024
MLP_HIDDEN_DIMS["custom_model"] = [512, 256]
DROPOUT_RATE["custom_model"] = 0.5
LEARNING_RATE["custom_model"] = 1e-4
WEIGHT_DECAY["custom_model"] = 1e-5
```

## Customizing Results Consolidation

The results consolidation script (`consolidate_results.py`) can be customized with command-line arguments:

```bash
python consolidate_results.py --no-highlight-best --no-performance
```

Available options:
- `--highlight-best` / `--no-highlight-best`: Toggle highlighting of best metrics
- `--include-performance` / `--no-performance`: Toggle inclusion of performance metrics
- `--include-efficiency` / `--no-efficiency`: Toggle inclusion of efficiency metrics

## Advanced Configuration

### Custom Cross-validation Folds

To use custom cross-validation folds:

1. Create a JSON file with fold assignments
2. Update `FOLD_ASSIGNMENTS_FILE` in `config.py`

Example fold assignments file:
```json
{
  "fold_1": ["sample1", "sample2", ...],
  "fold_2": ["sample3", "sample4", ...],
  ...
}
```

### Custom Annotations

To use custom annotations:

1. Create a CSV file with sample IDs and labels
2. Update `ANNOTATIONS_FILE` in `config.py`

Example annotations file:
```csv
sample_id,label
sample1,0
sample2,1
...
```

## Next Steps

- Learn about [results analysis](results_analysis.md)
