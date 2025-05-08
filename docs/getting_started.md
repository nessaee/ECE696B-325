# Getting Started with UA-SLSM MLP Training Pipeline

This guide will help you set up and run the UA-SLSM MLP Training Pipeline.

## Prerequisites

- Python 3.6+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nessaee/ECE696B-325.git
   cd ECE696B-325
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place raw data in `data/dataset/raw`
   - Run preprocessing scripts if needed:
     ```bash
     cd modules/preprocessing
     ./run.sh
     ```

## Dataset Structure

The pipeline expects the following dataset structure:

```
data/
├── dataset/
│   ├── raw/                # Raw images
│   ├── processed/          # Processed images (normalized and RGB versions)
│   │   ├── normalized/     # Normalized images
│   │   └── rgb/            # RGB images
│   └── labels/             # Dataset annotations
│       └── mlp/
│           ├── annotations.csv        # Class labels
│           └── fold_assignments.json  # Cross-validation fold assignments
```

## Basic Usage

### Running the Complete Pipeline

To run the complete pipeline (feature extraction and training) for all models and data versions:

```bash
cd modules/mlp
./sweep.sh
```

### Running Specific Stages

To run specific stages of the pipeline for a particular model and data version:

```bash
cd modules/mlp
./run.sh --version normalized --binary --model resnet50 --stage features
./run.sh --version normalized --binary --model resnet50 --stage train
```

### Command-line Options

The `run.sh` script accepts the following options:

- `--version VERSION`: Dataset version to use (`normalized` or `rgb`, default: `normalized`)
- `--binary`: Use binary classification mode (default)
- `--multi`: Use multi-class classification mode
- `--stage STAGE`: Pipeline stage to run (`features`, `train`, or `all`)
- `--model MODEL`: Model architecture to use (default: `mobilenet_v3`)
  - Options: `mobilenet_v3`, `resnet18`, `resnet50`, `efficientnet_v2_s`, `vit_b_16`
- `--help`: Show help message

## Output Structure

After running the pipeline, results will be organized as follows:

```
results/
├── mlp_training_output/
│   ├── normalized/
│   │   ├── mobilenet_v3/
│   │   │   ├── fold_1/
│   │   │   │   ├── model.pth
│   │   │   │   ├── training_curves.png
│   │   │   │   ├── roc_curve.png
│   │   │   │   └── confusion_matrix_*.png
│   │   │   ├── ...
│   │   │   ├── combined_training_curves.png
│   │   │   ├── combined_confusion_matrix_*.png
│   │   │   ├── model_info.json
│   │   │   └── cross_validation_summary.json
│   │   └── ...
│   └── rgb/
│       └── ...
└── consolidated_results_table.tex  # LaTeX table with all results
```

## Next Steps

- Learn about the [pipeline architecture](pipeline_overview.md)
- Understand how to [configure the pipeline](configuration.md)
- Explore [results analysis](results_analysis.md)
