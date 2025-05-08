# UA-SLSM MLP Training Pipeline

## Overview

This repository contains a comprehensive pipeline for training Multi-Layer Perceptron (MLP) classifiers on features extracted from various pre-trained CNN and transformer architectures. The pipeline supports multiple model architectures, data preprocessing versions, and provides extensive evaluation capabilities.

## Key Features

- **Multiple Feature Extractors**: Support for MobileNetV3, ResNet18, ResNet50, EfficientNetV2-S
- **Model-specific Hyperparameters**: Optimized hyperparameters for each architecture
- **Cross-validation**: K-fold cross-validation with comprehensive metrics
- **Performance Tracking**: Detailed metrics including F1 score, accuracy, AUC, and confusion matrices
- **Efficiency Metrics**: Parameter counts, training time, and inference time measurements
- **Results Consolidation**: Automated generation of publication-ready LaTeX tables

## Documentation

- [Getting Started](docs/getting_started.md): Installation and basic usage
- [Pipeline Overview](docs/pipeline_overview.md): Detailed explanation of the training pipeline
- [Configuration Guide](docs/configuration.md): How to configure the pipeline
- [Results Analysis](docs/results_analysis.md): Understanding and interpreting results

## Quick Start

```bash
# Clone the repository
git clone https://github.com/nessaee/ECE696B-325.git
cd ECE696B-325

# Run the complete pipeline for all models and data versions
cd modules/mlp
./sweep.sh

# Or run a specific model and data version
./run.sh --version normalized --binary --model resnet50 --stage all
```

## Reproducibility Example

**Note**: MobileNetV3 features have been pre-extracted and uploaded to the repository for reproducibility purposes. You can train the MobileNetV3 model directly without needing to run the feature extraction step.

```bash
# Clone the repository
git clone https://github.com/nessaee/ECE696B-325.git
cd ECE696B-325

# Train MobileNetV3 model with normalized data
cd modules/mlp
./run.sh --version normalized --binary --model mobilenet_v3 --stage train

# Train MobileNetV3 model with RGB data
./run.sh --version rgb --binary --model mobilenet_v3 --stage train

# Generate consolidated results table
./consolidate_results.sh
```

The pre-extracted features allow you to quickly evaluate the pipeline's training and evaluation capabilities without the computational overhead of feature extraction.

## Citation

If you use this codebase in your research, please cite:

```
@misc{UA-SLSM-Pipeline,
  author = {Ameer Nessaee},
  title = {SLSM Classification Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/nessaee/ECE696B-325}
}
```

<!-- ## License -->

<!-- This project is licensed under the MIT License - see the LICENSE file for details. -->
