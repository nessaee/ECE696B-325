#!/bin/bash
# UA-SLSM MLP Training Pipeline Runner

# Default settings
DATA_VERSION="normalized"
MODE="binary"  # binary or multi
STAGE="all"    # features, train, or all
MODEL="mobilenet_v3"  # mobilenet_v3, resnet18, resnet50, efficientnet_v2_s, vit_b_16

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --version)
      DATA_VERSION="$2"
      shift 2
      ;;
    --binary)
      MODE="binary"
      shift
      ;;
    --multi)
      MODE="multi"
      shift
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./run.sh [options]"
      echo "Options:"
      echo "  --version VERSION    Dataset version to use (default: normalized)"
      echo "  --binary             Use binary classification mode (default)"
      echo "  --multi              Use multi-class classification mode"
      echo "  --stage STAGE        Pipeline stage to run (features, train, all)"
      echo "  --model MODEL        Model type to use for feature extraction (default: mobilenet_v3)"
      echo "                       Options: mobilenet_v3, resnet18, resnet50, efficientnet_v2_s, vit_b_16"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './run.sh --help' for usage information."
      exit 1
      ;;
  esac
done

# Print settings
echo "Running UA-SLSM MLP pipeline with settings:"
echo "  - Data version: $DATA_VERSION"
echo "  - Classification mode: $MODE"
echo "  - Model: $MODEL"
echo "  - Stage: $STAGE"
echo ""

# Create required directories
mkdir -p data/features/$DATA_VERSION/$MODEL
mkdir -p data/mlp_training_output

# Run feature extraction if needed
if [[ "$STAGE" == "features" || "$STAGE" == "all" ]]; then
  echo "=== Running feature extraction using $MODEL ==="
  if [[ "$MODE" == "binary" ]]; then
    python features.py --version $DATA_VERSION --binary --model $MODEL
    FEATURE_STATUS=$?
  else
    python features.py --version $DATA_VERSION --multi --model $MODEL
    FEATURE_STATUS=$?
  fi
  
  if [ $FEATURE_STATUS -ne 0 ]; then
    echo "Feature extraction failed with status $FEATURE_STATUS. Exiting."
    exit 1
  fi
  echo "Feature extraction completed successfully."
fi

# Run training if needed
if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  # Only run training if features were extracted successfully or if we're only doing training
  if [[ "$STAGE" == "train" || $FEATURE_STATUS -eq 0 ]]; then
    echo "=== Running model training with features from $MODEL ==="
    python train.py --version $DATA_VERSION --model $MODEL
    TRAIN_STATUS=$?
    
    if [ $TRAIN_STATUS -ne 0 ]; then
      echo "Training failed with status $TRAIN_STATUS. Exiting."
      exit 1
    fi
    echo "Training completed successfully."
  else
    echo "Skipping training due to feature extraction failure."
    exit 1
  fi
fi

echo "All requested stages completed."
