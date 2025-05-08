# #!/bin/bash
# # Comprehensive sweep script for running feature extraction and training
# # for all model architectures and data versions

# # First, extract features for all models with normalized data
# echo "===== EXTRACTING FEATURES FOR NORMALIZED DATA ====="

# # Extract features using MobileNetV3
# echo "\n>> Extracting MobileNetV3 features (normalized)"
# ./run.sh --version normalized --binary --model mobilenet_v3 --stage features

# # Extract features using ResNet18
# echo "\n>> Extracting ResNet18 features (normalized)"
# ./run.sh --version normalized --binary --model resnet18 --stage features

# # Extract features using ResNet50
# echo "\n>> Extracting ResNet50 features (normalized)"
# ./run.sh --version normalized --binary --model resnet50 --stage features

# # Extract features using EfficientNetV2-S
# echo "\n>> Extracting EfficientNetV2-S features (normalized)"
# ./run.sh --version normalized --binary --model efficientnet_v2_s --stage features

# # Extract features for all models with RGB data
# echo "\n\n===== EXTRACTING FEATURES FOR RGB DATA ====="

# # Extract features using MobileNetV3
# echo "\n>> Extracting MobileNetV3 features (rgb)"
# ./run.sh --version rgb --binary --model mobilenet_v3 --stage features

# # Extract features using ResNet18
# echo "\n>> Extracting ResNet18 features (rgb)"
# ./run.sh --version rgb --binary --model resnet18 --stage features

# # Extract features using ResNet50
# echo "\n>> Extracting ResNet50 features (rgb)"
# ./run.sh --version rgb --binary --model resnet50 --stage features

# # Extract features using EfficientNetV2-S
# echo "\n>> Extracting EfficientNetV2-S features (rgb)"
# ./run.sh --version rgb --binary --model efficientnet_v2_s --stage features

# Now run training for all models with normalized data
echo "\n\n===== TRAINING MODELS WITH NORMALIZED DATA ====="

# Train with MobileNetV3 features
echo "\n>> Training with MobileNetV3 features (normalized)"
./run.sh --version normalized --binary --model mobilenet_v3 --stage train

# Train with ResNet18 features
echo "\n>> Training with ResNet18 features (normalized)"
./run.sh --version normalized --binary --model resnet18 --stage train

# Train with ResNet50 features
echo "\n>> Training with ResNet50 features (normalized)"
./run.sh --version normalized --binary --model resnet50 --stage train

# Train with EfficientNetV2-S features
echo "\n>> Training with EfficientNetV2-S features (normalized)"
./run.sh --version normalized --binary --model efficientnet_v2_s --stage train

# Now run training for all models with RGB data
echo "\n\n===== TRAINING MODELS WITH RGB DATA ====="

# Train with MobileNetV3 features
echo "\n>> Training with MobileNetV3 features (rgb)"
./run.sh --version rgb --binary --model mobilenet_v3 --stage train

# Train with ResNet18 features
echo "\n>> Training with ResNet18 features (rgb)"
./run.sh --version rgb --binary --model resnet18 --stage train

# Train with ResNet50 features
echo "\n>> Training with ResNet50 features (rgb)"
./run.sh --version rgb --binary --model resnet50 --stage train

# Train with EfficientNetV2-S features
echo "\n>> Training with EfficientNetV2-S features (rgb)"
./run.sh --version rgb --binary --model efficientnet_v2_s --stage train

# Run the consolidation script to generate the comparison table
echo "\n\n===== CONSOLIDATING RESULTS ====="
cd ..
./consolidate_results.sh

echo "\n\nAll processing completed successfully!"