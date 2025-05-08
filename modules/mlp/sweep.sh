# ./run.sh --version normalized --binary --stage features
# ./run.sh --version rgb --binary --stage features

# ./run.sh --version normalized --binary --stage train
# ./run.sh --version rgb --binary --stage train

# Extract features using MobileNetV3
./run.sh --version normalized --binary --model mobilenet_v3 --stage train

# Extract features using ResNet18
./run.sh --version normalized --binary --model resnet18 --stage train

# Extract features using ResNet50
./run.sh --version normalized --binary --model resnet50 --stage train

# Train with features from EfficientNetV2-S
./run.sh --version normalized --binary --model efficientnet_v2_s --stage train



# Extract features using MobileNetV3
./run.sh --version rgb --binary --model mobilenet_v3 --stage train

# Extract features using ResNet18
./run.sh --version rgb --binary --model resnet18 --stage train

# Extract features using ResNet50
./run.sh --version rgb --binary --model resnet50 --stage train

# Train with features from EfficientNetV2-S
./run.sh --version rgb --binary --model efficientnet_v2_s --stage train