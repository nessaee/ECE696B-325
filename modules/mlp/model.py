"""
Model definitions for the UA-SLSM MLP training pipeline.
Provides functions for creating and configuring models for feature extraction and classification.
"""

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from enum import Enum

from config import MOBILENET_FEATURE_DIM, MLP_HIDDEN_DIMS, DROPOUT_RATE, FEATURE_DIMS


class ModelType(Enum):
    """Enum for supported model types"""
    MOBILENET_V3 = "mobilenet_v3"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    EFFICIENTNET_V2_S = "efficientnet_v2_s"
    VIT_B_16 = "vit_b_16"

    @staticmethod
    def from_string(model_name):
        """Convert string to ModelType enum"""
        model_name = model_name.lower()
        for model_type in ModelType:
            if model_name == model_type.value:
                return model_type
        raise ValueError(f"Unknown model type: {model_name}. Supported types: {[m.value for m in ModelType]}")


def get_feature_dimensions(model_type):
    """Get the feature dimensions for different model types

    Args:
        model_type: The type of model (ModelType enum)

    Returns:
        The feature dimension for the model
    """
    feature_dims = {
        ModelType.MOBILENET_V3: 960,
        ModelType.RESNET18: 512,
        ModelType.RESNET50: 2048,
        ModelType.EFFICIENTNET_V2_S: 1280,
        ModelType.VIT_B_16: 768
    }
    return feature_dims.get(model_type, 960)  # Default to MobileNetV3 if not found


def get_model(model_type=ModelType.MOBILENET_V3, num_classes=2, freeze_backbone=True):
    """Load a pretrained model and configure it for transfer learning.

    Args:
        model_type: Type of model to load (ModelType enum)
        num_classes: Number of output classes for the classifier
        freeze_backbone: Whether to freeze the backbone layers for feature extraction

    Returns:
        A configured model
    """
    # Handle string input for model_type
    if isinstance(model_type, str):
        model_type = ModelType.from_string(model_type)

    # Load the appropriate pretrained model
    if model_type == ModelType.MOBILENET_V3:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
        feature_dim = model.classifier[0].in_features

    elif model_type == ModelType.RESNET18:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features

    elif model_type == ModelType.RESNET50:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features

    elif model_type == ModelType.EFFICIENTNET_V2_S:
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        feature_dim = model.classifier[1].in_features

    elif model_type == ModelType.VIT_B_16:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        feature_dim = model.heads[0].in_features

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Freeze backbone if requested (for transfer learning)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head with a new one for our task
    if model_type == ModelType.MOBILENET_V3:
        model.classifier = nn.Linear(feature_dim, num_classes)
    elif model_type in [ModelType.RESNET18, ModelType.RESNET50]:
        model.fc = nn.Linear(feature_dim, num_classes)
    elif model_type == ModelType.EFFICIENTNET_V2_S:
        model.classifier[1] = nn.Linear(feature_dim, num_classes)
    elif model_type == ModelType.VIT_B_16:
        model.heads[0] = nn.Linear(feature_dim, num_classes)

    return model, feature_dim


# Legacy function for backward compatibility
def get_mobilenet_model(num_classes: int, freeze_backbone: bool = True):
    """Load a pretrained MobileNetV3 Large model and configure it for transfer learning.

    Args:
        num_classes: Number of output classes for the classifier
        freeze_backbone: Whether to freeze the backbone layers for feature extraction

    Returns:
        A configured MobileNetV3 model
    """
    model, _ = get_model(ModelType.MOBILENET_V3, num_classes, freeze_backbone)
    return model


def get_model_feature_extractor(model, model_type=ModelType.MOBILENET_V3):
    """Create a feature extractor from a model by removing the classifier head.

    Args:
        model: A model created by get_model
        model_type: Type of model (ModelType enum)

    Returns:
        A model that extracts features (excluding the classifier head)
    """
    # Handle string input for model_type
    if isinstance(model_type, str):
        model_type = ModelType.from_string(model_type)

    if model_type == ModelType.MOBILENET_V3:
        class MobileNetFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super(MobileNetFeatureExtractor, self).__init__()
                self.features = original_model.features
                self.avgpool = original_model.avgpool

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)  # Flatten the feature maps
                return x

        return MobileNetFeatureExtractor(model)

    elif model_type in [ModelType.RESNET18, ModelType.RESNET50]:
        class ResNetFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super(ResNetFeatureExtractor, self).__init__()
                self.conv1 = original_model.conv1
                self.bn1 = original_model.bn1
                self.relu = original_model.relu
                self.maxpool = original_model.maxpool
                self.layer1 = original_model.layer1
                self.layer2 = original_model.layer2
                self.layer3 = original_model.layer3
                self.layer4 = original_model.layer4
                self.avgpool = original_model.avgpool

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x

        return ResNetFeatureExtractor(model)

    elif model_type == ModelType.EFFICIENTNET_V2_S:
        class EfficientNetFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super(EfficientNetFeatureExtractor, self).__init__()
                self.features = original_model.features
                self.avgpool = original_model.avgpool

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x

        return EfficientNetFeatureExtractor(model)

    elif model_type == ModelType.VIT_B_16:
        class ViTFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super(ViTFeatureExtractor, self).__init__()
                self.conv_proj = original_model.conv_proj
                self.encoder = original_model.encoder

            def forward(self, x):
                # Get patch embeddings
                x = self.conv_proj(x)
                # Add class token and position embeddings
                n = x.shape[0]
                batch_class_token = original_model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = x + original_model.encoder.pos_embedding
                # Apply transformer blocks
                x = self.encoder(x)
                # Extract class token for classification
                x = x[:, 0]
                return x

        return ViTFeatureExtractor(model)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model):
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLPClassifier(nn.Module):
    """MLP classifier with adaptive pooling for handling different input shapes.

    This classifier can handle both 2D feature vectors and 4D feature maps.
    For 4D inputs, it applies adaptive pooling before flattening.
    """
    def __init__(self, input_dim=MOBILENET_FEATURE_DIM, num_classes=2, 
                 hidden_dims=None, dropout_rate=None, model_type="mobilenet_v3"):
        super().__init__()

        # Handle model_type as string or enum
        if isinstance(model_type, str):
            model_type_str = model_type
        else:
            model_type_str = model_type.value
            
        # Set default values if None is provided
        if dropout_rate is None:
            dropout_rate = DROPOUT_RATE[model_type_str] if isinstance(DROPOUT_RATE, dict) else DROPOUT_RATE
            
        if hidden_dims is None:
            if isinstance(MLP_HIDDEN_DIMS, dict):
                hidden_dims = MLP_HIDDEN_DIMS[model_type_str]
            else:
                hidden_dims = MLP_HIDDEN_DIMS
                
        hidden_dim = hidden_dims[0] if isinstance(hidden_dims, list) and hidden_dims else 512
        output_dim = 1 if num_classes == 2 else num_classes

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

        # Store configuration for reference
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

    def forward(self, x):
        """
        Forward pass through the MLP classifier.
        
        Handles reshaping for 2D feature vectors to make them compatible with AdaptiveAvgPool2d.
        
        Args:
            x: Input features, either [batch_size, features] or [batch_size, channels, height, width]
            
        Returns:
            Output logits
        """
        if len(x.shape) == 2:  # [batch_size, features]
            # Handle reshaping for AdaptiveAvgPool2d which expects 4D input
            feature_dim = x.shape[1]
            height = int(feature_dim ** 0.5)
            if height * height != feature_dim:
                # If features can't be reshaped to a square, add dummy dimensions
                x = x.unsqueeze(-1).unsqueeze(-1)
            else:
                # Reshape to [batch_size, 1, height, height]
                x = x.view(x.shape[0], 1, height, height)
        
        return self.mlp(x)