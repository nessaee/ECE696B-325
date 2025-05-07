"""
Model definitions for the UA-SLSM MLP training pipeline.
Provides functions for creating and configuring models for feature extraction and classification.
"""

import torch
import torch.nn as nn
from torchvision import models

from config import MOBILENET_FEATURE_DIM, MLP_HIDDEN_DIMS, DROPOUT_RATE
def get_mobilenet_model(num_classes: int, freeze_backbone: bool = True) -> models.MobileNetV3:
    """
    Load a pretrained MobileNetV3 Large model and configure it for transfer learning.
    
    Args:
        num_classes: Number of output classes for the classifier
        freeze_backbone: Whether to freeze the backbone layers for feature extraction
        
    Returns:
        A configured MobileNetV3 model
    """
    # Load pretrained MobileNetV3 Large with ImageNet weights
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = models.mobilenet_v3_large(weights=weights)

    # Freeze backbone if requested (for transfer learning)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classifier head with a new one for our task
    # For MobileNetV3 Large, the feature dimension is 960 after pooling
    in_features = model.classifier[0].in_features  # Should be 960 for mobilenet_v3_large
    
    # Create a new classifier with the appropriate number of output classes
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model


def get_model_feature_extractor(model):
    """
    Create a feature extractor from a MobileNetV3 model by removing the classifier head.
    
    Args:
        model: A MobileNetV3 model created by get_mobilenet_model
        
    Returns:
        A model that extracts features (excluding the classifier head)
    """
    class FeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super(FeatureExtractor, self).__init__()
            self.features = original_model.features
            self.avgpool = original_model.avgpool
        
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # Flatten the feature maps
            return x
    
    return FeatureExtractor(model)


class MLPClassifier(nn.Module):
    """
    MLP classifier with adaptive pooling for handling different input shapes.
    
    This classifier can handle both 2D feature vectors and 4D feature maps.
    For 4D inputs, it applies adaptive pooling before flattening.
    """
    def __init__(self, input_dim=MOBILENET_FEATURE_DIM, num_classes=2, 
                 hidden_dims=MLP_HIDDEN_DIMS, dropout_rate=DROPOUT_RATE):
        super().__init__()
        
        # Set default values if None is provided
        dropout_rate = DROPOUT_RATE if dropout_rate is None else dropout_rate
        hidden_dim = MLP_HIDDEN_DIMS[0] if not hidden_dims else hidden_dims[0]
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