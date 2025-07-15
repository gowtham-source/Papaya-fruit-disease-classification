#!/usr/bin/env python3
"""
U-Net with ResNet Encoder and Cross Connections for Papaya Disease Segmentation
Implements a state-of-the-art segmentation model with skip connections and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import segmentation_models_pytorch as smp


class SimpleUNet(nn.Module):
    """Simple U-Net wrapper using segmentation_models_pytorch."""
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 9,
        activation: Optional[str] = None,
        **kwargs  # Ignore additional parameters
    ):
        super(SimpleUNet, self).__init__()
        
        # Create standard U-Net model from segmentation_models_pytorch
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple forward pass - no custom layers or modifications
        return self.model(x)


class SimpleUNetWithAuxiliaryLoss(nn.Module):
    """Simple U-Net with basic auxiliary loss."""
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 9,
        activation: Optional[str] = None,
        **kwargs  # Ignore additional parameters
    ):
        super(SimpleUNetWithAuxiliaryLoss, self).__init__()
        
        # Create standard U-Net model
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
        
        # Get encoder channels
        encoder_channels = self.model.encoder.out_channels
        
        # Use the second-to-last encoder feature for auxiliary loss
        aux_in_channels = encoder_channels[-2]
        
        # Simple auxiliary classifier
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(aux_in_channels, classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get encoder features
        features = self.model.encoder(x)
        
        # Get main output
        main_output = self.model(x)
        
        # Extract features for auxiliary loss (second-to-last encoder output)
        aux_features = features[-2]
        
        # Apply auxiliary classifier
        aux_output = self.aux_classifier(aux_features)
        
        # Resize auxiliary output to match input size
        aux_output = F.interpolate(
            aux_output, size=x.shape[2:], mode='bilinear', align_corners=True
        )
        
        # Return main output and auxiliary output
        return main_output, aux_output


def create_unet_resnet(
    encoder_name: str = 'resnet50',
    encoder_weights: str = 'imagenet',
    in_channels: int = 3,
    classes: int = 9,
    activation: Optional[str] = None,
    use_auxiliary_loss: bool = True,
    **kwargs
) -> nn.Module:
    """Create a U-Net ResNet model with optional auxiliary loss."""
    
    # Filter kwargs for base model
    base_kwargs = {
        'encoder_name': encoder_name,
        'encoder_weights': encoder_weights,
        'in_channels': in_channels,
        'classes': classes,
        'activation': activation
    }
    
    if use_auxiliary_loss:
        return SimpleUNetWithAuxiliaryLoss(**base_kwargs)
    else:
        return SimpleUNet(**base_kwargs)


# Model variants
def unet_resnet18(num_classes: int = 9, **kwargs) -> nn.Module:
    """U-Net with ResNet-18 encoder (faster training)."""
    return create_unet_resnet(encoder_name='resnet18', classes=num_classes, **kwargs)


def unet_resnet34(num_classes: int = 9, **kwargs) -> nn.Module:
    """U-Net with ResNet-34 encoder (balanced speed/accuracy)."""
    return create_unet_resnet(encoder_name='resnet34', classes=num_classes, **kwargs)


def unet_resnet50(num_classes: int = 9, **kwargs) -> nn.Module:
    """U-Net with ResNet-50 encoder (high accuracy)."""
    return create_unet_resnet(encoder_name='resnet50', classes=num_classes, **kwargs)


def unet_resnet101(num_classes: int = 9, **kwargs) -> nn.Module:
    """U-Net with ResNet-101 encoder (highest accuracy)."""
    return create_unet_resnet(encoder_name='resnet101', classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = unet_resnet50(num_classes=9, use_auxiliary_loss=True)
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        main_output, aux_output = model(x)
        
    print("Model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Main segmentation output shape: {main_output.shape}")
    print(f"Auxiliary segmentation output shape: {aux_output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")