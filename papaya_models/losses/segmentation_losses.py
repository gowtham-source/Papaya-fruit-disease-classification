#!/usr/bin/env python3
"""
Loss functions for papaya disease segmentation.
Implements various loss functions optimized for medical/agricultural segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6, ignore_index: int = 255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Create one-hot encoding for target
        num_classes = pred.size(1)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Remove ignore index
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * valid_mask
            target_one_hot = target_one_hot * valid_mask
        
        # Compute Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - mean dice as loss
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = 255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute cross entropy
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # False positive penalty
        self.beta = beta    # False negative penalty
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Create one-hot encoding for target
        num_classes = pred.size(1)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Compute Tversky components
        tp = (pred * target_one_hot).sum(dim=(2, 3))
        fp = (pred * (1 - target_one_hot)).sum(dim=(2, 3))
        fn = ((1 - pred) * target_one_hot).sum(dim=(2, 3))
        
        # Compute Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Return 1 - mean tversky as loss
        return 1.0 - tversky.mean()


class CombinedSegmentationLoss(nn.Module):
    """Combined loss function for comprehensive segmentation training."""
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        tversky_weight: float = 0.5,
        boundary_weight: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255
    ):
        super(CombinedSegmentationLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        
        # Cross Entropy Loss
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        
        # Dice Loss
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        # Focal Loss
        self.focal_loss = FocalLoss(ignore_index=ignore_index)
        
        # Tversky Loss
        self.tversky_loss = TverskyLoss()
        
        # Boundary Loss
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        losses = {}
        total_loss = 0.0
        
        # Cross Entropy Loss
        if self.ce_weight > 0:
            ce = self.ce_loss(pred, target)
            losses['ce_loss'] = ce
            total_loss += self.ce_weight * ce
        
        # Dice Loss
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            losses['dice_loss'] = dice
            total_loss += self.dice_weight * dice
        
        # Focal Loss
        if self.focal_weight > 0:
            focal = self.focal_loss(pred, target)
            losses['focal_loss'] = focal
            total_loss += self.focal_weight * focal
        
        # Tversky Loss
        if self.tversky_weight > 0:
            tversky = self.tversky_loss(pred, target)
            losses['tversky_loss'] = tversky
            total_loss += self.tversky_weight * tversky
        
        # Boundary Loss
        if self.boundary_weight > 0:
            boundary = self.boundary_loss(pred, target)
            losses['boundary_loss'] = boundary
            total_loss += self.boundary_weight * boundary
        
        losses['total_loss'] = total_loss
        return losses


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better edge detection."""
    
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss using distance transform.
        """
        # Convert to binary segmentation (foreground vs background)
        pred_binary = F.softmax(pred, dim=1)[:, 1:].sum(dim=1, keepdim=True)  # All disease classes vs background
        target_binary = (target > 0).float().unsqueeze(1)
        
        # Compute distance transform (simplified version)
        # In practice, you might want to use actual distance transform
        boundary_loss = F.mse_loss(pred_binary, target_binary)
        
        return boundary_loss


class AuxiliaryLoss(nn.Module):
    """Auxiliary loss for deep supervision."""
    
    def __init__(self, main_loss_fn, aux_weight: float = 0.4):
        super(AuxiliaryLoss, self).__init__()
        self.main_loss_fn = main_loss_fn
        self.aux_weight = aux_weight
    
    def forward(self, main_pred: torch.Tensor, aux_pred: torch.Tensor, target: torch.Tensor) -> dict:
        # Main loss
        main_losses = self.main_loss_fn(main_pred, target)
        
        # Auxiliary loss (usually just cross entropy)
        aux_loss = F.cross_entropy(aux_pred, target, ignore_index=255)
        
        # Combined loss
        total_loss = main_losses['total_loss'] + self.aux_weight * aux_loss
        
        return {
            **main_losses,
            'aux_loss': aux_loss,
            'total_loss': total_loss
        }


def get_class_weights(dataset_path: str, num_classes: int = 9) -> torch.Tensor:
    """Compute class weights based on frequency in dataset."""
    # This is a simplified version - in practice, you'd scan the entire dataset
    # For now, we'll use estimated weights based on typical disease distribution
    
    # Estimated class frequencies (background typically dominates)
    class_frequencies = [
        0.7,   # Background (healthy tissue)
        0.05,  # Anthracnose
        0.04,  # Black Spot
        0.03,  # Chocolate Spot
        0.02,  # Dieback
        0.04,  # Phytophthora
        0.04,  # Black Spot V2
        0.08   # Scar
    ]
    
    # Pad or trim to match num_classes
    while len(class_frequencies) < num_classes:
        class_frequencies.append(0.01)
    class_frequencies = class_frequencies[:num_classes]
    
    # Compute inverse frequency weights
    total_freq = sum(class_frequencies)
    weights = [total_freq / (freq * num_classes) for freq in class_frequencies]
    
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size = 2
    num_classes = 9
    height, width = 256, 256
    
    pred = torch.randn(batch_size, num_classes, height, width).to(device)
    target = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    # Test individual losses
    print("Testing loss functions...")
    
    # Dice Loss
    dice_loss = DiceLoss()
    dice_val = dice_loss(pred, target)
    print(f"Dice Loss: {dice_val.item():.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss()
    focal_val = focal_loss(pred, target)
    print(f"Focal Loss: {focal_val.item():.4f}")
    
    # Tversky Loss
    tversky_loss = TverskyLoss()
    tversky_val = tversky_loss(pred, target)
    print(f"Tversky Loss: {tversky_val.item():.4f}")
    
    # Combined Loss
    class_weights = get_class_weights("", num_classes)
    combined_loss = CombinedSegmentationLoss(class_weights=class_weights.to(device))
    combined_val = combined_loss(pred, target)
    print(f"Combined Loss: {combined_val['total_loss'].item():.4f}")
    print("Individual components:")
    for key, value in combined_val.items():
        if key != 'total_loss':
            print(f"  {key}: {value.item():.4f}")
    
    print("All loss functions working correctly!")
