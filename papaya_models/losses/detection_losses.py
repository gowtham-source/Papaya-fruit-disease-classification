#!/usr/bin/env python3
"""
Loss functions for papaya disease detection.
Implements optimized loss functions for fast hybrid detection model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class FastDetectionLoss(nn.Module):
    """
    Fast detection loss optimized for speed and accuracy.
    Combines classification, regression, and objectness losses.
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lambda_coord: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        iou_threshold: float = 0.5,
        anchor_free: bool = True
    ):
        super(FastDetectionLoss, self).__init__()
        
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.iou_threshold = iou_threshold
        self.anchor_free = anchor_free
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between predicted and target boxes."""
        # Convert from center format to corner format
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2
        
        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2
        
        # Compute intersection
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Compute union
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        
        # Compute IoU
        iou = inter_area / (union_area + 1e-8)
        return iou
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for classification."""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def smooth_l1_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute smooth L1 loss for bounding box regression."""
        diff = torch.abs(pred - target)
        loss = torch.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
        return (loss * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)
    
    def ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute Complete IoU loss for better localization."""
        # Extract coordinates
        pred_x, pred_y, pred_w, pred_h = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
        target_x, target_y, target_w, target_h = target_boxes[..., 0], target_boxes[..., 1], target_boxes[..., 2], target_boxes[..., 3]
        
        # Convert to corner coordinates
        pred_x1, pred_y1 = pred_x - pred_w / 2, pred_y - pred_h / 2
        pred_x2, pred_y2 = pred_x + pred_w / 2, pred_y + pred_h / 2
        target_x1, target_y1 = target_x - target_w / 2, target_y - target_h / 2
        target_x2, target_y2 = target_x + target_w / 2, target_y + target_h / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-8)
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        enclose_area = enclose_w * enclose_h
        
        # Distance between centers
        center_distance = (pred_x - target_x) ** 2 + (pred_y - target_y) ** 2
        diagonal_distance = enclose_w ** 2 + enclose_h ** 2
        
        # Aspect ratio consistency
        v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(target_w / (target_h + 1e-8)) - torch.atan(pred_w / (pred_h + 1e-8)), 2)
        alpha = v / (1 - iou + v + 1e-8)
        
        # CIoU
        ciou = iou - center_distance / (diagonal_distance + 1e-8) - alpha * v
        
        # CIoU loss
        ciou_loss = 1 - ciou
        
        return (ciou_loss * mask).sum() / (mask.sum() + 1e-8)
    
    def forward(self, predictions: Dict[str, List[torch.Tensor]], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: Dict containing lists of predictions for each FPN level
            targets: Ground truth targets [batch_size, max_objects, 5] (class, x, y, w, h)
        """
        batch_size = targets.size(0)
        device = targets.device
        
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_obj_loss = 0.0
        total_center_loss = 0.0 if self.anchor_free else None
        total_scale_loss = 0.0 if self.anchor_free else None
        
        num_levels = len(predictions['classification'])
        
        for level_idx in range(num_levels):
            # Get predictions for this level
            cls_pred = predictions['classification'][level_idx]  # [B, num_anchors * num_classes, H, W]
            reg_pred = predictions['regression'][level_idx]      # [B, num_anchors * 4, H, W]
            obj_pred = predictions['objectness'][level_idx]      # [B, num_anchors, H, W]
            
            B, _, H, W = cls_pred.shape
            num_anchors = obj_pred.size(1)
            
            # Reshape predictions
            cls_pred = cls_pred.view(B, num_anchors, self.num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
            reg_pred = reg_pred.view(B, num_anchors, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous()  # [B, H, W, num_anchors]
            
            # Create grid for this level
            grid_h, grid_w = H, W
            scale_h, scale_w = 1.0 / grid_h, 1.0 / grid_w
            
            # Assign targets to this level
            level_targets = self.assign_targets(targets, grid_h, grid_w, scale_h, scale_w, num_anchors)
            
            # Extract target components
            target_cls = level_targets['class']        # [B, H, W, num_anchors]
            target_reg = level_targets['regression']   # [B, H, W, num_anchors, 4]
            target_obj = level_targets['objectness']   # [B, H, W, num_anchors]
            obj_mask = target_obj > 0
            
            # Classification loss (only for positive samples)
            if obj_mask.sum() > 0:
                pos_cls_pred = cls_pred[obj_mask]
                pos_cls_target = target_cls[obj_mask].long()
                cls_loss = self.focal_loss(pos_cls_pred, pos_cls_target)
            else:
                cls_loss = torch.tensor(0.0, device=device)
            
            # Regression loss (only for positive samples)
            if obj_mask.sum() > 0:
                pos_reg_pred = reg_pred[obj_mask]
                pos_reg_target = target_reg[obj_mask]
                pos_mask = obj_mask[obj_mask]  # All True for positive samples
                reg_loss = self.ciou_loss(pos_reg_pred, pos_reg_target, pos_mask)
            else:
                reg_loss = torch.tensor(0.0, device=device)
            
            # Objectness loss
            obj_loss = self.bce_loss(obj_pred, target_obj.float()).mean()
            
            # Accumulate losses
            total_cls_loss += cls_loss
            total_reg_loss += reg_loss
            total_obj_loss += obj_loss
            
            # Anchor-free losses
            if self.anchor_free and 'center' in predictions:
                center_pred = predictions['center'][level_idx]
                scale_pred = predictions['scale'][level_idx]
                
                # Center loss (simplified)
                center_loss = F.mse_loss(center_pred, target_obj.unsqueeze(1).float())
                
                # Scale loss (simplified)
                if obj_mask.sum() > 0:
                    scale_target = target_reg[..., 2:4]  # width, height
                    scale_loss = F.mse_loss(scale_pred.permute(0, 2, 3, 1)[obj_mask], scale_target[obj_mask])
                else:
                    scale_loss = torch.tensor(0.0, device=device)
                
                total_center_loss += center_loss
                total_scale_loss += scale_loss
        
        # Average over levels
        total_cls_loss /= num_levels
        total_reg_loss /= num_levels
        total_obj_loss /= num_levels
        
        # Combine losses
        total_loss = (
            total_cls_loss +
            self.lambda_coord * total_reg_loss +
            self.lambda_obj * total_obj_loss
        )
        
        losses = {
            'total_loss': total_loss,
            'cls_loss': total_cls_loss,
            'reg_loss': total_reg_loss,
            'obj_loss': total_obj_loss
        }
        
        if self.anchor_free:
            total_center_loss /= num_levels
            total_scale_loss /= num_levels
            losses['center_loss'] = total_center_loss
            losses['scale_loss'] = total_scale_loss
            losses['total_loss'] += 0.5 * total_center_loss + 0.5 * total_scale_loss
        
        return losses
    
    def assign_targets(
        self, 
        targets: torch.Tensor, 
        grid_h: int, 
        grid_w: int, 
        scale_h: float, 
        scale_w: float,
        num_anchors: int
    ) -> Dict[str, torch.Tensor]:
        """Assign targets to grid cells."""
        batch_size = targets.size(0)
        device = targets.device
        
        # Initialize target tensors
        target_cls = torch.zeros(batch_size, grid_h, grid_w, num_anchors, device=device)
        target_reg = torch.zeros(batch_size, grid_h, grid_w, num_anchors, 4, device=device)
        target_obj = torch.zeros(batch_size, grid_h, grid_w, num_anchors, device=device)
        
        for b in range(batch_size):
            # Get valid targets for this batch
            batch_targets = targets[b]
            valid_mask = batch_targets[:, 0] >= 0  # Valid class IDs
            valid_targets = batch_targets[valid_mask]
            
            for target in valid_targets:
                class_id, x_center, y_center, width, height = target
                
                # Convert to grid coordinates
                grid_x = x_center * grid_w
                grid_y = y_center * grid_h
                
                # Get grid cell indices
                grid_x_int = int(grid_x.clamp(0, grid_w - 1))
                grid_y_int = int(grid_y.clamp(0, grid_h - 1))
                
                # Assign to all anchors in this cell (simplified assignment)
                for anchor_idx in range(num_anchors):
                    target_cls[b, grid_y_int, grid_x_int, anchor_idx] = class_id
                    target_reg[b, grid_y_int, grid_x_int, anchor_idx] = torch.tensor([x_center, y_center, width, height])
                    target_obj[b, grid_y_int, grid_x_int, anchor_idx] = 1.0
        
        return {
            'class': target_cls,
            'regression': target_reg,
            'objectness': target_obj
        }


class YOLOLoss(nn.Module):
    """YOLO-style loss for comparison."""
    
    def __init__(self, num_classes: int = 8, lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Standard YOLO loss computation."""
        # Simplified YOLO loss implementation
        # In practice, this would be more complex
        
        # Extract predictions
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4]
        pred_cls = predictions[..., 5:]
        
        # Extract targets
        target_boxes = targets[..., 1:5]
        target_conf = targets[..., 0] > 0
        target_cls = targets[..., 0].long()
        
        # Coordinate loss
        coord_loss = F.mse_loss(pred_boxes[target_conf], target_boxes[target_conf])
        
        # Confidence loss
        conf_loss = F.bce_loss(pred_conf, target_conf.float())
        
        # Classification loss
        if target_conf.sum() > 0:
            cls_loss = F.cross_entropy(pred_cls[target_conf], target_cls[target_conf])
        else:
            cls_loss = torch.tensor(0.0, device=predictions.device)
        
        total_loss = self.lambda_coord * coord_loss + conf_loss + cls_loss
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'conf_loss': conf_loss,
            'cls_loss': cls_loss
        }


if __name__ == "__main__":
    # Test detection loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy predictions
    batch_size = 2
    num_classes = 8
    
    predictions = {
        'classification': [
            torch.randn(batch_size, 3 * num_classes, 20, 20).to(device),
            torch.randn(batch_size, 3 * num_classes, 10, 10).to(device)
        ],
        'regression': [
            torch.randn(batch_size, 3 * 4, 20, 20).to(device),
            torch.randn(batch_size, 3 * 4, 10, 10).to(device)
        ],
        'objectness': [
            torch.randn(batch_size, 3, 20, 20).to(device),
            torch.randn(batch_size, 3, 10, 10).to(device)
        ]
    }
    
    # Create dummy targets
    targets = torch.zeros(batch_size, 10, 5).to(device)  # [class, x, y, w, h]
    targets[0, 0] = torch.tensor([1, 0.5, 0.5, 0.2, 0.3])  # One object in first image
    targets[1, 0] = torch.tensor([2, 0.3, 0.7, 0.1, 0.2])  # One object in second image
    
    # Test loss
    loss_fn = FastDetectionLoss(num_classes=num_classes, anchor_free=False)
    losses = loss_fn(predictions, targets)
    
    print("Detection loss test successful!")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Classification loss: {losses['cls_loss'].item():.4f}")
    print(f"Regression loss: {losses['reg_loss'].item():.4f}")
    print(f"Objectness loss: {losses['obj_loss'].item():.4f}")
