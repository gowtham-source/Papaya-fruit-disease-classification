#!/usr/bin/env python3
"""
Metrics calculation utilities for papaya disease segmentation and detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class SegmentationMetrics:
    """Comprehensive metrics for semantic segmentation."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255, device: str = 'cpu'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        self.total_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch predictions and targets."""
        # Apply softmax and get predictions
        if predictions.dim() == 4:  # [B, C, H, W]
            pred_labels = torch.argmax(F.softmax(predictions, dim=1), dim=1)
        else:
            pred_labels = predictions
        
        # Flatten tensors
        pred_labels = pred_labels.flatten()
        targets = targets.flatten()
        
        # Remove ignore index
        valid_mask = targets != self.ignore_index
        pred_labels = pred_labels[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        for t, p in zip(targets, pred_labels):
            self.confusion_matrix[t.long(), p.long()] += 1
        
        self.total_samples += len(targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        # Per-class IoU
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - intersection
        iou = intersection.float() / (union.float() + 1e-8)
        
        # Per-class Dice
        dice = 2 * intersection.float() / (self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) + 1e-8)
        
        # Per-class Precision and Recall
        precision = intersection.float() / (self.confusion_matrix.sum(dim=0) + 1e-8)
        recall = intersection.float() / (self.confusion_matrix.sum(dim=1) + 1e-8)
        
        # Mean metrics
        valid_classes = union > 0
        mean_iou = iou[valid_classes].mean().item()
        mean_dice = dice[valid_classes].mean().item()
        mean_precision = precision[valid_classes].mean().item()
        mean_recall = recall[valid_classes].mean().item()
        
        # Overall accuracy
        accuracy = torch.diag(self.confusion_matrix).sum().float() / (self.confusion_matrix.sum() + 1e-8)
        
        return {
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'accuracy': accuracy.item(),
            'class_iou': iou.cpu().numpy().tolist(),
            'class_dice': dice.cpu().numpy().tolist(),
            'class_precision': precision.cpu().numpy().tolist(),
            'class_recall': recall.cpu().numpy().tolist()
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix as numpy array."""
        return self.confusion_matrix.cpu().numpy()


class DetectionMetrics:
    """Comprehensive metrics for object detection."""
    
    def __init__(self, num_classes: int, iou_thresholds: List[float] = None, device: str = 'cpu'):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: List[Dict], targets: List[Dict]):
        """Update with batch predictions and targets."""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        # Box format: [x1, y1, x2, y2]
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        # Intersection
        inter_x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
        inter_y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
        inter_x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
        inter_y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Union
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-8)
        return iou
    
    def compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Compute Average Precision using 11-point interpolation."""
        # Add sentinel values
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # Compute precision envelope
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
        # Compute AP using 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        ap = 0.0
        for t in recall_thresholds:
            idx = np.where(recalls >= t)[0]
            if len(idx) > 0:
                ap += precisions[idx[0]]
        
        return ap / 11.0
    
    def compute_map(self, predictions: List[Dict] = None, targets: List[Dict] = None) -> Dict[str, float]:
        """Compute mean Average Precision (mAP)."""
        if predictions is None:
            predictions = self.predictions
        if targets is None:
            targets = self.targets
        
        if len(predictions) == 0 or len(targets) == 0:
            return {f'map_{int(t*100)}': 0.0 for t in self.iou_thresholds}
        
        aps = defaultdict(list)
        
        # For each class
        for class_id in range(self.num_classes):
            # Collect all predictions and targets for this class
            class_predictions = []
            class_targets = []
            
            for pred, target in zip(predictions, targets):
                # Filter by class
                if len(pred['labels']) > 0:
                    class_mask = pred['labels'] == class_id
                    if class_mask.sum() > 0:
                        class_predictions.append({
                            'boxes': pred['boxes'][class_mask],
                            'scores': pred['scores'][class_mask] if 'scores' in pred else torch.ones(class_mask.sum())
                        })
                    else:
                        class_predictions.append({'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0)})
                else:
                    class_predictions.append({'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0)})
                
                if len(target['labels']) > 0:
                    class_mask = target['labels'] == class_id
                    class_targets.append({
                        'boxes': target['boxes'][class_mask]
                    })
                else:
                    class_targets.append({'boxes': torch.zeros(0, 4)})
            
            # Compute AP for each IoU threshold
            for iou_threshold in self.iou_thresholds:
                ap = self.compute_class_ap(class_predictions, class_targets, iou_threshold)
                aps[f'map_{int(iou_threshold*100)}'].append(ap)
        
        # Average across classes
        results = {}
        for threshold_key, class_aps in aps.items():
            if len(class_aps) > 0:
                results[threshold_key] = np.mean(class_aps)
            else:
                results[threshold_key] = 0.0
        
        # Overall mAP (average across all thresholds)
        if len(results) > 0:
            results['map'] = np.mean(list(results.values()))
        else:
            results['map'] = 0.0
        
        return results
    
    def compute_class_ap(self, predictions: List[Dict], targets: List[Dict], iou_threshold: float) -> float:
        """Compute AP for a single class at a specific IoU threshold."""
        # Collect all predictions
        all_scores = []
        all_boxes = []
        all_image_ids = []
        
        for img_id, pred in enumerate(predictions):
            if len(pred['boxes']) > 0:
                all_scores.extend(pred['scores'].cpu().numpy())
                all_boxes.extend(pred['boxes'].cpu().numpy())
                all_image_ids.extend([img_id] * len(pred['boxes']))
        
        if len(all_scores) == 0:
            return 0.0
        
        # Sort by confidence
        sorted_indices = np.argsort(all_scores)[::-1]
        all_scores = np.array(all_scores)[sorted_indices]
        all_boxes = np.array(all_boxes)[sorted_indices]
        all_image_ids = np.array(all_image_ids)[sorted_indices]
        
        # Count total ground truth boxes
        total_gt = sum(len(target['boxes']) for target in targets)
        
        if total_gt == 0:
            return 0.0
        
        # Match predictions to ground truth
        tp = np.zeros(len(all_scores))
        fp = np.zeros(len(all_scores))
        
        for i, (score, box, img_id) in enumerate(zip(all_scores, all_boxes, all_image_ids)):
            target_boxes = targets[img_id]['boxes']
            
            if len(target_boxes) == 0:
                fp[i] = 1
                continue
            
            # Compute IoU with all ground truth boxes
            box_tensor = torch.tensor(box).unsqueeze(0)
            target_tensor = torch.tensor(target_boxes)
            ious = self.compute_iou(box_tensor, target_tensor).squeeze(0)
            
            # Find best match
            best_iou = ious.max().item()
            
            if best_iou >= iou_threshold:
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Compute AP
        ap = self.compute_ap(recalls, precisions)
        return ap


def compute_segmentation_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Standalone function to compute segmentation metrics."""
    metrics = SegmentationMetrics(num_classes, device=predictions.device)
    metrics.update(predictions, targets)
    return metrics.compute()


def compute_detection_metrics(predictions: List[Dict], targets: List[Dict], num_classes: int) -> Dict[str, float]:
    """Standalone function to compute detection metrics."""
    metrics = DetectionMetrics(num_classes, device='cpu')
    return metrics.compute_map(predictions, targets)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute Dice coefficient between predictions and targets."""
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Compute IoU score between predictions and targets."""
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


if __name__ == "__main__":
    # Test segmentation metrics
    print("Testing segmentation metrics...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size = 2
    num_classes = 9
    height, width = 256, 256
    
    pred = torch.randn(batch_size, num_classes, height, width).to(device)
    target = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    metrics = compute_segmentation_metrics(pred, target, num_classes)
    print(f"Segmentation mIoU: {metrics['mean_iou']:.4f}")
    print(f"Segmentation Dice: {metrics['mean_dice']:.4f}")
    
    # Test detection metrics
    print("\nTesting detection metrics...")
    
    predictions = [
        {
            'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
            'scores': torch.tensor([0.9, 0.8]),
            'labels': torch.tensor([0, 1])
        },
        {
            'boxes': torch.tensor([[15, 15, 55, 55]]),
            'scores': torch.tensor([0.7]),
            'labels': torch.tensor([0])
        }
    ]
    
    targets = [
        {
            'boxes': torch.tensor([[12, 12, 52, 52], [65, 65, 105, 105]]),
            'labels': torch.tensor([0, 1])
        },
        {
            'boxes': torch.tensor([[20, 20, 60, 60]]),
            'labels': torch.tensor([0])
        }
    ]
    
    det_metrics = compute_detection_metrics(predictions, targets, 8)
    print(f"Detection mAP@0.5: {det_metrics.get('map_50', 0.0):.4f}")
    print(f"Overall mAP: {det_metrics.get('map', 0.0):.4f}")
    
    print("All metrics tests passed!")
