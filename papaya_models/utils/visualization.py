#!/usr/bin/env python3
"""
Visualization utilities for papaya disease segmentation and detection results.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn.functional as F


# Disease class colors (BGR format for OpenCV)
DISEASE_COLORS = {
    0: (128, 128, 128),  # Background - Gray
    1: (0, 0, 255),      # Anthracnose - Red
    2: (0, 165, 255),    # Black Spot - Orange  
    3: (0, 255, 255),    # Chocolate Spot - Yellow
    4: (255, 0, 255),    # Dieback - Magenta
    5: (255, 255, 0),    # Phytophthora - Cyan
    6: (128, 0, 128),    # Black Spot V2 - Purple
    7: (0, 255, 0),      # Scar - Green
    8: (255, 0, 0),      # Reserved - Blue
}

# Disease class names
DISEASE_NAMES = {
    0: 'Background',
    1: 'Anthracnose',
    2: 'Black Spot',
    3: 'Chocolate Spot', 
    4: 'Dieback',
    5: 'Phytophthora',
    6: 'Black Spot V2',
    7: 'Scar',
    8: 'Reserved'
}


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def denormalize_image(image: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """Denormalize image tensor back to [0, 255] range."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet defaults
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet defaults
    
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    # Denormalize
    image = image * std + mean
    
    # Clip to [0, 1] and convert to [0, 255]
    image = np.clip(image, 0, 1) * 255
    return image.astype(np.uint8)


def create_segmentation_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of segmentation mask on original image."""
    # Ensure image is in the correct format (H, W, 3) and uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:  # If normalized to [0, 1]
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:  # Single channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Ensure mask is 2D and matches image dimensions
    if len(mask.shape) == 3 and mask.shape[2] == 1:  # Remove single channel dim if exists
        mask = mask.squeeze(2)
    
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_id in range(len(DISEASE_COLORS)):
        class_mask = mask == class_id
        if np.any(class_mask):
            colored_mask[class_mask] = DISEASE_COLORS[class_id]
    
    # Ensure both images have the same type before blending
    if image.dtype != colored_mask.dtype:
        image = image.astype(colored_mask.dtype)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay


def visualize_segmentation_prediction(
    image: Union[torch.Tensor, np.ndarray],
    prediction: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray] = None,
    save_path: str = None,
    title: str = None
) -> np.ndarray:
    """Visualize segmentation prediction with optional ground truth comparison."""
    
    # Convert to numpy
    image = tensor_to_numpy(image)
    prediction = tensor_to_numpy(prediction)
    if target is not None:
        target = tensor_to_numpy(target)
    
    # Handle different input formats
    if image.ndim == 4:  # Batch dimension
        image = image[0]
    if prediction.ndim == 4:  # Batch dimension
        prediction = prediction[0]
    if target is not None and target.ndim == 3:  # Batch dimension
        target = target[0] if target.ndim == 3 else target
    
    # Convert prediction probabilities to class indices
    if prediction.ndim == 3 and prediction.shape[0] > 1:  # [C, H, W]
        prediction = np.argmax(prediction, axis=0)
    elif prediction.ndim == 3:  # [H, W, C]
        prediction = np.argmax(prediction, axis=2)
    
    # Handle image format
    if image.ndim == 3 and image.shape[0] == 3:  # [C, H, W]
        image = np.transpose(image, (1, 2, 0))  # Convert to [H, W, C]
    
    # Denormalize image if needed
    if image.max() <= 1.0:
        image = denormalize_image(image)
    
    # Ensure image is BGR for OpenCV
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create visualization
    if target is not None:
        # Side-by-side comparison
        pred_overlay = create_segmentation_overlay(image.copy(), prediction)
        target_overlay = create_segmentation_overlay(image.copy(), target)
        
        # Combine images horizontally
        result = np.hstack([image, pred_overlay, target_overlay])
        
        # Add text labels
        cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, "Prediction", (image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, "Ground Truth", (2 * image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        # Just prediction overlay
        pred_overlay = create_segmentation_overlay(image.copy(), prediction)
        result = np.hstack([image, pred_overlay])
        
        # Add text labels
        cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, "Prediction", (image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add title if provided
    if title:
        cv2.putText(result, title, (10, result.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, result)
    
    return result


def visualize_detection_prediction(
    image: Union[torch.Tensor, np.ndarray],
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor] = None,
    confidence_threshold: float = 0.5,
    save_path: str = None,
    title: str = None
) -> np.ndarray:
    """Visualize detection predictions with optional ground truth comparison."""
    
    # Convert to numpy
    image = tensor_to_numpy(image)
    
    # Handle different input formats
    if image.ndim == 4:  # Batch dimension
        image = image[0]
    
    if image.ndim == 3 and image.shape[0] == 3:  # [C, H, W]
        image = np.transpose(image, (1, 2, 0))  # Convert to [H, W, C]
    
    # Denormalize image if needed
    if image.max() <= 1.0:
        image = denormalize_image(image)
    
    # Ensure image is BGR for OpenCV
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    result_image = image.copy()
    
    # Draw predictions
    if 'boxes' in predictions and len(predictions['boxes']) > 0:
        boxes = tensor_to_numpy(predictions['boxes'])
        scores = tensor_to_numpy(predictions.get('scores', torch.ones(len(boxes))))
        labels = tensor_to_numpy(predictions.get('labels', torch.zeros(len(boxes))))
        
        # Filter by confidence
        confident_mask = scores >= confidence_threshold
        boxes = boxes[confident_mask]
        scores = scores[confident_mask]
        labels = labels[confident_mask]
        
        # Draw prediction boxes
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            class_id = int(label)
            
            # Get color for this class
            color = DISEASE_COLORS.get(class_id, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and confidence
            label_text = f"{DISEASE_NAMES.get(class_id, f'Class {class_id}')}: {score:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw ground truth if provided
    if targets is not None and 'boxes' in targets and len(targets['boxes']) > 0:
        gt_boxes = tensor_to_numpy(targets['boxes'])
        gt_labels = tensor_to_numpy(targets.get('labels', torch.zeros(len(gt_boxes))))
        
        # Draw ground truth boxes with dashed lines
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.astype(int)
            class_id = int(label)
            
            # Get color for this class
            color = DISEASE_COLORS.get(class_id, (255, 255, 255))
            
            # Draw dashed rectangle (simplified as solid for now)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 1)
            
            # Draw GT label
            gt_text = f"GT: {DISEASE_NAMES.get(class_id, f'Class {class_id}')}"
            cv2.putText(result_image, gt_text, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Add title if provided
    if title:
        cv2.putText(result_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add legend
    legend_y = 50
    cv2.putText(result_image, "Predictions (thick), Ground Truth (thin)", 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, result_image)
    
    return result_image


def save_segmentation_results(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    images: List[torch.Tensor] = None,
    save_dir: Union[str, Path] = "visualizations/segmentation",
    max_samples: int = 10
):
    """Save segmentation visualization results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(len(predictions), max_samples)
    
    for i in range(num_samples):
        pred = predictions[i]
        target = targets[i] if i < len(targets) else None
        
        # Create dummy image if not provided
        if images is None or i >= len(images):
            # Create a simple grayscale image from target
            if target is not None:
                dummy_image = np.stack([target] * 3, axis=-1).astype(np.uint8) * 30
            else:
                dummy_image = np.zeros((pred.shape[-2], pred.shape[-1], 3), dtype=np.uint8)
        else:
            dummy_image = images[i]
        
        # Visualize
        result = visualize_segmentation_prediction(
            dummy_image,
            pred,
            target,
            title=f"Sample {i+1}"
        )
        
        # Save
        save_path = save_dir / f"segmentation_sample_{i+1:03d}.png"
        cv2.imwrite(str(save_path), result)
    
    print(f"Saved {num_samples} segmentation visualizations to {save_dir}")


def save_detection_results(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    images: List[torch.Tensor] = None,
    save_dir: str = "visualizations/detection",
    max_samples: int = 10,
    confidence_threshold: float = 0.5
):
    """Save detection visualization results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(len(predictions), max_samples)
    
    for i in range(num_samples):
        pred = predictions[i]
        target = targets[i] if i < len(targets) else None
        
        # Create dummy image if not provided
        if images is None or i >= len(images):
            dummy_image = np.ones((512, 512, 3), dtype=np.uint8) * 128  # Gray image
        else:
            dummy_image = images[i]
        
        # Visualize
        result = visualize_detection_prediction(
            dummy_image,
            pred,
            target,
            confidence_threshold=confidence_threshold,
            title=f"Sample {i+1}"
        )
        
        # Save
        save_path = save_dir / f"detection_sample_{i+1:03d}.png"
        cv2.imwrite(str(save_path), result)
    
    print(f"Saved {num_samples} detection visualizations to {save_dir}")


def create_confusion_matrix_plot(confusion_matrix: np.ndarray, class_names: List[str] = None, save_path: str = None):
    """Create and save confusion matrix plot."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    # Set labels
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                horizontalalignment="center",
                color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[Dict] = None,
    val_metrics: List[Dict] = None,
    save_dir: str = "plots"
):
    """Plot training curves for losses and metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Metrics curves
    if val_metrics and len(val_metrics) > 0:
        plt.subplot(1, 2, 2)
        
        # Extract metric values
        if 'mean_iou' in val_metrics[0]:  # Segmentation metrics
            metric_values = [m['mean_iou'] for m in val_metrics]
            plt.plot(epochs, metric_values, 'g-', label='Validation mIoU')
            plt.ylabel('mIoU')
        elif 'map_50' in val_metrics[0]:  # Detection metrics
            metric_values = [m['map_50'] for m in val_metrics]
            plt.plot(epochs, metric_values, 'g-', label='Validation mAP@0.5')
            plt.ylabel('mAP@0.5')
        
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Test segmentation visualization
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_pred = np.random.randint(0, 9, (256, 256))
    dummy_target = np.random.randint(0, 9, (256, 256))
    
    seg_result = visualize_segmentation_prediction(
        dummy_image, dummy_pred, dummy_target,
        title="Test Segmentation"
    )
    print(f"Segmentation visualization shape: {seg_result.shape}")
    
    # Test detection visualization
    dummy_predictions = {
        'boxes': torch.tensor([[50, 50, 150, 150], [200, 100, 250, 180]]),
        'scores': torch.tensor([0.9, 0.7]),
        'labels': torch.tensor([1, 2])
    }
    
    dummy_targets = {
        'boxes': torch.tensor([[55, 55, 155, 155]]),
        'labels': torch.tensor([1])
    }
    
    det_result = visualize_detection_prediction(
        dummy_image, dummy_predictions, dummy_targets,
        title="Test Detection"
    )
    print(f"Detection visualization shape: {det_result.shape}")
    
    print("Visualization tests completed!")
