#!/usr/bin/env python3
"""
Dataset loader for papaya disease segmentation.
Handles loading images and corresponding segmentation masks.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class PapayaSegmentationDataset(Dataset):
    """Dataset for papaya disease segmentation training."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 9,
        mask_suffix: str = "_mask.png"
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        self.mask_suffix = mask_suffix
        
        # Get all image files
        self.image_files = self._get_image_files()
        
        logger.info(f"Found {len(self.image_files)} images in {images_dir}")
        
    def _get_image_files(self) -> List[str]:
        """Get all image files that have corresponding masks."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for file in os.listdir(self.images_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Check if corresponding mask exists
                base_name = os.path.splitext(file)[0]
                mask_file = base_name + self.mask_suffix
                mask_path = os.path.join(self.masks_dir, mask_file)
                
                if os.path.exists(mask_path):
                    image_files.append(file)
                else:
                    logger.warning(f"No mask found for image {file}")
        
        return sorted(image_files)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        base_name = os.path.splitext(image_file)[0]
        mask_file = base_name + self.mask_suffix
        mask_path = os.path.join(self.masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # Mask is already a tensor from albumentations ToTensorV2
            mask = mask.long()
        else:
            # If no transform, convert to tensor manually
            mask = torch.from_numpy(mask).long()
        
        # Clamp mask values to valid range
        mask = torch.clamp(mask, 0, self.num_classes - 1)
        
        # Debug: Log mask value range
        unique_values = torch.unique(mask)
        if torch.any(unique_values >= self.num_classes) or torch.any(unique_values < 0):
            logger.warning(f"Invalid mask values found in {image_file}: {unique_values.tolist()}")
            logger.warning(f"Mask min: {mask.min()}, max: {mask.max()}, unique values: {unique_values.tolist()}")

        # Return tuple of (image, mask) as expected by training loop
        return image, mask


def get_segmentation_transforms(
    image_size: Tuple[int, int] = (512, 512),
    is_training: bool = True
) -> A.Compose:
    """Get augmentation transforms for segmentation."""
    
    if is_training:
        transforms = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.CLAHE(clip_limit=2, p=0.3),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transforms = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transforms


def create_segmentation_dataloaders(
    dataset_path: str,
    segmentation_output_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    num_classes: int = 9
) -> Dict[str, DataLoader]:
    """Create data loaders for segmentation training."""
    
    dataloaders = {}
    
    for split in ['Train', 'Test', 'Validation']:
        images_dir = os.path.join(dataset_path, split)
        masks_dir = os.path.join(segmentation_output_path, split, 'masks')
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            logger.warning(f"Skipping {split} split - directory not found")
            continue
        
        # Create dataset
        transform = get_segmentation_transforms(image_size, is_training=(split == 'Train'))
        
        dataset = PapayaSegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            transform=transform,
            image_size=image_size,
            num_classes=num_classes
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'Train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'Train')
        )
        
        dataloaders[split.lower()] = dataloader
        logger.info(f"Created {split} dataloader with {len(dataset)} samples")
    
    return dataloaders


class SegmentationMetrics:
    """Calculate segmentation metrics."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with batch predictions."""
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()
        
        # Remove ignore index
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute various segmentation metrics."""
        cm = self.confusion_matrix
        
        # Per-class metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        # Avoid division by zero
        epsilon = 1e-7
        
        # IoU per class
        iou = tp / (tp + fp + fn + epsilon)
        
        # Dice per class
        dice = 2 * tp / (2 * tp + fp + fn + epsilon)
        
        # Precision and Recall
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # Overall accuracy
        accuracy = tp.sum() / (cm.sum() + epsilon)
        
        # Mean IoU
        mean_iou = np.mean(iou)
        
        # Mean Dice
        mean_dice = np.mean(dice)
        
        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'iou_per_class': iou.tolist(),
            'dice_per_class': dice.tolist(),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist()
        }


if __name__ == "__main__":
    # Test dataset creation
    dataset_path = "c:/Users/gowth/Downloads/Sisfrutos-Papaya"
    segmentation_output_path = "c:/Users/gowth/Downloads/Sisfrutos-Papaya/segmentation_test"
    
    try:
        dataloaders = create_segmentation_dataloaders(
            dataset_path=dataset_path,
            segmentation_output_path=segmentation_output_path,
            batch_size=2,
            num_workers=0,  # Set to 0 for testing
            image_size=(256, 256)
        )
        
        print("Dataloaders created successfully!")
        for split, dataloader in dataloaders.items():
            print(f"{split}: {len(dataloader)} batches, {len(dataloader.dataset)} samples")
            
            # Test one batch
            batch = next(iter(dataloader))
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Mask shape: {batch['mask'].shape}")
            print(f"  Mask unique values: {torch.unique(batch['mask'])}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
