#!/usr/bin/env python3
"""
Dataset loader for papaya disease detection.
Handles loading images and YOLO format bounding box annotations.
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
logger.setLevel(logging.DEBUG)  # Ensure DEBUG messages are processed


class PapayaDetectionDataset(Dataset):
    """Dataset for papaya disease detection training.
    
    This dataset loads images and their corresponding YOLO-format annotation files
    from local directories (Train, Test, Validation).
    """
    
    def __init__(
        self,
        images_dir: str,
        annotations_dir: str = None,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (640, 640),
        num_classes: int = 9,  # Updated to match the 9 classes in the dataset
        max_objects: int = 50
    ):
        """Initialize the dataset.
        
        Args:
            images_dir: Path to directory containing images
            annotations_dir: Path to directory containing annotation files (default: same as images_dir)
            transform: Optional transform to be applied on a sample
            image_size: Target image size (height, width)
            num_classes: Number of object classes
            max_objects: Maximum number of objects per image
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir or images_dir  # If annotations_dir is None, use images_dir
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        
        # Create annotations directory if it doesn't exist
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Get all image files with corresponding annotations
        self.image_files = self._get_image_files()
        
        if not self.image_files:
            logger.warning(f"Ultimately, no valid image files found with corresponding annotations in images_dir='{self.images_dir}' (abs: '{os.path.abspath(self.images_dir)}') (annotations searched in annotations_dir='{self.annotations_dir}' (abs: '{os.path.abspath(self.annotations_dir)}')).")
        else:
            logger.info(f"Found {len(self.image_files)} images with annotations in {self.images_dir}")
    
    def _get_image_files(self) -> List[str]:
        """Get all image files that have corresponding annotation files."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        # Debug print for directories
        print(f"[DEBUG] Images directory: {self.images_dir}")
        print(f"[DEBUG] Absolute images path: {os.path.abspath(self.images_dir)}")
        print(f"[DEBUG] Annotations directory: {self.annotations_dir}")
        print(f"[DEBUG] Absolute annotations path: {os.path.abspath(self.annotations_dir)}")
        
        # Check if directories exist
        if not os.path.isdir(self.images_dir):
            msg = f"Images directory does not exist: {self.images_dir} (abs: {os.path.abspath(self.images_dir)})"
            logger.warning(msg)
            print(f"[ERROR] {msg}")
            return image_files
            
        if not os.path.isdir(self.annotations_dir):
            msg = f"Annotations directory does not exist: {self.annotations_dir} (abs: {os.path.abspath(self.annotations_dir)})"
            logger.warning(msg)
            print(f"[ERROR] {msg}")
            return image_files
            
        # List contents of images directory for debugging
        try:
            print(f"[DEBUG] Contents of {self.images_dir}:")
            for i, f in enumerate(os.listdir(self.images_dir)[:5]):  # Show first 5 files
                print(f"  {i+1}. {f}")
            if len(os.listdir(self.images_dir)) > 5:
                print(f"  ... and {len(os.listdir(self.images_dir)) - 5} more files")
        except Exception as e:
            print(f"[ERROR] Could not list contents of {self.images_dir}: {e}")
        
        # Find all image files with corresponding annotation files
        for file in os.listdir(self.images_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                base_name = os.path.splitext(file)[0]
                annotation_file = base_name + '.txt'
                annotation_path = os.path.join(self.annotations_dir, annotation_file)
                
                # Log the exact absolute path being checked
                abs_annotation_path = os.path.abspath(annotation_path)
                logger.debug(f"Checking for annotation: Image='{file}', AnnoFile='{annotation_file}', AnnoDir='{self.annotations_dir}', FullAnnoPath='{abs_annotation_path}'")

                if os.path.exists(annotation_path): # os.path.exists uses relative path correctly if CWD is set
                    image_files.append(file)
                else:
                    logger.warning(f"Annotation file NOT FOUND for '{file}' at '{annotation_path}' (abs: '{abs_annotation_path}') (Resolved from dir='{self.annotations_dir}', file='{annotation_file}')")
        
        return sorted(image_files)
    
    def _parse_yolo_annotation(self, annotation_path: str) -> List[Dict]:
        """Parse YOLO format annotation file."""
        annotations = []
        
        if not os.path.exists(annotation_path):
            return annotations
        
        with open(annotation_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # Load annotations
        base_name = os.path.splitext(image_file)[0]
        annotation_file = base_name + '.txt'
        annotation_path = os.path.join(self.annotations_dir, annotation_file)
        annotations = self._parse_yolo_annotation(annotation_path)
        
        # Convert YOLO format to bounding boxes for augmentation
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            # Convert from YOLO format (normalized) to pixel coordinates
            x_center = ann['x_center'] * original_w
            y_center = ann['y_center'] * original_h
            width = ann['width'] * original_w
            height = ann['height'] * original_h
            
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            
            # Clamp to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(original_w, x_max)
            y_max = min(original_h, y_max)
            
            bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(ann['class_id'])
        
        # Apply transformations
        if self.transform and len(bboxes) > 0:
            augmented = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_labels = augmented['class_labels']
        elif self.transform:
            # No bboxes, just transform image
            augmented = self.transform(image=image, bboxes=[], class_labels=[])
            image = augmented['image']
            bboxes = []
            class_labels = []
        
        # Resize image if transform doesn't handle it
        if not self.transform:
            image = cv2.resize(image, self.image_size)
            # Also need to scale bboxes
            scale_x = self.image_size[0] / original_w
            scale_y = self.image_size[1] / original_h
            
            scaled_bboxes = []
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                scaled_bboxes.append([
                    x_min * scale_x,
                    y_min * scale_y,
                    x_max * scale_x,
                    y_max * scale_y
                ])
            bboxes = scaled_bboxes
            
            # Convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Convert bboxes back to YOLO format for training
        target_boxes = torch.zeros((self.max_objects, 5))  # [class, x_center, y_center, width, height]
        
        for i, (bbox, class_id) in enumerate(zip(bboxes, class_labels)):
            if i >= self.max_objects:
                break
                
            x_min, y_min, x_max, y_max = bbox
            
            # Convert to YOLO format (normalized)
            x_center = (x_min + x_max) / 2 / self.image_size[0]
            y_center = (y_min + y_max) / 2 / self.image_size[1]
            width = (x_max - x_min) / self.image_size[0]
            height = (y_max - y_min) / self.image_size[1]
            
            target_boxes[i] = torch.tensor([class_id, x_center, y_center, width, height])
        
        # Number of valid objects
        num_objects = min(len(bboxes), self.max_objects)
        
        return {
            'image': image,
            'targets': target_boxes,
            'num_objects': torch.tensor(num_objects),
            'filename': image_file
        }


def get_detection_transforms(
    image_size: Tuple[int, int] = (640, 640),
    is_training: bool = True
) -> A.Compose:
    """Get augmentation transforms for detection."""
    
    if is_training:
        transforms = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 30), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.2),
            A.CLAHE(clip_limit=2, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    else:
        transforms = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    return transforms


def create_detection_dataloaders(
    dataset_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 640),
    num_classes: int = 9,  # Updated to match the 9 classes in the dataset
    train_dir: str = 'Train',
    val_dir: str = 'Validation',
    test_dir: str = 'Test'
) -> Dict[str, DataLoader]:
    """Create data loaders for detection training.
    
    This function creates dataloaders for the Train, Validation, and Test splits
    using the specified directory structure.
    
    Each split directory should contain images and a 'labels' subdirectory with YOLO-format annotations.
    
    Args:
        dataset_path: Path to the root directory containing the dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        image_size: Target image size (height, width)
        num_classes: Number of object classes
        train_dir: Name of training data directory (default: 'Train')
        val_dir: Name of validation data directory (default: 'Validation')
        test_dir: Name of test data directory (default: 'Test')
        
    Returns:
        Dictionary containing 'train', 'val', and 'test' DataLoaders
    """
    import os
    from pathlib import Path
    
    # Resolve dataset path relative to the current working directory
    original_dataset_path = dataset_path
    dataset_path = os.path.abspath(os.path.join(os.getcwd(), dataset_path))
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Original dataset path: {original_dataset_path}")
    logger.info(f"Resolved dataset path: {dataset_path}")
    logger.info(f"Directory exists: {os.path.exists(dataset_path)}")
    
    if os.path.exists(dataset_path):
        logger.info(f"Contents of dataset directory: {os.listdir(dataset_path)[:20]}")
    
    # Define split directories
    split_dirs = {
        'train': os.path.join(dataset_path, train_dir),
        'val': os.path.join(dataset_path, val_dir),
        'test': os.path.join(dataset_path, test_dir)
    }
    
    # Log the full paths being used
    for split, path in split_dirs.items():
        logger.info(f"{split.capitalize()} directory: {path}")
        logger.info(f"Directory exists: {os.path.exists(path)}")
        if os.path.exists(path):
            logger.info(f"Contents of {path}: {os.listdir(path)[:10]}")
    
    # Check if any directories exist
    if not any(os.path.exists(path) for path in split_dirs.values()):
        logger.error(f"No dataset directories found in {dataset_path}")
        logger.error(f"Expected directories: {train_dir}, {val_dir}, {test_dir}")
        return {}
    
    dataloaders = {}
    
    for split, split_dir in split_dirs.items():
        # Check if split directory exists
        if not os.path.isdir(split_dir):
            logger.warning(f"Skipping {split} split - directory not found: {split_dir}")
            continue
            
        # Create dataset
        transform = get_detection_transforms(image_size, is_training=(split == 'train'))
        
        try:
            # Use the same directory for images and annotations
            dataset = PapayaDetectionDataset(
                images_dir=split_dir,
                annotations_dir=split_dir,  # Look in the same directory as images
                transform=transform,
                image_size=image_size,
                num_classes=num_classes
            )
            
            if len(dataset) == 0:
                logger.warning(f"No valid samples found in {split_dir}")
                continue
                
            # Create dataloader with pin_memory=True for better GPU transfer
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,  # Enable faster data transfer to GPU
                persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                drop_last=(split == 'train'),
                collate_fn=detection_collate_fn
            )
            
            dataloaders[split] = dataloader
            logger.info(f"Created {split} dataloader with {len(dataset)} samples from {split_dir}")
            
        except Exception as e:
            logger.error(f"Error creating {split} dataloader: {str(e)}")
            continue
    
    return dataloaders


def detection_collate_fn(batch):
    """Custom collate function for detection data."""
    images = torch.stack([item['image'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    num_objects = torch.stack([item['num_objects'] for item in batch])
    filenames = [item['filename'] for item in batch]
    
    return {
        'images': images,
        'targets': targets,
        'num_objects': num_objects,
        'filenames': filenames
    }


class DetectionMetrics:
    """Calculate detection metrics (mAP, precision, recall)."""
    
    def __init__(self, num_classes: int, iou_thresholds: List[float] = None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, pred_boxes: List[np.ndarray], pred_scores: List[np.ndarray], 
               pred_classes: List[np.ndarray], target_boxes: List[np.ndarray], 
               target_classes: List[np.ndarray]):
        """Update metrics with batch predictions."""
        for i in range(len(pred_boxes)):
            self.predictions.append({
                'boxes': pred_boxes[i],
                'scores': pred_scores[i],
                'classes': pred_classes[i]
            })
            self.targets.append({
                'boxes': target_boxes[i],
                'classes': target_classes[i]
            })
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Compute Average Precision using 11-point interpolation."""
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute mAP and other detection metrics."""
        all_predictions = []
        all_targets = []
        
        # Flatten all predictions and targets
        for pred, target in zip(self.predictions, self.targets):
            for i in range(len(pred['boxes'])):
                all_predictions.append({
                    'box': pred['boxes'][i],
                    'score': pred['scores'][i],
                    'class': pred['classes'][i],
                    'image_id': len(all_targets)
                })
            
            for i in range(len(target['boxes'])):
                all_targets.append({
                    'box': target['boxes'][i],
                    'class': target['classes'][i],
                    'image_id': len(all_targets)
                })
        
        # Sort predictions by confidence
        all_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Compute mAP for each class and IoU threshold
        aps = []
        
        for class_id in range(self.num_classes):
            for iou_thresh in self.iou_thresholds:
                # Get predictions and targets for this class
                class_preds = [p for p in all_predictions if p['class'] == class_id]
                class_targets = [t for t in all_targets if t['class'] == class_id]
                
                if len(class_targets) == 0:
                    continue
                
                # Compute precision and recall
                tp = np.zeros(len(class_preds))
                fp = np.zeros(len(class_preds))
                
                for i, pred in enumerate(class_preds):
                    # Find best matching target
                    best_iou = 0
                    best_target_idx = -1
                    
                    for j, target in enumerate(class_targets):
                        if target['image_id'] == pred['image_id']:
                            iou = self.compute_iou(pred['box'], target['box'])
                            if iou > best_iou:
                                best_iou = iou
                                best_target_idx = j
                    
                    if best_iou >= iou_thresh:
                        tp[i] = 1
                        # Mark target as used
                        class_targets.pop(best_target_idx)
                    else:
                        fp[i] = 1
                
                # Compute precision and recall curves
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                recalls = tp_cumsum / len(class_targets) if len(class_targets) > 0 else np.zeros_like(tp_cumsum)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                
                # Compute AP
                ap = self.compute_ap(recalls, precisions)
                aps.append(ap)
        
        # Compute overall metrics
        map_score = np.mean(aps) if aps else 0.0
        
        return {
            'mAP': map_score,
            'mAP@0.5': np.mean([ap for i, ap in enumerate(aps) if i % len(self.iou_thresholds) == 0]),
            'num_detections': len(all_predictions),
            'num_targets': len(all_targets)
        }


if __name__ == "__main__":
    # Test dataset creation
    dataset_path = "c:/Users/gowth/Downloads/Sisfrutos-Papaya"
    
    try:
        dataloaders = create_detection_dataloaders(
            dataset_path=dataset_path,
            batch_size=2,
            num_workers=0,  # Set to 0 for testing
            image_size=(416, 416)
        )
        
        print("Detection dataloaders created successfully!")
        for split, dataloader in dataloaders.items():
            print(f"{split}: {len(dataloader)} batches, {len(dataloader.dataset)} samples")
            
            # Test one batch
            batch = next(iter(dataloader))
            print(f"  Images shape: {batch['images'].shape}")
            print(f"  Targets shape: {batch['targets'].shape}")
            print(f"  Num objects: {batch['num_objects']}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
