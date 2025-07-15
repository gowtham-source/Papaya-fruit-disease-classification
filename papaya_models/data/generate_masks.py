#!/usr/bin/env python3
"""
Script to generate initial segmentation masks from YOLO bounding box annotations.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_yolo_annotation(annotation_path: str, image_width: int, image_height: int):
    """Read YOLO format annotation and convert to pixel coordinates."""
    boxes = []
    classes = []
    
    with open(annotation_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert normalized coordinates to pixel coordinates
            x_center = int(x_center * image_width)
            y_center = int(y_center * image_height)
            width = int(width * image_width)
            height = int(height * image_height)
            
            # Calculate box coordinates
            x1 = max(0, int(x_center - width/2))
            y1 = max(0, int(y_center - height/2))
            x2 = min(image_width, int(x_center + width/2))
            y2 = min(image_height, int(y_center + height/2))
            
            boxes.append((x1, y1, x2, y2))
            classes.append(int(class_id))
    
    return boxes, classes

def create_segmentation_mask(image_shape: tuple, boxes: list, classes: list):
    """Create a segmentation mask from bounding boxes."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for (x1, y1, x2, y2), class_id in zip(boxes, classes):
        # Fill the bounding box region with the class ID
        mask[y1:y2, x1:x2] = class_id + 1  # Add 1 to reserve 0 for background
    
    return mask

def process_split(split_dir: str, output_images_dir: str, output_masks_dir: str):
    """Process a dataset split (train/val/test)."""
    split_path = Path(split_dir)
    
    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in split_path.glob("*.jpg")]
    logger.info(f"Found {len(image_files)} images in {split_dir}")
    
    for image_file in tqdm(image_files, desc=f"Processing {split_path.name}"):
        # Check if annotation exists
        annotation_file = image_file.with_suffix('.txt')
        if not annotation_file.exists():
            logger.warning(f"No annotation found for {image_file}")
            continue
        
        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            logger.warning(f"Failed to read image {image_file}")
            continue
        
        # Read annotation
        boxes, classes = read_yolo_annotation(
            str(annotation_file),
            image.shape[1],
            image.shape[0]
        )
        
        # Create mask
        mask = create_segmentation_mask(image.shape, boxes, classes)
        
        # Save image and mask
        output_image_path = os.path.join(output_images_dir, image_file.name)
        output_mask_path = os.path.join(
            output_masks_dir,
            image_file.stem + "_mask.png"
        )
        
        cv2.imwrite(output_image_path, image)
        cv2.imwrite(output_mask_path, mask)

def main():
    # Base directories
    data_dir = Path(__file__).parent
    splits = ['Train', 'Validation', 'Test']
    
    # Output directories
    seg_images_dir = data_dir / 'segmentation' / 'images'
    seg_masks_dir = data_dir / 'segmentation' / 'masks'
    
    # Process each split
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} not found")
            continue
        
        # Create split-specific output directories
        split_images_dir = seg_images_dir / split.lower()
        split_masks_dir = seg_masks_dir / split.lower()
        
        process_split(str(split_dir), str(split_images_dir), str(split_masks_dir))
    
    logger.info("Mask generation complete!")

if __name__ == '__main__':
    main()
