#!/usr/bin/env python3
"""
Script to prepare segmentation dataset by sampling 120 images per class from original dataset.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import re

def read_class_from_label(label_file):
    """Read class ID from a label file (first number in first line)."""
    try:
        with open(label_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                return int(first_line.split()[0])
    except (IOError, ValueError, IndexError):
        pass
    return None

def get_image_base_from_label(label_file):
    """Get image base name from label file (e.g., TR000001-8-8-8)."""
    return label_file.stem  # Remove .txt extension

def prepare_dataset(original_data_dir, source_masks_dir, target_base_dir, samples_per_class=120):
    """Prepare segmentation dataset by sampling from original dataset and finding corresponding masks."""
    original_data_dir = Path(original_data_dir)
    source_masks_dir = Path(source_masks_dir)
    target_base_dir = Path(target_base_dir)
    
    print(f"Original data directory: {original_data_dir}")
    print(f"Source masks directory: {source_masks_dir}")
    print(f"Target directory: {target_base_dir}")
    
    # Create target directories
    target_images_dir = target_base_dir / 'images'
    target_masks_dir = target_base_dir / 'masks'
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_masks_dir, exist_ok=True)
    
    # Group label files by class
    class_to_files = defaultdict(list)
    label_files = list(original_data_dir.glob('*.txt'))
    print(f"Found {len(label_files)} total label files")
    
    for label_file in label_files:
        class_id = read_class_from_label(label_file)
        if class_id is not None and class_id != 0:  # Skip class 0 (papaya)
            class_to_files[class_id].append(label_file)
    
    # Sample and copy files
    print("\nProcessing classes:")
    for class_id, label_files in sorted(class_to_files.items()):
        print(f"\nClass {class_id}: Found {len(label_files)} samples")
        
        # Sample files (or take all if less than requested)
        num_samples = min(samples_per_class, len(label_files))
        selected_labels = random.sample(label_files, num_samples)
        
        copied_count = 0
        for label_file in selected_labels:
            # Get base name and find corresponding files
            image_base = get_image_base_from_label(label_file)
            original_image = original_data_dir / f"{image_base}.jpg"
            mask_file = source_masks_dir / f"{image_base}_mask.png"
            
            if not original_image.exists():
                print(f"Warning: Missing original image: {original_image}")
                continue
                
            if not mask_file.exists():
                print(f"Warning: Missing mask file: {mask_file}")
                continue
            
            # Copy files
            try:
                shutil.copy2(original_image, target_images_dir / f"{image_base}.jpg")
                shutil.copy2(mask_file, target_masks_dir / f"{image_base}_mask.png")
                copied_count += 1
            except Exception as e:
                print(f"Error copying files for {image_base}: {e}")
        
        print(f"Class {class_id}: Successfully copied {copied_count} samples")
    
    # Print final statistics
    print("\nDataset preparation complete!")
    print(f"Target images directory: {target_images_dir}")
    print(f"Target masks directory: {target_masks_dir}")

def main():
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Paths
    original_data_dir = project_root / "papaya_models" / "data" / "Train"
    source_masks_dir = project_root / "segmentation_test" / "Train" / "masks"
    target_base_dir = project_root / "papaya_models" / "data" / "segmentation" / "train"
    
    # Prepare dataset
    prepare_dataset(original_data_dir, source_masks_dir, target_base_dir, samples_per_class=120)

if __name__ == '__main__':
    main()
