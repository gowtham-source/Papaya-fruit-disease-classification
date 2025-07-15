#!/usr/bin/env python3
"""
Script to split segmentation data into train, validation, and test sets.
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_directory_structure(base_dir: Path):
    """Create directory structure for train/val/test splits."""
    for split in ['train', 'val', 'test']:
        # Create main split directory
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (split_dir / 'images').mkdir(exist_ok=True)
        (split_dir / 'masks').mkdir(exist_ok=True)

def get_image_mask_pairs(masks_dir: Path):
    """Get pairs of (image_path, mask_path) from the masks directory."""
    # Get all mask files
    mask_files = [f for f in masks_dir.glob('*_mask.png')]
    
    # Create pairs (image_path, mask_path)
    pairs = []
    for mask_path in mask_files:
        # Extract base name (without _mask.png)
        base_name = mask_path.stem.replace('_mask', '')
        # Find corresponding image file (could be .jpg, .jpeg, .png)
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = mask_path.parent / f"{base_name}{ext}"
            if image_path.exists():
                pairs.append((image_path, mask_path))
                break
    
    return pairs

def copy_files(pairs, output_dir: Path, split_name: str):
    """Copy image-mask pairs to the specified split directory."""
    for img_path, mask_path in pairs:
        # Copy image
        dest_img = output_dir / split_name / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # Copy mask
        dest_mask = output_dir / split_name / 'masks' / mask_path.name
        shutil.copy2(mask_path, dest_mask)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    base_dir = Path('segmentation_test')
    input_dir = base_dir / 'Train'
    output_dir = base_dir / 'split_data'
    
    # Create output directory structure
    create_directory_structure(output_dir)
    
    # Get all image-mask pairs
    masks_dir = input_dir / 'masks'
    pairs = get_image_mask_pairs(masks_dir)
    
    print(f"Found {len(pairs)} image-mask pairs")
    
    if not pairs:
        print("No image-mask pairs found. Exiting.")
        return
    
    # Split into train (70%), val (15%), test (15%)
    train_pairs, test_pairs = train_test_split(
        pairs, test_size=0.3, random_state=42
    )
    val_pairs, test_pairs = train_test_split(
        test_pairs, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(train_pairs)} samples")
    print(f"Validation: {len(val_pairs)} samples")
    print(f"Test: {len(test_pairs)} samples")
    
    # Copy files to respective directories
    print("Copying files...")
    copy_files(train_pairs, output_dir, 'train')
    copy_files(val_pairs, output_dir, 'val')
    copy_files(test_pairs, output_dir, 'test')
    
    print(f"Data split completed. Output directory: {output_dir}")

if __name__ == "__main__":
    main()
