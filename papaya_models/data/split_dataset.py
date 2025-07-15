import os
import shutil
import random
from pathlib import Path

def split_dataset(src_images_dir, src_masks_dir, target_base_dir, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train/val/test sets."""
    # Create target directories if they don't exist
    splits = ['train', 'val', 'test']
    for split in splits:
        Path(os.path.join(target_base_dir, split, 'images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(target_base_dir, split, 'masks')).mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(src_images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(image_files)
    
    # Calculate split sizes
    total = len(image_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split files
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    # Copy files to respective directories
    splits_files = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split, files in splits_files.items():
        print(f"Copying {len(files)} files to {split} split...")
        for img_file in files:
            # Copy image
            src_img = os.path.join(src_images_dir, img_file)
            dst_img = os.path.join(target_base_dir, split, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy mask
            mask_file = os.path.splitext(img_file)[0] + '_mask.png'
            src_mask = os.path.join(src_masks_dir, mask_file)
            dst_mask = os.path.join(target_base_dir, split, 'masks', mask_file)
            shutil.copy2(src_mask, dst_mask)

if __name__ == '__main__':
    # Source directories
    src_images_dir = 'papaya_models/data/segmentation/train/images'
    src_masks_dir = 'papaya_models/data/segmentation/train/masks'
    
    # Target base directory
    target_base_dir = 'papaya_models/data/segmentation/train'
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Split dataset
    split_dataset(src_images_dir, src_masks_dir, target_base_dir)
