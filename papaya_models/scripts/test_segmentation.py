#!/usr/bin/env python3
"""
Standalone script to test a trained papaya disease segmentation model.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from segmentation.unet_resnet import create_unet_resnet
from data.segmentation_dataset import (
    PapayaSegmentationDataset,
    get_segmentation_transforms
)
from utils.metrics import SegmentationMetrics
from utils.visualization import save_segmentation_results

def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load the trained model from checkpoint."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = create_unet_resnet(
        encoder_name=config['model']['encoder_name'],
        classes=config['model']['num_classes'],
        encoder_weights=config['model']['encoder_weights'],
        use_auxiliary_loss=config['model']['aux_classifier']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return model, config

def main():
    parser = argparse.ArgumentParser(description='Test papaya disease segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/segmentation/train', 
                        help='Root directory containing test/images/ and test/masks/')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and config
    model, config = load_model(args.config, args.checkpoint, device)
    model.eval()
    
    # Set up test dataset paths
    test_image_dir = os.path.join(args.data_dir, 'test', 'images')
    test_mask_dir = os.path.join(args.data_dir, 'test', 'masks')
    
    # Verify directories exist
    if not os.path.exists(test_image_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_image_dir}")
    if not os.path.exists(test_mask_dir):
        raise FileNotFoundError(f"Test mask directory not found: {test_mask_dir}")
    
    print(f"Using test images from: {test_image_dir}")
    print(f"Using test masks from: {test_mask_dir}")
    
    # Create test dataset and dataloader
    test_transform = get_segmentation_transforms(
        image_size=tuple(config['data']['image_size']),
        is_training=False
    )
    
    test_dataset = PapayaSegmentationDataset(
        images_dir=test_image_dir,
        masks_dir=test_mask_dir,
        transform=test_transform,
        image_size=tuple(config['data']['image_size']),
        num_classes=config['model']['num_classes']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Initialize metrics calculator
    metrics_calculator = SegmentationMetrics(num_classes=config['model']['num_classes'])
    
    # Lists to store predictions for visualization
    test_predictions = []
    test_targets = []
    test_images = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            if hasattr(model, 'aux_classifier') and model.aux_classifier:
                main_output, _ = model(images)
                predictions = main_output
            else:
                predictions = model(images)
            
            # Update metrics
            metrics_calculator.update(predictions, masks)
            
            # Store for visualization
            if len(test_predictions) < args.num_samples:
                test_predictions.extend([p for p in predictions.cpu()])
                test_targets.extend([m for m in masks.cpu()])
                test_images.extend([img for img in images.cpu()])
    
    # Compute metrics
    print("\nComputing metrics...")
    test_metrics = metrics_calculator.compute()
    
    print("\nTest Results:")
    print(f"Test mIoU: {test_metrics['mean_iou']:.4f}")
    print(f"Test Dice: {test_metrics['mean_dice']:.4f}")
    print(f"Test Precision: {test_metrics['mean_precision']:.4f}")
    print(f"Test Recall: {test_metrics['mean_recall']:.4f}")
    
    # Create results directory
    results_dir = Path(config.get('results_dir', 'results/segmentation'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = results_dir / 'test_results.json'
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nTest metrics saved to: {metrics_path}")
    
    # Save visualizations if we have predictions and targets
    if len(test_predictions) > 0 and len(test_targets) > 0:
        vis_dir = results_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        print(f"Saving visualizations to: {vis_dir}")
        save_segmentation_results(
            predictions=test_predictions[:args.num_samples],
            targets=test_targets[:args.num_samples],
            images=test_images[:args.num_samples] if test_images else None,
            save_dir=vis_dir
        )
    else:
        print("No predictions or targets available for visualization")

if __name__ == '__main__':
    main()
