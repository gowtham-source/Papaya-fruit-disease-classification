#!/usr/bin/env python3
"""
Training script for papaya disease segmentation model.
Implements comprehensive training with metrics, logging, and checkpointing.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from segmentation.unet_resnet import SimpleUNet, SimpleUNetWithAuxiliaryLoss, create_unet_resnet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from papaya_models.data.segmentation_dataset import PapayaSegmentationDataset, get_segmentation_transforms
from losses.segmentation_losses import CombinedSegmentationLoss, AuxiliaryLoss, get_class_weights
from utils.metrics import SegmentationMetrics
from utils.visualization import save_segmentation_results


class SegmentationTrainer:
    """Comprehensive trainer for papaya disease segmentation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_miou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Logging
        self.logger = self._setup_logging()
        self.writer = None
        if config.get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        if config.get('use_wandb', False):
            wandb.init(
                project="papaya-segmentation",
                config=config,
                name=config.get('experiment_name', 'papaya_seg_experiment')
            )
    
    def _setup_logging(self):
        """Setup logging."""
        import logging
        
        # Create logs directory
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger('SegmentationTrainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_dir / 'training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def setup_model(self):
        """Initialize model, loss, optimizer, and scheduler."""
        # Model
        self.model = create_unet_resnet(
            encoder_name=self.config['model']['encoder_name'],
            classes=self.config['model']['num_classes'],
            encoder_weights=self.config['model']['encoder_weights'],
            use_auxiliary_loss=self.config['model']['aux_classifier']
        ).to(self.device)
        
        # Loss function
        class_weights = None
        if self.config['loss']['use_class_weights']:
            class_weights = get_class_weights(
                self.config['data']['dataset_path'],
                self.config['model']['num_classes']
            ).to(self.device)
        
        main_loss = CombinedSegmentationLoss(
            ce_weight=self.config['loss']['ce_weight'],
            dice_weight=self.config['loss']['dice_weight'],
            focal_weight=self.config['loss']['focal_weight'],
            tversky_weight=self.config['loss']['tversky_weight'],
            boundary_weight=self.config['loss']['boundary_weight'],
            class_weights=class_weights
        )
        
        if self.config['model']['aux_classifier']:
            self.loss_fn = AuxiliaryLoss(main_loss, aux_weight=self.config['loss']['aux_weight'])
        else:
            self.loss_fn = main_loss
        
        # Optimizer
        if self.config['optimizer']['name'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )
        elif self.config['optimizer']['name'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )
        elif self.config['optimizer']['name'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                momentum=self.config['optimizer']['momentum'],
                weight_decay=self.config['optimizer']['weight_decay']
            )
        
        # Scheduler
        if self.config['scheduler']['name'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif self.config['scheduler']['name'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['scheduler']['step_size'],
                gamma=self.config['scheduler']['gamma']
            )
        elif self.config['scheduler']['name'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config['scheduler']['factor'],
                patience=self.config['scheduler']['patience']
            )
        
        self.logger.info(f"Model initialized: {self.config['model']['encoder_name']}")
        self.logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Device: {self.device}")
    
    def setup_data(self):
        """Setup data loaders."""
        # Get base dataset path
        base_path = Path(self.config['data']['dataset_path'])
        
        # Create dataloaders for each split
        dataloaders = {}
        for split in ['train', 'val', 'test']:
            images_dir = base_path / split / 'images'
            masks_dir = base_path / split / 'masks'
            
            if not images_dir.exists() or not masks_dir.exists():
                self.logger.warning(f"Skipping {split} split - directory not found")
                continue
                
            # Create dataset
            transform = get_segmentation_transforms(
                image_size=tuple(self.config['data']['image_size']),
                is_training=(split == 'train' and self.config['data']['train_augmentations'])
            )
            
            dataset = PapayaSegmentationDataset(
                images_dir=str(images_dir),
                masks_dir=str(masks_dir),
                transform=transform,
                image_size=tuple(self.config['data']['image_size'])
            )
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=(split == 'train'),
                num_workers=self.config['data']['num_workers'],
                pin_memory=True,
                drop_last=(split == 'train')
            )
            
            dataloaders[split] = dataloader
            self.logger.info(f"Created {split} dataloader with {len(dataset)} samples")
        
        # Assign dataloaders
        self.train_loader = dataloaders.get('train')
        self.val_loader = dataloaders.get('val')
        self.test_loader = dataloaders.get('test')
        
        if not self.train_loader:
            raise RuntimeError("Failed to create training dataloader. Check dataset paths.")
            
        if not self.val_loader:
            self.logger.warning("No validation dataloader created. Using test set for validation.")
            self.val_loader = self.test_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch}')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config['model']['aux_classifier']:
                main_output, aux_output = self.model(images)
                losses = self.loss_fn(main_output, aux_output, masks)
            else:
                output = self.model(images)
                losses = self.loss_fn(output, masks)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total_loss'].item()
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log batch metrics
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('train/batch_loss', losses['total_loss'].item(), global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'avg_loss': avg_loss, **loss_components}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        
        # Initialize metrics
        metrics_calculator = SegmentationMetrics(
            num_classes=self.config['model']['num_classes'],
            device=self.device
        )
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validation Epoch {self.current_epoch}')
            
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                if self.config['model']['aux_classifier']:
                    main_output, aux_output = self.model(images)
                    losses = self.loss_fn(main_output, aux_output, masks)
                    predictions = main_output
                else:
                    output = self.model(images)
                    losses = self.loss_fn(output, masks)
                    predictions = output
                
                # Accumulate losses
                total_loss += losses['total_loss'].item()
                for key, value in losses.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
                
                # Update metrics
                metrics_calculator.update(predictions, masks)
                
                # Update progress bar
                pbar.set_postfix({'val_loss': f"{losses['total_loss'].item():.4f}"})
        
        # Calculate final metrics
        metrics = metrics_calculator.compute()
        
        # Average losses
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'avg_loss': avg_loss, **loss_components, **metrics}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_miou': self.best_val_miou,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        
        # Save latest checkpoint
        checkpoint_dir = Path(self.config['checkpoints_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            self.logger.info(f"Best model saved with mIoU: {self.best_val_miou:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_miou = checkpoint['best_val_miou']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_results = self.train_epoch()
            self.train_losses.append(train_results)
            
            # Validate epoch
            val_results = self.validate_epoch()
            self.val_losses.append(val_results)
            self.val_metrics.append(val_results)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['mean_iou'])
                else:
                    self.scheduler.step()
            
            # Log results
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_results['avg_loss']:.4f}, "
                f"Val Loss: {val_results['avg_loss']:.4f}, "
                f"Val mIoU: {val_results['mean_iou']:.4f}, "
                f"Val Dice: {val_results['mean_dice']:.4f}"
            )
            
            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar('train/epoch_loss', train_results['avg_loss'], epoch)
                self.writer.add_scalar('val/epoch_loss', val_results['avg_loss'], epoch)
                self.writer.add_scalar('val/mean_iou', val_results['mean_iou'], epoch)
                self.writer.add_scalar('val/mean_dice', val_results['mean_dice'], epoch)
                
                # Log individual class metrics
                for i, (iou, dice) in enumerate(zip(val_results['class_iou'], val_results['class_dice'])):
                    self.writer.add_scalar(f'val/class_{i}_iou', iou, epoch)
                    self.writer.add_scalar(f'val/class_{i}_dice', dice, epoch)
            
            # Wandb logging
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_results['avg_loss'],
                    'val_loss': val_results['avg_loss'],
                    'val_miou': val_results['mean_iou'],
                    'val_dice': val_results['mean_dice'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_results['mean_iou'] > self.best_val_miou
            if is_best:
                self.best_val_miou = val_results['mean_iou']
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config['training']['early_stopping_patience'] > 0:
                if len(self.val_metrics) >= self.config['training']['early_stopping_patience']:
                    recent_scores = [m['mean_iou'] for m in self.val_metrics[-self.config['training']['early_stopping_patience']:]]
                    if all(score <= self.best_val_miou for score in recent_scores):
                        self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
        
        self.logger.info("Training completed!")
        
        # Final test evaluation
        self.test()
    
    def test(self):
        """Test the best model."""
        self.logger.info("Starting test evaluation...")
        
        # Load best checkpoint
        best_checkpoint_path = Path(self.config['checkpoints_dir']) / 'best.pth'
        if best_checkpoint_path.exists():
            self.load_checkpoint(str(best_checkpoint_path))
        
        self.model.eval()
        
        # Initialize metrics
        metrics_calculator = SegmentationMetrics(
            num_classes=self.config['model']['num_classes'],
            device=self.device
        )
        
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                if self.config['model']['aux_classifier']:
                    main_output, _ = self.model(images)
                    predictions = main_output
                else:
                    predictions = self.model(images)
                
                # Update metrics
                metrics_calculator.update(predictions, masks)
                
                # Store for visualization
                test_predictions.append(predictions.cpu())
                test_targets.append(masks.cpu())
        
        # Calculate final test metrics
        test_metrics = metrics_calculator.compute()
        
        self.logger.info("Test Results:")
        self.logger.info(f"Test mIoU: {test_metrics['mean_iou']:.4f}")
        self.logger.info(f"Test Dice: {test_metrics['mean_dice']:.4f}")
        self.logger.info(f"Test Precision: {test_metrics['mean_precision']:.4f}")
        self.logger.info(f"Test Recall: {test_metrics['mean_recall']:.4f}")
        
        # Save test results
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Save visualizations
        if self.config.get('save_visualizations', True):
            save_segmentation_results(
                test_predictions[:10],  # Save first 10 samples
                test_targets[:10],
                results_dir / 'visualizations'
            )
        
        return test_metrics


def get_default_config() -> Dict:
    """Get default training configuration."""
    return {
        'model': {
            'encoder_name': 'resnet50',
            'num_classes': 8,  # Excluding class 0 (papaya)
            'encoder_weights': 'imagenet',
            'aux_classifier': True
        },
        'data': {
            'dataset_path': 'data',
            'masks_path': 'segmentation_masks',
            'batch_size': 8,
            'image_size': (512, 512),
            'num_workers': 4,
            'train_augmentations': True
        },
        'loss': {
            'ce_weight': 1.0,
            'dice_weight': 1.0,
            'focal_weight': 0.5,
            'tversky_weight': 0.5,
            'boundary_weight': 0.3,
            'aux_weight': 0.4,
            'use_class_weights': True
        },
        'optimizer': {
            'name': 'adamw',
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'momentum': 0.9
        },
        'scheduler': {
            'name': 'cosine',
            'step_size': 30,
            'gamma': 0.1,
            'factor': 0.5,
            'patience': 10
        },
        'training': {
            'num_epochs': 100,
            'gradient_clip': 1.0,
            'early_stopping_patience': 15
        },
        'logging': {
            'log_interval': 10
        },
        'checkpoints_dir': 'checkpoints/segmentation',
        'log_dir': 'logs/segmentation',
        'results_dir': 'results/segmentation',
        'use_tensorboard': True,
        'use_wandb': False,
        'save_visualizations': True,
        'experiment_name': 'papaya_segmentation'
    }


def main():
    parser = argparse.ArgumentParser(description='Train papaya disease segmentation model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true', help='Only run testing')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Initialize trainer
    trainer = SegmentationTrainer(config)
    trainer.setup_model()
    trainer.setup_data()
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training or testing
    if args.test_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
