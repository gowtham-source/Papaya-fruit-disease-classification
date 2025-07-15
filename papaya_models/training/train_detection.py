#!/usr/bin/env python3
"""
Training script for papaya disease detection model.
Implements comprehensive training with metrics, logging, and checkpointing.
"""

import os
import sys
import argparse
import json
import time
import logging
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

from detection.fast_hybrid_detector import FastHybridDetector, create_fast_detector
from data.detection_dataset import PapayaDetectionDataset, create_detection_dataloaders
from losses.detection_losses import FastDetectionLoss, YOLOLoss
from utils.metrics import DetectionMetrics
from utils.visualization import save_detection_results


class DetectionTrainer:
    """Comprehensive trainer for papaya disease detection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging first
        self.logger = self._setup_logging()
        self.writer = None
        if config.get('use_tensorboard', True):
            self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        if config.get('use_wandb', False):
            wandb.init(
                project="papaya-detection",
                config=config,
                name=config.get('experiment_name', 'papaya_det_experiment')
            )
            
        # Log GPU information
        self._log_gpu_info()
        
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
        self.best_val_map = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Configure logger for detection_dataset to ensure its messages are captured
        dataset_logger = logging.getLogger('data.detection_dataset')
        dataset_logger.setLevel(logging.DEBUG)
        
        # Add handlers from DetectionTrainer's logger if they aren't already there
        if self.logger.hasHandlers():
            # Clear any existing handlers to avoid duplicates
            for handler in dataset_logger.handlers[:]:
                dataset_logger.removeHandler(handler)
            # Add the handlers from the main trainer logger
            for handler in self.logger.handlers:
                dataset_logger.addHandler(handler)
            # Prevent propagation to avoid duplicate logs
            dataset_logger.propagate = False
    
    def _setup_logging(self):
        """Setup logging."""
        import logging
        
        # Create logs directory
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger('DetectionTrainer')
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
        # Get input channels from config or default to 3 (RGB)
        in_channels = self.config['model'].get('in_channels', 3)
        
        # Log model configuration
        self.logger.info(f"Initializing model with {in_channels} input channels")
        
        # Model
        self.model = create_fast_detector(
            backbone_name=self.config['model']['backbone'],
            num_classes=self.config['model']['num_classes'],
            anchor_free=self.config['model']['anchor_free'],
            in_channels=in_channels
        ).to(self.device)
        
        # Loss function
        if self.config['loss']['type'] == 'fast_detection':
            self.loss_fn = FastDetectionLoss(
                num_classes=self.config['model']['num_classes'],
                alpha=self.config['loss']['focal_alpha'],
                gamma=self.config['loss']['focal_gamma'],
                lambda_coord=self.config['loss']['lambda_coord'],
                lambda_obj=self.config['loss']['lambda_obj'],
                lambda_noobj=self.config['loss']['lambda_noobj'],
                anchor_free=self.config['model']['anchor_free']
            )
        elif self.config['loss']['type'] == 'yolo':
            self.loss_fn = YOLOLoss(
                num_classes=self.config['model']['num_classes'],
                lambda_coord=self.config['loss']['lambda_coord'],
                lambda_noobj=self.config['loss']['lambda_noobj']
            )
        
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
        elif self.config['scheduler']['name'] == 'warmup_cosine':
            from utils.schedulers import WarmupCosineScheduler
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=self.config['scheduler']['warmup_epochs'],
                max_epochs=self.config['training']['num_epochs']
            )
        
        self.logger.info(f"Model initialized: {self.config['model']['backbone']}")
        self.logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Device: {self.device}")
    
    def setup_data(self):
        """Setup data loaders."""
        # Get directory names from config or use defaults
        train_dir = self.config['data'].get('train_dir', 'Train')
        val_dir = self.config['data'].get('val_dir', 'Validation')
        test_dir = self.config['data'].get('test_dir', 'Test')
        
        self.logger.info(f"Loading data with directories - Train: {train_dir}, Val: {val_dir}, Test: {test_dir}")
        
        # Create all dataloaders at once
        dataloaders = create_detection_dataloaders(
            dataset_path=self.config['data']['dataset_path'],
            batch_size=self.config['data']['batch_size'],
            image_size=self.config['data']['image_size'],
            num_workers=self.config['data']['num_workers'],
            num_classes=self.config['model']['num_classes'],
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir
        )
        
        # Assign the dataloaders
        self.train_loader = dataloaders.get('train')
        self.val_loader = dataloaders.get('val')  # Changed from 'validation' to 'val' to match the key in dataloaders
        self.test_loader = dataloaders.get('test')
        
        # Log dataset sizes
        if self.train_loader is not None:
            self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        else:
            self.logger.warning("No training data found!")
            
        if self.val_loader is not None:
            self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        else:
            self.logger.warning("No validation data found!")
            
        if self.test_loader is not None:
            self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        else:
            self.logger.warning("No test data found!")
    
    def _log_gpu_info(self):
        """Log information about available GPUs."""
        if torch.cuda.is_available():
            self.logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
            self.logger.info(f'CUDA Version: {torch.version.cuda}')
            self.logger.info(f'PyTorch Version: {torch.__version__}')
            self.logger.info(f'GPU Memory Usage:')
            self.logger.info(f'Allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB')
            self.logger.info(f'Cached:    {torch.cuda.memory_reserved(0)/1024**2:.1f} MB')
        else:
            self.logger.warning('No GPU available, using CPU. Training will be slow.')
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Clear CUDA cache at the start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=self.config['training'].get('mixed_precision', True))
        
        progress_bar = tqdm(self.train_loader, total=num_batches, 
                          desc=f'Epoch {self.current_epoch + 1}', dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device (non-blocking for better performance)
            images = batch['images'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            num_objects = batch['num_objects']
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=self.config['training'].get('mixed_precision', True)):
                # Forward pass
                outputs = self.model(images)
                
                # Prepare targets for loss computation
                target_list = []
                for i in range(images.size(0)):
                    n = num_objects[i].item()
                    if n > 0:
                        target_list.append(targets[i, :n])
                    else:
                        # Add empty tensor if no objects
                        target_list.append(torch.zeros((0, 5), device=targets.device))
                
                # Compute loss
                loss_dict = self.loss_fn(outputs, target_list)
                loss = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize with gradient scaling for mixed precision
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config['training']['grad_clip']
                )
            
            scaler.step(self.optimizer)
            scaler.update()
            
            # Update progress
            loss_value = loss.item()
            epoch_loss += loss_value
            avg_loss = epoch_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar('train/loss', loss_value, self.current_epoch * num_batches + batch_idx)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], 
                                      self.current_epoch * num_batches + batch_idx)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v.item(), self.current_epoch * num_batches + batch_idx)
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train/loss': loss_value,
                    'train/avg_loss': avg_loss,
                    'train/lr': self.optimizer.param_groups[0]["lr"],
                    **{f'train/{k}': v.item() for k, v in loss_dict.items()},
                    'epoch': self.current_epoch,
                    'batch': batch_idx,
                    'step': self.current_epoch * num_batches + batch_idx
                })
            
            # Free up memory
            del images, targets, outputs, loss_dict, loss
            
        # Update learning rate scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # For ReduceLROnPlateau, we need to pass a validation metric
                # We'll update it after validation
                pass
            else:
                self.scheduler.step()
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        
        # Log epoch metrics
        self.logger.info(f'Epoch {self.current_epoch + 1} - Train Loss: {avg_epoch_loss:.4f}')
        
        # Free up memory
        torch.cuda.empty_cache()
        
        return avg_epoch_loss
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch with GPU optimizations."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        if num_batches == 0:
            self.logger.warning("No validation data available.")
            return {}
            
        val_metrics = DetectionMetrics(num_classes=self.config['model']['num_classes'])
        
        # Clear CUDA cache before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Disable gradient calculation
        torch.set_grad_enabled(False)
        
        try:
            progress_bar = tqdm(self.val_loader, desc=f'Validation Epoch {self.current_epoch + 1}', 
                             dynamic_ncols=True)
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device (non-blocking)
                images = batch['images'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                num_objects = batch['num_objects']
                
                # Prepare targets for loss computation
                target_list = []
                for i in range(images.size(0)):
                    n = num_objects[i].item()
                    if n > 0:
                        target_list.append(targets[i, :n])
                    else:
                        # Add empty tensor if no objects
                        target_list.append(torch.zeros((0, 5), device=targets.device))
                
                # Mixed precision for validation
                with torch.cuda.amp.autocast(enabled=self.config['training'].get('mixed_precision', True)):
                    # Forward pass
                    predictions = self.model(images)
                    
                    # Compute loss
                    loss_dict = self.loss_fn(predictions, targets)
                    loss = sum(loss for loss in loss_dict.values())
                
                # Update metrics
                processed_preds = self.process_predictions(predictions, images.size()[-2:])
                processed_targets = self.process_targets(target_list)
                val_metrics.update(processed_preds, processed_targets)
                
                # Accumulate loss
                loss_value = loss.item()
                total_loss += loss_value
                avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}'
                })
                
                # Free up memory
                del images, targets, predictions, loss_dict, loss
            
            # Calculate metrics
            metrics = val_metrics.compute()
            avg_epoch_loss = total_loss / num_batches
            
            # Log metrics
            self.logger.info(f'Epoch {self.current_epoch + 1} - Validation Loss: {avg_epoch_loss:.4f}')
            self.logger.info(f'mAP@0.5: {metrics["map_50"]:.4f}, mAP@0.5:0.95: {metrics["map"]:.4f}')
            
            # Log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar('val/loss', avg_epoch_loss, self.current_epoch)
                for key, value in metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'val/loss': avg_epoch_loss,
                    **{f'val/{k}': v for k, v in metrics.items()},
                    'epoch': self.current_epoch
                })
            
            # Update learning rate scheduler if using ReduceLROnPlateau
            if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics['map'])
            
            return {'loss': avg_epoch_loss, **metrics}
            
        finally:
            # Re-enable gradient calculation
            torch.set_grad_enabled(True)
            
            # Free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process_predictions(self, predictions: Dict[str, List[torch.Tensor]], image_shape: torch.Size) -> List[Dict]:
        """Process model predictions into format for metrics calculation."""
        # This would implement NMS and convert predictions to proper format
        # For now, simplified implementation
        batch_size = image_shape[0]
        processed = []
        
        for b in range(batch_size):
            # Extract predictions for this batch item
            # Apply NMS, convert to [x1, y1, x2, y2, score, class] format
            # This is a placeholder - actual implementation would be more complex
            pred_dict = {
                'boxes': torch.zeros(0, 4),  # [x1, y1, x2, y2]
                'scores': torch.zeros(0),
                'labels': torch.zeros(0, dtype=torch.long)
            }
            processed.append(pred_dict)
        
        return processed
    
    def process_targets(self, targets: torch.Tensor) -> List[Dict]:
        """Process targets into format for metrics calculation."""
        batch_size = targets.size(0)
        processed = []
        
        for b in range(batch_size):
            batch_targets = targets[b]
            valid_mask = batch_targets[:, 0] >= 0  # Valid class IDs
            valid_targets = batch_targets[valid_mask]
            
            if len(valid_targets) > 0:
                # Convert from [class, x_center, y_center, width, height] to [x1, y1, x2, y2]
                boxes = valid_targets[:, 1:5].clone()
                boxes[:, 0] = valid_targets[:, 1] - valid_targets[:, 3] / 2  # x1
                boxes[:, 1] = valid_targets[:, 2] - valid_targets[:, 4] / 2  # y1
                boxes[:, 2] = valid_targets[:, 1] + valid_targets[:, 3] / 2  # x2
                boxes[:, 3] = valid_targets[:, 2] + valid_targets[:, 4] / 2  # y2
                
                target_dict = {
                    'boxes': boxes,
                    'labels': valid_targets[:, 0].long()
                }
            else:
                target_dict = {
                    'boxes': torch.zeros(0, 4),
                    'labels': torch.zeros(0, dtype=torch.long)
                }
            
            processed.append(target_dict)
        
        return processed
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_map': self.best_val_map,
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
            self.logger.info(f"Best model saved with mAP: {self.best_val_map:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_map = checkpoint['best_val_map']
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
                    self.scheduler.step(val_results.get('map_50', val_results['avg_loss']))
                else:
                    self.scheduler.step()
            
            # Log results
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_results['avg_loss']:.4f}, "
                f"Val Loss: {val_results['avg_loss']:.4f}, "
                f"Val mAP@0.5: {val_results.get('map_50', 0.0):.4f}, "
                f"Val mAP@0.75: {val_results.get('map_75', 0.0):.4f}"
            )
            
            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar('train/epoch_loss', train_results['avg_loss'], epoch)
                self.writer.add_scalar('val/epoch_loss', val_results['avg_loss'], epoch)
                if 'map_50' in val_results:
                    self.writer.add_scalar('val/map_50', val_results['map_50'], epoch)
                if 'map_75' in val_results:
                    self.writer.add_scalar('val/map_75', val_results['map_75'], epoch)
            
            # Wandb logging
            if self.config.get('use_wandb', False):
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_results['avg_loss'],
                    'val_loss': val_results['avg_loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                if 'map_50' in val_results:
                    log_dict['val_map_50'] = val_results['map_50']
                if 'map_75' in val_results:
                    log_dict['val_map_75'] = val_results['map_75']
                
                wandb.log(log_dict)
            
            # Save checkpoint
            current_map = val_results.get('map_50', 0.0)
            is_best = current_map > self.best_val_map
            if is_best:
                self.best_val_map = current_map
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config['training']['early_stopping_patience'] > 0:
                if len(self.val_metrics) >= self.config['training']['early_stopping_patience']:
                    recent_scores = [m.get('map_50', 0.0) for m in self.val_metrics[-self.config['training']['early_stopping_patience']:]]
                    if all(score <= self.best_val_map for score in recent_scores):
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
        metrics_calculator = DetectionMetrics(
            num_classes=self.config['model']['num_classes'],
            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            device=self.device
        )
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for batch in pbar:
                # Move data to device
                images = batch['images'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                num_objects = batch['num_objects']
                
                # Prepare targets for metrics
                target_list = []
                for i in range(images.size(0)):
                    n = num_objects[i].item()
                    if n > 0:
                        target_list.append(targets[i, :n])
                    else:
                        # Add empty tensor if no objects
                        target_list.append(torch.zeros((0, 5), device=targets.device))
                
                # Forward pass
                predictions = self.model(images)
                
                # Process predictions and targets
                processed_preds = self.process_predictions(predictions, images.shape[-2:])
                processed_targets = self.process_targets(target_list)
                
                all_predictions.extend(processed_preds)
                all_targets.extend(processed_targets)
        
        # Calculate final test metrics
        test_metrics = metrics_calculator.compute_map(all_predictions, all_targets)
        
        self.logger.info("Test Results:")
        for key, value in test_metrics.items():
            self.logger.info(f"Test {key}: {value:.4f}")
        
        # Save test results
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Save visualizations
        if self.config.get('save_visualizations', True):
            save_detection_results(
                all_predictions[:10],  # Save first 10 samples
                all_targets[:10],
                results_dir / 'visualizations'
            )
        
        return test_metrics


def get_default_config() -> Dict:
    """Get default training configuration."""
    return {
        'model': {
            'backbone': 'efficientnet-b3',
            'num_classes': 8,
            'anchor_free': True,
            'compound_coef': 3,
            'in_channels': 3  # Default to 3 for RGB images
        },
        'data': {
            'dataset_path': 'data',
            'batch_size': 8,
            'image_size': (512, 512),
            'num_workers': 0,  # Changed from 4 to 0 for debugging
            'train_augmentations': True
        },
        'loss': {
            'type': 'fast_detection',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'lambda_coord': 50.0,
            'lambda_obj': 1.0,
            'lambda_noobj': 0.5
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
            'patience': 10,
            'warmup_epochs': 5
        },
        'training': {
            'num_epochs': 100,
            'gradient_clip': 1.0,
            'early_stopping_patience': 15
        },
        'logging': {
            'log_interval': 10
        },
        'checkpoints_dir': 'checkpoints/detection',
        'log_dir': 'logs/detection',
        'results_dir': 'results/detection',
        'use_tensorboard': True,
        'use_wandb': False,
        'save_visualizations': True,
        'experiment_name': 'papaya_detection'
    }


def main():
    parser = argparse.ArgumentParser(description='Train papaya disease detection model')
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
    trainer = DetectionTrainer(config)
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
