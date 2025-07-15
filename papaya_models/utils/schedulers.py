#!/usr/bin/env python3
"""
Custom learning rate schedulers for papaya disease models.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate decay scheduler.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        power: float = 0.9,
        last_epoch: int = -1
    ):
        self.max_epochs = max_epochs
        self.power = power
        super(PolynomialLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class OneCycleLRScheduler(_LRScheduler):
    """
    One cycle learning rate scheduler (simplified version).
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        
        super(OneCycleLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        step = self.last_epoch
        
        if step < self.pct_start * self.total_steps:
            # Increasing phase
            progress = step / (self.pct_start * self.total_steps)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Decreasing phase
            progress = (step - self.pct_start * self.total_steps) / ((1 - self.pct_start) * self.total_steps)
            lr = self.max_lr - (self.max_lr - self.final_lr) * progress
        
        return [lr for _ in self.base_lrs]


if __name__ == "__main__":
    # Test schedulers
    import torch.nn as nn
    
    # Create dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test warmup cosine scheduler
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, max_epochs=100)
    
    print("Testing WarmupCosineScheduler:")
    for epoch in range(20):
        lr = scheduler.get_lr()[0]
        print(f"Epoch {epoch}: LR = {lr:.6f}")
        scheduler.step()
    
    print("\nScheduler tests completed!")
