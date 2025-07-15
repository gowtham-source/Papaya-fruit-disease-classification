#!/usr/bin/env python3
"""
Quick test to verify our fixes are working.
"""

import sys
import os
from pathlib import Path

# Add papaya_models to path
sys.path.append(str(Path(__file__).parent / "papaya_models"))

def test_metrics_fix():
    """Test that SegmentationMetrics works correctly."""
    print("Testing SegmentationMetrics fix...")
    try:
        import torch
        from utils.metrics import SegmentationMetrics
        
        # Create metrics instance
        metrics = SegmentationMetrics(num_classes=9)
        
        # Create dummy data
        pred = torch.randint(0, 9, (2, 64, 64))
        target = torch.randint(0, 9, (2, 64, 64))
        
        # Update and compute metrics
        metrics.update(pred, target)
        result = metrics.compute()
        
        print(f"  SUCCESS: mIoU = {result['mean_iou']:.4f}")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

def test_config_files():
    """Test that config files can be found."""
    print("Testing config file paths...")
    try:
        script_dir = Path(__file__).parent / "papaya_models"
        
        configs = [
            script_dir / "configs" / "segmentation_config.json",
            script_dir / "configs" / "detection_config.json"
        ]
        
        for config_file in configs:
            if config_file.exists():
                print(f"  SUCCESS: {config_file.name} found")
            else:
                print(f"  FAILED: {config_file.name} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA availability...")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  SUCCESS: CUDA available with {torch.cuda.device_count()} device(s)")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
        else:
            print("  INFO: CUDA not available (CPU mode)")
        
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

if __name__ == "__main__":
    print("=== Quick Validation Test ===")
    
    tests = [
        ("Metrics Fix", test_metrics_fix),
        ("Config Files", test_config_files), 
        ("CUDA Check", test_cuda)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
    
    print(f"\n=== Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("All critical fixes are working!")
    else:
        print("Some issues remain.")
