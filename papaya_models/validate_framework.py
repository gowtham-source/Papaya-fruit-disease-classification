#!/usr/bin/env python3
"""
Validation script for the papaya disease detection and segmentation framework.
This script validates that all components are properly installed and configured.
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def check_imports():
    """Check if all required packages can be imported."""
    print("Checking imports...")
    
    required_packages = [
        'torch', 'torchvision', 'cv2', 'numpy', 'PIL', 'albumentations',
        'segmentation_models_pytorch', 'timm', 'efficientnet_pytorch',
        'sklearn', 'scipy', 'matplotlib', 'tqdm'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'sklearn':
                import sklearn
            elif package == 'segmentation_models_pytorch':
                import segmentation_models_pytorch as smp
            elif package == 'efficientnet_pytorch':
                from efficientnet_pytorch import EfficientNet
            else:
                __import__(package)
            print(f"  OK {package}")
        except ImportError as e:
            print(f"  Failed {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {', '.join(failed_imports)}")
        print("Please install missing packages using: uv pip install <package_name>")
        return False
    else:
        print("All required packages imported successfully!")
        return True


def check_framework_imports():
    """Check if framework modules can be imported."""
    print("\nChecking framework imports...")
    
    framework_modules = [
        'segmentation.unet_resnet',
        'detection.fast_hybrid_detector',
        'data.segmentation_dataset',
        'data.detection_dataset',
        'losses.segmentation_losses',
        'losses.detection_losses',
        'utils.metrics',
        'utils.visualization',
        'utils.schedulers'
    ]
    
    failed_imports = []
    
    for module in framework_modules:
        try:
            __import__(module)
            print(f"  OK {module}")
        except ImportError as e:
            print(f"  Failed {module}: {e}")
            failed_imports.append(module)
            traceback.print_exc()
    
    if failed_imports:
        print(f"\nFailed framework imports: {', '.join(failed_imports)}")
        return False
    else:
        print("All framework modules imported successfully!")
        return True


def validate_configs():
    """Validate configuration files."""
    print("\nValidating configuration files...")
    
    # Get the directory of this script to find config files relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config_files = [
        os.path.join(script_dir, 'configs', 'segmentation_config.json'),
        os.path.join(script_dir, 'configs', 'detection_config.json')
    ]
    
    valid_configs = True
    
    for config_file in config_files:
        config_name = os.path.basename(config_file)
        if not os.path.exists(config_file):
            print(f"  Failed {config_name} - File not found")
            valid_configs = False
            continue
            
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"  OK {config_name} - Valid JSON")
            
            # Basic validation
            required_keys = ['model', 'data', 'loss', 'optimizer', 'training']
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                print(f"    Warning  Missing keys: {missing_keys}")
                valid_configs = False
            else:
                print(f"    OK Required keys present")
                
        except json.JSONDecodeError as e:
            print(f"  Failed {config_name} - Invalid JSON: {e}")
            valid_configs = False
        except Exception as e:
            print(f"  Failed {config_name} - Error: {e}")
            valid_configs = False
    
    return valid_configs


def test_model_creation():
    """Test model creation functions."""
    try:
        print("Testing model creation...")
        
        # Test segmentation model
        from segmentation.unet_resnet import create_unet_resnet
        seg_model = create_unet_resnet(num_classes=9, use_auxiliary=False)
        print("  OK Segmentation model created")
        
        # Test detection model  
        from detection.fast_hybrid_detector import create_fast_detector
        det_model = create_fast_detector(num_classes=8)
        print("  OK Detection model created")
        
        return True
    except Exception as e:
        print(f"  Failed Model creation failed: {str(e)}")
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss function creation."""
    print("\nTesting loss functions...")
    
    try:
        import torch
        
        # Test segmentation losses
        from losses.segmentation_losses import CombinedSegmentationLoss
        seg_loss = CombinedSegmentationLoss()
        
        # Create dummy data
        pred = torch.randn(1, 9, 64, 64)
        target = torch.randint(0, 9, (1, 64, 64))
        
        loss_value = seg_loss(pred, target)
        if isinstance(loss_value, dict):
            total_loss = loss_value.get('total_loss', sum(loss_value.values()))
            print(f"  OK Segmentation loss computed: {total_loss.item():.4f}")
        else:
            print(f"  OK Segmentation loss computed: {loss_value.item():.4f}")
        
        # Test detection losses
        from losses.detection_losses import FastDetectionLoss
        det_loss = FastDetectionLoss(num_classes=8)
        print("  OK Detection loss created successfully")
        
        return True
        
    except Exception as e:
        print(f"  Failed Loss function test failed: {e}")
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")
    
    try:
        import torch
        from utils.metrics import SegmentationMetrics, DetectionMetrics
        
        # Test segmentation metrics
        seg_metrics = SegmentationMetrics(num_classes=9)
        
        pred = torch.randint(0, 9, (10, 64, 64))
        target = torch.randint(0, 9, (10, 64, 64))
        
        # Update metrics with predictions first
        seg_metrics.update(pred, target)
        metrics = seg_metrics.compute()
        print(f"  OK Segmentation metrics: mIoU = {metrics['mean_iou']:.4f}")
        
        # Test detection metrics
        det_metrics = DetectionMetrics(num_classes=8)
        print("  OK Detection metrics created successfully")
        
        return True
        
    except Exception as e:
        print(f"  Failed Metrics test failed: {e}")
        traceback.print_exc()
        return False


def check_dataset_structure():
    """Check if dataset structure is correct."""
    print("\nChecking dataset structure...")
    
    dataset_path = Path("../data")  # Assuming data is in parent directory
    
    if not dataset_path.exists():
        print(f"  Warning  Dataset path not found: {dataset_path}")
        print(f"  Tip Expected structure: data/Train/, data/Test/, data/Validation/")
        return False
    
    required_splits = ['Train', 'Test', 'Validation']
    missing_splits = []
    
    for split in required_splits:
        split_path = dataset_path / split
        if split_path.exists():
            # Check for images and annotations
            images = list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))
            annotations = list(split_path.glob("*.json"))
            
            print(f"  OK {split}: {len(images)} images, {len(annotations)} annotations")
        else:
            print(f"  Failed {split}: Missing")
            missing_splits.append(split)
    
    # Check for segmentation masks if they exist
    masks_path = Path("../segmentation_masks")
    if masks_path.exists():
        print(f"  OK Segmentation masks found")
    else:
        print(f"  Warning  Segmentation masks not found - run auto_segmentation.py first")
    
    return len(missing_splits) == 0


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
            
            print(f"  OK CUDA available")
            print(f"  Devices: {device_count}")
            print(f"  Current device: {current_device} ({device_name})")
            print(f"  Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        else:
            print(f"  Warning  CUDA not available - using CPU")
            print(f"  Tip Training will be slower on CPU")
        
        return True
        
    except Exception as e:
        print(f"  Failed CUDA check failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("Papaya Disease Framework Validation")
    print("=" * 50)
    
    checks = [
        ("Package Imports", check_imports),
        ("Framework Imports", check_framework_imports),
        ("Configuration Files", validate_configs),
        ("Model Creation", test_model_creation),
        ("Loss Functions", test_loss_functions),
        ("Metrics", test_metrics),
        ("Dataset Structure", check_dataset_structure),
        ("CUDA Support", check_cuda)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            success = check_func()
            if success:
                passed += 1
        except Exception as e:
            print(f"  Failed {name} check failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("Framework is ready for training!")
        print("\nNext steps:")
        print("1. Generate segmentation masks: python ../auto_segmentation.py")
        print("2. Train segmentation model: python training/train_segmentation.py")
        print("3. Train detection model: python training/train_detection.py")
    else:
        print("Some checks failed. Please address the issues above.")
        print("\nFor help:")
        print("1. Check the README.md for installation instructions")
        print("2. Ensure all dependencies are installed")
        print("3. Verify dataset structure")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
