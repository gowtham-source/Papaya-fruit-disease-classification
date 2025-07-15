# Papaya Disease Segmentation and Detection Framework

A comprehensive PyTorch-based framework for automated papaya disease detection and segmentation using deep learning models.

## 🎯 Overview

This project implements an end-to-end pipeline for:
1. **Automated Segmentation**: Generating segmentation masks from bounding box annotations using multi-method pipeline
2. **Disease Segmentation**: U-Net with ResNet encoder for pixel-level disease classification
3. **Disease Detection**: Fast hybrid detector for bounding box detection with improved speed and accuracy

## 📁 Project Structure

```
papaya_models/
├── configs/                    # Configuration files
│   ├── segmentation_config.json
│   └── detection_config.json
├── data/                      # Data loading utilities
│   ├── segmentation_dataset.py
│   └── detection_dataset.py
├── detection/                 # Detection models
│   └── fast_hybrid_detector.py
├── losses/                    # Loss functions
│   ├── segmentation_losses.py
│   └── detection_losses.py
├── segmentation/              # Segmentation models
│   └── unet_resnet.py
├── training/                  # Training scripts
│   ├── train_segmentation.py
│   └── train_detection.py
├── utils/                     # Utility modules
│   ├── metrics.py
│   ├── visualization.py
│   └── schedulers.py
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
uv pip install -r requirements.txt

# For segmentation pipeline dependencies
uv pip install -r ../segmentation_requirements.txt
```

### 2. Generate Segmentation Masks (First Time Only)

```bash
# Run the automated segmentation pipeline
cd ..
python auto_segmentation.py --dataset_path data --output_path segmentation_masks
```

### 3. Train Segmentation Model

```bash
# Using default configuration
python training/train_segmentation.py

# Using custom configuration
python training/train_segmentation.py --config configs/segmentation_config.json

# Resume from checkpoint
python training/train_segmentation.py --resume checkpoints/segmentation/latest.pth

# Test only
python training/train_segmentation.py --test-only --resume checkpoints/segmentation/best.pth
```

### 4. Train Detection Model

```bash
# Using default configuration
python training/train_detection.py

# Using custom configuration
python training/train_detection.py --config configs/detection_config.json

# Resume from checkpoint
python training/train_detection.py --resume checkpoints/detection/latest.pth
```

## 🏗️ Model Architectures

### Segmentation Model (U-Net + ResNet)
- **Encoder**: ResNet backbone (18/34/50/101) with ImageNet pretraining
- **Decoder**: U-Net decoder with skip connections and attention mechanisms
- **Features**: Pyramid pooling, auxiliary classifier, deep supervision
- **Output**: Multi-class segmentation masks (9 classes: background + 8 diseases)

### Detection Model (Fast Hybrid Detector)
- **Backbone**: EfficientNet (B0-B7) with compound scaling
- **Neck**: Fast Feature Pyramid Network with depthwise separable convolutions
- **Head**: Optimized detection head with squeeze-excitation blocks
- **Features**: Anchor-free detection, faster than YOLO, high accuracy

## 📊 Disease Classes

The framework supports 8 papaya disease classes:
1. **Anthracnose** - Fungal disease causing dark lesions
2. **Black Spot** - Bacterial infection with black spots
3. **Chocolate Spot** - Brown/chocolate colored lesions
4. **Dieback** - Branch/stem dying back from tips
5. **Phytophthora** - Water mold causing rot
6. **Black Spot V2** - Variant of black spot disease
7. **Scar** - Physical damage or healed lesions
8. **Background** - Healthy tissue (segmentation only)

## ⚙️ Configuration

### Segmentation Configuration
```json
{
  "model": {
    "encoder_name": "resnet50",
    "num_classes": 9,
    "encoder_weights": "imagenet",
    "aux_classifier": true
  },
  "data": {
    "dataset_path": "data",
    "masks_path": "segmentation_masks",
    "batch_size": 8,
    "image_size": [512, 512]
  },
  "loss": {
    "ce_weight": 1.0,
    "dice_weight": 1.0,
    "focal_weight": 0.5,
    "use_class_weights": true
  }
}
```

### Detection Configuration
```json
{
  "model": {
    "backbone": "efficientnet-b3",
    "num_classes": 8,
    "anchor_free": true
  },
  "data": {
    "dataset_path": "data",
    "batch_size": 8,
    "image_size": [512, 512]
  },
  "loss": {
    "type": "fast_detection",
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "lambda_coord": 50.0
  }
}
```

## 🔧 Advanced Usage

### Custom Data Loading
```python
from data.segmentation_dataset import create_segmentation_dataloader
from data.detection_dataset import create_detection_dataloader

# Segmentation data loader
seg_loader = create_segmentation_dataloader(
    dataset_path="data",
    masks_path="segmentation_masks",
    split="Train",
    batch_size=16,
    image_size=(512, 512),
    augmentations=True
)

# Detection data loader
det_loader = create_detection_dataloader(
    dataset_path="data",
    split="Train",
    batch_size=16,
    image_size=(512, 512),
    augmentations=True
)
```

### Custom Model Creation
```python
from segmentation.unet_resnet import create_unet_model
from detection.fast_hybrid_detector import create_detector_model

# Segmentation model
seg_model = create_unet_model(
    encoder_name="resnet101",
    num_classes=9,
    encoder_weights="imagenet"
)

# Detection model
det_model = create_detector_model(
    backbone="efficientnet-b5",
    num_classes=8,
    anchor_free=True
)
```

### Custom Loss Functions
```python
from losses.segmentation_losses import CombinedSegmentationLoss
from losses.detection_losses import FastDetectionLoss

# Segmentation loss
seg_loss = CombinedSegmentationLoss(
    ce_weight=1.0,
    dice_weight=1.0,
    focal_weight=0.5
)

# Detection loss
det_loss = FastDetectionLoss(
    num_classes=8,
    anchor_free=True
)
```

## 📈 Monitoring and Logging

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir logs/

# View at http://localhost:6006
```

### Weights & Biases (Optional)
```python
# Enable in config
"use_wandb": true

# Set API key
export WANDB_API_KEY=your_api_key
```

## 🎯 Performance Optimization

### Speed Optimizations
- Mixed precision training (automatic)
- Gradient accumulation
- Efficient data loading with multiple workers
- Optimized model architectures

### Memory Optimizations
- Gradient checkpointing
- Batch size adjustment
- Image size scaling
- Model pruning support

## 📊 Results and Evaluation

### Metrics Computed
**Segmentation:**
- Mean IoU (Intersection over Union)
- Dice Coefficient
- Pixel Accuracy
- Per-class metrics

**Detection:**
- mAP (mean Average Precision)
- mAP@0.5, mAP@0.75
- Precision, Recall
- Per-class AP

### Visualization
Results include:
- Segmentation mask overlays
- Detection bounding boxes
- Training curves
- Confusion matrices
- Per-class performance

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller image size
   - Enable gradient checkpointing

2. **Low Performance**
   - Check data augmentations
   - Verify class weights
   - Adjust learning rate

3. **Training Instability**
   - Enable gradient clipping
   - Use learning rate warmup
   - Check loss function weights

### Performance Tips
- Use mixed precision for faster training
- Increase num_workers for faster data loading
- Use larger batch sizes when possible
- Enable early stopping to prevent overfitting


---

**Note**: This framework is designed for research and educational purposes. For production use, additional validation and testing may be required.
