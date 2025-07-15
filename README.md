# Papaya Fruit Disease Classification

A comprehensive Deep learning project for detecting and classifying diseases in papaya fruits using EfficientNet and UNet Resnet models.

## Overview

This repository contains the implementation of a deep learning system for detecting and segmenting diseases in papaya fruits. The project includes both detection and segmentation models, along with tools for training, validation, and inference.

## Features

- Object Detection for identifying diseased areas in papaya fruits
- Instance Segmentation for precise disease boundary detection
- Support for training on custom datasets
- Pre-trained models for quick inference
- Comprehensive evaluation metrics
- Integration with modern deep learning frameworks

## Project Structure

```
papaya_models/
├── detection/         # Object detection models and utilities
├── segmentation/      # Instance segmentation models and utilities
├── losses/           # Custom loss functions
├── utils/            # Utility functions and helpers
├── models/           # Core model implementations
├── data/             # Data processing and augmentation
└── weights/          # Pre-trained model weights
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- albumentations
- opencv-python
- numpy
- Pillow
- tqdm
- tensorboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gowtham-source/Papaya-fruit-disease-classification.git
cd Papaya-fruit-disease-classification
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

To train the model:
```bash
python train.py --config-path cfg/train_config.yaml
```

### Inference

To run inference on new images:
```bash
python detect_and_segment.py --input-path path/to/image.jpg
```

### Exporting Models

To export trained models:
```bash
python export.py --model-path runs/train/exp/weights/best.pt
```


## Acknowledgments

- Thanks to the PyTorch community for their excellent documentation and support
- Special thanks to the contributors who have helped improve this project

