# Automated Papaya Disease Segmentation Pipeline

This pipeline automatically generates segmentation masks for papaya disease regions from bounding box annotations. It converts your existing detection dataset into a segmentation dataset without requiring manual annotation.

## Overview

The pipeline uses multiple computer vision techniques to create accurate segmentation masks:

1. **GrabCut Algorithm** - Interactive foreground/background segmentation
2. **Color-Based Thresholding** - Disease-specific color pattern detection
3. **Watershed Segmentation** - Region growing from disease centers
4. **Intelligent Combination** - Weighted fusion of all methods

## Features

- ✅ Processes YOLO format bounding box annotations
- ✅ Supports all 9 papaya disease classes
- ✅ Generates both individual and combined segmentation masks
- ✅ Creates visual overlays for quality assessment
- ✅ Maintains original dataset structure
- ✅ Batch processing with progress tracking
- ✅ Comprehensive logging and error handling

## Disease Classes Supported

| Class ID | Disease Name | Segmentation Strategy |
|----------|--------------|----------------------|
| 0 | Papaya (Healthy) | General segmentation |
| 1 | Anthracnose | Balanced approach |
| 2 | Black Spot | Color-focused |
| 3 | Chocolate Spot | Color-focused |
| 4 | Dieback | Balanced approach |
| 6 | Phytophthora | Balanced approach |
| 7 | Black Spot V2 | Color-focused |
| 8 | Scar | GrabCut-focused |

## Installation

1. Install required dependencies:
```bash
pip install -r segmentation_requirements.txt
```

## Usage

### Process Entire Dataset
```bash
python auto_segmentation.py --dataset_path . --output_path ./segmentation_output
```

### Process Specific Split
```bash
# Process only training data
python auto_segmentation.py --dataset_path . --output_path ./segmentation_output --split Train

# Process only test data
python auto_segmentation.py --dataset_path . --output_path ./segmentation_output --split Test

# Process only validation data
python auto_segmentation.py --dataset_path . --output_path ./segmentation_output --split Validation
```

## Output Structure

The pipeline creates the following directory structure:

```
segmentation_output/
├── Train/
│   ├── masks/
│   │   ├── TR000001-8-8-8_mask.png          # Combined multi-class mask
│   │   ├── TR000001-8-8-8_8_0.png           # Individual disease mask
│   │   └── ...
│   └── visualizations/
│       ├── TR000001-8-8-8_viz.jpg           # Overlay visualization
│       └── ...
├── Test/
│   ├── masks/
│   └── visualizations/
├── Validation/
│   ├── masks/
│   └── visualizations/
└── segmentation_report.txt                   # Summary report
```

## Mask Format

- **Combined masks**: Multi-class masks where each pixel value represents `class_id + 1`
  - 0 = Background
  - 1 = Papaya (class 0)
  - 2 = Anthracnose (class 1)
  - etc.

- **Individual masks**: Binary masks (0 or 255) for each detected object

## Algorithm Details

### Method Selection by Disease Type

The pipeline intelligently selects the best combination of segmentation methods based on disease characteristics:

- **Black Spot diseases (classes 2, 7)**: Emphasizes color-based segmentation (60% weight)
- **Chocolate Spot (class 3)**: Emphasizes color-based segmentation (60% weight)  
- **Scar disease (class 8)**: Emphasizes GrabCut segmentation (60% weight)
- **Other diseases**: Balanced combination of all methods

### Color-Based Segmentation Strategies

- **Black Spot**: Targets dark regions (low brightness in HSV)
- **Chocolate Spot**: Targets brown/chocolate colors (specific hue range)
- **Scar**: Targets lighter, damaged regions (high brightness, low saturation)
- **General**: Identifies regions different from healthy papaya colors

## Quality Assessment

The pipeline generates visualization images showing:
- Original image
- Segmentation mask overlay
- Bounding box annotations
- Class labels

Review these visualizations to assess segmentation quality and adjust parameters if needed.

## Performance Tips

1. **Memory Usage**: For large datasets, consider processing splits separately
2. **Speed**: The pipeline processes ~100-200 images per minute depending on image size
3. **Quality**: Check visualizations regularly to ensure good segmentation quality

## Troubleshooting

### Common Issues

1. **"Could not load image" errors**: Check image file paths and formats
2. **Poor segmentation quality**: Review the visualization outputs and consider adjusting color thresholds
3. **Missing annotations**: Ensure .txt files exist for all .jpg images

### Parameter Tuning

You can modify the segmentation parameters in the script:

- **Color thresholds**: Adjust HSV ranges in `color_based_segmentation()`
- **Morphological operations**: Modify kernel sizes for mask cleanup
- **Combination weights**: Adjust weights in `process_single_image()`

## Integration with Your Existing Workflow

The generated masks can be used with:

- **PyTorch**: Load as tensor datasets for segmentation training
- **TensorFlow**: Convert to TFRecord format
- **YOLO**: Convert to YOLO segmentation format
- **COCO**: Convert to COCO segmentation format

## Example Code for Loading Masks

```python
import cv2
import numpy as np

# Load combined mask
mask = cv2.imread('segmentation_output/Train/masks/TR000001-8-8-8_mask.png', cv2.IMREAD_GRAYSCALE)

# Extract specific class mask
class_id = 8  # Scar disease
class_mask = (mask == (class_id + 1)).astype(np.uint8)

# Load individual mask
individual_mask = cv2.imread('segmentation_output/Train/masks/TR000001-8-8-8_8_0.png', cv2.IMREAD_GRAYSCALE)
```

## Next Steps

After generating segmentation masks, you can:

1. Train segmentation models (U-Net, ResNet, etc.)
2. Perform detailed disease analysis
3. Calculate precise disease area measurements
4. Develop automated disease severity assessment tools

## Support

For issues or improvements, refer to the logging output in the console and the generated `segmentation_report.txt` file for detailed processing information.
