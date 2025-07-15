#!/usr/bin/env python3
"""
Papaya Disease Detection and Segmentation
Combines YOLO detection with SAM and CV-based segmentation
"""

import argparse
import time
import os
from pathlib import Path
import urllib.request

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from segment_anything import sam_model_registry, SamPredictor

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized

class PapayaDiseaseProcessor:
    def __init__(self, detection_weights, segmentation_checkpoint, device=''):
        """Initialize EfficientNetB3 and UNet_ResNet models"""
        self.device = select_device(device)
        
        # Initialize EfficientNetB3 for detection with weights_only=False for compatibility
        self.detection_model = attempt_load(detection_weights, map_location=self.device, weights_only=False)
        self.stride = int(self.detection_model.stride.max())
        self.img_size = check_img_size(640, s=self.stride)
        self.half = self.device.type != 'cpu'
        if self.half:
            self.detection_model.half()
            
        # Initialize UNet_ResNet for segmentation
        unet = sam_model_registry["default"](checkpoint=segmentation_checkpoint)
        unet.to(device=self.device)
        self.segmentor = SamPredictor(unet)
        
        # Get class names
        self.names = self.detection_model.module.names if hasattr(self.detection_model, 'module') else self.detection_model.names
        
        # Disease class colors for visualization
        self.colors = {
            0: (255, 255, 255),  # Papaya - White
            1: (255, 0, 0),      # Anthracnose - Red
            2: (0, 0, 0),        # Black Spot - Black
            3: (139, 69, 19),    # Chocolate Spot - Brown
            4: (128, 0, 0),      # Dieback - Maroon
            6: (0, 255, 0),      # Phytophthora - Green
            7: (64, 64, 64),     # Black Spot V2 - Dark Gray
            8: (255, 165, 0)     # Scar - Orange
        }

    def preprocess_image(self, img0):
        """Preprocess image for YOLO inference"""
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect_diseases(self, img):
        """Run EfficientNetB3 detection"""
        with torch.no_grad():
            pred = self.detection_model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        return pred[0] if pred else None

    def segment_disease_region(self, image, bbox):
        """Apply UNet_ResNet segmentation within detected box"""
        # Set image for UNet_ResNet
        self.segmentor.set_image(image)
        
        # Convert bbox to prompt format
        x1, y1, x2, y2 = map(int, bbox)
        input_box = np.array([x1, y1, x2, y2])
        
        # Get UNet_ResNet prediction
        masks, _, _ = self.segmentor.predict(
            box=input_box[None, :],
            multimask_output=False
        )
        return masks[0]  # Return first mask

    def apply_cv_refinement(self, image, mask, class_id):
        """Refine SAM mask using traditional CV methods"""
        # Convert mask to uint8
        mask = mask.astype(np.uint8) * 255
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # For specific diseases, apply additional processing
        if class_id in [1, 3]:  # Anthracnose and Chocolate spot
            # Use color thresholding
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if class_id == 1:  # Anthracnose - reddish
                lower = np.array([0, 50, 50])
                upper = np.array([10, 255, 255])
            else:  # Chocolate spot - brownish
                lower = np.array([10, 50, 50])
                upper = np.array([20, 255, 255])
            
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_and(mask, color_mask)
        
        return mask > 127

    def process_image(self, source_path, save_dir):
        """Process a single image: detect, segment, and visualize"""
        # Load image
        img0 = cv2.imread(str(source_path))
        if img0 is None:
            print(f"Error loading image: {source_path}")
            return
        
        # Prepare image for YOLO
        img = self.preprocess_image(img0)
        
        # Run detection
        t1 = time_synchronized()
        pred = self.detect_diseases(img)
        t2 = time_synchronized()
        
        # Process detections
        results = []
        if pred is not None:
            # Scale boxes to original image
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
            
            # Process each detection
            for *xyxy, conf, cls in pred:
                # Get integer coordinates
                bbox = [int(x.item()) for x in xyxy]
                class_id = int(cls.item())
                confidence = float(conf.item())
                
                # Apply SAM segmentation
                mask = self.segment_disease_region(img0, bbox)
                
                # Refine mask using CV methods
                refined_mask = self.apply_cv_refinement(img0, mask, class_id)
                
                results.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': self.names[class_id],
                    'confidence': confidence,
                    'mask': refined_mask
                })
                
                # Visualize result
                color = self.colors[class_id]
                mask_overlay = np.zeros_like(img0)
                mask_overlay[refined_mask] = color
                img0 = cv2.addWeighted(img0, 1.0, mask_overlay, 0.5, 0)
                
                # Draw bbox and label
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                label = f'{self.names[class_id]} {confidence:.2f}'
                cv2.putText(img0, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save results
        save_path = Path(save_dir) / source_path.name
        cv2.imwrite(str(save_path), img0)
        
        print(f'Done. ({t2 - t1:.3f}s) Found {len(results)} diseases')
        return results

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def download_segmentation_model(model_type='vit_h'):
    """Download UNet_ResNet model checkpoint if not exists"""
    model_urls = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    }
    
    # Create weights directory if not exists
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    # Set model path
    model_path = weights_dir / f'sam_{model_type}.pth'
    
    # Download if not exists
    if not model_path.exists():
        print(f'Downloading Supporting layers and preprocessor {model_type}...')
        urllib.request.urlretrieve(model_urls[model_type], model_path)
        print('Download completed!')
    
    return str(model_path)

def main():
    # Set default device (GPU if available, else CPU)
    default_device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Get the directory where the script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Set default weights path relative to the script location
    default_weights = script_dir / 'weights' / 'unet_resnet-papaya.pt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=str(default_weights), 
                        help=f'Model weights path (default: {default_weights})')
    parser.add_argument('--segmentation-type', type=str, default='vit_h', 
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='UNet_ResNet model type: vit_h (high quality), vit_l (medium), vit_b (fast)')
    parser.add_argument('--source', type=str, default='inference/images', 
                        help='Source path (image/folder)')
    parser.add_argument('--device', default=default_device, 
                        help=f'Device to run on: cuda device, i.e. 0 or 0,1,2,3 or cpu (default: {default_device})')
    parser.add_argument('--output', type=str, default='runs/detect_segment', 
                        help='Output folder')
    args = parser.parse_args()
    
    # Download UNet_ResNet model if needed
    segmentation_checkpoint = download_segmentation_model(args.segmentation_type)

    # Create output folder
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = PapayaDiseaseProcessor(args.weights, segmentation_checkpoint, args.device)

    # Process source path
    source = Path(args.source)
    if source.is_file():
        files = [source]
    else:
        files = list(source.glob('*.jpg')) + list(source.glob('*.png'))

    # Process each image
    for file in files:
        print(f'Processing {file}...')
        processor.process_image(file, save_dir)

if __name__ == '__main__':
    main()
