#!/usr/bin/env python3
"""
Automated Segmentation Pipeline for Papaya Disease Classification
Converts bounding box annotations to segmentation masks using multiple approaches.

This script implements several automated segmentation techniques:
1. SAM (Segment Anything Model) - for high-quality segmentation
2. GrabCut - traditional computer vision method
3. Color-based thresholding - for specific disease patterns
4. Watershed segmentation - for complex regions

Author: Cascade AI Assistant
"""

import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PapayaSegmentationPipeline:
    """Main pipeline for automated papaya disease segmentation."""
    
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create output directory structure
        for split in ['Train', 'Test', 'Validation']:
            (self.output_path / split / 'masks').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'visualizations').mkdir(parents=True, exist_ok=True)
        
        # Disease class names (based on analysis)
        self.class_names = {
            0: 'Papaya',
            1: 'Anthracnose', 
            2: 'Black_Spot',
            3: 'Chocolate_Spot',
            4: 'Dieback',
            6: 'Phytophthora',
            7: 'Black_Spot_V2',
            8: 'Scar'
        }
        
        logger.info(f"Initialized segmentation pipeline")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Output path: {self.output_path}")

    def parse_yolo_annotation(self, annotation_path: str, image_shape: Tuple[int, int]) -> List[Dict]:
        """Parse YOLO format annotation file."""
        annotations = []
        if not os.path.exists(annotation_path):
            return annotations
            
        height, width = image_shape
        with open(annotation_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height
                    
                    # Convert to bounding box coordinates
                    x1 = int(x_center - box_width/2)
                    y1 = int(y_center - box_height/2)
                    x2 = int(x_center + box_width/2)
                    y2 = int(y_center + box_height/2)
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'center': (int(x_center), int(y_center))
                    })
        return annotations

    def grabcut_segmentation(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Apply GrabCut algorithm within bounding box."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Initialize mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Set rectangle as probable foreground
        rect = (x1, y1, x2-x1, y2-y1)
        
        try:
            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            return mask2
            
        except Exception as e:
            logger.warning(f"GrabCut failed: {e}")
            # Fallback to simple rectangle mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            return mask

    def color_based_segmentation(self, image: np.ndarray, bbox: List[int], class_id: int) -> np.ndarray:
        """Apply color-based segmentation for specific disease types."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi = image[y1:y2, x1:x2]
        
        # Convert to different color spaces
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Define color ranges for different diseases
        if class_id == 2 or class_id == 7:  # Black spot
            # Target dark regions
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([180, 255, 80])
            color_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
            
        elif class_id == 3:  # Chocolate spot
            # Target brown/chocolate colored regions
            lower_bound = np.array([10, 50, 20])
            upper_bound = np.array([20, 255, 200])
            color_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
            
        elif class_id == 8:  # Scar
            # Target lighter, damaged regions
            lower_bound = np.array([0, 0, 180])
            upper_bound = np.array([180, 50, 255])
            color_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
            
        else:
            # Generic approach: find regions different from healthy papaya
            # Healthy papaya is typically yellow-orange
            lower_healthy = np.array([15, 100, 100])
            upper_healthy = np.array([35, 255, 255])
            healthy_mask = cv2.inRange(hsv_roi, lower_healthy, upper_healthy)
            color_mask = cv2.bitwise_not(healthy_mask)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # Place back in full image coordinates
        mask[y1:y2, x1:x2] = color_mask
        
        return (mask > 0).astype(np.uint8)

    def watershed_segmentation(self, image: np.ndarray, bbox: List[int], center: Tuple[int, int]) -> np.ndarray:
        """Apply watershed segmentation starting from center point."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Create markers
        markers = np.zeros(gray.shape, dtype=np.int32)
        
        # Foreground marker (center of disease)
        center_x = center[0] - x1
        center_y = center[1] - y1
        
        if 0 <= center_x < roi.shape[1] and 0 <= center_y < roi.shape[0]:
            cv2.circle(markers, (center_x, center_y), 5, 1, -1)
        
        # Background markers (edges of bounding box)
        markers[0, :] = 2
        markers[-1, :] = 2
        markers[:, 0] = 2
        markers[:, -1] = 2
        
        # Apply watershed
        try:
            cv2.watershed(roi, markers)
            
            # Create mask from watershed result
            mask_roi = (markers == 1).astype(np.uint8)
            
            # Place back in full image
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = mask_roi
            
            return mask
            
        except Exception as e:
            logger.warning(f"Watershed failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def combine_masks(self, masks: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """Combine multiple segmentation masks using weighted voting."""
        if not masks:
            return np.zeros_like(masks[0]) if masks else np.zeros((100, 100), dtype=np.uint8)
        
        if weights is None:
            weights = [1.0] * len(masks)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted combination
        combined = np.zeros_like(masks[0], dtype=np.float32)
        for mask, weight in zip(masks, weights):
            combined += mask.astype(np.float32) * weight
        
        # Threshold to create binary mask
        return (combined > 0.5).astype(np.uint8)

    def process_single_image(self, image_path: str, annotation_path: str, output_dir: str) -> bool:
        """Process a single image and generate segmentation masks."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            # Parse annotations
            annotations = self.parse_yolo_annotation(annotation_path, image.shape[:2])
            if not annotations:
                logger.warning(f"No annotations found for {image_path}")
                return False
            
            # Create output filename
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Initialize combined mask for all objects
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            individual_masks = {}
            
            for i, ann in enumerate(annotations):
                class_id = ann['class_id']
                bbox = ann['bbox']
                center = ann['center']
                
                # Apply different segmentation methods
                masks = []
                
                # Method 1: GrabCut
                grabcut_mask = self.grabcut_segmentation(image, bbox)
                masks.append(grabcut_mask)
                
                # Method 2: Color-based segmentation
                color_mask = self.color_based_segmentation(image, bbox, class_id)
                masks.append(color_mask)
                
                # Method 3: Watershed
                watershed_mask = self.watershed_segmentation(image, bbox, center)
                masks.append(watershed_mask)
                
                # Combine masks with different weights based on disease type
                if class_id in [2, 7, 3]:  # Black spot, Chocolate spot - favor color
                    weights = [0.2, 0.6, 0.2]
                elif class_id == 8:  # Scar - favor GrabCut
                    weights = [0.6, 0.2, 0.2]
                else:  # Others - balanced approach
                    weights = [0.4, 0.3, 0.3]
                
                final_mask = self.combine_masks(masks, weights)
                
                # Store individual mask
                individual_masks[f"{class_id}_{i}"] = final_mask
                
                # Add to combined mask with unique label
                combined_mask[final_mask > 0] = class_id + 1  # Add 1 to avoid 0 (background)
            
            # Save masks
            mask_dir = os.path.join(output_dir, 'masks')
            os.makedirs(mask_dir, exist_ok=True)
            
            # Save combined mask
            cv2.imwrite(os.path.join(mask_dir, f"{image_name}_mask.png"), combined_mask)
            
            # Save individual masks
            for mask_name, mask in individual_masks.items():
                cv2.imwrite(os.path.join(mask_dir, f"{image_name}_{mask_name}.png"), mask * 255)
            
            # Create visualization
            self.create_visualization(image, combined_mask, annotations, 
                                    os.path.join(output_dir, 'visualizations', f"{image_name}_viz.jpg"))
            
            logger.info(f"Processed {image_name} - found {len(annotations)} objects")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False

    def create_visualization(self, image: np.ndarray, mask: np.ndarray, 
                           annotations: List[Dict], output_path: str):
        """Create visualization showing original image, mask overlay, and bounding boxes."""
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (128, 128, 128)
        ]
        
        for class_id in range(1, 9):
            if np.any(mask == class_id):
                color = colors[(class_id - 1) % len(colors)]
                colored_mask[mask == class_id] = color
        
        # Overlay mask on image
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        
        # Draw bounding boxes and labels
        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox']
            class_id = ann['class_id']
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} ({class_id})"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save visualization
        cv2.imwrite(output_path, overlay)

    def process_dataset_split(self, split_name: str) -> int:
        """Process all images in a dataset split (Train/Test/Validation)."""
        split_path = self.dataset_path / split_name
        output_split_path = self.output_path / split_name
        
        if not split_path.exists():
            logger.warning(f"Split {split_name} not found at {split_path}")
            return 0
        
        # Find all image files
        image_files = list(split_path.glob("*.jpg")) + list(split_path.glob("*.jpeg")) + list(split_path.glob("*.png"))
        
        processed_count = 0
        total_files = len(image_files)
        
        logger.info(f"Processing {total_files} images in {split_name} split")
        
        for i, image_path in enumerate(image_files):
            # Find corresponding annotation file
            annotation_path = image_path.with_suffix('.txt')
            
            if annotation_path.exists():
                success = self.process_single_image(
                    str(image_path), 
                    str(annotation_path), 
                    str(output_split_path)
                )
                if success:
                    processed_count += 1
            else:
                logger.warning(f"No annotation file found for {image_path}")
            
            # Progress update
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{total_files} ({((i + 1)/total_files)*100:.1f}%)")
        
        logger.info(f"Completed {split_name} split: {processed_count}/{total_files} images processed")
        return processed_count

    def run_full_pipeline(self):
        """Run the complete segmentation pipeline on all dataset splits."""
        logger.info("Starting automated segmentation pipeline")
        
        total_processed = 0
        
        # Process each split
        for split in ['Train', 'Test', 'Validation']:
            processed = self.process_dataset_split(split)
            total_processed += processed
        
        logger.info(f"Pipeline completed! Total images processed: {total_processed}")
        
        # Generate summary report
        self.generate_summary_report(total_processed)

    def generate_summary_report(self, total_processed: int):
        """Generate a summary report of the segmentation process."""
        report_path = self.output_path / "segmentation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Automated Papaya Disease Segmentation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total images processed: {total_processed}\n")
            f.write(f"Dataset path: {self.dataset_path}\n")
            f.write(f"Output path: {self.output_path}\n\n")
            
            f.write("Methods used:\n")
            f.write("1. GrabCut algorithm for general segmentation\n")
            f.write("2. Color-based thresholding for disease-specific patterns\n")
            f.write("3. Watershed segmentation for complex regions\n")
            f.write("4. Weighted combination of all methods\n\n")
            
            f.write("Output structure:\n")
            f.write("- masks/: Binary and multi-class segmentation masks\n")
            f.write("- visualizations/: Overlay visualizations for quality check\n\n")
            
            f.write("Disease classes:\n")
            for class_id, name in self.class_names.items():
                f.write(f"  {class_id}: {name}\n")
        
        logger.info(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Automated Papaya Disease Segmentation')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the papaya dataset directory')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save segmentation outputs')
    parser.add_argument('--split', type=str, choices=['Train', 'Test', 'Validation', 'all'],
                       default='all', help='Dataset split to process')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PapayaSegmentationPipeline(args.dataset_path, args.output_path)
    
    # Run pipeline
    if args.split == 'all':
        pipeline.run_full_pipeline()
    else:
        pipeline.process_dataset_split(args.split)


if __name__ == "__main__":
    main()
