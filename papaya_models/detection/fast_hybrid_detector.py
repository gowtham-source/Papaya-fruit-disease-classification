#!/usr/bin/env python3
"""
Fast Hybrid Detection Model for Papaya Disease Detection
Novel architecture combining EfficientNet backbone with FPN and optimized detection head.
Designed to be faster than YOLO while maintaining high accuracy.

Fixed version with proper channel matching for EfficientNet backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import Dict, List, Tuple, Optional
from efficientnet_pytorch import EfficientNet


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient computation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, channels // reduction)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class FastFPNBlock(nn.Module):
    """Fast Feature Pyramid Network block with reduced computation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(FastFPNBlock, self).__init__()
        
        # Ensure output channels match expected dimensions
        self.out_channels = out_channels
        
        # Skip connection if input and output channels match
        self.has_skip = (in_channels == out_channels)
        
        # Channel alignment (1x1 conv to match dimensions if needed)
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.se = SqueezeExcitation(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Main path
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.conv2(x)
        x = self.se(x)
        
        # Skip connection
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        
        return x + identity


class FastFPN(nn.Module):
    """Fast Feature Pyramid Network for multi-scale feature extraction."""
    
    def __init__(self, backbone_channels: List[int], fpn_channels: int = 256):
        super(FastFPN, self).__init__()
        
        self.backbone_channels = backbone_channels
        self.fpn_channels = fpn_channels
        
        # Lateral connections - map backbone features to FPN channels
        self.lateral_convs = nn.ModuleList()
        for i, ch in enumerate(backbone_channels):
            # Create a sequential block for each lateral connection
            lateral_block = []
            
            # First conv to match channel dimensions if needed
            if ch != fpn_channels:
                lateral_block.extend([
                    nn.Conv2d(ch, fpn_channels, 1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True)
                ])
            else:
                lateral_block.append(nn.Identity())
                
            self.lateral_convs.append(nn.Sequential(*lateral_block))
        
        # Top-down pathway with feature fusion
        self.fpn_convs = nn.ModuleList()
        for i in range(len(backbone_channels)):
            # For the first level, just use a single conv
            if i == 0:
                self.fpn_convs.append(DepthwiseSeparableConv(fpn_channels, fpn_channels))
            else:
                # For higher levels, use a block that fuses with upsampled features
                self.fpn_convs.append(nn.Sequential(
                    DepthwiseSeparableConv(fpn_channels, fpn_channels),
                    nn.Upsample(scale_factor=2, mode='nearest')
                ))
        
        # Feature fusion for top-down pathway
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels * 2, fpn_channels, 1, bias=False)
            for _ in range(1, len(backbone_channels))
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.
        
        Args:
            features: List of backbone features from bottom to top level
            
        Returns:
            List of FPN features from highest to lowest resolution
        """
        # Apply lateral convolutions
        laterals = []
        for feat, lateral_conv in zip(features, self.lateral_convs):
            lat = lateral_conv(feat)
            laterals.append(lat)
        
        # Top-down pathway
        # Start with the topmost feature map
        outs = [self.fpn_convs[0](laterals[-1])]
        
        # Build top-down path
        for i in range(1, len(laterals)):
            prev_feat = outs[-1]
            lateral = laterals[-(i+1)]
            
            # Upsample previous feature to match current lateral size
            target_size = (lateral.shape[2], lateral.shape[3])
            prev_feat = F.interpolate(prev_feat, size=target_size, mode='nearest')
            
            # Concatenate features along channel dimension
            fused = torch.cat([laterals[-(i+1)], prev_feat], dim=1)
            
            # Apply fusion conv to reduce channels
            fused = self.fusion_convs[i-1](fused)
            
            # Apply FPN convolution
            out = self.fpn_convs[i](fused)
            outs.append(out)
        
        # Return features from highest to lowest resolution
        return outs[::-1]


class OptimizedDetectionHead(nn.Module):
    """Optimized detection head with shared weights and efficient computation."""
    
    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        num_anchors: int = 3,
        num_layers: int = 2
    ):
        super(OptimizedDetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared feature extraction
        shared_layers = []
        for _ in range(num_layers):
            shared_layers.extend([
                DepthwiseSeparableConv(in_channels, in_channels),
                nn.Dropout2d(0.1)
            ])
        
        self.shared_conv = nn.Sequential(*shared_layers)
        
        # Detection heads
        self.cls_head = nn.Conv2d(
            in_channels, num_anchors * num_classes, 3, padding=1
        )
        self.reg_head = nn.Conv2d(
            in_channels, num_anchors * 4, 3, padding=1
        )
        self.obj_head = nn.Conv2d(
            in_channels, num_anchors, 3, padding=1
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize detection head weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for classification head (focal loss)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shared feature extraction
        shared_features = self.shared_conv(x)
        
        # Predictions
        cls_pred = self.cls_head(shared_features)
        reg_pred = self.reg_head(shared_features)
        obj_pred = self.obj_head(shared_features)
        
        return {
            'classification': cls_pred,
            'regression': reg_pred,
            'objectness': obj_pred
        }


class FastHybridDetector(nn.Module):
    """
    Fast Hybrid Detector for Papaya Disease Detection.
    
    Architecture:
    - EfficientNet backbone for feature extraction
    - Fast FPN for multi-scale features
    - Optimized detection head with shared weights
    - Anchor-free detection option for simplicity
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet-b3',
        num_classes: int = 8,  # 8 disease classes (excluding background)
        fpn_channels: int = 128,  # Reduced from typical 256 for speed
        num_anchors: int = 3,
        pretrained: bool = True,
        anchor_free: bool = True,
        in_channels: int = 3  # RGB images by default
    ):
        super(FastHybridDetector, self).__init__()
        
        self.num_classes = num_classes
        self.anchor_free = anchor_free
        self.backbone_name = backbone_name
        
        # Backbone
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(backbone_name, in_channels=in_channels)
        else:
            self.backbone = EfficientNet.from_name(backbone_name)
            
            # Initialize first conv layer with correct input channels
            if in_channels != 3:
                old_conv = self.backbone._conv_stem
                new_conv = nn.Conv2d(
                    in_channels, 
                    old_conv.out_channels, 
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
                
                # Initialize weights for the new conv layer
                if in_channels == 3:
                    new_conv.weight.data = old_conv.weight.data
                else:
                    # Handle different input channels
                    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                
                self.backbone._conv_stem = new_conv
        
        # Get actual backbone output channels by doing a forward pass
        self.backbone_channels = self._get_actual_backbone_channels()
        
        # Feature Pyramid Network
        self.fpn = FastFPN(self.backbone_channels, fpn_channels)
        
        # Detection heads for different scales
        self.detection_heads = nn.ModuleList([
            OptimizedDetectionHead(fpn_channels, num_classes, num_anchors)
            for _ in range(len(self.backbone_channels))
        ])
        
        # Anchor-free center and scale prediction
        if anchor_free:
            self.center_head = nn.ModuleList([
                nn.Conv2d(fpn_channels, 1, 3, padding=1)
                for _ in range(len(self.backbone_channels))
            ])
            self.scale_head = nn.ModuleList([
                nn.Conv2d(fpn_channels, 2, 3, padding=1)  # width, height
                for _ in range(len(self.backbone_channels))
            ])
        
        # Efficient post-processing
        self.max_detections = 1000
        self.nms_threshold = 0.5
        self.confidence_threshold = 0.1
        
    def _get_actual_backbone_channels(self) -> List[int]:
        """Get the actual number of channels for each backbone feature level by doing a forward pass."""
        self.backbone.eval()
        
        # Create a dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Extract features to get actual channel dimensions
        features = []
        
        with torch.no_grad():
            # Initial convolution and batch norm
            x = self.backbone._conv_stem(dummy_input)
            x = self.backbone._bn0(x)
            x = self.backbone._swish(x)
            
            # We'll collect features at specific block indices
            feature_indices = self._get_feature_indices()
            
            # Store the initial feature if needed
            if 0 in feature_indices:
                features.append(x.shape[1])  # Store channel count
            
            # Pass through all blocks
            for i, block in enumerate(self.backbone._blocks):
                x = block(x)
                if (i + 1) in feature_indices:  # +1 because blocks are 0-indexed
                    features.append(x.shape[1])  # Store channel count
        
        # Take the last 4 feature maps if we have more than 4
        if len(features) > 4:
            features = features[-4:]
        elif len(features) < 4:
            # If we have fewer than 4, duplicate the last feature
            while len(features) < 4:
                features.append(features[-1])
        
        return features
    
    def _get_feature_indices(self) -> List[int]:
        """Get the indices where we'll extract features based on backbone type."""
        if 'b0' in self.backbone_name:
            return [2, 4, 10, 15]  # For B0
        elif 'b1' in self.backbone_name:
            return [2, 5, 12, 18]  # For B1
        elif 'b2' in self.backbone_name:
            return [2, 5, 12, 18]  # For B2
        elif 'b3' in self.backbone_name:
            return [2, 5, 12, 18]  # For B3
        elif 'b4' in self.backbone_name:
            return [2, 6, 15, 22]  # For B4
        elif 'b5' in self.backbone_name:
            return [2, 7, 17, 25]  # For B5
        else:
            # Default to B3 indices
            return [2, 5, 12, 18]
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from backbone."""
        features = []
        feature_indices = self._get_feature_indices()
        
        # Initial convolution and batch norm
        x = self.backbone._conv_stem(x)
        x = self.backbone._bn0(x)
        x = self.backbone._swish(x)
        
        # Store the initial feature if needed
        if 0 in feature_indices:
            features.append(x)
        
        # Pass through all blocks
        for i, block in enumerate(self.backbone._blocks):
            x = block(x)
            if (i + 1) in feature_indices:  # +1 because blocks are 0-indexed
                features.append(x)
        
        # Take the last 4 feature maps if we have more than 4
        if len(features) > 4:
            features = features[-4:]
        elif len(features) < 4:
            # If we have fewer than 4, duplicate the last feature
            while len(features) < 4:
                features.append(features[-1])
            
        return features
    
    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        # Extract backbone features
        backbone_features = self.extract_features(x)
        
        # FPN features
        fpn_features = self.fpn(backbone_features)
        
        # Detection predictions
        all_predictions = {
            'classification': [],
            'regression': [],
            'objectness': []
        }
        
        if self.anchor_free:
            all_predictions['center'] = []
            all_predictions['scale'] = []
        
        for i, (feature, head) in enumerate(zip(fpn_features, self.detection_heads)):
            predictions = head(feature)
            
            for key in predictions:
                all_predictions[key].append(predictions[key])
            
            # Anchor-free predictions
            if self.anchor_free:
                center_pred = torch.sigmoid(self.center_head[i](feature))
                scale_pred = self.scale_head[i](feature)
                
                all_predictions['center'].append(center_pred)
                all_predictions['scale'].append(scale_pred)
        
        return all_predictions
    
    def decode_predictions(
        self, 
        predictions: Dict[str, List[torch.Tensor]], 
        input_size: Tuple[int, int]
    ) -> List[Dict]:
        """Decode predictions to bounding boxes."""
        # This would be implemented for inference
        # For training, we return raw predictions
        return predictions


def create_fast_detector(
    backbone_name: str = 'efficientnet-b3',
    num_classes: int = 8,
    in_channels: int = 3,
    **kwargs
) -> FastHybridDetector:
    """Factory function to create Fast Hybrid Detector.
    
    Args:
        backbone_name: Name of the backbone architecture
        num_classes: Number of output classes
        in_channels: Number of input channels
        **kwargs: Additional arguments to pass to FastHybridDetector
    """
    return FastHybridDetector(
        backbone_name=backbone_name,
        num_classes=num_classes,
        in_channels=in_channels,
        **kwargs
    )


# Model variants for different speed/accuracy trade-offs
def fast_detector_nano(num_classes: int = 8, in_channels: int = 3):
    """Ultra-fast detector (EfficientNet-B0)."""
    return create_fast_detector('efficientnet-b0', num_classes, fpn_channels=96, in_channels=in_channels)


def fast_detector_small(num_classes: int = 8, in_channels: int = 3):
    """Small fast detector (EfficientNet-B1)."""
    return create_fast_detector('efficientnet-b1', num_classes, fpn_channels=112, in_channels=in_channels)


def fast_detector_medium(num_classes: int = 8, in_channels: int = 3):
    """Medium detector (EfficientNet-B3)."""
    return create_fast_detector('efficientnet-b3', num_classes, fpn_channels=128, in_channels=in_channels)


def fast_detector_large(num_classes: int = 8, in_channels: int = 3):
    """Large detector for highest accuracy (EfficientNet-B5)."""
    return create_fast_detector('efficientnet-b5', num_classes, fpn_channels=160, in_channels=in_channels)


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = fast_detector_medium(num_classes=8)
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 640, 640).to(device)
    
    with torch.no_grad():
        predictions = model(x)
    
    print("Fast Hybrid Detector created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Number of FPN levels: {len(predictions['classification'])}")
    
    for i, (cls_pred, reg_pred) in enumerate(zip(predictions['classification'], predictions['regression'])):
        print(f"Level {i}: Classification {cls_pred.shape}, Regression {reg_pred.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Estimate model size
    param_size = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"Estimated model size: {param_size:.1f} MB")