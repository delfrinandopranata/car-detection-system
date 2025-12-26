"""
Custom Object Detection Model for Car Detection
Implements a YOLO-inspired architecture with custom improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


class ConvBlock(nn.Module):
    """Standard Convolution Block with BatchNorm and activation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: str = 'silu'
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BottleneckBlock(nn.Module):
    """Bottleneck block with residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(hidden_channels, out_channels, 3)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.add else out


class CSPBlock(nn.Module):
    """Cross Stage Partial block - key component of CSPDarknet"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv3 = ConvBlock(2 * hidden_channels, out_channels, 1)
        
        self.blocks = nn.Sequential(*[
            BottleneckBlock(hidden_channels, hidden_channels, shortcut, expansion=1.0)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat([y1, y2], dim=1))


class SPPFBlock(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPDarknetBackbone(nn.Module):
    """
    CSPDarknet53 Backbone - Feature Extractor
    Used as the backbone for YOLO-based detection
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0
    ):
        super().__init__()
        
        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor
        
        def make_round(x):
            return max(round(x * depth_multiple), 1)
        
        c1 = make_divisible(base_channels * width_multiple)
        c2 = make_divisible(base_channels * 2 * width_multiple)
        c3 = make_divisible(base_channels * 4 * width_multiple)
        c4 = make_divisible(base_channels * 8 * width_multiple)
        c5 = make_divisible(base_channels * 16 * width_multiple)
        
        # Stem
        self.stem = ConvBlock(in_channels, c1, 6, 2, 2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(c1, c2, 3, 2),
            CSPBlock(c2, c2, make_round(3))
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(c2, c3, 3, 2),
            CSPBlock(c3, c3, make_round(6))
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(c3, c4, 3, 2),
            CSPBlock(c4, c4, make_round(9))
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBlock(c4, c5, 3, 2),
            CSPBlock(c5, c5, make_round(3)),
            SPPFBlock(c5, c5)
        )
        
        self.out_channels = [c3, c4, c5]
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)  # P3
        c4 = self.stage3(c3)  # P4
        c5 = self.stage4(c4)  # P5
        return [c3, c4, c5]


class PANet(nn.Module):
    """
    Path Aggregation Network
    Feature Pyramid Network with bottom-up path augmentation
    """
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        c3, c4, c5 = in_channels
        
        # Top-down path
        self.lateral5 = ConvBlock(c5, out_channels, 1)
        self.lateral4 = ConvBlock(c4, out_channels, 1)
        self.lateral3 = ConvBlock(c3, out_channels, 1)
        
        self.fpn_conv5 = CSPBlock(out_channels, out_channels, 3)
        self.fpn_conv4 = CSPBlock(out_channels * 2, out_channels, 3)
        self.fpn_conv3 = CSPBlock(out_channels * 2, out_channels, 3)
        
        # Bottom-up path
        self.down_conv3 = ConvBlock(out_channels, out_channels, 3, 2)
        self.pan_conv4 = CSPBlock(out_channels * 2, out_channels, 3)
        
        self.down_conv4 = ConvBlock(out_channels, out_channels, 3, 2)
        self.pan_conv5 = CSPBlock(out_channels * 2, out_channels, 3)
        
        self.out_channels = out_channels
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        c3, c4, c5 = features
        
        # Top-down
        p5 = self.lateral5(c5)
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = self.lateral4(c4)
        p4 = self.fpn_conv4(torch.cat([p4, p5_up], dim=1))
        
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.lateral3(c3)
        p3 = self.fpn_conv3(torch.cat([p3, p4_up], dim=1))
        
        # Bottom-up
        p3_down = self.down_conv3(p3)
        p4 = self.pan_conv4(torch.cat([p3_down, p4], dim=1))
        
        p4_down = self.down_conv4(p4)
        p5 = self.pan_conv5(torch.cat([p4_down, p5], dim=1))
        
        return [p3, p4, p5]


class DetectionHead(nn.Module):
    """
    YOLO Detection Head
    Predicts bounding boxes, objectness, and class probabilities
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,  # Just 'car' for detection
        num_anchors: int = 3,
        strides: List[int] = [8, 16, 32]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.strides = strides
        self.num_outputs = 5 + num_classes  # x, y, w, h, obj, classes
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels, in_channels, 3),
                nn.Conv2d(in_channels, self.num_outputs * num_anchors, 1)
            )
            for _ in strides
        ])
        
        # Default anchors for car detection (optimized for vehicle sizes)
        self.register_buffer('anchors', torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # P3
            [[30, 61], [62, 45], [59, 119]],     # P4
            [[116, 90], [156, 198], [373, 326]]  # P5
        ]).float())
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i, (feature, head) in enumerate(zip(features, self.heads)):
            output = head(feature)
            batch_size, _, h, w = output.shape
            output = output.view(batch_size, self.num_anchors, self.num_outputs, h, w)
            output = output.permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(output)
        return outputs
    
    def decode_predictions(
        self,
        outputs: List[torch.Tensor],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """Decode raw predictions to bounding boxes"""
        batch_size = outputs[0].shape[0]
        device = outputs[0].device
        
        all_predictions = []
        
        for batch_idx in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []
            
            for scale_idx, output in enumerate(outputs):
                pred = output[batch_idx]  # [num_anchors, h, w, num_outputs]
                h, w = pred.shape[1:3]
                stride = self.strides[scale_idx]
                anchors = self.anchors[scale_idx]
                
                # Create grid
                yv, xv = torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing='ij'
                )
                grid = torch.stack([xv, yv], dim=-1).float()
                
                # Decode predictions
                pred_xy = (pred[..., :2].sigmoid() * 2 - 0.5 + grid) * stride
                pred_wh = (pred[..., 2:4].sigmoid() * 2) ** 2 * anchors.view(1, 1, 1, -1, 2).permute(0, 3, 1, 2, 4)
                pred_wh = pred_wh.squeeze(0)
                
                pred_conf = pred[..., 4].sigmoid()
                pred_cls = pred[..., 5:].sigmoid()
                
                # Filter by confidence
                conf_mask = pred_conf > conf_threshold
                
                if conf_mask.sum() > 0:
                    pred_xy = pred_xy[conf_mask]
                    pred_wh = pred_wh[conf_mask]
                    pred_conf = pred_conf[conf_mask]
                    pred_cls = pred_cls[conf_mask]
                    
                    # Convert to x1y1x2y2
                    boxes = torch.cat([
                        pred_xy - pred_wh / 2,
                        pred_xy + pred_wh / 2
                    ], dim=-1)
                    
                    scores = pred_conf * pred_cls.max(dim=-1)[0]
                    labels = pred_cls.argmax(dim=-1)
                    
                    boxes_list.append(boxes)
                    scores_list.append(scores)
                    labels_list.append(labels)
            
            if boxes_list:
                boxes = torch.cat(boxes_list, dim=0)
                scores = torch.cat(scores_list, dim=0)
                labels = torch.cat(labels_list, dim=0)
                
                # NMS
                from torchvision.ops import nms
                keep = nms(boxes, scores, iou_threshold)
                
                all_predictions.append({
                    'boxes': boxes[keep],
                    'scores': scores[keep],
                    'labels': labels[keep]
                })
            else:
                all_predictions.append({
                    'boxes': torch.empty(0, 4, device=device),
                    'scores': torch.empty(0, device=device),
                    'labels': torch.empty(0, dtype=torch.long, device=device)
                })
        
        return all_predictions


class CarDetectionModel(nn.Module):
    """
    Complete Car Detection Model
    CSPDarknet Backbone + PANet Neck + YOLO Head
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        backbone_channels: int = 64,
        neck_channels: int = 256,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.5,
        pretrained_backbone: bool = True
    ):
        super().__init__()
        
        self.backbone = CSPDarknetBackbone(
            base_channels=backbone_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple
        )
        
        self.neck = PANet(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels
        )
        
        self.head = DetectionHead(
            in_channels=neck_channels,
            num_classes=num_classes
        )
        
        self.num_classes = num_classes
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)
        
        if self.training and targets is not None:
            loss = self.compute_loss(outputs, targets)
            return outputs, loss
        
        return outputs, None
    
    def compute_loss(
        self,
        outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute YOLO loss
        targets: [batch_idx, class, x_center, y_center, width, height]
        """
        device = outputs[0].device
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        bce_obj = nn.BCEWithLogitsLoss(reduction='mean')
        bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
        
        for scale_idx, output in enumerate(outputs):
            batch_size, num_anchors, h, w, _ = output.shape
            stride = self.head.strides[scale_idx]
            
            # Build targets for this scale
            obj_target = torch.zeros(batch_size, num_anchors, h, w, device=device)
            
            if targets is not None and len(targets) > 0:
                # Simplified target assignment
                for target in targets:
                    bi = int(target[0])
                    cx, cy = target[2] * w, target[3] * h
                    gi, gj = int(cx), int(cy)
                    
                    if 0 <= gi < w and 0 <= gj < h:
                        obj_target[bi, :, gj, gi] = 1.0
            
            # Objectness loss
            obj_loss += bce_obj(output[..., 4], obj_target)
        
        total_loss = box_loss + obj_loss + cls_loss
        return total_loss
    
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """Run inference and return decoded predictions"""
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(x)
            predictions = self.head.decode_predictions(
                outputs, conf_threshold, iou_threshold
            )
        return predictions


def create_car_detector(
    model_size: str = 'small',
    num_classes: int = 1,
    pretrained: bool = True
) -> CarDetectionModel:
    """
    Factory function to create car detection model
    
    Args:
        model_size: 'nano', 'small', 'medium', 'large', 'xlarge'
        num_classes: Number of classes (1 for car-only detection)
        pretrained: Whether to use pretrained backbone
    """
    configs = {
        'nano': {'depth': 0.33, 'width': 0.25, 'neck': 128},
        'small': {'depth': 0.33, 'width': 0.5, 'neck': 256},
        'medium': {'depth': 0.67, 'width': 0.75, 'neck': 384},
        'large': {'depth': 1.0, 'width': 1.0, 'neck': 512},
        'xlarge': {'depth': 1.33, 'width': 1.25, 'neck': 640}
    }
    
    cfg = configs.get(model_size, configs['small'])
    
    return CarDetectionModel(
        num_classes=num_classes,
        neck_channels=cfg['neck'],
        depth_multiple=cfg['depth'],
        width_multiple=cfg['width'],
        pretrained_backbone=pretrained
    )


if __name__ == '__main__':
    # Test the model
    model = create_car_detector('small')
    x = torch.randn(2, 3, 640, 640)
    outputs, loss = model(x)
    
    print("Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
