"""
Car Type Classifier
Multiple architectures: CNN-based (ResNet, EfficientNet) and Transformer-based (ViT, Swin)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


# ============================================================================
# Custom CNN Blocks for Building from Scratch
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    """Basic ResNet Block"""
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """Bottleneck ResNet Block"""
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = False
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = SEBlock(out_channels * self.expansion) if use_se else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class CustomResNet(nn.Module):
    """
    Custom ResNet Implementation for Car Type Classification
    Built from scratch with optional SE blocks
    """
    
    def __init__(
        self,
        block_type: str = 'bottleneck',
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 7,
        use_se: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        Block = BottleneckBlock if block_type == 'bottleneck' else BasicBlock
        self.expansion = Block.expansion
        self.in_channels = 64
        self.use_se = use_se
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Stages
        self.layer1 = self._make_layer(Block, 64, layers[0])
        self.layer2 = self._make_layer(Block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Block, 512, layers[3], stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = [block(self.in_channels, out_channels, stride, downsample, self.use_se)]
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, use_se=self.use_se))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ============================================================================
# Vision Transformer (ViT) - Attention-based Architecture
# ============================================================================

class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP Block for Transformer"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for Car Type Classification
    Attention-based architecture built from scratch
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 7,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use CLS token
        x = self.head(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head"""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]


# ============================================================================
# Wrapper for Pretrained Models
# ============================================================================

class PretrainedClassifier(nn.Module):
    """
    Wrapper for using pretrained models from timm
    Supports both CNN and Transformer architectures
    """
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.model_name = model_name
        
        try:
            import timm
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # Remove classifier
            )
            self.feature_dim = self.backbone.num_features
        except ImportError:
            print("timm not available, using custom model")
            self.backbone = CustomResNet(num_classes=0)
            self.feature_dim = 2048
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ============================================================================
# Ensemble Classifier
# ============================================================================

class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple classifiers for improved accuracy
    Combines CNN and Transformer predictions
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        num_classes: int = 7,
        ensemble_method: str = 'average'  # 'average', 'weighted', 'learned'
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        elif ensemble_method == 'learned':
            total_features = sum(m.feature_dim if hasattr(m, 'feature_dim') else 512 for m in models)
            self.meta_classifier = nn.Sequential(
                nn.Linear(total_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ensemble_method == 'average':
            outputs = [F.softmax(model(x), dim=1) for model in self.models]
            return torch.stack(outputs).mean(dim=0)
        
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            outputs = [F.softmax(model(x), dim=1) for model in self.models]
            weighted = sum(w * o for w, o in zip(weights, outputs))
            return weighted
        
        elif self.ensemble_method == 'learned':
            features = [model.get_features(x) for model in self.models]
            combined = torch.cat(features, dim=1)
            return self.meta_classifier(combined)


# ============================================================================
# Car Type Classifier with Multiple Architectures
# ============================================================================

CAR_TYPES = ['sedan', 'suv', 'mpv', 'hatchback', 'pickup', 'minivan', 'crossover']


class CarTypeClassifier:
    """
    Factory class to create different classifier architectures
    """
    
    @staticmethod
    def create_resnet(
        variant: str = 'resnet50',
        num_classes: int = 7,
        pretrained: bool = True
    ) -> nn.Module:
        """Create ResNet-based classifier"""
        layers_config = {
            'resnet18': [2, 2, 2, 2],
            'resnet34': [3, 4, 6, 3],
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3]
        }
        
        block_type = 'bottleneck' if 'resnet50' in variant or int(variant[-3:] if variant[-3:].isdigit() else variant[-2:]) >= 50 else 'basic'
        layers = layers_config.get(variant, [3, 4, 6, 3])
        
        if pretrained:
            return PretrainedClassifier(variant, num_classes, pretrained=True)
        else:
            return CustomResNet(block_type, layers, num_classes)
    
    @staticmethod
    def create_efficientnet(
        variant: str = 'efficientnet_b0',
        num_classes: int = 7,
        pretrained: bool = True
    ) -> nn.Module:
        """Create EfficientNet-based classifier"""
        return PretrainedClassifier(variant, num_classes, pretrained)
    
    @staticmethod
    def create_vit(
        variant: str = 'vit_small',
        num_classes: int = 7,
        pretrained: bool = True,
        img_size: int = 224
    ) -> nn.Module:
        """Create Vision Transformer classifier"""
        configs = {
            'vit_tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
            'vit_small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
            'vit_base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
            'vit_large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16}
        }
        
        if pretrained:
            timm_name = f'{variant}_patch16_{img_size}'
            return PretrainedClassifier(timm_name, num_classes, pretrained)
        else:
            cfg = configs.get(variant, configs['vit_small'])
            return VisionTransformer(
                img_size=img_size,
                num_classes=num_classes,
                **cfg
            )
    
    @staticmethod
    def create_swin(
        variant: str = 'swin_tiny',
        num_classes: int = 7,
        pretrained: bool = True
    ) -> nn.Module:
        """Create Swin Transformer classifier"""
        timm_names = {
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_small': 'swin_small_patch4_window7_224',
            'swin_base': 'swin_base_patch4_window7_224'
        }
        return PretrainedClassifier(
            timm_names.get(variant, timm_names['swin_tiny']),
            num_classes,
            pretrained
        )
    
    @staticmethod
    def create_ensemble(
        model_configs: List[Dict],
        num_classes: int = 7
    ) -> EnsembleClassifier:
        """Create ensemble of multiple classifiers"""
        models = []
        for cfg in model_configs:
            model_type = cfg.get('type', 'resnet')
            variant = cfg.get('variant', 'resnet50')
            pretrained = cfg.get('pretrained', True)
            
            if model_type == 'resnet':
                model = CarTypeClassifier.create_resnet(variant, num_classes, pretrained)
            elif model_type == 'efficientnet':
                model = CarTypeClassifier.create_efficientnet(variant, num_classes, pretrained)
            elif model_type == 'vit':
                model = CarTypeClassifier.create_vit(variant, num_classes, pretrained)
            elif model_type == 'swin':
                model = CarTypeClassifier.create_swin(variant, num_classes, pretrained)
            
            models.append(model)
        
        return EnsembleClassifier(models, num_classes)


def get_classifier(
    architecture: str,
    num_classes: int = 7,
    pretrained: bool = True
) -> nn.Module:
    """
    Main function to get a classifier
    
    Args:
        architecture: One of 'resnet50', 'efficientnet_b0', 'vit_base', 'swin_tiny'
        num_classes: Number of car types
        pretrained: Use pretrained weights
    """
    if 'resnet' in architecture.lower():
        return CarTypeClassifier.create_resnet(architecture, num_classes, pretrained)
    elif 'efficientnet' in architecture.lower():
        return CarTypeClassifier.create_efficientnet(architecture, num_classes, pretrained)
    elif 'vit' in architecture.lower():
        return CarTypeClassifier.create_vit(architecture, num_classes, pretrained)
    elif 'swin' in architecture.lower():
        return CarTypeClassifier.create_swin(architecture, num_classes, pretrained)
    else:
        # Default to ResNet50
        return CarTypeClassifier.create_resnet('resnet50', num_classes, pretrained)


if __name__ == '__main__':
    # Test different architectures
    print("Testing Car Type Classifiers...")
    
    x = torch.randn(2, 3, 224, 224)
    
    # Test Custom ResNet
    print("\n1. Custom ResNet50 (from scratch):")
    model = CustomResNet(num_classes=7)
    out = model(x)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Output shape: {out.shape}")
    
    # Test Custom ViT
    print("\n2. Custom Vision Transformer (from scratch):")
    model = VisionTransformer(num_classes=7, embed_dim=384, depth=6, num_heads=6)
    out = model(x)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Output shape: {out.shape}")
    
    print("\nAll models tested successfully!")
