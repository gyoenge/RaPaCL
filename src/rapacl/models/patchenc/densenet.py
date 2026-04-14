import torch
import torch.nn as nn
from torchvision import models


class PatchDenseNetEncoder(nn.Module):
    """DenseNet121-based patch image encoder.

    Outputs a global feature vector per patch.

    Example:
        input:  [B, 3, H, W]
        output: [B, 1024]
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_dim: int = 1024,
        freeze_backbone: bool = False,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        # -------------------------------------------------
        # 1. Load DenseNet121 backbone
        # -------------------------------------------------
        if pretrained:
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            backbone = models.densenet121(weights=None)

        # DenseNet 구조:
        # backbone.features -> convolutional feature extractor
        # backbone.classifier -> final FC (삭제할 것)
        self.features = backbone.features

        # -------------------------------------------------
        # 2. Global pooling
        # -------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # DenseNet121 feature dim = 1024
        self.backbone_dim = backbone.classifier.in_features  # usually 1024

        # -------------------------------------------------
        # 3. Optional projection to out_dim
        # -------------------------------------------------
        if out_dim != self.backbone_dim:
            self.proj = nn.Linear(self.backbone_dim, out_dim)
        else:
            self.proj = nn.Identity()

        # optional normalization
        self.norm = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()

        # -------------------------------------------------
        # 4. Freeze backbone if needed
        # -------------------------------------------------
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            feat: [B, out_dim]
        """

        # DenseNet feature extraction
        feat = self.features(x)                 # [B, 1024, H', W']

        # ReLU (DenseNet 공식 forward에도 포함됨)
        feat = torch.relu(feat)

        # Global average pooling
        feat = self.pool(feat)                  # [B, 1024, 1, 1]

        # Flatten
        feat = feat.view(feat.size(0), -1)      # [B, 1024]

        # Projection (optional)
        feat = self.proj(feat)                  # [B, out_dim]

        # Normalization (optional)
        feat = self.norm(feat)

        return feat
