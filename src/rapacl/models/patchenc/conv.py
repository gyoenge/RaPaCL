import torch
import torch.nn as nn
from torchvision import models


class PatchImageConvEncoder(nn.Module):
    """Unified patch encoder supporting multiple CNN backbones.

    Supported:
        - densenet121
        - resnet18
        - resnet34
        - resnet50
        - resnet101
    """

    def __init__(
        self,
        backbone_name: str = "densenet121",
        pretrained: bool = True,
        out_dim: int = 1024,
        freeze_backbone: bool = False,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "densenet121":
            backbone = models.densenet121(
                weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
            )
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            backbone_dim = backbone.classifier.in_features  # 1024
            self._forward_backbone = self._forward_densenet

        elif self.backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            backbone_dim = backbone.fc.in_features
            self._forward_backbone = self._forward_resnet

        elif self.backbone_name == "resnet34":
            backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            backbone_dim = backbone.fc.in_features
            self._forward_backbone = self._forward_resnet

        elif self.backbone_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            backbone_dim = backbone.fc.in_features
            self._forward_backbone = self._forward_resnet

        elif self.backbone_name == "resnet101":
            backbone = models.resnet101(
                weights=models.ResNet101_Weights.DEFAULT if pretrained else None
            )
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            backbone_dim = backbone.fc.in_features
            self._forward_backbone = self._forward_resnet

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone_dim = backbone_dim

        if out_dim != backbone_dim:
            self.proj = nn.Linear(backbone_dim, out_dim)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.out_dim = out_dim

    def _forward_densenet(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.relu(feat)
        return feat

    def _forward_resnet(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._forward_backbone(x)   # [B, C, H', W']
        feat = self.pool(feat)             # [B, C, 1, 1]
        feat = feat.flatten(1)             # [B, C]
        feat = self.proj(feat)             # [B, out_dim]
        feat = self.norm(feat)
        return feat
