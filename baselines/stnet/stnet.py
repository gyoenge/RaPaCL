from __future__ import annotations

import torch
import torch.nn as nn

from src.baselines.stnet.densenet import build_densenet121_backbone


class STNet(nn.Module):
    def __init__(
        self,
        num_genes: int = 250,
        pretrained: bool = True,
        backbone_name: str = "densenet121",
    ):
        super().__init__()

        if backbone_name != "densenet121":
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = build_densenet121_backbone(pretrained=pretrained)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)          # [B, 1024, H, W]
        x = self.pool(x)              # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)       # [B, 1024]
        x = self.classifier(x)        # [B, num_genes]
        return x


def build_model(model_cfg: dict) -> STNet:
    return STNet(
        num_genes=model_cfg.get("num_genes", 250),
        pretrained=model_cfg.get("pretrained", True),
        backbone_name=model_cfg.get("backbone", "densenet121"),
    )
