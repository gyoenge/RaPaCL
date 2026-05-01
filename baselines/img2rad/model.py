from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class PatchImgEncoder(nn.Module):
    def __init__(self, backbone_weight_path: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        self.backbone = models.densenet121(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 1024

        if backbone_weight_path is not None:
            if not os.path.exists(backbone_weight_path):
                raise FileNotFoundError(
                    f"Backbone checkpoint not found: {backbone_weight_path}"
                )
            state_dict = torch.load(backbone_weight_path, map_location=device)
            self.backbone.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class ImgToRadiomicsModel(nn.Module):
    def __init__(
        self,
        radiomics_dim: int,
        backbone_weight_path: Optional[str] = None,
        device: str = "cpu",
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_encoder = PatchImgEncoder(backbone_weight_path, device)

        h1, h2 = hidden_dims
        self.rad_hidden_dim = h2
        self.radiomics_dim = radiomics_dim

        self.rad_fc1 = nn.Linear(self.img_encoder.output_dim, h1)
        self.rad_fc2 = nn.Linear(h1, h2)
        self.rad_fc3 = nn.Linear(h2, radiomics_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, rad_pred = self.forward_with_feature(x)
        return rad_pred

    def forward_with_feature(self, x: torch.Tensor):
        img_emb = self.img_encoder(x)

        h1 = self.rad_fc1(img_emb)
        h1 = self.relu(h1)
        h1 = self.dropout(h1)

        rad_hidden = self.rad_fc2(h1)
        rad_hidden = self.relu(rad_hidden)

        rad_pred = self.rad_fc3(rad_hidden)
        return img_emb, rad_hidden, rad_pred
    

class FusionGeneModel(nn.Module):
    def __init__(
        self,
        pretrained_img2rad_model: ImgToRadiomicsModel,
        num_genes: int,
        radiomics_dim: int,
        fusion_mode: str = "img_radpred",  # ["img_radpred", "img_radhidden", "img_rawrad"]
        freeze_img2rad: bool = False,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img2rad = pretrained_img2rad_model
        self.fusion_mode = fusion_mode

        if freeze_img2rad:
            for p in self.img2rad.parameters():
                p.requires_grad = False

        if fusion_mode == "img_radpred":
            fusion_dim = self.img2rad.img_encoder.output_dim + radiomics_dim
        elif fusion_mode == "img_radhidden":
            fusion_dim = self.img2rad.img_encoder.output_dim + self.img2rad.rad_hidden_dim
        elif fusion_mode == "img_rawrad":
            fusion_dim = self.img2rad.img_encoder.output_dim + radiomics_dim
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        h1, h2 = hidden_dims
        self.gene_head = nn.Sequential(
            nn.Linear(fusion_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_genes),
        )

    def forward(
        self,
        x: torch.Tensor,
        raw_radiomics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_emb, rad_hidden, rad_pred = self.img2rad.forward_with_feature(x)

        if self.fusion_mode == "img_radpred":
            fused = torch.cat([img_emb, rad_pred], dim=1)

        elif self.fusion_mode == "img_radhidden":
            fused = torch.cat([img_emb, rad_hidden], dim=1)

        elif self.fusion_mode == "img_rawrad":
            if raw_radiomics is None:
                raise ValueError("raw_radiomics must be provided for fusion_mode='img_rawrad'")
            fused = torch.cat([img_emb, raw_radiomics], dim=1)

        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        gene_pred = self.gene_head(fused)
        return gene_pred