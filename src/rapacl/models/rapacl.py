import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from src.rapacl.models.patchenc.conv import PatchImageConvEncoder
from src.rapacl.models.transtab.radiomics import TransTabRadiomicsEncoder
from src.rapacl.models.heads.projection import ProjectionHead
from src.rapacl.models.heads.classifier import LinearClassifier
from src.rapacl.models.heads.batch_discriminator import AdversarialDiscriminator


class RaPaCL(nn.Module):
    """RaPaCL: RadiomicsFeature-Pathomics Contrastive Learning.

    This module manages the full multimodal model:
      1) patch/pathomics encoder
      2) radiomics/tabular encoder
      3) contrastive projection heads
      4) classification heads
      5) optional batch-correction discriminators

    Notes
    -----
    - This class does NOT compute losses directly.
    - It only returns all outputs required by criterion/loss modules.
    """

    def __init__(
        self,
        # -------------------------------------------------
        # patch encoder
        # -------------------------------------------------
        patch_backbone_name: str = "densenet121",
        patch_pretrained: bool = True,
        patch_feat_dim: int = 1024,
        patch_freeze_backbone: bool = False,
        patch_use_bn: bool = False,

        # -------------------------------------------------
        # radiomics encoder
        # -------------------------------------------------
        radiomics_hidden_dim: int = 128,
        separate_contrast_token: bool = True,

        # -------------------------------------------------
        # shared contrastive projection
        # -------------------------------------------------
        proj_dim: int = 128,
        projection_use_mlp: bool = False,
        projection_hidden_dim: Optional[int] = None,
        projection_dropout: float = 0.0,

        # -------------------------------------------------
        # classification
        # -------------------------------------------------
        num_classes: int = 6,
        classifier_dropout: float = 0.0,

        # -------------------------------------------------
        # batch correction
        # -------------------------------------------------
        use_batch_correction: bool = True,
        num_batch_labels: Optional[int] = None,
        batch_disc_hidden_dim: Optional[int] = None,
        batch_disc_nlayers: int = 2,
        batch_disc_dropout: float = 0.1,
        batch_disc_grl_lambda: float = 1.0,

        # -------------------------------------------------
        # radiomics encoder kwargs
        # -------------------------------------------------
        **radiomics_kwargs,
    ) -> None:
        super().__init__()

        # -------------------------------------------------
        # 1. Patch encoder
        # -------------------------------------------------
        self.patch_encoder = PatchImageConvEncoder(
            backbone_name=patch_backbone_name,
            pretrained=patch_pretrained,
            out_dim=patch_feat_dim,
            freeze_backbone=patch_freeze_backbone,
            use_bn=patch_use_bn,
        )

        # -------------------------------------------------
        # 2. Radiomics encoder
        # -------------------------------------------------
        self.radiomics_encoder = TransTabRadiomicsEncoder(
            separate_contrast_token=separate_contrast_token,
            hidden_dim=radiomics_hidden_dim,
            **radiomics_kwargs,
        )

        # -------------------------------------------------
        # 3. Projection heads for multimodal contrastive learning
        # -------------------------------------------------
        self.patch_projection = ProjectionHead(
            in_dim=patch_feat_dim,
            out_dim=proj_dim,
            hidden_dim=projection_hidden_dim,
            use_mlp=projection_use_mlp,
            dropout=projection_dropout,
        )

        self.radiomics_projection = ProjectionHead(
            in_dim=radiomics_hidden_dim,
            out_dim=proj_dim,
            hidden_dim=projection_hidden_dim,
            use_mlp=projection_use_mlp,
            dropout=projection_dropout,
        )

        # -------------------------------------------------
        # 4. Classification heads
        # -------------------------------------------------
        self.patch_classifier = LinearClassifier(
            in_dim=patch_feat_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

        self.radiomics_classifier = LinearClassifier(
            in_dim=radiomics_hidden_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

        # -------------------------------------------------
        # 5. Optional batch correction discriminators
        # -------------------------------------------------
        self.use_batch_correction = use_batch_correction

        if self.use_batch_correction:
            if num_batch_labels is None:
                raise ValueError(
                    "`num_batch_labels` must be provided when `use_batch_correction=True`."
                )

            self.patch_batch_discriminator = AdversarialDiscriminator(
                d_model=patch_feat_dim,
                n_cls=num_batch_labels,
                hidden_dim=batch_disc_hidden_dim,
                nlayers=batch_disc_nlayers,
                dropout=batch_disc_dropout,
                reverse_grad=True,
                grl_lambda=batch_disc_grl_lambda,
            )

            self.radiomics_batch_discriminator = AdversarialDiscriminator(
                d_model=radiomics_hidden_dim,
                n_cls=num_batch_labels,
                hidden_dim=batch_disc_hidden_dim,
                nlayers=batch_disc_nlayers,
                dropout=batch_disc_dropout,
                reverse_grad=True,
                grl_lambda=batch_disc_grl_lambda,
            )
        else:
            self.patch_batch_discriminator = None
            self.radiomics_batch_discriminator = None

        # -------------------------------------------------
        # metadata / config-ish attrs
        # -------------------------------------------------
        self.patch_backbone_name = patch_backbone_name
        self.patch_feat_dim = patch_feat_dim
        self.radiomics_hidden_dim = radiomics_hidden_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.num_batch_labels = num_batch_labels

    def forward(
        self,
        patch_x: torch.Tensor,
        radiomics_x: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        patch_x : torch.Tensor
            Patch images. Expected shape depends on patch backbone,
            typically [B, C, H, W].

        radiomics_x : Any
            Radiomics/tabular input, typically pd.DataFrame for
            TransTabRadiomicsEncoder.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing all outputs needed for training/inference.
        """

        # -------------------------------------------------
        # 1. Patch/pathomics branch
        # -------------------------------------------------
        patch_feat = self.patch_encoder(patch_x)           # [B, patch_feat_dim]
        patch_proj = self.patch_projection(patch_feat)     # [B, proj_dim]
        patch_logits = self.patch_classifier(patch_feat)   # [B, C] or [B, 1]

        # -------------------------------------------------
        # 2. Radiomics branch
        # -------------------------------------------------
        rad_out = self.radiomics_encoder(radiomics_x)

        # expected keys from TransTabRadiomicsEncoder:
        # - cls_embedding: [B, H]
        # - contrastive_embedding: [B, H] or None
        # - multiview_embeddings: [B, V, H]
        rad_cls_feat = rad_out["cls_embedding"]                    # [B, H]
        rad_ctr_feat = rad_out.get("contrastive_embedding", None)  # [B, H] or None
        rad_multiview = rad_out["multiview_embeddings"]            # [B, V, H]

        radiomics_logits = self.radiomics_classifier(rad_cls_feat)  # [B, C] or [B, 1]

        # project multi-view radiomics embeddings for multimodal contrastive learning
        if rad_multiview is not None and rad_multiview.dim() == 3:
            rad_multiview_proj = self.radiomics_projection(rad_multiview)  # [B, V, P]
        else:
            rad_multiview_proj = None

        # optional single contrastive embedding projection
        if rad_ctr_feat is not None:
            radiomics_ctr_proj = self.radiomics_projection(rad_ctr_feat)    # [B, P]
        else:
            radiomics_ctr_proj = None

        # -------------------------------------------------
        # 3. Collect outputs
        # -------------------------------------------------
        outputs: Dict[str, torch.Tensor] = {
            # patch branch
            "patch_feat": patch_feat,
            "patch_proj": patch_proj,
            "patch_logits": patch_logits,

            # radiomics branch
            "radiomics_cls_feat": rad_cls_feat,
            "radiomics_ctr_feat": rad_ctr_feat,
            "radiomics_ctr_proj": radiomics_ctr_proj,
            "radiomics_multiview_feat": rad_multiview,
            "radiomics_multiview_proj": rad_multiview_proj,
            "radiomics_logits": radiomics_logits,
        }

        # -------------------------------------------------
        # 4. Optional batch correction
        # -------------------------------------------------
        if self.use_batch_correction:
            outputs["patch_batch_logits"] = self.patch_batch_discriminator(patch_feat)
            outputs["radiomics_batch_logits"] = self.radiomics_batch_discriminator(rad_cls_feat)

        return outputs
