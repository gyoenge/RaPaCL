import torch
import torch.nn as nn

from src.rapacl.models.patchenc.densenet import PatchDenseNetEncoder
from src.rapacl.models.transtab.radiomics import TransTabRadiomicsEncoder
from src.rapacl.models.heads.projection import ProjectionHead
from src.rapacl.models.heads.classifier import LinearClassifier
from src.rapacl.models.heads.batch_discriminator import AdversarialDiscriminator


class RaPaCL(nn.Module):
    def __init__(
        self,
        patch_pretrained=True,
        patch_feat_dim=1024,
        radiomics_hidden_dim=128,
        proj_dim=128,
        num_classes=6,
        num_batch_labels=None,
        use_batch_correction=True,
        separate_contrast_token=True,
        **radiomics_kwargs,
    ):
        super().__init__()

        self.patch_encoder = PatchDenseNetEncoder(
            pretrained=patch_pretrained,
            out_dim=patch_feat_dim,
        )

        self.radiomics_encoder = TransTabRadiomicsEncoder(
            separate_contrast_token=separate_contrast_token,
            **radiomics_kwargs,
        )

        self.patch_projection = ProjectionHead(
            in_dim=patch_feat_dim,
            out_dim=proj_dim,
        )
        self.radiomics_projection = ProjectionHead(
            in_dim=radiomics_hidden_dim,
            out_dim=proj_dim,
        )

        self.patch_classifier = LinearClassifier(
            in_dim=patch_feat_dim,
            num_classes=num_classes,
        )
        self.radiomics_classifier = LinearClassifier(
            in_dim=radiomics_hidden_dim,
            num_classes=num_classes,
        )

        self.use_batch_correction = use_batch_correction
        if use_batch_correction:
            self.patch_batch_discriminator = AdversarialDiscriminator(
                d_model=patch_feat_dim,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )
            self.radiomics_batch_discriminator = AdversarialDiscriminator(
                d_model=radiomics_hidden_dim,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )
        else:
            self.patch_batch_discriminator = None
            self.radiomics_batch_discriminator = None

    def forward(self, patch_x, radiomics_x):
        patch_feat = self.patch_encoder(patch_x)  # [B, patch_feat_dim]
        rad_out = self.radiomics_encoder(radiomics_x)

        rad_cls_feat = rad_out["cls_embedding"]                    # [B, H]
        rad_multiview = rad_out["multiview_embeddings"]            # [B, V, H] or [B, V, P]
        rad_ctr_feat = rad_out.get("contrastive_embedding", None)

        patch_proj = self.patch_projection(patch_feat)

        if rad_multiview is not None and rad_multiview.dim() == 3:
            rad_multiview_proj = self.radiomics_projection(rad_multiview)
        else:
            rad_multiview_proj = None

        patch_logits = self.patch_classifier(patch_feat)
        radiomics_logits = self.radiomics_classifier(rad_cls_feat)

        outputs = {
            "patch_feat": patch_feat,
            "patch_proj": patch_proj,
            "patch_logits": patch_logits,
            "radiomics_cls_feat": rad_cls_feat,
            "radiomics_multiview_proj": rad_multiview_proj,
            "radiomics_logits": radiomics_logits,
        }

        if self.use_batch_correction:
            outputs["patch_batch_logits"] = self.patch_batch_discriminator(patch_feat)
            outputs["radiomics_batch_logits"] = self.radiomics_batch_discriminator(rad_cls_feat)

        return outputs
