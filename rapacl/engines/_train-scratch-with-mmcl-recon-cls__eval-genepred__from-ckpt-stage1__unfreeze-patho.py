
# stage2
# train: pathomics_encoder + pathomics_proj + gene_head
# freeze: radiomics_model + recon_head + cls_head
# 
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from rapacl.data._dataset import HestRadiomicsDataset
from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES
from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import set_seed
import rapacl.engines.constants as constants


WARMUP_RECON_EPOCHS = 5
MMCL_LAMBDA = 0.3
RECON_LAMBDA = 1.0
CLS_LAMBDA = 0.5


# =========================================================
# Heads
# =========================================================
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseNet121PathomicsEncoder(nn.Module):
    """Frozen DenseNet121 encoder. Returns pathomics CLS-like global image embedding."""

    def __init__(self, out_dim: int = 1024, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
        self.features = backbone.features
        self.out_dim = 1024
        self.proj = nn.Identity() if out_dim == self.out_dim else nn.Linear(self.out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expected: [B, 3, H, W]. If grayscale, repeat to 3 channels.
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.features(x)
        feat = F.relu(feat, inplace=True)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return self.proj(feat)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


# =========================================================
# Main model wrapper
# =========================================================
class MMCLReconClsModel(nn.Module):
    """
    Train stage:
      - pathomics CLS <-> radiomics contrastive token: MMCL loss
      - radiomics contrastive token -> radiomics reconstruction
      - radiomics CLS token -> cell-type classification

    Eval stage:
      - freeze pathomics encoder + trained radiomics TransTab
      - train gene head using concat(pathomics CLS, radiomics contrastive)
    """

    def __init__(
        self,
        radiomics_model: nn.Module,
        pathomics_encoder: nn.Module,
        pathomics_proj: nn.Module,
        recon_head: nn.Module,
        cls_head: nn.Module,
        gene_head: nn.Module,
        feature_cols: list[str],
    ):
        super().__init__()
        self.radiomics_model = radiomics_model
        self.pathomics_encoder = pathomics_encoder
        self.pathomics_proj = pathomics_proj
        self.recon_head = recon_head
        self.cls_head = cls_head
        self.gene_head = gene_head
        self.feature_cols = feature_cols

    def _to_dataframe(self, radiomics: torch.Tensor | pd.DataFrame) -> pd.DataFrame:
        if isinstance(radiomics, pd.DataFrame):
            return radiomics
        return pd.DataFrame(radiomics.detach().cpu().numpy(), columns=self.feature_cols)

    def encode_radiomics(self, radiomics: torch.Tensor | pd.DataFrame) -> dict[str, torch.Tensor]:
        x_df = self._to_dataframe(radiomics)
        feat = self.radiomics_model.input_encoder(x_df)
        feat = self.radiomics_model.contrastive_token(**feat)  # appended after feature tokens
        feat = self.radiomics_model.cls_token(**feat)          # prepended before all tokens
        enc = self.radiomics_model.encoder(**feat)

        # Final sequence: [CLS, original feature tokens..., CONTRASTIVE]
        # In your current implementation, cls_token is applied after contrastive_token.
        # Therefore CLS = index 0, Contrastive = index -1.
        rad_cls_h = enc[:, 0, :]
        rad_contrast_h = enc[:, 1, :] ######################## 
        rad_contrast_z = self.radiomics_model.projection_head(rad_contrast_h)

        return {
            "rad_cls_h": rad_cls_h,
            "rad_contrast_h": rad_contrast_h,
            "rad_contrast_z": rad_contrast_z,
        }

    @torch.no_grad()
    def encode_pathomics_frozen(self, image: torch.Tensor) -> torch.Tensor:
        self.pathomics_encoder.eval()
        return self.pathomics_encoder(image)

    def encode_pathomics_projected(
        self,
        image: torch.Tensor,
        freeze_encoder: bool = True,
    ) -> dict[str, torch.Tensor]:
        if freeze_encoder:
            with torch.no_grad():
                path_cls = self.encode_pathomics_frozen(image)
        else:
            path_cls = self.pathomics_encoder(image)

        path_z = self.pathomics_proj(path_cls)
        return {"path_cls": path_cls, "path_z": path_z}

    def forward_pretrain(self, image: torch.Tensor, radiomics: torch.Tensor | pd.DataFrame):
        rad = self.encode_radiomics(radiomics)
        path = self.encode_pathomics_projected(image, freeze_encoder=True)
        pred_radiomics = self.recon_head(rad["rad_contrast_z"])
        pred_class_logits = self.cls_head(rad["rad_cls_h"])
        return {**rad, **path, "pred_radiomics": pred_radiomics, "pred_class_logits": pred_class_logits}

    def forward_gene(self, image: torch.Tensor, radiomics: torch.Tensor | pd.DataFrame):
        rad = self.encode_radiomics(radiomics)
        path = self.encode_pathomics_projected(image, freeze_encoder=False)
        fused = torch.cat([path["path_cls"], rad["rad_contrast_z"]], dim=1)
        pred_gene = self.gene_head(fused)
        return {**rad, **path, "fused": fused, "pred_gene": pred_gene}


# =========================================================
# Loss / Metrics
# =========================================================
def symmetric_info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def compute_genewise_pcc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    denom = torch.sqrt((pred_c ** 2).sum(dim=0) * (target_c ** 2).sum(dim=0)) + eps
    pcc_per_gene = (pred_c * target_c).sum(dim=0) / denom
    return pcc_per_gene.mean().item(), pcc_per_gene.numpy()


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == target).float().mean().item()


# =========================================================
# Data helpers
# =========================================================
def build_dataset(split_csv_path: str):
    return HestRadiomicsDataset(
        bench_data_root=constants.ROOT_DIR,
        split_csv_path=split_csv_path,
        gene_list_path=constants.GENE_LIST_PATH,
        feature_list_path=constants.FEATURE_LIST_PATH,
        radiomics_dir=getattr(constants, "RADIOMICS_DIR", "radiomics_features"),
    )


def build_loader(dataset, shuffle: bool, drop_last: bool = False):
    return DataLoader(
        dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=constants.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_batch_tensor(batch: dict[str, Any], names: tuple[str, ...], device: torch.device) -> torch.Tensor:
    for name in names:
        if name in batch:
            return batch[name].to(device, non_blocking=True).float()
    raise KeyError(f"None of keys {names} found in batch. Available keys: {list(batch.keys())}")


def get_target_label(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    for name in ("target_label", "label", "celltype_label"):
        if name in batch:
            return batch[name].to(device, non_blocking=True).long()
    raise KeyError(f"target label key not found. Available keys: {list(batch.keys())}")


# =========================================================
# Build
# =========================================================
def build_scratch_radiomics_model(device: torch.device):
    return build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=getattr(constants, "NUM_CELLTYPE_CLASSES", 5),
        hidden_dropout_prob=constants.DROPOUT,
        projection_dim=constants.PROJECTION_DIM,
        activation=constants.ACTIVATION,
        ape_drop_rate=getattr(constants, "APE_DROP_RATE", 0.0),
        device=device,
        num_sub_cols=getattr(constants, "NUM_SUB_COLS", [72, 54, 36, 18, 9, 3, 1]),
    ).to(device)


def build_model(device: torch.device, num_genes: int, num_radiomics_features: int):
    radiomics_model = build_scratch_radiomics_model(device)

    pathomics_encoder = DenseNet121PathomicsEncoder(
        out_dim=getattr(constants, "PATHOMICS_DIM", 1024),
        pretrained=True,
    ).to(device)
    freeze_module(pathomics_encoder)

    pathomics_proj = MLPHead(
        in_dim=getattr(constants, "PATHOMICS_DIM", 1024),
        out_dim=constants.PROJECTION_DIM,
        hidden_dim=getattr(constants, "PATH_PROJ_HIDDEN_DIM", 512),
        dropout=getattr(constants, "HEAD_DROPOUT", 0.1),
    ).to(device)

    recon_head = MLPHead(
        in_dim=constants.PROJECTION_DIM,
        out_dim=num_radiomics_features,
        hidden_dim=getattr(constants, "RECON_HIDDEN_DIM", 512),
        dropout=getattr(constants, "HEAD_DROPOUT", 0.1),
    ).to(device)

    cls_head = MLPHead(
        in_dim=getattr(constants, "HIDDEN_DIM", 128),
        out_dim=getattr(constants, "NUM_CELLTYPE_CLASSES", 5),
        hidden_dim=getattr(constants, "CLS_HIDDEN_DIM", 256),
        dropout=getattr(constants, "HEAD_DROPOUT", 0.1),
    ).to(device)

    gene_head = MLPHead(
        in_dim=getattr(constants, "PATHOMICS_DIM", 1024) + constants.PROJECTION_DIM,
        out_dim=num_genes,
        hidden_dim=getattr(constants, "GENE_HIDDEN_DIM", 512),
        dropout=getattr(constants, "HEAD_DROPOUT", 0.1),
    ).to(device)

    return MMCLReconClsModel(
        radiomics_model=radiomics_model,
        pathomics_encoder=pathomics_encoder,
        pathomics_proj=pathomics_proj,
        recon_head=recon_head,
        cls_head=cls_head,
        gene_head=gene_head,
        feature_cols=RADIOMICS_FEATURES_NAMES,
    ).to(device)




# =========================================================
# Checkpoint helpers
# =========================================================
def get_existing_stage1_checkpoint_path(save_dir: str) -> str | None:
    """
    Resolve Stage1 checkpoint path.

    Priority:
      1. constants.STAGE1_CHECKPOINT_PATH, if defined and exists
      2. best full MMCL+Recon+Cls checkpoint
      3. best checkpoint from any Stage1 phase, including recon warmup
    """
    explicit_path = getattr(constants, "STAGE1_CHECKPOINT_PATH", None)
    if explicit_path is not None and os.path.exists(explicit_path):
        return explicit_path

    candidates = [
        os.path.join(save_dir, "best_stage1_full_mmcl_recon_cls.pt"),
        os.path.join(save_dir, "best_stage1_mmcl_recon_cls.pt"),
    ]
    for ckpt_path in candidates:
        if os.path.exists(ckpt_path):
            return ckpt_path
    return None


def load_stage1_checkpoint_if_available(
    model: nn.Module,
    save_dir: str,
    device: torch.device,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """
    Load Stage1 checkpoint if enabled and available.

    constants options:
      - LOAD_STAGE1_CHECKPOINT_IF_EXISTS: bool, default True
      - STAGE1_CHECKPOINT_PATH: optional explicit checkpoint path
    """
    if not getattr(constants, "LOAD_STAGE1_CHECKPOINT_IF_EXISTS", True):
        return False, None, None

    ckpt_path = get_existing_stage1_checkpoint_path(save_dir)
    if ckpt_path is None:
        return False, None, None

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(
        f"[INFO] loaded existing Stage1 checkpoint: {ckpt_path} "
        f"(epoch={ckpt.get('epoch')}, stage={ckpt.get('stage_name')}, recon_only={ckpt.get('recon_only')})"
    )
    return True, ckpt_path, ckpt

# =========================================================
# Stage 1: train TransTab with MMCL + recon + cls
# =========================================================
def train_pretrain_epoch(model, loader, optimizer, device, recon_only: bool = False):
    model.train()
    model.pathomics_encoder.eval()

    mmcl_w = getattr(constants, "MMCL_LAMBDA", MMCL_LAMBDA)
    recon_w = getattr(constants, "RECON_LAMBDA", RECON_LAMBDA)
    cls_w = getattr(constants, "CLS_LAMBDA", CLS_LAMBDA)
    temperature = getattr(constants, "CONTRASTIVE_TEMPERATURE", 0.07)

    meter = {"loss": 0.0, "mmcl": 0.0, "recon": 0.0, "cls": 0.0, "acc": 0.0}
    for batch in tqdm(loader, desc="stage1_train", leave=False):
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        target_label = get_target_label(batch, device)

        out = model.forward_pretrain(image=image, radiomics=radiomics)
        
        recon_loss = F.mse_loss(out["pred_radiomics"], radiomics)
        
        if recon_only:
            mmcl_loss = torch.zeros((), device=device)
            cls_loss = torch.zeros((), device=device)
            loss = recon_loss
        else:
            mmcl_loss = symmetric_info_nce(
                out["path_z"],
                out["rad_contrast_z"],
                temperature=temperature,
            )
            cls_loss = F.cross_entropy(out["pred_class_logits"], target_label)
            loss = mmcl_w * mmcl_loss + recon_w * recon_loss + cls_w * cls_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = image.size(0)
        meter["loss"] += loss.item() * bs
        meter["mmcl"] += mmcl_loss.item() * bs
        meter["recon"] += recon_loss.item() * bs
        meter["cls"] += cls_loss.item() * bs
        meter["acc"] += accuracy(out["pred_class_logits"].detach(), target_label) * bs

    n = len(loader.dataset)
    return {k: v / n for k, v in meter.items()}


@torch.no_grad()
def eval_pretrain_epoch(model, loader, device):
    model.eval()
    temperature = getattr(constants, "CONTRASTIVE_TEMPERATURE", 0.07)
    meter = {"loss": 0.0, "mmcl": 0.0, "recon": 0.0, "cls": 0.0, "acc": 0.0}
    mmcl_w = getattr(constants, "MMCL_LAMBDA", MMCL_LAMBDA)
    recon_w = getattr(constants, "RECON_LAMBDA", RECON_LAMBDA)
    cls_w = getattr(constants, "CLS_LAMBDA", CLS_LAMBDA)

    for batch in tqdm(loader, desc="stage1_val", leave=False):
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        target_label = get_target_label(batch, device)

        out = model.forward_pretrain(image=image, radiomics=radiomics)
        mmcl_loss = symmetric_info_nce(out["path_z"], out["rad_contrast_z"], temperature=temperature)
        recon_loss = F.mse_loss(out["pred_radiomics"], radiomics)
        cls_loss = F.cross_entropy(out["pred_class_logits"], target_label)
        loss = mmcl_w * mmcl_loss + recon_w * recon_loss + cls_w * cls_loss

        bs = image.size(0)
        meter["loss"] += loss.item() * bs
        meter["mmcl"] += mmcl_loss.item() * bs
        meter["recon"] += recon_loss.item() * bs
        meter["cls"] += cls_loss.item() * bs
        meter["acc"] += accuracy(out["pred_class_logits"], target_label) * bs

    n = len(loader.dataset)
    return {k: v / n for k, v in meter.items()}


# =========================================================
# Stage 2: train gene head + pathomics only
# =========================================================
def set_gene_eval_trainable(model: MMCLReconClsModel):
    # freeze
    freeze_module(model.radiomics_model)
    freeze_module(model.recon_head)
    freeze_module(model.cls_head)

    # train
    model.pathomics_encoder.train()
    model.pathomics_proj.train()
    model.gene_head.train()

    for p in model.pathomics_encoder.parameters():
        p.requires_grad_(True)
    for p in model.pathomics_proj.parameters():
        p.requires_grad_(True)
    for p in model.gene_head.parameters():
        p.requires_grad_(True)


def train_gene_epoch(model, loader, optimizer, device):
    set_gene_eval_trainable(model)
    meter = {"mse": 0.0}
    for batch in tqdm(loader, desc="stage2_gene_train", leave=False):
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        out = model.forward_gene(image=image, radiomics=radiomics)
        loss = F.mse_loss(out["pred_gene"], gene)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        meter["mse"] += loss.item() * image.size(0)

    return {k: v / len(loader.dataset) for k, v in meter.items()}


@torch.no_grad()
def eval_gene_epoch(model, loader, device):
    model.eval()
    mse_sum = 0.0
    preds, targets = [], []
    for batch in tqdm(loader, desc="stage2_gene_val", leave=False):
        image = get_batch_tensor(batch, ("image", "img", "patch"), device)
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        out = model.forward_gene(image=image, radiomics=radiomics)
        pred = out["pred_gene"]
        mse_sum += F.mse_loss(pred, gene, reduction="sum").item()
        preds.append(pred.cpu())
        targets.append(gene.cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    mean_pcc, pcc_per_gene = compute_genewise_pcc(pred_all, target_all)
    mse = mse_sum / target_all.numel()
    return {"mse": mse, "mean_pcc": mean_pcc, "pcc_per_gene": pcc_per_gene}


# =========================================================
# Main
# =========================================================
def main():
    set_seed(constants.SEED)
    device = torch.device(constants.DEVICE)
    print(f"[INFO] device: {device}")

    trainset = build_dataset(constants.TRAIN_SPLIT_CSV)
    valset = build_dataset(constants.VAL_SPLIT_CSV)
    train_loader = build_loader(trainset, shuffle=True, drop_last=False)
    val_loader = build_loader(valset, shuffle=False, drop_last=False)

    print(f"[INFO] train samples: {len(trainset)}")
    print(f"[INFO] val samples: {len(valset)}")

    num_genes = len(trainset.genes)
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)
    print(f"[INFO] num_genes: {num_genes}")
    print(f"[INFO] num_radiomics_features: {num_radiomics_features}")

    model = build_model(device, num_genes=num_genes, num_radiomics_features=num_radiomics_features)

    save_dir = os.path.join(constants.CHECKPOINT_PATH, "scratch_mmcl_recon_cls__eval_genepred")
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Stage 1 ----------
    best_stage1_path = os.path.join(save_dir, "best_stage1_mmcl_recon_cls.pt")
    best_stage1_full_path = os.path.join(save_dir, "best_stage1_full_mmcl_recon_cls.pt")

    loaded_stage1, loaded_stage1_path, loaded_stage1_ckpt = load_stage1_checkpoint_if_available(
        model=model,
        save_dir=save_dir,
        device=device,
    )

    skip_stage1_if_loaded = getattr(constants, "SKIP_STAGE1_IF_LOADED", True)
    run_stage1 = not (loaded_stage1 and skip_stage1_if_loaded)

    if loaded_stage1 and skip_stage1_if_loaded:
        print("[INFO] Stage1 checkpoint found. Skip Stage1 training and move to Stage2 gene prediction.")

    if run_stage1:
        # Include all Stage1-related params in optimizer. Warmup/full mode is controlled by requires_grad.
        # Exclude frozen DenseNet encoder and gene head from Stage1 optimizer.
        stage1_params = (
            list(model.radiomics_model.parameters())
            + list(model.pathomics_proj.parameters())
            + list(model.recon_head.parameters())
            + list(model.cls_head.parameters())
        )

        optimizer_stage1 = torch.optim.AdamW(
            stage1_params,
            lr=getattr(constants, "LR", 1e-4),
            weight_decay=getattr(constants, "WEIGHT_DECAY", 1e-4),
        )

        best_stage1_val = float("inf")
        best_stage1_full_val = float("inf")
        stage1_start_epoch = 0

        # If checkpoint was loaded but SKIP_STAGE1_IF_LOADED=False, resume Stage1 training.
        if loaded_stage1 and loaded_stage1_ckpt is not None:
            stage1_start_epoch = int(loaded_stage1_ckpt.get("epoch", -1)) + 1
            val_metrics = loaded_stage1_ckpt.get("val_metrics", {}) or {}
            loaded_val_loss = float(val_metrics.get("loss", float("inf")))
            best_stage1_val = loaded_val_loss
            if not bool(loaded_stage1_ckpt.get("recon_only", False)):
                best_stage1_full_val = loaded_val_loss
            if "optimizer_state_dict" in loaded_stage1_ckpt:
                optimizer_stage1.load_state_dict(loaded_stage1_ckpt["optimizer_state_dict"])
                print("[INFO] loaded Stage1 optimizer state for resume.")
            print(f"[INFO] resume Stage1 training from epoch {stage1_start_epoch}")

        stage1_epochs = getattr(constants, "PRETRAIN_EPOCHS", getattr(constants, "EPOCHS", 50))
        warmup_recon_epochs = getattr(constants, "WARMUP_RECON_EPOCHS", WARMUP_RECON_EPOCHS)
        print(f"[INFO] Stage1 recon warmup epochs: {warmup_recon_epochs}")

        if stage1_start_epoch >= stage1_epochs:
            print(
                f"[INFO] Stage1 already completed: start_epoch={stage1_start_epoch}, "
                f"PRETRAIN_EPOCHS={stage1_epochs}."
            )
        else:
            for epoch in range(stage1_start_epoch, stage1_epochs):
                recon_only = epoch < warmup_recon_epochs
                stage_name = "ReconWarmup" if recon_only else "MMCLReconCls"

                train_m = train_pretrain_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer_stage1,
                    device=device,
                    recon_only=recon_only,
                )
                val_m = eval_pretrain_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    recon_only=recon_only,
                )

                print(
                    f"[Stage1:{stage_name}][Epoch {epoch}] "
                    f"train_loss={train_m['loss']:.4f} mmcl={train_m['mmcl']:.4f} recon={train_m['recon']:.4f} cls={train_m['cls']:.4f} acc={train_m['acc']:.4f} | "
                    f"val_loss={val_m['loss']:.4f} mmcl={val_m['mmcl']:.4f} recon={val_m['recon']:.4f} cls={val_m['cls']:.4f} acc={val_m['acc']:.4f}"
                )

                # Save best checkpoint under the metric actually used in the current phase.
                if val_m["loss"] < best_stage1_val:
                    best_stage1_val = val_m["loss"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "stage_name": stage_name,
                            "recon_only": recon_only,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer_stage1.state_dict(),
                            "val_metrics": val_m,
                        },
                        best_stage1_path,
                    )

                # Prefer a full MMCL+Recon+Cls checkpoint for Stage2 if at least one full epoch was trained.
                if (not recon_only) and val_m["loss"] < best_stage1_full_val:
                    best_stage1_full_val = val_m["loss"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "stage_name": stage_name,
                            "recon_only": recon_only,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer_stage1.state_dict(),
                            "val_metrics": val_m,
                        },
                        best_stage1_full_path,
                    )

    # Load best full pretraining checkpoint before gene eval.
    # If an existing checkpoint was loaded and Stage1 was skipped, reuse that path.
    # Otherwise, prefer full MMCL+Recon+Cls checkpoint; fall back to recon-only checkpoint.
    if loaded_stage1 and skip_stage1_if_loaded:
        load_path = loaded_stage1_path
    else:
        load_path = best_stage1_full_path if os.path.exists(best_stage1_full_path) else best_stage1_path

    if load_path is None or not os.path.exists(load_path):
        raise FileNotFoundError(
            "No Stage1 checkpoint found for Stage2. "
            "Run Stage1 first or set constants.STAGE1_CHECKPOINT_PATH."
        )

    ckpt = torch.load(load_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"[INFO] loaded Stage1 checkpoint for gene eval: {load_path}")

    # ---------- Stage 2 ----------
    set_gene_eval_trainable(model)
    optimizer_stage2 = torch.optim.AdamW(
        [
            {"params": model.gene_head.parameters(), "lr": getattr(constants, "GENE_LR", 1e-4)},
            {"params": model.pathomics_proj.parameters(), "lr": getattr(constants, "PATH_PROJ_LR", 1e-4)},
            {"params": model.pathomics_encoder.parameters(), "lr": getattr(constants, "PATH_ENCODER_LR", 1e-4)},
        ],
        weight_decay=getattr(constants, "GENE_WEIGHT_DECAY", 1e-4),
    )

    best_pcc = -float("inf")
    best_record = None
    stage2_epochs = getattr(constants, "GENE_EPOCHS", 50)
    for epoch in range(stage2_epochs):
        train_gene_m = train_gene_epoch(model, train_loader, optimizer_stage2, device)
        val_gene_m = eval_gene_epoch(model, val_loader, device)
        val_pcc = val_gene_m["mean_pcc"]
        print(
            f"[Stage2][Epoch {epoch}] "
            f"train_gene_mse={train_gene_m['mse']:.6f} | "
            f"val_gene_mse={val_gene_m['mse']:.6f} val_genewise_PCC={val_pcc:.4f} best_PCC={best_pcc:.4f}"
        )

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            best_record = {
                "epoch": epoch,
                "val_gene_mse": val_gene_m["mse"],
                "val_genewise_pcc": val_pcc,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "gene_head_state_dict": model.gene_head.state_dict(),
                    "best_record": best_record,
                    "feature_cols": RADIOMICS_FEATURES_NAMES,
                    "pcc_per_gene": val_gene_m["pcc_per_gene"],
                },
                os.path.join(save_dir, "best_stage2_genepred.pt"),
            )
            print(f"[INFO] saved best gene model: PCC={best_pcc:.4f}")

    print("\n========== Final Result ==========")
    print(best_record)


if __name__ == "__main__":
    main()


