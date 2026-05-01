from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim

from .engine import evaluate_loss, train_epoch
from .loader import build_gene_dataloaders, build_radiomics_dataloaders
from .model import FusionGeneModel, ImgToRadiomicsModel


def build_optimizer(parameters, cfg: dict):
    train_cfg = cfg["train"]
    optimizer_name = str(train_cfg.get("optimizer_name", "adamw")).lower()
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def _build_backbone_weight_path(cfg: dict, outer_fold: int) -> str:
    stnet_ckpt_dir = cfg["paths"]["stnet_ckpt_dir"]
    genes_criteria = cfg["model"]["genes_criteria"]
    num_genes = cfg["model"]["num_genes"]
    backbone_name = cfg["model"].get("backbone_name", "densenet121")

    return os.path.join(
        stnet_ckpt_dir,
        f"stnet_backbone_fold{outer_fold}_{backbone_name}_{genes_criteria}{num_genes}.pth",
    )


def _build_gene_ckpt_name(fusion_mode: str, outer_fold: int) -> str:
    if fusion_mode == "img_radpred":
        tag = "imgfeat_radpred_gene"
    elif fusion_mode == "img_radhidden":
        tag = "imgfeat_radhidden_gene"
    elif fusion_mode == "img_rawrad":
        tag = "imgfeat_rawrad_gene"
    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    return f"{tag}_best_fold{outer_fold}.pth"


def train_one_fold(
    outer_fold: int,
    cfg: dict,
    gene_list_path: str,
    device: torch.device,
    logger: logging.Logger,
):
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    num_epochs_img2rad = int(
        cfg["train"].get("num_epochs_img2rad", cfg["train"].get("max_epochs", 50))
    )
    num_epochs_gene = int(
        cfg["train"].get("num_epochs_gene", cfg["train"].get("max_epochs", 50))
    )

    rad_hidden_dims = tuple(cfg["model"].get("radiomics_head_hidden_dims", [512, 256]))
    gene_hidden_dims = tuple(cfg["model"].get("gene_head_hidden_dims", [512, 256]))
    dropout = float(cfg["model"].get("dropout", 0.1))
    freeze_img2rad = bool(cfg["model"].get("freeze_img2rad", False))
    fusion_mode = str(cfg["model"].get("fusion_mode", "img_radpred"))

    logger.info("=" * 100)
    logger.info("[Fold %d] Start training", outer_fold)
    logger.info("[Fold %d] fusion_mode=%s", outer_fold, fusion_mode)
    logger.info("[Fold %d] freeze_img2rad=%s", outer_fold, freeze_img2rad)

    train_loader_rad, val_loader_rad, radiomics_dim, _ = build_radiomics_dataloaders(
        cfg=cfg,
        gene_list_path=gene_list_path,
        outer_fold=outer_fold,
        logger=logger,
    )

    backbone_weight_path = _build_backbone_weight_path(cfg, outer_fold)
    logger.info("[Fold %d] backbone_weight_path = %s", outer_fold, backbone_weight_path)

    if cfg["model"].get("radiomics_dim") is not None:
        radiomics_dim = cfg["model"]["radiomics_dim"]

    img2rad_model = ImgToRadiomicsModel(
        radiomics_dim=radiomics_dim,
        backbone_weight_path=backbone_weight_path,
        device=str(device),
        hidden_dims=rad_hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion_rad = nn.MSELoss()
    optimizer_rad = build_optimizer(img2rad_model.parameters(), cfg)

    best_val_loss = float("inf")
    img2rad_ckpt_path = os.path.join(checkpoint_dir, f"img2rad_best_fold{outer_fold}.pth")

    for epoch in range(1, num_epochs_img2rad + 1):
        train_loss = train_epoch(
            model=img2rad_model,
            data_loader=train_loader_rad,
            optimizer=optimizer_rad,
            criterion=criterion_rad,
            device=device,
            fusion_mode="img_radpred",  # img2rad 단계는 rawrad 입력 안 씀
            logger=logger,
        )
        val_loss = evaluate_loss(
            model=img2rad_model,
            data_loader=val_loader_rad,
            criterion=criterion_rad,
            device=device,
            fusion_mode="img_radpred",
        )

        logger.info(
            "[Fold %d] [Img2Rad] Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f",
            outer_fold,
            epoch,
            num_epochs_img2rad,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(img2rad_model.state_dict(), img2rad_ckpt_path)
            logger.info("[Fold %d] Saved best img2rad -> %s", outer_fold, img2rad_ckpt_path)

    logger.info("[Fold %d] Best Img2Rad val_loss = %.6f", outer_fold, best_val_loss)

    train_loader_gene, val_loader_gene, num_genes = build_gene_dataloaders(
        cfg=cfg,
        gene_list_path=gene_list_path,
        outer_fold=outer_fold,
        logger=logger,
    )

    pretrained_img2rad_model = ImgToRadiomicsModel(
        radiomics_dim=radiomics_dim,
        backbone_weight_path=None,
        device=str(device),
        hidden_dims=rad_hidden_dims,
        dropout=dropout,
    ).to(device)
    pretrained_img2rad_model.load_state_dict(
        torch.load(img2rad_ckpt_path, map_location=device)
    )

    fusion_gene_model = FusionGeneModel(
        pretrained_img2rad_model=pretrained_img2rad_model,
        num_genes=num_genes,
        radiomics_dim=radiomics_dim,
        fusion_mode=fusion_mode,
        freeze_img2rad=freeze_img2rad,
        hidden_dims=gene_hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion_gene = nn.MSELoss()
    optimizer_gene = build_optimizer(fusion_gene_model.parameters(), cfg)

    best_gene_val_loss = float("inf")
    gene_ckpt_path = os.path.join(
        checkpoint_dir,
        _build_gene_ckpt_name(fusion_mode, outer_fold),
    )

    for epoch in range(1, num_epochs_gene + 1):
        train_loss = train_epoch(
            model=fusion_gene_model,
            data_loader=train_loader_gene,
            optimizer=optimizer_gene,
            criterion=criterion_gene,
            device=device,
            fusion_mode=fusion_mode,
            logger=logger,
        )
        val_loss = evaluate_loss(
            model=fusion_gene_model,
            data_loader=val_loader_gene,
            criterion=criterion_gene,
            device=device,
            fusion_mode=fusion_mode,
        )

        logger.info(
            "[Fold %d] [Gene] Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f",
            outer_fold,
            epoch,
            num_epochs_gene,
            train_loss,
            val_loss,
        )

        if val_loss < best_gene_val_loss:
            best_gene_val_loss = val_loss
            torch.save(fusion_gene_model.state_dict(), gene_ckpt_path)
            logger.info(
                "[Fold %d] Saved best fusion gene -> %s",
                outer_fold,
                gene_ckpt_path,
            )

    logger.info("[Fold %d] Best Gene val_loss = %.6f", outer_fold, best_gene_val_loss)
    logger.info("[Fold %d] Done training", outer_fold)

    return {
        "outer_fold": outer_fold,
        "fusion_mode": fusion_mode,
        "img2rad_ckpt": img2rad_ckpt_path,
        "fusion_gene_ckpt": gene_ckpt_path,
        "radiomics_dim": radiomics_dim,
        "num_genes": num_genes,
        "best_img2rad_val_loss": float(best_val_loss),
        "best_gene_val_loss": float(best_gene_val_loss),
    }


def run_all_folds_training(
    cfg: dict,
    gene_list_path: str,
    device: torch.device,
    logger: logging.Logger,
):
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    folds_to_train = list(cfg["runtime"]["folds"])

    logger.info("=" * 100)
    logger.info("Start all-fold training")

    reports = []
    for outer_fold in folds_to_train:
        report = train_one_fold(
            outer_fold=outer_fold,
            cfg=cfg,
            gene_list_path=gene_list_path,
            device=device,
            logger=logger,
        )
        reports.append(report)

    with open(
        os.path.join(checkpoint_dir, "train_reports_all_folds.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(reports, f, indent=2)

    logger.info("All-fold training done")
    return reports