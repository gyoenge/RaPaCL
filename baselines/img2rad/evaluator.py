from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from .cache import safe_load_gene_names
from .engine import predict_all
from .loader import build_test_loader
from .metrics import compute_genewise_pcc
from .model import FusionGeneModel, ImgToRadiomicsModel


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


def build_model_and_load(
    cfg: dict,
    device: torch.device,
    radiomics_dim: int,
    num_genes: int,
    img2rad_ckpt: str,
    fusion_gene_ckpt: str,
):
    rad_hidden_dims = tuple(cfg["model"].get("radiomics_head_hidden_dims", [512, 256]))
    gene_hidden_dims = tuple(cfg["model"].get("gene_head_hidden_dims", [512, 256]))
    dropout = float(cfg["model"].get("dropout", 0.1))
    freeze_img2rad = bool(cfg["model"].get("freeze_img2rad", False))
    fusion_mode = str(cfg["model"].get("fusion_mode", "img_radpred"))

    pretrained_img2rad_model = ImgToRadiomicsModel(
        radiomics_dim=radiomics_dim,
        backbone_weight_path=None,
        device=str(device),
        hidden_dims=rad_hidden_dims,
        dropout=dropout,
    ).to(device)
    pretrained_img2rad_model.load_state_dict(torch.load(img2rad_ckpt, map_location=device))

    fusion_gene_model = FusionGeneModel(
        pretrained_img2rad_model=pretrained_img2rad_model,
        num_genes=num_genes,
        radiomics_dim=radiomics_dim,
        fusion_mode=fusion_mode,
        freeze_img2rad=freeze_img2rad,
        hidden_dims=gene_hidden_dims,
        dropout=dropout,
    ).to(device)
    fusion_gene_model.load_state_dict(torch.load(fusion_gene_ckpt, map_location=device))
    fusion_gene_model.eval()
    return fusion_gene_model


def evaluate_one_fold(
    outer_fold: int,
    cfg: dict,
    gene_list_path: str,
    radiomics_dim: int,
    configured_num_genes: int,
    device: torch.device,
    save_dir: str,
    logger: logging.Logger,
) -> Optional[Dict]:
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    fusion_mode = str(cfg["model"].get("fusion_mode", "img_radpred"))

    img2rad_ckpt = os.path.join(checkpoint_dir, f"img2rad_best_fold{outer_fold}.pth")
    fusion_gene_ckpt = os.path.join(
        checkpoint_dir,
        _build_gene_ckpt_name(fusion_mode, outer_fold),
    )

    if not os.path.exists(img2rad_ckpt):
        logger.warning("[Fold %d] Missing checkpoint: %s -> skip", outer_fold, img2rad_ckpt)
        return None
    if not os.path.exists(fusion_gene_ckpt):
        logger.warning("[Fold %d] Missing checkpoint: %s -> skip", outer_fold, fusion_gene_ckpt)
        return None

    logger.info("=" * 80)
    logger.info("[Fold %d] Start evaluation", outer_fold)
    logger.info("[Fold %d] fusion_mode      : %s", outer_fold, fusion_mode)
    logger.info("[Fold %d] img2rad_ckpt    : %s", outer_fold, img2rad_ckpt)
    logger.info("[Fold %d] fusion_gene_ckpt: %s", outer_fold, fusion_gene_ckpt)

    test_loader, inferred_num_genes = build_test_loader(
        cfg=cfg,
        gene_list_path=gene_list_path,
        outer_fold=outer_fold,
        logger=logger,
    )

    if configured_num_genes != inferred_num_genes:
        logger.warning(
            "[Fold %d] configured_num_genes=%d, but dataset infers %d. Using inferred value.",
            outer_fold,
            configured_num_genes,
            inferred_num_genes,
        )
    num_genes = inferred_num_genes

    model = build_model_and_load(
        cfg=cfg,
        device=device,
        radiomics_dim=radiomics_dim,
        num_genes=num_genes,
        img2rad_ckpt=img2rad_ckpt,
        fusion_gene_ckpt=fusion_gene_ckpt,
    )

    preds, targets = predict_all(
        model=model,
        data_loader=test_loader,
        device=device,
        fusion_mode=fusion_mode,
    )
    mean_pcc, gene_pccs = compute_genewise_pcc(targets, preds)

    logger.info("[Fold %d] num_test_samples = %d", outer_fold, targets.shape[0])
    logger.info("[Fold %d] num_genes        = %d", outer_fold, num_genes)
    logger.info("[Fold %d] mean PCC         = %.6f", outer_fold, mean_pcc)

    fold_dir = os.path.join(save_dir, f"fold_{outer_fold}")
    os.makedirs(fold_dir, exist_ok=True)

    gene_names = safe_load_gene_names(gene_list_path)
    per_gene_rows = []
    for idx, pcc in enumerate(gene_pccs):
        gene_name = gene_names[idx] if idx < len(gene_names) else f"gene_{idx}"
        per_gene_rows.append(
            {
                "gene_idx": idx,
                "gene_name": gene_name,
                "pcc": float(pcc),
            }
        )

    per_gene_rows_sorted = sorted(per_gene_rows, key=lambda x: x["pcc"], reverse=True)

    np.save(os.path.join(fold_dir, "preds.npy"), preds)
    np.save(os.path.join(fold_dir, "targets.npy"), targets)

    with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "outer_fold": outer_fold,
                "fusion_mode": fusion_mode,
                "mean_pcc": mean_pcc,
                "num_test_samples": int(targets.shape[0]),
                "num_genes": int(num_genes),
            },
            f,
            indent=2,
        )

    with open(os.path.join(fold_dir, "per_gene_pcc.json"), "w", encoding="utf-8") as f:
        json.dump(per_gene_rows_sorted, f, indent=2)

    logger.info("[Fold %d] saved results to %s", outer_fold, fold_dir)
    logger.info("[Fold %d] Top 5 genes by PCC:", outer_fold)
    for row in per_gene_rows_sorted[:5]:
        logger.info("    %s: %.6f", row["gene_name"], row["pcc"])

    return {
        "outer_fold": outer_fold,
        "mean_pcc": float(mean_pcc),
        "num_test_samples": int(targets.shape[0]),
        "num_genes": int(num_genes),
        "preds": preds,
        "targets": targets,
        "per_gene_rows": per_gene_rows_sorted,
    }


def aggregate_fold_results(
    results: List[Dict],
    save_dir: str,
    gene_list_path: str,
    logger: logging.Logger,
):
    fold_rows = []
    all_preds = []
    all_targets = []
    gene_to_fold_pccs = {}

    for result in results:
        fold_rows.append(
            {
                "outer_fold": result["outer_fold"],
                "mean_pcc": result["mean_pcc"],
                "num_test_samples": result["num_test_samples"],
                "num_genes": result["num_genes"],
            }
        )
        all_preds.append(result["preds"])
        all_targets.append(result["targets"])

        for row in result["per_gene_rows"]:
            gene_to_fold_pccs.setdefault(row["gene_name"], []).append(row["pcc"])

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    macro_mean_pcc = float(np.mean([x["mean_pcc"] for x in fold_rows]))
    std_mean_pcc = float(np.std([x["mean_pcc"] for x in fold_rows]))
    pooled_mean_pcc, pooled_gene_pccs = compute_genewise_pcc(all_targets, all_preds)

    pooled_gene_names = safe_load_gene_names(gene_list_path)

    aggregate_per_gene_rows = []
    for idx, pooled_pcc in enumerate(pooled_gene_pccs):
        gene_name = pooled_gene_names[idx] if idx < len(pooled_gene_names) else f"gene_{idx}"
        fold_pccs = gene_to_fold_pccs.get(gene_name, [])
        aggregate_per_gene_rows.append(
            {
                "gene_idx": idx,
                "gene_name": gene_name,
                "mean_pcc_across_folds": float(np.mean(fold_pccs)) if fold_pccs else 0.0,
                "std_pcc_across_folds": float(np.std(fold_pccs)) if fold_pccs else 0.0,
                "pooled_pcc": float(pooled_pcc),
            }
        )

    aggregate_per_gene_rows = sorted(
        aggregate_per_gene_rows,
        key=lambda x: x["mean_pcc_across_folds"],
        reverse=True,
    )

    aggregate_summary = {
        "evaluated_folds": [x["outer_fold"] for x in fold_rows],
        "macro_mean_pcc_across_folds": macro_mean_pcc,
        "std_mean_pcc_across_folds": std_mean_pcc,
        "pooled_mean_pcc_all_samples": pooled_mean_pcc,
        "total_test_samples": int(all_targets.shape[0]),
    }

    with open(os.path.join(save_dir, "fold_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(fold_rows, f, indent=2)

    with open(os.path.join(save_dir, "aggregate_summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2)

    with open(os.path.join(save_dir, "aggregate_per_gene_pcc.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate_per_gene_rows, f, indent=2)

    np.save(os.path.join(save_dir, "all_preds.npy"), all_preds)
    np.save(os.path.join(save_dir, "all_targets.npy"), all_targets)

    logger.info("=" * 80)
    logger.info("[Aggregate] Done")
    logger.info("[Aggregate] evaluated folds               = %s", aggregate_summary["evaluated_folds"])
    logger.info("[Aggregate] macro mean PCC across folds = %.6f", macro_mean_pcc)
    logger.info("[Aggregate] std of fold mean PCC        = %.6f", std_mean_pcc)
    logger.info("[Aggregate] pooled mean PCC all samples = %.6f", pooled_mean_pcc)
    logger.info("[Aggregate] total test samples          = %d", all_targets.shape[0])
    logger.info("[Aggregate] Top 10 genes by mean PCC across folds:")
    for row in aggregate_per_gene_rows[:10]:
        logger.info(
            "    %s: mean=%.6f, std=%.6f, pooled=%.6f",
            row["gene_name"],
            row["mean_pcc_across_folds"],
            row["std_pcc_across_folds"],
            row["pooled_pcc"],
        )

    return aggregate_summary


def run_all_folds_pcc_eval(
    cfg: dict,
    gene_list_path: str,
    radiomics_dim: int,
    device: torch.device,
    timestamp: str,
    logger: logging.Logger,
):
    checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    save_dir = os.path.join(checkpoint_dir, f"pcceval_run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    folds_to_eval = list(cfg["runtime"]["folds"])
    num_genes = int(cfg["model"]["num_genes"])
    fusion_mode = str(cfg["model"].get("fusion_mode", "img_radpred"))

    logger.info("=" * 100)
    logger.info("Start all-fold PCC evaluation")
    logger.info("gene_list_path  = %s", gene_list_path)
    logger.info("checkpoint_dir  = %s", checkpoint_dir)
    logger.info("save_dir        = %s", save_dir)
    logger.info("folds_to_eval   = %s", folds_to_eval)
    logger.info("radiomics_dim   = %d", radiomics_dim)
    logger.info("fusion_mode     = %s", fusion_mode)
    logger.info("batch_size      = %d", int(cfg["train"]["batch_size"]))
    logger.info("num_genes(conf) = %d", num_genes)

    results = []
    for outer_fold in folds_to_eval:
        result = evaluate_one_fold(
            outer_fold=outer_fold,
            cfg=cfg,
            gene_list_path=gene_list_path,
            radiomics_dim=radiomics_dim,
            configured_num_genes=num_genes,
            device=device,
            save_dir=save_dir,
            logger=logger,
        )
        if result is not None:
            results.append(result)

    if not results:
        logger.error("No folds were evaluated. Check checkpoint filenames and paths.")
        return None

    aggregate_summary = aggregate_fold_results(
        results=results,
        save_dir=save_dir,
        gene_list_path=gene_list_path,
        logger=logger,
    )

    final_report = {
        "timestamp": timestamp,
        "save_dir": save_dir,
        "aggregate_summary": aggregate_summary,
    }
    with open(os.path.join(save_dir, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    return aggregate_summary