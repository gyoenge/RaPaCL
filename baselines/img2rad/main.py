from __future__ import annotations

import os
from pathlib import Path

import torch

from src.baselines.img2rad.evaluator import run_all_folds_pcc_eval
from src.baselines.img2rad.trainer import run_all_folds_training
from src.common.config import apply_cli_overrides, load_yaml, parse_common_args
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything


def build_gene_list_path(cfg: dict) -> str:
    bench_data_root = cfg["paths"]["bench_data_root"]
    genes_criteria = cfg["model"]["genes_criteria"]
    num_genes = cfg["model"]["num_genes"]
    return os.path.join(
        bench_data_root,
        f"{genes_criteria}_{num_genes}genes.json",
    )


def resolve_device(cfg: dict) -> torch.device:
    requested = cfg["runtime"].get("device", "cpu")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def resolve_run_flags(args, cfg: dict) -> tuple[bool, bool]:
    """
    args.mode 우선.
    mode가 명시되면 config의 run_train/run_eval보다 우선한다.
    """
    mode = getattr(args, "mode", None)

    if mode == "train":
        return True, False
    if mode == "eval":
        return False, True
    if mode == "all":
        return True, True

    # mode가 없거나 애매하면 config 따름
    run_train = bool(cfg["runtime"].get("run_train", True))
    run_eval = bool(cfg["runtime"].get("run_eval", True))
    return run_train, run_eval


def main() -> None:
    args = parse_common_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    log_dir = cfg["paths"].get("log_dir", "logs")
    timestamp, logger = setup_logger(log_dir=log_dir, name="img2rad")

    device = resolve_device(cfg)
    logger.info("device: %s", device)

    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    gene_list_path = build_gene_list_path(cfg)

    # 실험 config snapshot 저장
    save_yaml(cfg, Path(checkpoint_dir) / f"config_{timestamp}.yaml")

    logger.info("Configuration:")
    for top_key, top_value in cfg.items():
        logger.info("  %s: %s", top_key, top_value)

    fusion_mode = cfg["model"].get("fusion_mode", "img_radpred")
    freeze_img2rad = bool(cfg["model"].get("freeze_img2rad", False))

    logger.info("fusion_mode: %s", fusion_mode)
    logger.info("freeze_img2rad: %s", freeze_img2rad)

    run_train, run_eval = resolve_run_flags(args, cfg)
    logger.info("run_train: %s", run_train)
    logger.info("run_eval: %s", run_eval)

    train_reports = None

    if run_train:
        logger.info("=" * 100)
        logger.info("Start training")
        train_reports = run_all_folds_training(
            cfg=cfg,
            gene_list_path=gene_list_path,
            device=device,
            logger=logger,
        )

    if run_eval:
        logger.info("=" * 100)
        logger.info("Start evaluation")

        radiomics_dim = cfg["model"].get("radiomics_dim")
        if radiomics_dim is None:
            if train_reports is None or len(train_reports) == 0:
                raise ValueError(
                    "radiomics_dim is not available. "
                    "Either run training first or set model.radiomics_dim in config."
                )

            if "radiomics_dim" not in train_reports[0]:
                raise ValueError(
                    "train_reports[0]['radiomics_dim'] is missing. "
                    "Please include radiomics_dim in the training report."
                )

            radiomics_dim = int(train_reports[0]["radiomics_dim"])

        aggregate_summary = run_all_folds_pcc_eval(
            cfg=cfg,
            gene_list_path=gene_list_path,
            radiomics_dim=int(radiomics_dim),
            device=device,
            timestamp=timestamp,
            logger=logger,
        )

        logger.info("=" * 100)
        logger.info("Evaluation done")
        if aggregate_summary is not None:
            logger.info(
                "Final macro mean PCC across folds = %.6f",
                aggregate_summary["macro_mean_pcc_across_folds"],
            )

    logger.info("=" * 100)
    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()