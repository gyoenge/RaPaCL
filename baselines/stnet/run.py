from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.baselines.stnet import build_model
from src.baselines.stnet.dataset import STNetDataset
from src.baselines.stnet.trainer import (
    build_optimizer,
    eval_fold,
    retrain_full_train,
    select_best_epoch,
    train_one_epoch,
)
from src.common.config import apply_cli_overrides, load_yaml, parse_common_args
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything


def print_config(cfg: dict[str, Any], logger) -> None:
    logger.info("========== CONFIG ==========")
    for section, value in cfg.items():
        logger.info("%s: %s", section, value)
    logger.info("============================")


def get_device(runtime_cfg: dict[str, Any]) -> torch.device:
    requested = runtime_cfg.get("device", "cuda")
    if isinstance(requested, str) and requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def resolve_gene_list_path(paths_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> Path:
    if paths_cfg.get("gene_list_path"):
        return Path(paths_cfg["gene_list_path"])

    bench_data_root = Path(paths_cfg["bench_data_root"])
    genes_criteria = model_cfg.get("genes_criteria", "var")
    num_genes = model_cfg.get("num_genes", 250)
    return bench_data_root / f"{genes_criteria}_{num_genes}genes.json"


def resolve_split_path(
    paths_cfg: dict[str, Any],
    split_kind: str,
    outer_fold: int | None = None,
) -> Path:
    """
    split_kind: 'train' or 'test'
    priority:
    1) paths.<split_kind>_split_csv
    2) {bench_data_root}/splits/{split_kind}_{outer_fold}.csv
    """
    direct_key = f"{split_kind}_split_csv"
    if paths_cfg.get(direct_key):
        return Path(paths_cfg[direct_key])

    if outer_fold is None:
        raise ValueError(
            f"{direct_key} is not set and outer_fold is None, so split path cannot be resolved."
        )

    bench_data_root = Path(paths_cfg["bench_data_root"])
    return bench_data_root / "splits" / f"{split_kind}_{outer_fold}.csv"


def create_dataset(
    bench_data_root: str,
    gene_list_path: str,
    split_csv_path: str,
    transforms=None,
) -> STNetDataset:
    return STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_csv_path=split_csv_path,
        transforms=transforms,
    )


def create_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def save_model_checkpoint(model: torch.nn.Module, save_path: Path, logger) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info("Saved checkpoint: %s", save_path)


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device, logger) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info("Loaded checkpoint: %s", ckpt_path)


def run_train_mode(
    cfg: dict[str, Any],
    args,
    logger,
    device: torch.device,
    run_root: Path,
    ckpt_dir: Path,
    result_dir: Path,
) -> None:
    paths_cfg = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    bench_data_root = paths_cfg["bench_data_root"]
    gene_list_path = resolve_gene_list_path(paths_cfg, model_cfg)

    train_split_path = resolve_split_path(paths_cfg, "train", outer_fold=cfg.get("default_outer_fold", 0))
    test_split_path = None
    if paths_cfg.get("test_split_csv"):
        test_split_path = resolve_split_path(paths_cfg, "test")

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    max_epochs = train_cfg.get("max_epochs", 50)

    logger.info("Train split: %s", train_split_path)
    logger.info("Gene list path: %s", gene_list_path)

    train_dataset = create_dataset(
        bench_data_root=bench_data_root,
        gene_list_path=str(gene_list_path),
        split_csv_path=str(train_split_path),
        transforms=None,
    )
    train_loader = create_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model = build_model(model_cfg).to(device)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=train_cfg.get("optimizer_name", "sgd"),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        momentum=train_cfg.get("momentum", 0.9),
    )
    criterion = torch.nn.MSELoss()

    train_history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
        }

        if test_split_path is not None:
            test_dataset = create_dataset(
                bench_data_root=bench_data_root,
                gene_list_path=str(gene_list_path),
                split_csv_path=str(test_split_path),
                transforms=None,
            )
            test_loader = create_loader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            mean_pcc, _ = eval_fold(
                model=model,
                test_loader=test_loader,
                device=device,
            )
            row["eval_mean_pcc"] = float(mean_pcc)
            logger.info(
                "Epoch %02d/%d | train_loss=%.6f | eval_mean_pcc=%.6f",
                epoch,
                max_epochs,
                train_loss,
                mean_pcc,
            )
        else:
            logger.info(
                "Epoch %02d/%d | train_loss=%.6f",
                epoch,
                max_epochs,
                train_loss,
            )

        train_history.append(row)

    final_ckpt_path = ckpt_dir / "final_model.pth"
    save_model_checkpoint(model, final_ckpt_path, logger)

    history_df = pd.DataFrame(train_history)
    history_csv = result_dir / "train_history.csv"
    history_df.to_csv(history_csv, index=False)
    logger.info("Saved train history: %s", history_csv)

    summary = {
        "mode": "train",
        "checkpoint": str(final_ckpt_path),
        "num_epochs": max_epochs,
        "train_split_path": str(train_split_path),
        "test_split_path": str(test_split_path) if test_split_path else None,
    }
    with open(result_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved train summary: %s", result_dir / "train_summary.json")


def run_eval_mode(
    cfg: dict[str, Any],
    logger,
    device: torch.device,
    ckpt_dir: Path,
    result_dir: Path,
    pred_dir: Path,
) -> None:
    paths_cfg = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    runtime_cfg = cfg["runtime"]

    bench_data_root = paths_cfg["bench_data_root"]
    gene_list_path = resolve_gene_list_path(paths_cfg, model_cfg)
    test_split_path = resolve_split_path(paths_cfg, "test", outer_fold=cfg.get("default_outer_fold", 0))

    checkpoint_path = runtime_cfg.get("checkpoint_path")
    if checkpoint_path is None:
        checkpoint_path = str(ckpt_dir / "final_model.pth")

    checkpoint_path = Path(checkpoint_path)

    logger.info("Eval split: %s", test_split_path)
    logger.info("Checkpoint: %s", checkpoint_path)

    model = build_model(model_cfg).to(device)
    load_model_checkpoint(model, checkpoint_path, device, logger)

    test_dataset = create_dataset(
        bench_data_root=bench_data_root,
        gene_list_path=str(gene_list_path),
        split_csv_path=str(test_split_path),
        transforms=None,
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )

    mean_pcc, gene_pccs = eval_fold(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    logger.info("Eval mean PCC: %.6f", mean_pcc)

    gene_pcc_df = pd.DataFrame({
        "gene_idx": list(range(len(gene_pccs))),
        "pearson_corr": gene_pccs,
    })
    gene_pcc_csv = result_dir / "eval_gene_pcc.csv"
    gene_pcc_df.to_csv(gene_pcc_csv, index=False)

    summary = {
        "mode": "eval",
        "checkpoint": str(checkpoint_path),
        "test_split_path": str(test_split_path),
        "mean_pcc": float(mean_pcc),
        "num_genes": len(gene_pccs),
    }
    with open(result_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(pred_dir / "eval_gene_pcc.json", "w", encoding="utf-8") as f:
        json.dump(gene_pccs, f, ensure_ascii=False, indent=2)

    logger.info("Saved eval summary: %s", result_dir / "eval_summary.json")
    logger.info("Saved gene-wise PCC CSV: %s", gene_pcc_csv)
    logger.info("Saved gene-wise PCC JSON: %s", pred_dir / "eval_gene_pcc.json")


def run_tuning_mode(
    cfg: dict[str, Any],
    logger,
    device: torch.device,
    ckpt_dir: Path,
    result_dir: Path,
) -> None:
    paths_cfg = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    cv_cfg = cfg["cv"]

    bench_data_root = paths_cfg["bench_data_root"]
    gene_list_path = resolve_gene_list_path(paths_cfg, model_cfg)
    outer_folds = cv_cfg.get("outer_folds", [0, 1, 2, 3])
    use_inner_selection = cv_cfg.get("use_inner_selection", True)

    all_rows = []

    for outer_fold in outer_folds:
        logger.info("=============== OUTER FOLD %s ===============", outer_fold)

        train_csv = resolve_split_path(paths_cfg, "train", outer_fold=outer_fold)
        test_csv = resolve_split_path(paths_cfg, "test", outer_fold=outer_fold)

        outer_train_df = pd.read_csv(train_csv)
        outer_test_df = pd.read_csv(test_csv)

        logger.info("Outer train csv: %s", train_csv)
        logger.info("Outer test  csv: %s", test_csv)
        logger.info("Outer train rows: %d", len(outer_train_df))
        logger.info("Outer test rows: %d", len(outer_test_df))

        if use_inner_selection:
            best_epoch, mean_epoch_scores = select_best_epoch(
                train_df=outer_train_df,
                bench_data_root=bench_data_root,
                gene_list_path=str(gene_list_path),
                device=device,
                num_genes=model_cfg.get("num_genes", 250),
                pretrained=model_cfg.get("pretrained", True),
                max_epochs=train_cfg.get("max_epochs", 50),
                batch_size=train_cfg.get("batch_size", 32),
                num_workers=train_cfg.get("num_workers", 0),
                seed=cfg.get("seed", 42),
                optimizer_name=train_cfg.get("optimizer_name", "sgd"),
                lr=train_cfg.get("lr", 1e-4),
                weight_decay=train_cfg.get("weight_decay", 0.0),
                momentum=train_cfg.get("momentum", 0.9),
                logger=logger,
            )
        else:
            best_epoch = train_cfg.get("max_epochs", 50)
            mean_epoch_scores = {best_epoch: None}
            logger.info("Inner epoch selection disabled. Using max_epochs=%d", best_epoch)

        logger.info("Selected best_epoch=%d", best_epoch)

        final_model = retrain_full_train(
            train_df=outer_train_df,
            bench_data_root=bench_data_root,
            gene_list_path=str(gene_list_path),
            device=device,
            num_genes=model_cfg.get("num_genes", 250),
            pretrained=model_cfg.get("pretrained", True),
            num_epochs=best_epoch,
            batch_size=train_cfg.get("batch_size", 32),
            num_workers=train_cfg.get("num_workers", 0),
            optimizer_name=train_cfg.get("optimizer_name", "sgd"),
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 0.0),
            momentum=train_cfg.get("momentum", 0.9),
            logger=logger,
        )

        fold_ckpt_path = ckpt_dir / f"fold_{outer_fold}_best_epoch_{best_epoch}.pth"
        save_model_checkpoint(final_model, fold_ckpt_path, logger)

        outer_test_dataset = STNetDataset(
            bench_data_root=bench_data_root,
            gene_list_path=str(gene_list_path),
            split_df=outer_test_df,
            transforms=None,
        )
        outer_test_loader = create_loader(
            outer_test_dataset,
            batch_size=train_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
        )

        test_mean_pcc, gene_pccs = eval_fold(
            model=final_model,
            test_loader=outer_test_loader,
            device=device,
        )

        row = {
            "outer_fold": outer_fold,
            "best_epoch": best_epoch,
            "test_mean_pcc": float(test_mean_pcc),
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            "checkpoint_path": str(fold_ckpt_path),
        }
        all_rows.append(row)

        fold_gene_pcc_csv = result_dir / f"fold_{outer_fold}_gene_pcc.csv"
        pd.DataFrame({
            "gene_idx": list(range(len(gene_pccs))),
            "pearson_corr": gene_pccs,
        }).to_csv(fold_gene_pcc_csv, index=False)

        logger.info(
            "Outer fold %d finished | best_epoch=%d | test_mean_pcc=%.6f",
            outer_fold,
            best_epoch,
            test_mean_pcc,
        )

        epoch_score_json = result_dir / f"fold_{outer_fold}_epoch_scores.json"
        with open(epoch_score_json, "w", encoding="utf-8") as f:
            json.dump(mean_epoch_scores, f, ensure_ascii=False, indent=2)

    result_df = pd.DataFrame(all_rows)
    result_csv = result_dir / "tuning_fold_results.csv"
    result_df.to_csv(result_csv, index=False)

    summary = {
        "mode": "tuning",
        "outer_folds": outer_folds,
        "mean_test_pcc": float(result_df["test_mean_pcc"].mean()) if len(result_df) > 0 else None,
        "std_test_pcc": float(result_df["test_mean_pcc"].std()) if len(result_df) > 1 else None,
        "num_folds": int(len(result_df)),
    }
    with open(result_dir / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Saved tuning fold results: %s", result_csv)
    logger.info("Saved tuning summary: %s", result_dir / "tuning_summary.json")


def main() -> None:
    args = parse_common_args()
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    seed_everything(cfg.get("seed", 42))

    paths_cfg = cfg.setdefault("paths", {})
    cfg.setdefault("runtime", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("cv", {})

    log_dir = paths_cfg.get("log_dir", "logs/stnet")
    output_root = paths_cfg.get("output_root", "outputs/stnet")

    timestamp, logger = setup_logger(log_dir=log_dir, name="stnet")

    run_root = ensure_dir(Path(output_root) / f"run_{timestamp}")
    ckpt_dir = ensure_dir(run_root / "checkpoints")
    result_dir = ensure_dir(run_root / "results")
    pred_dir = ensure_dir(run_root / "predictions")

    save_yaml(cfg, run_root / "resolved_config.yaml")

    device = get_device(cfg["runtime"])

    logger.info("Mode: %s", args.mode)
    logger.info("Device: %s", device)
    print_config(cfg, logger)

    if "bench_data_root" not in cfg["paths"]:
        raise ValueError("cfg['paths']['bench_data_root'] is required.")

    gene_list_path = resolve_gene_list_path(cfg["paths"], cfg["model"])
    if not gene_list_path.exists():
        raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")

    logger.info("Resolved gene list path: %s", gene_list_path)

    if args.mode == "train":
        run_train_mode(
            cfg=cfg,
            args=args,
            logger=logger,
            device=device,
            run_root=run_root,
            ckpt_dir=ckpt_dir,
            result_dir=result_dir,
        )
    elif args.mode == "eval":
        run_eval_mode(
            cfg=cfg,
            logger=logger,
            device=device,
            ckpt_dir=ckpt_dir,
            result_dir=result_dir,
            pred_dir=pred_dir,
        )
    elif args.mode == "tuning":
        run_tuning_mode(
            cfg=cfg,
            logger=logger,
            device=device,
            ckpt_dir=ckpt_dir,
            result_dir=result_dir,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    logger.info("Run finished successfully.")
    logger.info("Run root: %s", run_root)
    logger.info("Checkpoint dir: %s", ckpt_dir)
    logger.info("Result dir: %s", result_dir)
    logger.info("Prediction dir: %s", pred_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise