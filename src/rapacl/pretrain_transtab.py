from __future__ import annotations
"""
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  -m src.rapacl.pretrain_transtab \
  --config configs/pretrain_transtab/idc_allxenium.yaml \
  --mode train
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.common.config import load_yaml, parse_common_args, apply_cli_overrides
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything

from transtab import load_data, build_contrastive_learner, train


def save_column_info(
    run_dir: Path,
    categorical_columns: list[str],
    numerical_columns: list[str],
    binary_columns: list[str],
) -> None:
    info = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "binary_columns": binary_columns,
        "num_categorical": len(categorical_columns),
        "num_numerical": len(numerical_columns),
        "num_binary": len(binary_columns),
    }

    with open(run_dir / "column_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def setup_distributed(distributed: bool):
    if not distributed:
        return 0, 0, 1, "cuda:0" if torch.cuda.is_available() else "cpu"

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, f"cuda:{local_rank}"


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def main() -> None:
    args = parse_common_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg["mode"] = args.mode

    distributed = cfg["runtime"].get("distributed", False)
    rank, local_rank, world_size, device = setup_distributed(distributed)
    logger.info("Distributed: %s", distributed)
    logger.info("Rank: %s", rank)
    logger.info("Local rank: %s", local_rank)
    logger.info("World size: %s", world_size)
    logger.info("Device: %s", device)

    seed = cfg.get("seed", 42)
    seed_everything(seed + rank)

    log_dir = cfg["paths"]["log_dir"]
    timestamp, logger = setup_logger(log_dir, name=f"pretrain_transtab_rank{rank}")

    if is_main_process(rank):
        logger.info("Loaded config from: %s", args.config)
        logger.info("Execution mode: %s", args.mode)
        logger.info("Distributed: %s", distributed)
        logger.info("Device: %s", device)
        logger.info("World size: %s", world_size)
        logger.info("Batch size: %s", cfg["train"].get("batch_size"))
        logger.info("Learning rate: %s", cfg["train"].get("lr"))

    output_root = ensure_dir(cfg["paths"]["output_root"])
    run_dir = ensure_dir(output_root / f"run_{timestamp}")
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])

    if is_main_process(rank) and cfg["experiment"].get("save_config", True):
        save_yaml(cfg, run_dir / "config_final.yaml")
        logger.info("Final config saved to: %s", run_dir / "config_final.yaml")

    if is_main_process(rank):
        logger.info("Preparing custom TransTab dataset from: %s", cfg["paths"]["data_root"])

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(
        [f'{cfg["paths"]["data_root"]}']
    )

    if is_main_process(rank):
        logger.info("Detected column types:")
        logger.info("  categorical: %d", len(cat_cols))
        logger.info("  numerical  : %d", len(num_cols))
        logger.info("  binary     : %d", len(bin_cols))

        save_column_info(
            run_dir=run_dir,
            categorical_columns=cat_cols,
            numerical_columns=num_cols,
            binary_columns=bin_cols,
        )

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    model, collate_fn = build_contrastive_learner(
        cat_cols=cat_cols,
        num_cols=num_cols,
        bin_cols=bin_cols,
        supervised=model_cfg["supervised"],
        num_partition=model_cfg["num_partition"],
        overlap_ratio=model_cfg["overlap_ratio"],
        device=device,
    )

    if args.mode in {"all", "train"}:
        if is_main_process(rank):
            logger.info("Start training...")

        train(
            model=model,
            trainset=trainset,
            valset=valset,
            num_epoch=train_cfg["max_epochs"],
            batch_size=train_cfg["batch_size"],
            eval_batch_size=train_cfg["eval_batch_size"],
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            patience=train_cfg["patience"],
            warmup_ratio=train_cfg["warmup_ratio"],
            warmup_steps=train_cfg["warmup_steps"],
            eval_metric=train_cfg["eval_metric"],
            output_dir=str(checkpoint_dir),
            collate_fn=collate_fn,
            num_workers=train_cfg["num_workers"],
            ignore_duplicate_cols=model_cfg["ignore_duplicate_cols"],
            eval_less_is_better=train_cfg["eval_less_is_better"],
            distributed=distributed,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        if is_main_process(rank):
            logger.info("Training finished.")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
