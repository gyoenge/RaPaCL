from __future__ import annotations
# distributed learning version of pretrain_transtab.py
"""
For single node 2GPU: 
torchrun --nproc_per_node=2 -m src.rapacl.pretrain_transtab_distributed \ 
    --config configs/pretrain_transtab/idc_allxenium.yaml \ 
    --mode train
"""


import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import transtab
from src.common.config import apply_cli_overrides, load_yaml, parse_common_args
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything


class RankFilter:
    """Allow logs only on rank 0 to avoid duplicated messages."""

    def __init__(self, rank: int):
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


class TabularTupleDataset(Dataset):
    """Wrap a (X, y) tuple from transtab into a PyTorch Dataset."""

    def __init__(self, xy_tuple: tuple[pd.DataFrame, pd.Series]):
        x, y = xy_tuple
        self.x = x.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        row_x = self.x.iloc[idx].to_dict()
        row_y = self.y.iloc[idx]
        return row_x, row_y


class DDPTransTabCollator:
    """Apply TransTab collator to features, and keep labels separately."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, batch):
        x_list, y_list = zip(*batch)
        inputs = self.base_collator(list(x_list))
        labels = torch.as_tensor(np.asarray(y_list))
        return inputs, labels


def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.cuda.is_available():
        raise RuntimeError("DDP script requires CUDA GPUs. torch.cuda.is_available() is False.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_everything_ddp(seed: int, rank: int) -> None:
    final_seed = seed + rank
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)
    seed_everything(final_seed)


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def move_to_device(batch_inputs: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch_inputs):
        return batch_inputs.to(device, non_blocking=True)
    if isinstance(batch_inputs, dict):
        return {k: move_to_device(v, device) for k, v in batch_inputs.items()}
    if isinstance(batch_inputs, (list, tuple)):
        moved = [move_to_device(v, device) for v in batch_inputs]
        return type(batch_inputs)(moved)
    return batch_inputs


def read_feature_list(txt_path: str | Path) -> list[str]:
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Feature txt file not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_custom_transtab_dataset(
    data_root: str | Path,
    data_csv: str,
    numerical_feature_file: str,
    target_col: str,
    binary_feature_file: str | None = None,
):
    data_root = Path(data_root)

    csv_path = data_root / data_csv
    num_path = data_root / numerical_feature_file

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not num_path.exists():
        raise FileNotFoundError(f"Numerical feature file not found: {num_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).lower() for c in df.columns]
    target_col = target_col.lower()

    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    numerical_columns = read_feature_list(num_path)

    binary_columns = []
    if binary_feature_file is not None:
        bin_path = data_root / binary_feature_file
        if bin_path.exists():
            binary_columns = read_feature_list(bin_path)

    numerical_columns = [c for c in numerical_columns if c in df.columns and c != target_col]
    binary_columns = [c for c in binary_columns if c in df.columns and c != target_col]

    used = set(numerical_columns) | set(binary_columns) | {target_col}
    categorical_columns = [c for c in df.columns if c not in used]

    x = df.drop(columns=[target_col])
    y = df[target_col]

    return (x, y), categorical_columns, numerical_columns, binary_columns


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


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def build_ddp_dataloader(trainset, collate_fn, batch_size: int, num_workers: int):
    dataset = TabularTupleDataset(trainset)
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=DDPTransTabCollator(collate_fn),
    )
    return loader, sampler


def call_model_for_loss(model, batch_inputs, batch_labels):
    """
    Try several calling conventions because transtab versions may differ.
    Returns a scalar loss tensor.
    """
    attempts = [
        lambda: model(batch_inputs, labels=batch_labels),
        lambda: model(batch_inputs, y=batch_labels),
        lambda: model(batch_inputs, batch_labels),
        lambda: model(batch_inputs),
    ]

    last_exc = None
    for fn in attempts:
        try:
            outputs = fn()
            if isinstance(outputs, dict):
                if "loss" in outputs:
                    return outputs["loss"]
                if "train_loss" in outputs:
                    return outputs["train_loss"]
            if hasattr(outputs, "loss"):
                return outputs.loss
            if torch.is_tensor(outputs):
                return outputs
        except TypeError as e:
            last_exc = e
            continue

    raise RuntimeError(
        "Failed to obtain loss from model.forward(...). "
        "Please check your installed transtab version and forward signature."
    ) from last_exc


@torch.no_grad()
def maybe_log_batch_shapes(logger, batch_inputs, batch_labels, done_flag: dict[str, bool]) -> None:
    if done_flag.get("done", False):
        return
    if not is_main_process():
        return

    def describe(obj):
        if torch.is_tensor(obj):
            return tuple(obj.shape)
        if isinstance(obj, dict):
            return {k: describe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [describe(v) for v in obj]
        return type(obj).__name__

    logger.info("First batch input structure: %s", describe(batch_inputs))
    logger.info("First batch label shape: %s", tuple(batch_labels.shape))
    done_flag["done"] = True


def train_ddp(
    model,
    train_loader,
    train_sampler,
    logger,
    local_rank: int,
    world_size: int,
    num_epoch: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    output_dir: str = "./ckpt",
    max_grad_norm: float | None = None,
):
    device = torch.device(f"cuda:{local_rank}")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ensure_dir(output_dir)
    best_loss = float("inf")
    first_batch_logged = {"done": False}

    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        model.train()

        running_loss = torch.zeros(1, device=device)
        num_batches = 0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = move_to_device(batch_inputs, device)
            batch_labels = batch_labels.to(device, non_blocking=True)

            maybe_log_batch_shapes(logger, batch_inputs, batch_labels, first_batch_logged)

            optimizer.zero_grad(set_to_none=True)
            loss = call_model_for_loss(model, batch_inputs, batch_labels)
            loss.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            running_loss += loss.detach()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        epoch_loss = reduce_mean(epoch_loss, world_size)

        if is_main_process():
            logger.info(
                "Epoch %d/%d | train loss: %.6f | lr: %.6e",
                epoch + 1,
                num_epoch,
                epoch_loss.item(),
                optimizer.param_groups[0]["lr"],
            )

            if epoch_loss.item() < best_loss:
                best_loss = epoch_loss.item()
                save_path = os.path.join(output_dir, "best_model.pt")
                torch.save(model.module.state_dict(), save_path)
                logger.info("Saved best model to: %s", save_path)

        barrier()

    if is_main_process():
        last_path = os.path.join(output_dir, "last_model.pt")
        torch.save(model.module.state_dict(), last_path)
        logger.info("Saved last model to: %s", last_path)


def main() -> None:
    args = parse_common_args()
    rank, world_size, local_rank = setup_ddp()

    try:
        cfg = load_yaml(args.config)
        cfg = apply_cli_overrides(cfg, args)
        cfg["mode"] = args.mode

        seed = cfg.get("seed", 42)
        seed_everything_ddp(seed, rank)

        log_dir = cfg["paths"]["log_dir"]
        timestamp, logger = setup_logger(log_dir, name="pretrain_transtab_distributed")
        logger.logger.addFilter(RankFilter(rank)) if hasattr(logger, "logger") else None
        if hasattr(logger, "handlers"):
            for h in logger.handlers:
                h.addFilter(RankFilter(rank))

        if is_main_process():
            logger.info("Loaded config from: %s", args.config)
            logger.info("Execution mode: %s", args.mode)
            logger.info("DDP world size: %d", world_size)
            logger.info("Rank / Local rank: %d / %d", rank, local_rank)
            logger.info("Batch size per GPU: %s", cfg["train"].get("batch_size"))
            logger.info("Effective batch size: %s", cfg["train"].get("batch_size") * world_size)
            logger.info("Learning rate: %s", cfg["train"].get("lr"))

        output_root = ensure_dir(cfg["paths"]["output_root"])
        checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
        run_dir = ensure_dir(output_root / f"run_{timestamp}") if is_main_process() else None

        if is_main_process() and cfg["experiment"].get("save_config", True):
            save_yaml(cfg, run_dir / "config_final.yaml")
            logger.info("Final config saved to: %s", run_dir / "config_final.yaml")

        if is_main_process():
            logger.info("Preparing custom TransTab dataset from: %s", cfg["paths"]["data_root"])

        # Keep the same loading path as the original script.
        allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(
            [f'{cfg["paths"]["data_root"]}']
        )

        if is_main_process():
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

        model, collate_fn = transtab.build_contrastive_learner(
            cat_cols=cat_cols,
            num_cols=num_cols,
            bin_cols=bin_cols,
            supervised=model_cfg["supervised"],
            num_partition=model_cfg["num_partition"],
            overlap_ratio=model_cfg["overlap_ratio"],
        )

        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        train_loader, train_sampler = build_ddp_dataloader(
            trainset=trainset,
            collate_fn=collate_fn,
            batch_size=train_cfg["batch_size"],
            num_workers=train_cfg["num_workers"],
        )

        if args.mode in {"all", "train"}:
            if is_main_process():
                logger.info("Start DDP training...")
            train_ddp(
                model=model,
                train_loader=train_loader,
                train_sampler=train_sampler,
                logger=logger,
                local_rank=local_rank,
                world_size=world_size,
                num_epoch=train_cfg["max_epochs"],
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
                output_dir=str(checkpoint_dir),
                max_grad_norm=train_cfg.get("max_grad_norm"),
            )
            if is_main_process():
                logger.info("Training finished.")
        else:
            if is_main_process():
                logger.info("Mode '%s' does not start training in this distributed script.", args.mode)

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
