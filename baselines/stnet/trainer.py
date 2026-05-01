from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import STNetDataset
from .stnet import build_model


def build_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "sgd",
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def build_train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    optimizer,
    criterion,
) -> float:
    model.train()
    epoch_loss = 0.0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(train_loader), 1)


def eval_fold(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    pearson_corrs = []
    for gene_idx in range(all_targets.shape[1]):
        corr, _ = pearsonr(all_targets[:, gene_idx], all_preds[:, gene_idx])
        pearson_corrs.append(0.0 if np.isnan(corr) else float(corr))

    mean_pearson = float(np.mean(pearson_corrs))
    return mean_pearson, pearson_corrs


def select_best_epoch(
    train_df,
    bench_data_root: str,
    gene_list_path: str,
    device: torch.device,
    num_genes: int,
    pretrained: bool = True,
    max_epochs: int = 50,
    n_inner_folds: int = 4,  # 현재는 LOO 구조 유지
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    optimizer_name: str = "sgd",
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    logger: Optional[object] = None,
):
    """
    Split outer-train into inner folds and pick the epoch with the best
    mean validation Pearson correlation.

    Note:
        The original code uses LeaveOneOut() regardless of n_inner_folds.
        To preserve behavior, this function also keeps LOO.
    """
    _ = seed
    _ = n_inner_folds

    loo = LeaveOneOut()
    fold_indices = list(loo.split(train_df))
    epoch_scores = {epoch: [] for epoch in range(1, max_epochs + 1)}

    for inner_fold, (tr_idx, val_idx) in enumerate(fold_indices):
        if logger:
            logger.info("\n  [Inner Fold %d/%d]", inner_fold + 1, len(fold_indices))

        inner_train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        inner_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        inner_train_dataset = STNetDataset(
            bench_data_root=bench_data_root,
            gene_list_path=gene_list_path,
            split_df=inner_train_df,
            transforms=build_train_transform(),
        )
        inner_val_dataset = STNetDataset(
            bench_data_root=bench_data_root,
            gene_list_path=gene_list_path,
            split_df=inner_val_df,
            transforms=None,
        )

        inner_train_loader = build_dataloader(
            inner_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        inner_val_loader = build_dataloader(
            inner_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model = build_model(
            num_genes=num_genes,
            pretrained=pretrained,
            backbone_name="densenet121",
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = build_optimizer(
            model,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                train_loader=inner_train_loader,
                device=device,
                optimizer=optimizer,
                criterion=criterion,
            )
            val_pearson, _ = eval_fold(model, inner_val_loader, device)
            epoch_scores[epoch].append(val_pearson)

            if logger:
                logger.info(
                    "    Epoch %02d/%d | Train Loss: %.4f | Val Pearson: %.4f",
                    epoch,
                    max_epochs,
                    train_loss,
                    val_pearson,
                )

    mean_epoch_scores = {
        epoch: float(np.mean(scores))
        for epoch, scores in epoch_scores.items()
    }

    best_epoch = max(mean_epoch_scores, key=mean_epoch_scores.get)
    best_score = mean_epoch_scores[best_epoch]

    if logger:
        logger.info("\n  [Epoch Selection Summary]")
        for epoch in range(1, max_epochs + 1):
            logger.info(
                "    Epoch %02d: Mean Val Pearson = %.4f",
                epoch,
                mean_epoch_scores[epoch],
            )
        logger.info(
            "  >>> Selected Best Epoch = %d (Mean Val Pearson = %.4f)",
            best_epoch,
            best_score,
        )

    return best_epoch, mean_epoch_scores


def retrain_full_train(
    train_df,
    bench_data_root: str,
    gene_list_path: str,
    device: torch.device,
    num_genes: int,
    pretrained: bool = True,
    num_epochs: int = 50,
    batch_size: int = 32,
    num_workers: int = 0,
    optimizer_name: str = "sgd",
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    logger: Optional[object] = None,
):
    full_train_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_df=train_df,
        transforms=build_train_transform(),
    )
    full_train_loader = build_dataloader(
        full_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model = build_model({
        "num_genes": num_genes,
        "backbone": "densenet121",
        "pretrained": pretrained,
    }).to(device)

    criterion = nn.MSELoss()
    optimizer = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=full_train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
        )
        if logger:
            logger.info(
                "  [Retrain] Epoch %02d/%d | Train Loss: %.4f",
                epoch,
                num_epochs,
                train_loss,
            )

    return model