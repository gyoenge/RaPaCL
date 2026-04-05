from __future__ import annotations

import numpy as np
import torch


def train_epoch(model, data_loader, optimizer, criterion, device, logger=None):
    model.train()
    epoch_loss = 0.0

    for step, (imgs, targets) in enumerate(data_loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(imgs)

        # [Debug] 
        if logger is not None and step == 0:
            logger.info(
                "[Debug] (for first batch) pred mean=%.4f std=%.4f | target mean=%.4f std=%.4f",
                preds.mean().item(),
                preds.std().item(),
                targets.mean().item(),
                targets.std().item(),
            )

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0

    for imgs, targets in data_loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(imgs)
        loss = criterion(preds, targets)
        epoch_loss += loss.item()

    return epoch_loss / max(1, len(data_loader))


@torch.no_grad()
def predict_all(model, data_loader, device):
    model.eval()

    preds_all = []
    targets_all = []

    for imgs, targets in data_loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(targets.numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    return preds_all, targets_all