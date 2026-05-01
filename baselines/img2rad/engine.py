from __future__ import annotations

import numpy as np
import torch


def _unpack_batch(batch, fusion_mode: str, device):
    """
    Supports:
      - (imgs, targets)
      - (imgs, raw_radiomics, targets)

    fusion_mode:
      - img_radpred
      - img_radhidden
      - img_rawrad
    """
    if fusion_mode == "img_rawrad":
        if len(batch) != 3:
            raise ValueError(
                "For fusion_mode='img_rawrad', batch must be (imgs, raw_radiomics, targets)."
            )
        imgs, raw_radiomics, targets = batch
        imgs = imgs.to(device, non_blocking=True)
        raw_radiomics = raw_radiomics.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        return imgs, raw_radiomics, targets

    else:
        if len(batch) != 2:
            raise ValueError(
                f"For fusion_mode='{fusion_mode}', batch must be (imgs, targets)."
            )
        imgs, targets = batch
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        return imgs, None, targets


def _forward_model(model, imgs, raw_radiomics, fusion_mode: str):
    if fusion_mode == "img_rawrad":
        return model(imgs, raw_radiomics=raw_radiomics)
    return model(imgs)


def train_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    fusion_mode: str = "img_radpred",
    logger=None,
):
    model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(data_loader):
        imgs, raw_radiomics, targets = _unpack_batch(batch, fusion_mode, device)

        optimizer.zero_grad()
        preds = _forward_model(model, imgs, raw_radiomics, fusion_mode)

        # [Debug]
        if logger is not None and step == 0:
            logger.info(
                "[Debug] fusion_mode=%s | pred mean=%.4f std=%.4f | target mean=%.4f std=%.4f",
                fusion_mode,
                preds.mean().item(),
                preds.std().item(),
                targets.mean().item(),
                targets.std().item(),
            )
            if fusion_mode == "img_rawrad":
                logger.info(
                    "[Debug] raw_radiomics mean=%.4f std=%.4f shape=%s",
                    raw_radiomics.mean().item(),
                    raw_radiomics.std().item(),
                    tuple(raw_radiomics.shape),
                )

        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate_loss(
    model,
    data_loader,
    criterion,
    device,
    fusion_mode: str = "img_radpred",
):
    model.eval()
    epoch_loss = 0.0

    for batch in data_loader:
        imgs, raw_radiomics, targets = _unpack_batch(batch, fusion_mode, device)

        preds = _forward_model(model, imgs, raw_radiomics, fusion_mode)
        loss = criterion(preds, targets)
        epoch_loss += loss.item()

    return epoch_loss / max(1, len(data_loader))


@torch.no_grad()
def predict_all(
    model,
    data_loader,
    device,
    fusion_mode: str = "img_radpred",
):
    model.eval()

    preds_all = []
    targets_all = []

    for batch in data_loader:
        imgs, raw_radiomics, targets = _unpack_batch(batch, fusion_mode, device)

        preds = _forward_model(model, imgs, raw_radiomics, fusion_mode)

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    return preds_all, targets_all