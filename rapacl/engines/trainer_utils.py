import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_model_radiomics_from_full_checkpoint(
    model_radiomics,
    checkpoint_path,
    device,
    strict=False,
):
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    else:
        model_path = checkpoint_path

    print(f"[INFO] loading model weights from: {model_path}")

    state_dict = torch.load(model_path, map_location=device)

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # classifier head 제외
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("clf.")
    }

    missing_keys, unexpected_keys = model_radiomics.load_state_dict(
        filtered_state_dict,
        strict=False,
    )

    print("[INFO] checkpoint loaded except classifier head")
    print(f"[INFO] loaded keys: {len(filtered_state_dict)}")
    print(f"[INFO] missing keys: {len(missing_keys)}")
    print(f"[INFO] unexpected keys: {len(unexpected_keys)}")

    for k in missing_keys:
        print("  [MISSING]", k)

    for k in unexpected_keys:
        print("  [UNEXPECTED]", k)

    return missing_keys, unexpected_keys


def save_checkpoint(
    output_dir: str,
    model_radiomics: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
    name: str = "checkpoint",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"{name}_epoch{epoch}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_radiomics": model_radiomics.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": args or {},
        },
        save_path,
    )

    return save_path


def train_one_epoch(
    model_radiomics,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    scaler=None,
):
    model_radiomics.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")

    for batch in pbar:
        radiomics_features = batch["radiomics_features"]
        labels = torch.tensor(batch["labels"], device=device).long()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                embeddings, logits = model_radiomics(radiomics_features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            embeddings, logits = model_radiomics(radiomics_features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        pbar.set_postfix(
            {
                "loss": loss_meter.avg,
                "acc": acc_meter.avg,
            }
        )

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
    }


@torch.no_grad()
def evaluate(
    model_radiomics,
    loader,
    criterion,
    device,
    epoch: int,
):
    model_radiomics.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")

    for batch in pbar:
        radiomics_features = batch["radiomics_features"]
        labels = torch.tensor(batch["labels"], device=device).long()

        embeddings, logits = model_radiomics(radiomics_features)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        pbar.set_postfix(
            {
                "loss": loss_meter.avg,
                "acc": acc_meter.avg,
            }
        )

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
    }

