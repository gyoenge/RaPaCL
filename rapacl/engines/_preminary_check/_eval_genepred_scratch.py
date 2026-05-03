# rapacl/engines/_train_genehead_from_scratch_radtranstab.py

from __future__ import annotations

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rapacl.data._dataset import HestRadiomicsDataset
from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES

from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import set_seed

import rapacl.configs.default.train as train


# =========================================================
# Gene Head
# =========================================================
class GeneHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_genes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_genes),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# Radiomics TransTab + GeneHead from Scratch
# =========================================================
class RadiomicsGenePredictor(nn.Module):
    def __init__(
        self,
        radiomics_model,
        gene_head,
        feature_cols,
    ):
        super().__init__()

        self.radiomics_model = radiomics_model
        self.gene_head = gene_head
        self.feature_cols = feature_cols

    def encode_projection(self, radiomics_features):
        if isinstance(radiomics_features, torch.Tensor):
            x_df = pd.DataFrame(
                radiomics_features.detach().cpu().numpy(),
                columns=self.feature_cols,
            )
        else:
            x_df = radiomics_features

        feat = self.radiomics_model.input_encoder(x_df)
        feat = self.radiomics_model.contrastive_token(**feat)
        feat = self.radiomics_model.cls_token(**feat)
        enc = self.radiomics_model.encoder(**feat)

        z = enc[:, 1, :]
        z = self.radiomics_model.projection_head(z)

        return z

    def forward(self, radiomics_features):
        z = self.encode_projection(radiomics_features)
        pred_gene = self.gene_head(z)
        return pred_gene


# =========================================================
# PCC
# =========================================================
def compute_pcc(pred, target, eps=1e-8):
    pred = pred.detach()
    target = target.detach()

    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=0)

    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=0)
        * (target_centered ** 2).sum(dim=0)
    ) + eps

    pcc_per_gene = numerator / denominator
    mean_pcc = pcc_per_gene.mean()

    return mean_pcc.item(), pcc_per_gene.cpu().numpy()


# =========================================================
# Dataset
# =========================================================
def build_dataset(split_csv_path):
    return HestRadiomicsDataset(
        bench_data_root=train.ROOT_DIR,
        split_csv_path=split_csv_path,
        gene_list_path=train.GENE_LIST_PATH,
        feature_list_path=train.FEATURE_LIST_PATH,
        radiomics_dir="radiomics_features",
    )


def build_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=train.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=train.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )


# =========================================================
# Build Radiomics TransTab from Scratch
# =========================================================
def build_scratch_radiomics_model(device):
    model = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=train.NUM_CLASS,
        hidden_dropout_prob=train.DROPOUT,
        projection_dim=train.PROJECTION_DIM,
        activation=train.ACTIVATION,
        ape_drop_rate=train.APE_DROP_RATE,
        device=device,
        # add
        num_sub_cols = [72, 36, 24]
    )

    return model.to(device)


# =========================================================
# Train
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for batch in loader:
        radiomics = batch["radiomics"]
        gene = batch["gene"].to(device)

        pred = model(radiomics)
        loss = criterion(pred, gene)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        radiomics = batch["radiomics"]
        gene = batch["gene"].to(device)

        pred = model(radiomics)
        loss = criterion(pred, gene)

        total_loss += loss.item()

        all_preds.append(pred.cpu())
        all_targets.append(gene.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mean_pcc, _ = compute_pcc(preds, targets)

    return total_loss / len(loader), mean_pcc


# =========================================================
# Main
# =========================================================
def main():
    set_seed(train.SEED)

    device = torch.device(train.DEVICE)
    print(f"[INFO] device: {device}")

    trainset = build_dataset(train.TRAIN_SPLIT_CSV)
    valset = build_dataset(train.VAL_SPLIT_CSV)

    train_loader = build_loader(trainset, shuffle=True)
    val_loader = build_loader(valset, shuffle=False)

    print(f"[INFO] train samples: {len(trainset)}")
    print(f"[INFO] val samples: {len(valset)}")

    num_genes = len(trainset.genes)
    print(f"[INFO] num_genes: {num_genes}")

    radiomics_model = build_scratch_radiomics_model(device)

    gene_head = GeneHead(
        in_dim=train.PROJECTION_DIM,
        num_genes=num_genes,
        hidden_dim=512,
        dropout=0.1,
    ).to(device)

    model = RadiomicsGenePredictor(
        radiomics_model=radiomics_model,
        gene_head=gene_head,
        feature_cols=RADIOMICS_FEATURES_NAMES,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    save_dir = os.path.join(train.CHECKPOINT_PATH, "scratch_radtranstab_gene")
    os.makedirs(save_dir, exist_ok=True)

    best_pcc = -1.0
    best_record = None

    # for epoch in tqdm(range(constants.EPOCHS), desc="scratch_radtranstab_gene"):
    for epoch in tqdm(range(50), desc="scratch_radtranstab_gene"):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_pcc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            best_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pcc": val_pcc,
            }

            save_path = os.path.join(
                save_dir,
                "best_scratch_radtranstab_gene_model.pt",
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_record": best_record,
                },
                save_path,
            )

            print(f"[INFO] saved best model: {save_path}")

        print(
            f"[INFO] Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_PCC={val_pcc:.4f}, "
            f"best_PCC={best_pcc:.4f}"
        )

    print("\n========== Final Result ==========")
    print(best_record)


if __name__ == "__main__":
    main()
