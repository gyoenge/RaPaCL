# rapacl/engines/_train_genehead_from_pretrained_radtranstab.py

from __future__ import annotations

import os 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rapacl.data._dataset import HestRadiomicsDataset
from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES

from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import (
    set_seed,
    load_model_radiomics_from_full_checkpoint,
)

import rapacl.engines.constants as constants


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
# Radiomics TransTab + GeneHead
# =========================================================
class RadiomicsGenePredictor(nn.Module):
    def __init__(
        self,
        radiomics_model,
        gene_head,
        feature_cols,
        freeze_radiomics=True,
    ):
        super().__init__()

        self.radiomics_model = radiomics_model
        self.gene_head = gene_head
        self.feature_cols = feature_cols

        if freeze_radiomics:
            for p in self.radiomics_model.parameters():
                p.requires_grad = False

    def encode_projection(self, radiomics_tensor):
        x_df = pd.DataFrame(
            radiomics_tensor.detach().cpu().numpy(),
            columns=self.feature_cols,
        )

        feat = self.radiomics_model.input_encoder(x_df)
        feat = self.radiomics_model.contrastive_token(**feat)
        feat = self.radiomics_model.cls_token(**feat)
        enc = self.radiomics_model.encoder(**feat)

        # contrastive token embedding
        z = enc[:, 1, :]
        z = self.radiomics_model.projection_head(z)

        return z

    def forward(self, radiomics_tensor):
        z = self.encode_projection(radiomics_tensor)
        pred_gene = self.gene_head(z)
        return pred_gene


# =========================================================
# Raw Radiomics + GeneHead
# =========================================================
class RawRadiomicsGenePredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_genes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gene_head = GeneHead(
            in_dim=in_dim,
            num_genes=num_genes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, radiomics_tensor):
        return self.gene_head(radiomics_tensor)
    

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
    dataset = HestRadiomicsDataset(
        bench_data_root=constants.ROOT_DIR,
        split_csv_path=split_csv_path,
        gene_list_path=constants.GENE_LIST_PATH,
        feature_list_path=constants.FEATURE_LIST_PATH,
        radiomics_dir="radiomics_features",
    )
    return dataset


def build_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=constants.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )


# =========================================================
# Build pretrained radiomics TransTab
# =========================================================
def build_pretrained_radiomics_model(device):
    model = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=constants.NUM_CLASS,
        hidden_dropout_prob=constants.DROPOUT,
        projection_dim=constants.PROJECTION_DIM,
        activation=constants.ACTIVATION,
        ape_drop_rate=constants.APE_DROP_RATE,
        device=device,
    ).to(device)

    load_model_radiomics_from_full_checkpoint(
        model_radiomics=model,
        checkpoint_path=constants.CHECKPOINT_PATH,
        device=device,
        strict=False,
    )

    return model


# =========================================================
# Train
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for batch in loader:
        radiomics = batch["radiomics"].to(device)
        gene = batch["gene"].to(device)

        pred = model(radiomics)

        loss = criterion(pred, gene)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# Eval
# =========================================================
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader: 
            radiomics = batch["radiomics"].to(device)
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
    set_seed(constants.SEED)

    device = torch.device(constants.DEVICE)

    print(f"[INFO] device: {device}")

    # ------------------------------------
    # Dataset
    # ------------------------------------
    trainset = build_dataset(constants.TRAIN_SPLIT_CSV)
    valset = build_dataset(constants.VAL_SPLIT_CSV)

    train_loader = build_loader(trainset, shuffle=True)
    val_loader = build_loader(valset, shuffle=False)

    print(f"[INFO] train_loader: {len(train_loader)} batches, {len(trainset)} samples")
    print(f"[INFO] val_loader: {len(val_loader)} batches, {len(valset)} samples")

    num_genes = len(trainset.genes)

    print(f"[INFO] num_genes: {num_genes}")

    criterion = nn.MSELoss()

    genehead_dir = os.path.join(constants.CHECKPOINT_PATH, "genehead")
    os.makedirs(genehead_dir, exist_ok=True)

    experiments = {}

    # =====================================================
    # Experiment A: Pretrained TransTab Projection -> GeneHead
    # =====================================================
    pretrained_model = build_pretrained_radiomics_model(device)

    projection_gene_head = GeneHead(
        in_dim=constants.PROJECTION_DIM,
        num_genes=num_genes,
        hidden_dim=512,
    ).to(device)

    experiments["pretrained_projection"] = RadiomicsGenePredictor(
        radiomics_model=pretrained_model,
        gene_head=projection_gene_head,
        feature_cols=RADIOMICS_FEATURES_NAMES,
        # freeze_radiomics=True,
        freeze_radiomics=False,  # for quick testing, change to True for full training
    ).to(device)

    # =====================================================
    # Experiment B: Raw Radiomics Feature -> GeneHead
    # =====================================================
    experiments["raw_feature"] = RawRadiomicsGenePredictor(
        in_dim=len(RADIOMICS_FEATURES_NAMES),
        num_genes=num_genes,
        hidden_dim=512,
        dropout=0.1,
    ).to(device)

    results = {}

    for exp_name, model in experiments.items():
        print(f"\n========== Experiment: {exp_name} ==========")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=1e-4,
        )

        best_pcc = -1.0
        best_record = None

        pbar = tqdm(
            # range(constants.EPOCHS),
            range(50),  # for quick testing, change to constants.EPOCHS for full training
            desc=f"{exp_name}",
            leave=True,
        )

        for epoch in pbar:
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
                    genehead_dir,
                    f"best_{exp_name}_genehead_model.pt",
                )

                torch.save(model.state_dict(), save_path)
                print(f"[INFO] saved best model: {save_path}")
            
            print(
                f"[INFO] {exp_name}: "
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"PCC={val_pcc:.4f}, "
                f"best={best_pcc:.4f}"
            )

        results[exp_name] = best_record

    print("\n========== Final Comparison ==========")
    for exp_name, record in results.items():
        print(
            f"{exp_name}: "
            f"best_epoch={record['epoch']}, "
            f"val_loss={record['val_loss']:.4f}, "
            f"val_PCC={record['val_pcc']:.4f}"
        )


if __name__ == "__main__":
    main()

