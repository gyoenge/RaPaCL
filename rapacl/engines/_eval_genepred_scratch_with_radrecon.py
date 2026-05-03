# rapacl/engines/_train_genehead_with_radrecon.py

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
# Gene Prediction Head
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================
# Radiomics Reconstruction Head
# =========================================================
class RadiomicsReconstructionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================
# Radiomics TransTab + GeneHead + ReconstructionHead
# =========================================================
class RadiomicsGeneReconPredictor(nn.Module):
    def __init__(
        self,
        radiomics_model: nn.Module,
        gene_head: nn.Module,
        recon_head: nn.Module,
        feature_cols: list[str],
    ):
        super().__init__()

        self.radiomics_model = radiomics_model
        self.gene_head = gene_head
        self.recon_head = recon_head
        self.feature_cols = feature_cols

    def encode_projection(self, radiomics_features):
        """
        radiomics_features:
            - Tensor: shape [B, num_radiomics_features]
            - or pandas DataFrame

        returns:
            - z: projected embedding, shape [B, projection_dim]
        """
        if isinstance(radiomics_features, torch.Tensor):
            x_df = pd.DataFrame(
                radiomics_features.detach().cpu().numpy(),
                columns=self.feature_cols,
            )
        else:
            x_df = radiomics_features

        feat = self.radiomics_model.input_encoder(x_df)
        # feat = self.radiomics_model.contrastive_token(**feat)
        feat = self.radiomics_model.cls_token(**feat)
        enc = self.radiomics_model.encoder(**feat)

        z = enc[:, 0, :] ### 
        z = self.radiomics_model.projection_head(z)

        return z

    def forward(self, radiomics_features):
        z = self.encode_projection(radiomics_features)

        pred_gene = self.gene_head(z)
        pred_radiomics = self.recon_head(z)

        return {
            "pred_gene": pred_gene,
            "pred_radiomics": pred_radiomics,
            "embedding": z,
        }


# =========================================================
# PCC
# =========================================================
def compute_pcc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
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
# Dataset / Loader
# =========================================================
def build_dataset(split_csv_path: str):
    return HestRadiomicsDataset(
        bench_data_root=train.ROOT_DIR,
        split_csv_path=split_csv_path,
        gene_list_path=train.GENE_LIST_PATH,
        feature_list_path=train.FEATURE_LIST_PATH,
        radiomics_dir="radiomics_features",
    )


def build_loader(dataset, shuffle: bool):
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
def build_scratch_radiomics_model(device: torch.device):
    model = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=train.NUM_CLASS,
        hidden_dropout_prob=train.DROPOUT,
        projection_dim=train.PROJECTION_DIM,
        activation=train.ACTIVATION,
        ape_drop_rate=train.APE_DROP_RATE,
        device=device,
        num_sub_cols=[72, 36, 24],
    )

    return model.to(device)


# =========================================================
# Loss
# =========================================================
def compute_total_loss(
    outputs: dict[str, torch.Tensor],
    gene: torch.Tensor,
    radiomics: torch.Tensor,
    gene_criterion: nn.Module,
    recon_criterion: nn.Module,
    recon_lambda: float,
):
    pred_gene = outputs["pred_gene"]
    pred_radiomics = outputs["pred_radiomics"]

    gene_loss = gene_criterion(pred_gene, gene)
    recon_loss = recon_criterion(pred_radiomics, radiomics)

    total_loss = gene_loss + recon_lambda * recon_loss

    return total_loss, gene_loss, recon_loss


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gene_criterion: nn.Module,
    recon_criterion: nn.Module,
    device: torch.device,
    recon_lambda: float,
):
    model.train()

    total_loss_sum = 0.0
    gene_loss_sum = 0.0
    recon_loss_sum = 0.0

    for batch in loader:
        radiomics = batch["radiomics"].to(device)
        gene = batch["gene"].to(device)

        outputs = model(radiomics)

        total_loss, gene_loss, recon_loss = compute_total_loss(
            outputs=outputs,
            gene=gene,
            radiomics=radiomics,
            gene_criterion=gene_criterion,
            recon_criterion=recon_criterion,
            recon_lambda=recon_lambda,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        gene_loss_sum += gene_loss.item()
        recon_loss_sum += recon_loss.item()

    num_batches = len(loader)

    return {
        "total_loss": total_loss_sum / num_batches,
        "gene_loss": gene_loss_sum / num_batches,
        "recon_loss": recon_loss_sum / num_batches,
    }


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    gene_criterion: nn.Module,
    recon_criterion: nn.Module,
    device: torch.device,
    recon_lambda: float,
):
    model.eval()

    total_loss_sum = 0.0
    gene_loss_sum = 0.0
    recon_loss_sum = 0.0

    all_preds = []
    all_targets = []

    for batch in loader:
        radiomics = batch["radiomics"].to(device)
        gene = batch["gene"].to(device)

        outputs = model(radiomics)

        total_loss, gene_loss, recon_loss = compute_total_loss(
            outputs=outputs,
            gene=gene,
            radiomics=radiomics,
            gene_criterion=gene_criterion,
            recon_criterion=recon_criterion,
            recon_lambda=recon_lambda,
        )

        total_loss_sum += total_loss.item()
        gene_loss_sum += gene_loss.item()
        recon_loss_sum += recon_loss.item()

        all_preds.append(outputs["pred_gene"].cpu())
        all_targets.append(gene.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mean_pcc, _ = compute_pcc(preds, targets)

    num_batches = len(loader)

    return {
        "total_loss": total_loss_sum / num_batches,
        "gene_loss": gene_loss_sum / num_batches,
        "recon_loss": recon_loss_sum / num_batches,
        "mean_pcc": mean_pcc,
    }


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
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)

    print(f"[INFO] num_genes: {num_genes}")
    print(f"[INFO] num_radiomics_features: {num_radiomics_features}")

    radiomics_model = build_scratch_radiomics_model(device)

    gene_head = GeneHead(
        in_dim=train.PROJECTION_DIM,
        num_genes=num_genes,
        hidden_dim=512,
        dropout=0.1,
    ).to(device)

    recon_head = RadiomicsReconstructionHead(
        in_dim=train.PROJECTION_DIM,
        out_dim=num_radiomics_features,
        hidden_dim=512,
        dropout=0.1,
    ).to(device)

    model = RadiomicsGeneReconPredictor(
        radiomics_model=radiomics_model,
        gene_head=gene_head,
        recon_head=recon_head,
        feature_cols=RADIOMICS_FEATURES_NAMES,
    ).to(device)

    gene_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    # Auxiliary reconstruction loss weight.
    # Start small because the main target is gene prediction.
    recon_lambda = getattr(train, "RECON_LAMBDA", 0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    save_dir = os.path.join(train.CHECKPOINT_PATH, "scratch_radtranstab_gene_recon")
    os.makedirs(save_dir, exist_ok=True)

    best_pcc = -1.0
    best_record = None

    num_epochs = 50 # getattr(constants, "EPOCHS", 50)

    for epoch in tqdm(range(num_epochs), desc="scratch_radtranstab_gene_recon"):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            gene_criterion=gene_criterion,
            recon_criterion=recon_criterion,
            device=device,
            recon_lambda=recon_lambda,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            gene_criterion=gene_criterion,
            recon_criterion=recon_criterion,
            device=device,
            recon_lambda=recon_lambda,
        )

        val_pcc = val_metrics["mean_pcc"]

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            best_record = {
                "epoch": epoch,
                "recon_lambda": recon_lambda,
                "train_total_loss": train_metrics["total_loss"],
                "train_gene_loss": train_metrics["gene_loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "val_total_loss": val_metrics["total_loss"],
                "val_gene_loss": val_metrics["gene_loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_pcc": val_pcc,
            }

            save_path = os.path.join(
                save_dir,
                "best_scratch_radtranstab_gene_recon_model.pt",
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_record": best_record,
                    "recon_lambda": recon_lambda,
                    "feature_cols": RADIOMICS_FEATURES_NAMES,
                },
                save_path,
            )

            print(f"[INFO] saved best model: {save_path}")

        print(
            f"[INFO] Epoch {epoch}: "
            f"train_total={train_metrics['total_loss']:.4f}, "
            f"train_gene={train_metrics['gene_loss']:.4f}, "
            f"train_recon={train_metrics['recon_loss']:.4f}, "
            f"val_total={val_metrics['total_loss']:.4f}, "
            f"val_gene={val_metrics['gene_loss']:.4f}, "
            f"val_recon={val_metrics['recon_loss']:.4f}, "
            f"val_PCC={val_pcc:.4f}, "
            f"best_PCC={best_pcc:.4f}"
        )

    print("\n========== Final Result ==========")
    print(best_record)


if __name__ == "__main__":
    main()

