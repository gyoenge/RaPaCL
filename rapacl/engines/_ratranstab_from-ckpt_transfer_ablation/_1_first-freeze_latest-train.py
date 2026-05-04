from __future__ import annotations

import os 
from typing import Any 
import numpy as np 
import pandas as pd 
from scipy.stats import pearsonr
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from rapacl.data._dataset import HestRadiomicsDataset
from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES
from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import set_seed
import rapacl.configs.default.train as CONSTANTS 

NUM_CELLTYPE_CLASS = 5 # 고정 
NUM_SUB_COLS = [72, 54, 36, 18, 9, 3, 1] # 실험해볼거 ################################ 
PROJECTION_DIM = 384
NUM_RADIOMICS = len(RADIOMICS_FEATURES_NAMES)
NUM_GENES = 250 


def build_radiomics_model_from_ckpt(device: torch.device):
    """
    from ckpt except clf (classifier)
    """

    model = build_radiomics_learner(
        checkpoint=None, # for loading with excluding clf
        numerical_columns=RADIOMICS_FEATURES_NAMES, 
        num_class=NUM_CELLTYPE_CLASS, 
        projection_dim=PROJECTION_DIM, 
        activation="leakyrelu",
        ape_drop_rate=0.0, # exclude APE 
        num_sub_cols=NUM_SUB_COLS, 
        device=device, 
    ).to(device)

    # load ckpt except clf 
    ckpt_path = "/root/workspace/RaPaCL/rapacl/checkpoints/radiomics_retrieval/transtab/pytorch_model.bin"
    if ckpt_path is not None: 
        print(f"[INFO] Load RadTransTab checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)

    filtered_k = "clf."
    filtered = {
        k: v for k, v in state_dict.items()
        if not k.startswith(filtered_k)
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print(f"[INFO] Loaded RadTransTab, ")
    print(f"[INFO] Skipped keys: {filtered_k}*")
    print(f"[INFO] Missing keys: {missing}")
    print(f"[INFO] Unexpected keys: {unexpected}")
    
    return model 


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RadTransTabGenePredModel(nn.Module):
    def __init__(self,
        device: torch.device, 
    ):
        super().__init__()

        self.radiomics_model = build_radiomics_model_from_ckpt(
            device=device
        ).to(device) # use only backbone ? 

        self.recon_head = MLPHead(
            in_dim=128, # radtranstab hidden dim  # PROJECTION_DIM,
            out_dim=NUM_RADIOMICS,
            hidden_dim=512, 
            dropout=0.1,
        ).to(device)
        self.cls_head = MLPHead(
            in_dim=128, # radtranstab hidden dim  
            out_dim=NUM_CELLTYPE_CLASS, 
            hidden_dim=256, 
            dropout=0.1, 
        ).to(device) 
        self.gene_head = MLPHead(
            in_dim=PROJECTION_DIM,
            out_dim=NUM_GENES,
            hidden_dim=512, 
            dropout=0.1,
        ).to(device) 
        # radtranstab (backbone) -> projhead>recon / cls>cls / genehead>gene

    def _encode_radiomics(self, radiomics: torch.Tensor | pd.DataFrame):
        if isinstance(radiomics, pd.DataFrame):
            x_df = radiomics 
        else: 
            x_df = pd.DataFrame(radiomics.detach().cpu().numpy(), columns=RADIOMICS_FEATURES_NAMES)

        feat = self.radiomics_model.input_encoder(x_df)
        feat = self.radiomics_model.contrastive_token(**feat)
        feat = self.radiomics_model.cls_token(**feat)
        enc = self.radiomics_model.encoder(**feat)
        
        rad_cls_h = enc[:, 0, :]
        rad_contrast_h = enc[:, 1, :]
        rad_contrast_z = self.radiomics_model.projection_head(rad_contrast_h)

        return {
            "rad_cls_h": rad_cls_h,
            "rad_contrast_h": rad_contrast_h, 
            "rad_contrast_z": rad_contrast_z, 
        }

    def forward(self, radiomics: torch.Tensor | pd.DataFrame):
        rad = self._encode_radiomics(radiomics) 

        pred_radiomics = self.recon_head(rad["rad_contrast_h"])
        pred_class_logits = self.cls_head(rad["rad_cls_h"])
        pred_gene = self.gene_head(rad["rad_contrast_z"])

        return {
            **rad, 
            "pred_radiomics": pred_radiomics,
            "pred_class_logits": pred_class_logits, 
            "pred_gene": pred_gene, 
        }


# Loss
def symmetric_info_nce(
    a: torch.Tensor, 
    b: torch.Tensor, 
    temperature: float=0.7, 
)-> torch.Tensor: 
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.t() / temperature 
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) # bi-directional 

# Metric
def __compute_genewise_pcc_pearsonr(
    pred: torch.Tensor, 
    target: torch.Tensor, 
): 
    pred = pred.detach().float().cpu().numpy() 
    target = target.detach().float().cpu().numpy() 

    num_genes = pred.shape[1]
    pcc_list = []

    for i in range(num_genes):
        r, _ = pearsonr(pred[:, i], pred[:, i])
        pcc_list.append(r)
    
    pcc_array = np.array(pcc_list)
    return pcc_array.mean().item(), pcc_array 
    

def compute_genewise_pcc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    denom = torch.sqrt((pred_c ** 2).sum(dim=0) * (target_c ** 2).sum(dim=0)) + eps
    pcc_per_gene = (pred_c * target_c).sum(dim=0) / denom
    return pcc_per_gene.mean().item(), pcc_per_gene.numpy()


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == target).float().mean().item()


# Data helper 
def get_batch_tensor(
    batch: dict[str, Any], 
    names: tuple[str, ...], 
    device: torch.device, 
)-> torch.Tensor: 
    for name in names: 
        if name in batch: 
            return batch[name].to(device, non_blocking=True).float() 
    raise KeyError(f"None of keys {names} found in batch. Available keys: {list(batch.keys())}")


# Train epoch
def train_epoch(
    model, 
    loader, 
    optimizer, 
    device, 
): 
    model.train() # layer 별 freezing 추가하기 ###### 

    # 나중에 Loss 별 weighting 추가하기 ###### 
    temperature = 0.07 

    metrics = {"loss": 0.0, "loss-recon": 0.0, "loss-cls": 0.0, "loss-gene": 0.0, "cls-acc": 0.0,}
    for batch in tqdm(loader, desc="Train", leave=False):
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        target_label = get_batch_tensor(
            batch,
            ("target_label", "label", "celltype_label"),
            device
        ).long()
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        out = model.forward(radiomics=radiomics)

        recon_loss = F.mse_loss(out["pred_radiomics"], radiomics)
        cls_loss = F.cross_entropy(out["pred_class_logits"], target_label)
        gene_loss = F.mse_loss(out["pred_gene"], gene)

        loss = 1 * recon_loss + 1 * cls_loss + 1 * gene_loss  # 나중에 Loss 별 weighting 추가하기 ###### 
    
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step() 

        bs = radiomics.size(0)
        metrics["loss"] += loss.item() * bs 
        metrics["loss-recon"] += recon_loss.item() * bs 
        metrics["loss-cls"] += cls_loss.item() * bs 
        metrics["loss-gene"] += gene_loss.item() * bs 
        metrics["cls-acc"] += accuracy(out["pred_class_logits"].detach(), target_label) * bs 

    n = len(loader.dataset)
    return {k: v/n for k,v in metrics.items()}


# Eval epoch 
@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
):
    model.eval()

    metrics = {
        "loss": 0.0,
        "loss-recon": 0.0,
        "loss-cls": 0.0,
        "loss-gene": 0.0,
        "cls-acc": 0.0,
    }

    preds = []
    targets = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        target_label = get_batch_tensor(batch, ("target_label", "label", "celltype_label"), device).long()
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        out = model.forward(radiomics=radiomics)

        recon_loss = F.mse_loss(out["pred_radiomics"], radiomics)
        cls_loss = F.cross_entropy(out["pred_class_logits"], target_label)
        gene_loss = F.mse_loss(out["pred_gene"], gene)

        loss = recon_loss + cls_loss + gene_loss

        bs = radiomics.size(0)
        metrics["loss"] += loss.item() * bs
        metrics["loss-recon"] += recon_loss.item() * bs
        metrics["loss-cls"] += cls_loss.item() * bs
        metrics["loss-gene"] += gene_loss.item() * bs
        metrics["cls-acc"] += accuracy(out["pred_class_logits"], target_label) * bs

        preds.append(out["pred_gene"].detach().cpu())
        targets.append(gene.detach().cpu())

    n = len(loader.dataset)
    metrics = {k: v / n for k, v in metrics.items()}

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)

    mean_pcc, pcc_per_gene = compute_genewise_pcc(pred_all, target_all)

    metrics["mean_pcc"] = mean_pcc
    metrics["pcc_per_gene"] = pcc_per_gene

    return metrics

# Main 
def main():
    set_seed(CONSTANTS.SEED)
    device = torch.device(CONSTANTS.DEVICE)
    print(f"[INFO] device: {device}")

    trainset = HestRadiomicsDataset(
        bench_data_root=CONSTANTS.ROOT_DIR,
        split_csv_path=CONSTANTS.TRAIN_SPLIT_CSV,
        gene_list_path=CONSTANTS.GENE_LIST_PATH,
        feature_list_path=CONSTANTS.FEATURE_LIST_PATH,
        radiomics_dir=getattr(CONSTANTS, "RADIOMICS_DIR", "radiomics_features"),
    )
    evalset = HestRadiomicsDataset(
        bench_data_root=CONSTANTS.ROOT_DIR,
        split_csv_path=CONSTANTS.VAL_SPLIT_CSV,
        gene_list_path=CONSTANTS.GENE_LIST_PATH,
        feature_list_path=CONSTANTS.FEATURE_LIST_PATH,
        radiomics_dir=getattr(CONSTANTS, "RADIOMICS_DIR", "radiomics_features"),
    )
    train_loader = DataLoader(
        trainset,
        batch_size=CONSTANTS.BATCH_SIZE,
        shuffle=True,
        num_workers=CONSTANTS.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        evalset,
        batch_size=CONSTANTS.BATCH_SIZE,
        shuffle=False,
        num_workers=CONSTANTS.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[INFO] train samples: {len(trainset)}")
    print(f"[INFO] val samples: {len(evalset)}")
    num_genes = len(trainset.genes)
    num_radiomics_features = len(RADIOMICS_FEATURES_NAMES)
    print(f"[INFO] num_genes: {num_genes}")
    print(f"[INFO] num_radiomics_features: {num_radiomics_features}")

    model = RadTransTabGenePredModel(device)

    save_dir = os.path.join(CONSTANTS.CHECKPOINT_PATH, "1_first_freeze_latest_train")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] trained ckpt save directory: {save_dir}")

    # train & eval loop
    num_full_epochs = 20  # CONSTANTS.EPOCHS | CONSTANTS.PRETRAIN_EPOCHS
    # num_warmup_epochs = 5  # CONSTANTS.WARMUP_RECON_EPOCHS

    best_path = os.path.join(save_dir, "best.pt")
    best_val = float("inf")

    train_params = (
        list(model.radiomics_model.parameters())
        + list(model.recon_head.parameters())
        + list(model.cls_head.parameters())
    )
    train_optimizer = torch.optim.AdamW(
        train_params, 
        lr=1e-4,  # CONSTANTS.LR 
        weight_decay=1e-4,  # CONSTANTS.WEIGHT_DECAY 
    )

    for epoch in range(1,num_full_epochs+1):
        # recon_only = epoch < num_warmup_epochs 

        train_m = train_epoch(
            model=model, 
            loader=train_loader,
            optimizer=train_optimizer,
            device=device,
        )

        eval_m = eval_epoch(
            model=model, 
            loader=eval_loader, 
            device=device, 
        )

        print(
            f"[INFO][Epoch {epoch}][Train Metrics] "
            f"train_loss={train_m['loss']:.4f} "
            f"loss-recon={train_m['loss-recon']:.4f} "
            f"loss-cls={train_m['loss-cls']:.4f} "
            f"cls-acc={train_m['cls-acc']:.4f} "
            f"loss-gene={train_m['loss-gene']:.4f} "
            f"\n"
            f"[INFO][Epoch {epoch}][Eval Metraics] "
            f"val_loss={eval_m['loss']:.4f} "
            f"loss-recon={eval_m['loss-recon']:.4f} "
            f"loss-cls={eval_m['loss-cls']:.4f} "
            f"cls-acc={eval_m['cls-acc']:.4f} "
            f"loss-gene={eval_m['loss-gene']:.4f} "
            f"mean_pcc={eval_m['mean_pcc']:.4f}"
        )

        if eval_m["loss"] < best_val:
            best_val = eval_m["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": train_optimizer.state_dict(),
                    "val_metrics": eval_m,
                    "pcc_per_gene": eval_m["pcc_per_gene"],
                },
                best_path,
            )


if __name__ == "__main__":
    main()


# 여기 LARS 추가해보기 
