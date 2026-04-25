from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from hest.bench.st_dataset import H5PatchDataset, load_adata
from radtranstab.models.build_transtab import build_contrastive_learner


@dataclass
class Config:
    parquet_dir: str
    gene_list_json: str
    bench_data_root: str
    output_dir: str

    barcode_col: str = "barcode"
    sample_id_col: str = "sample_id"
    label_col: str = "target_label"
    label_to_sample_id: Optional[dict[int, str]] = None

    meta_columns: Optional[list[str]] = None
    exclude_prefixes: Optional[list[str]] = None
    radiomics_valid_prefixes: Optional[list[str]] = None
    exclude_keywords: Optional[list[str]] = None

    batch_size: int = 256
    eval_batch_size: int = 512
    num_workers: int = 0
    max_epochs: int = 50
    patience: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5

    embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1

    num_partition: int = 3
    overlap_ratio: float = 0.5
    supervised: bool = False
    ignore_duplicate_cols: bool = True

    use_one_label_for_val: bool = True
    val_label_shift: int = 1
    random_seed: int = 42

    device: str = "cuda:0"
    normalize_gene: bool = True

    resume_pretrained: bool = True
    skip_pretrain_if_exists: bool = True
    start_fold: int = 0
    end_fold: Optional[int] = None

    # multitask pretrain
    pretrain_gene_loss_weight: float = 0.1
    pretrain_gene_warmup_epochs: int = 0
    pretrain_temperature: float = 0.1
    pretrain_gene_use_both_views: bool = True
    pretrain_checkpoint_name: str = "pretrain_multitask.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(cfg_dict: dict) -> Config:
    paths = cfg_dict["paths"]
    data = cfg_dict["data"]
    train_cfg = cfg_dict["train"]
    model_cfg = cfg_dict["model"]
    cv_cfg = cfg_dict["cv"]
    runtime = cfg_dict["runtime"]

    raw_label_to_sample_id = data.get("label_to_sample_id", None)
    label_to_sample_id = None
    if raw_label_to_sample_id is not None:
        label_to_sample_id = {int(k): str(v) for k, v in raw_label_to_sample_id.items()}

    return Config(
        parquet_dir=paths["parquet_dir"],
        gene_list_json=paths["gene_list_json"],
        bench_data_root=paths["bench_data_root"],
        output_dir=paths["output_dir"],
        barcode_col=data.get("barcode_col", "barcode"),
        sample_id_col=data.get("sample_id_col", "sample_id"),
        label_col=data.get("label_col", "target_label"),
        label_to_sample_id=label_to_sample_id,
        meta_columns=data.get("meta_columns", []),
        exclude_prefixes=data.get("exclude_prefixes", []),
        radiomics_valid_prefixes=data.get("radiomics_valid_prefixes", []),
        exclude_keywords=data.get("exclude_keywords", []),
        batch_size=int(train_cfg.get("batch_size", 256)),
        eval_batch_size=int(train_cfg.get("eval_batch_size", 512)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        max_epochs=int(train_cfg.get("max_epochs", 50)),
        patience=int(train_cfg.get("patience", 8)),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_partition=int(model_cfg.get("num_partition", 3)),
        overlap_ratio=float(model_cfg.get("overlap_ratio", 0.5)),
        supervised=bool(model_cfg.get("supervised", False)),
        ignore_duplicate_cols=bool(model_cfg.get("ignore_duplicate_cols", True)),
        use_one_label_for_val=bool(cv_cfg.get("use_one_label_for_val", True)),
        val_label_shift=int(cv_cfg.get("val_label_shift", 1)),
        random_seed=int(cv_cfg.get("random_seed", 42)),
        device=runtime.get("device", "cuda:0"),
        normalize_gene=bool(runtime.get("normalize_gene", True)),
        resume_pretrained=bool(train_cfg.get("resume_pretrained", True)),
        skip_pretrain_if_exists=bool(train_cfg.get("skip_pretrain_if_exists", True)),
        start_fold=int(train_cfg.get("start_fold", 0)),
        end_fold=train_cfg.get("end_fold", None),
        pretrain_gene_loss_weight=float(train_cfg.get("pretrain_gene_loss_weight", 0.1)),
        pretrain_gene_warmup_epochs=int(train_cfg.get("pretrain_gene_warmup_epochs", 0)),
        pretrain_temperature=float(train_cfg.get("pretrain_temperature", 0.1)),
        pretrain_gene_use_both_views=bool(train_cfg.get("pretrain_gene_use_both_views", True)),
        pretrain_checkpoint_name=str(train_cfg.get("pretrain_checkpoint_name", "pretrain_multitask.pt")),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_gene_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "genes" not in payload:
        raise KeyError(f"'genes' key not found in {path}")
    return payload["genes"]


def is_feature_column(
    col: str,
    meta_columns: set[str],
    exclude_prefixes: tuple[str, ...],
    valid_prefixes: tuple[str, ...],
    exclude_keywords: tuple[str, ...],
) -> bool:
    lowered = col.lower()
    if lowered in meta_columns:
        return False
    for prefix in exclude_prefixes:
        if lowered.startswith(prefix):
            return False
    for kw in exclude_keywords:
        if kw in lowered:
            return False
    return any(lowered.startswith(prefix) for prefix in valid_prefixes)


def infer_feature_columns_from_parquet(df: pd.DataFrame, cfg: Config) -> list[str]:
    meta_columns = {c.lower() for c in (cfg.meta_columns or [])}
    exclude_prefixes = tuple(p.lower() for p in (cfg.exclude_prefixes or []))
    exclude_keywords = tuple(p.lower() for p in (cfg.exclude_keywords or []))
    valid_prefixes = tuple(p.lower() for p in (cfg.radiomics_valid_prefixes or []))

    feature_cols = []
    for c in df.columns:
        if is_feature_column(c, meta_columns, exclude_prefixes, valid_prefixes, exclude_keywords):
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)

    if not feature_cols:
        raise ValueError("No valid feature columns found from parquet.")
    return feature_cols


def load_slide_barcodes(patches_h5_path: str) -> list[str]:
    patch_dataset = H5PatchDataset(patches_h5_path)
    slide_barcodes: list[str] = []
    for i in range(len(patch_dataset)):
        chunk = patch_dataset[i]
        chunk_barcodes = chunk["barcodes"]
        if isinstance(chunk_barcodes, torch.Tensor):
            chunk_barcodes = chunk_barcodes.numpy()
        if isinstance(chunk_barcodes, (bytes, str)):
            chunk_barcodes = [chunk_barcodes]

        for barcode in chunk_barcodes:
            slide_barcodes.append(barcode.decode("utf-8") if isinstance(barcode, bytes) else str(barcode))
    return slide_barcodes


def build_expression_df_for_slide(
    bench_data_root: str,
    sample_id: str,
    genes: list[str],
    normalize_gene: bool,
) -> pd.DataFrame:
    patches_h5_path = os.path.join(bench_data_root, "patches", f"{sample_id}.h5")
    expr_path = os.path.join(bench_data_root, "adata", f"{sample_id}.h5ad")

    if not os.path.isfile(patches_h5_path):
        raise FileNotFoundError(f"Patch file not found: {patches_h5_path}")
    if not os.path.isfile(expr_path):
        raise FileNotFoundError(f"Expr file not found: {expr_path}")

    slide_barcodes = load_slide_barcodes(patches_h5_path)
    adata_df = load_adata(expr_path, genes=genes, barcodes=slide_barcodes, normalize=normalize_gene)
    adata_df = adata_df.copy()
    adata_df["barcode"] = list(adata_df.index.astype(str))
    return adata_df.reset_index(drop=True)


def load_tabular_df_from_sample_parquet(
    parquet_path: str,
    sample_id: str,
    label: int,
    cfg: Config,
    feature_cols_ref: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(parquet_path).copy()
    df.columns = [c.lower() for c in df.columns]

    if cfg.barcode_col not in df.columns:
        raise KeyError(f"{cfg.barcode_col} not found in parquet: {parquet_path}")

    feature_cols = infer_feature_columns_from_parquet(df, cfg)
    if feature_cols_ref is not None:
        ref_set = set(feature_cols_ref)
        cur_set = set(feature_cols)
        if ref_set != cur_set:
            missing = sorted(ref_set - cur_set)
            extra = sorted(cur_set - ref_set)
            raise ValueError(
                f"Feature columns mismatch in {sample_id}\nMissing: {missing[:10]}\nExtra: {extra[:10]}"
            )

    out = df[[cfg.barcode_col] + feature_cols].copy()
    out[cfg.sample_id_col] = sample_id
    out[cfg.label_col] = label
    out[cfg.barcode_col] = out[cfg.barcode_col].astype(str)
    return out, feature_cols


def build_joint_dataframe(cfg: Config) -> tuple[pd.DataFrame, list[str], list[str]]:
    if cfg.label_to_sample_id is None:
        raise ValueError("data.label_to_sample_id must be provided for parquet-based loading.")

    genes = read_gene_list(cfg.gene_list_json)
    all_rows: list[pd.DataFrame] = []
    feature_cols_ref: Optional[list[str]] = None

    for label, sample_id in sorted(cfg.label_to_sample_id.items(), key=lambda x: x[0]):
        parquet_path = os.path.join(cfg.parquet_dir, f"{sample_id}.parquet")
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        slide_tab, feature_cols = load_tabular_df_from_sample_parquet(
            parquet_path=parquet_path,
            sample_id=sample_id,
            label=label,
            cfg=cfg,
            feature_cols_ref=feature_cols_ref,
        )
        if feature_cols_ref is None:
            feature_cols_ref = feature_cols

        expr_df = build_expression_df_for_slide(
            bench_data_root=cfg.bench_data_root,
            sample_id=sample_id,
            genes=genes,
            normalize_gene=cfg.normalize_gene,
        )
        expr_df["barcode"] = expr_df["barcode"].astype(str)

        merged = slide_tab.merge(
            expr_df,
            how="inner",
            left_on=cfg.barcode_col,
            right_on="barcode",
            suffixes=("", "_expr"),
        )
        if merged.empty:
            raise ValueError(f"No matched rows after barcode merge for sample_id={sample_id}")

        all_rows.append(merged)
        print(
            f"[build_joint_dataframe] sample_id={sample_id} "
            f"parquet_rows={len(slide_tab)} expr_rows={len(expr_df)} matched={len(merged)}"
        )

    if feature_cols_ref is None:
        raise ValueError("No features found from parquet files.")

    joint_df = pd.concat(all_rows, axis=0, ignore_index=True)
    gene_cols = genes

    print("before dropna:", len(joint_df))
    print("rows with any feature NaN:", joint_df[feature_cols_ref].isna().any(axis=1).sum())
    print("rows with any gene NaN:", joint_df[gene_cols].isna().any(axis=1).sum())
    print("rows with any feature/gene NaN:", joint_df[feature_cols_ref + gene_cols].isna().any(axis=1).sum())
    print(joint_df[feature_cols_ref + gene_cols].isna().sum().sort_values(ascending=False).head(30))

    joint_df = joint_df.replace([np.inf, -np.inf], np.nan)
    joint_df = joint_df.dropna(subset=feature_cols_ref + gene_cols).reset_index(drop=True)

    print(f"[build_joint_dataframe] final rows={len(joint_df)}")
    print(f"[build_joint_dataframe] num_features={len(feature_cols_ref)} num_genes={len(gene_cols)}")
    return joint_df, feature_cols_ref, gene_cols


def build_leave_one_label_out_folds(
    df: pd.DataFrame,
    label_col: str,
    use_one_label_for_val: bool,
    val_label_shift: int,
) -> list[dict]:
    labels = sorted(df[label_col].unique().tolist())
    folds: list[dict] = []
    for test_label in labels:
        remaining = [x for x in labels if x != test_label]
        if use_one_label_for_val and len(remaining) >= 2:
            idx = val_label_shift % len(remaining)
            val_label = remaining[idx]
            train_labels = [x for x in remaining if x != val_label]
        else:
            val_label = remaining[0]
            train_labels = remaining[1:]
        folds.append({"test_label": test_label, "val_label": val_label, "train_labels": train_labels})
    return folds


class TabularGeneDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], gene_cols: list[str]) -> None:
        self.x = df[feature_cols].reset_index(drop=True).copy()
        self.y = df[gene_cols].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x.iloc[idx], self.y[idx]


class EmbeddingGeneDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class GeneHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskPretrainModel(nn.Module):
    def __init__(self, base_model: nn.Module, gene_out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.base_model = base_model
        self.gene_head = GeneHead(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=gene_out_dim, dropout=dropout)

    def encode_cls(self, x_df: pd.DataFrame) -> torch.Tensor:
        encoded = self.base_model.input_encoder(x_df)
        encoded = self.base_model.cls_token(**encoded)
        z = self.base_model.encoder(**encoded)
        if not isinstance(z, torch.Tensor):
            raise RuntimeError(f"Unexpected encoder output type: {type(z)}")
        return z[:, 0, :]

    def project(self, cls_emb: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, "projection_head"):
            return self.base_model.projection_head(cls_emb)
        return cls_emb

    def gene_predict(self, cls_emb: torch.Tensor) -> torch.Tensor:
        return self.gene_head(cls_emb)


class PretrainedEmbeddingExtractor(nn.Module):
    def __init__(self, feature_cols: list[str], checkpoint_dir: str, cfg: Config, device: str) -> None:
        super().__init__()
        self.feature_cols = feature_cols
        self.device_name = device
        self.model, _ = build_contrastive_learner(
            categorical_columns=[],
            numerical_columns=feature_cols,
            binary_columns=[],
            supervised=cfg.supervised,
            num_partition=cfg.num_partition,
            overlap_ratio=cfg.overlap_ratio,
            hidden_dim=cfg.embedding_dim,
            projection_dim=cfg.embedding_dim,
            device=device,
        )
        ckpt_path = Path(checkpoint_dir) / cfg.pretrain_checkpoint_name
        if ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(payload["model_state_dict"], strict=False)
        else:
            # fallback to original TransTab directory checkpoint layout if present
            self.model, _ = build_contrastive_learner(
                categorical_columns=[],
                numerical_columns=feature_cols,
                binary_columns=[],
                supervised=cfg.supervised,
                num_partition=cfg.num_partition,
                overlap_ratio=cfg.overlap_ratio,
                hidden_dim=cfg.embedding_dim,
                projection_dim=cfg.embedding_dim,
                device=device,
                checkpoint=checkpoint_dir,
            )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, x_df: pd.DataFrame) -> torch.Tensor:
        encoded = self.model.input_encoder(x_df)
        encoded = self.model.cls_token(**encoded)
        z = self.model.encoder(**encoded)
        if not isinstance(z, torch.Tensor):
            raise RuntimeError(f"Unexpected encoder output: {type(z)}")
        return z[:, 0, :].to(self.device_name, dtype=torch.float32)


def batch_collate_rows(rows):
    x_df = pd.DataFrame([r[0] for r in rows])
    y = torch.tensor(np.stack([r[1] for r in rows], axis=0), dtype=torch.float32)
    return x_df, y


def make_masked_view(df: pd.DataFrame, feature_cols: list[str], num_partition: int, overlap_ratio: float) -> pd.DataFrame:
    n_cols = len(feature_cols)
    base_size = max(1, math.ceil(n_cols / max(num_partition, 2)))
    overlap = max(0, int(round(base_size * overlap_ratio)))
    keep_size = min(n_cols, base_size + overlap)
    keep_cols = random.sample(feature_cols, keep_size)

    view = df.copy()
    mask_cols = [c for c in feature_cols if c not in keep_cols]
    if mask_cols:
        view.loc[:, mask_cols] = np.nan

    if keep_cols:
        noise = np.random.normal(loc=0.0, scale=0.01, size=(len(view), len(keep_cols)))
        view.loc[:, keep_cols] = view.loc[:, keep_cols].to_numpy(dtype=np.float32) + noise
    return view


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_12 + loss_21)


def compute_gene_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    pcc_list = []
    for g in range(true.shape[1]):
        a = pred[:, g]
        b = true[:, g]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        pcc = np.corrcoef(a, b)[0, 1]
        if np.isfinite(pcc):
            pcc_list.append(float(pcc))
    mean_pcc = float(np.mean(pcc_list)) if pcc_list else float("nan")
    return {"mse": mse, "mae": mae, "mean_pcc": mean_pcc}


def run_multitask_pretrain_epoch(
    model: MultiTaskPretrainModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    feature_cols: list[str],
    cfg: Config,
    device: str,
    epoch: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_total = 0.0
    total_ctr = 0.0
    total_gene = 0.0
    total_count = 0

    for x_df, y_gene in loader:
        x_df = x_df[feature_cols].reset_index(drop=True)
        y_gene = y_gene.to(device)

        view1 = make_masked_view(x_df, feature_cols, cfg.num_partition, cfg.overlap_ratio)
        view2 = make_masked_view(x_df, feature_cols, cfg.num_partition, cfg.overlap_ratio)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        cls1 = model.encode_cls(view1).to(device)
        cls2 = model.encode_cls(view2).to(device)
        proj1 = model.project(cls1)
        proj2 = model.project(cls2)
        loss_ctr = nt_xent_loss(proj1, proj2, cfg.pretrain_temperature)

        if cfg.pretrain_gene_use_both_views:
            gene_pred = 0.5 * (model.gene_predict(cls1) + model.gene_predict(cls2))
        else:
            gene_pred = model.gene_predict(cls1)
        loss_gene = F.mse_loss(gene_pred, y_gene)

        gene_weight = cfg.pretrain_gene_loss_weight if epoch >= cfg.pretrain_gene_warmup_epochs else 0.0
        loss = loss_ctr + gene_weight * loss_gene

        if is_train:
            loss.backward()
            optimizer.step()

        bs = y_gene.size(0)
        total_total += float(loss.item()) * bs
        total_ctr += float(loss_ctr.item()) * bs
        total_gene += float(loss_gene.item()) * bs
        total_count += bs

    denom = max(total_count, 1)
    return {
        "total_loss": total_total / denom,
        "contrastive_loss": total_ctr / denom,
        "gene_loss": total_gene / denom,
    }


def save_pretrain_checkpoint(model: MultiTaskPretrainModel, pretrain_dir: Path, cfg: Config, feature_cols: list[str], gene_cols: list[str]) -> None:
    ckpt_path = pretrain_dir / cfg.pretrain_checkpoint_name
    payload = {
        "model_state_dict": model.base_model.state_dict(),
        "gene_head_state_dict": model.gene_head.state_dict(),
        "feature_cols": feature_cols,
        "gene_cols": gene_cols,
        "embedding_dim": cfg.embedding_dim,
    }
    torch.save(payload, ckpt_path)



def has_pretrain_checkpoint(pretrain_dir: Path, cfg: Config) -> bool:
    candidates = [
        pretrain_dir / cfg.pretrain_checkpoint_name,
        pretrain_dir / "pytorch_model.bin",
        pretrain_dir / "model.safetensors",
        pretrain_dir / "config.json",
    ]
    return any(p.exists() for p in candidates)


def pretrain_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    gene_cols: list[str],
    cfg: Config,
    pretrain_dir: Path,
    device: str,
):
    base_model, _ = build_contrastive_learner(
        categorical_columns=[],
        numerical_columns=feature_cols,
        binary_columns=[],
        supervised=cfg.supervised,
        num_partition=cfg.num_partition,
        overlap_ratio=cfg.overlap_ratio,
        hidden_dim=cfg.embedding_dim,
        projection_dim=cfg.embedding_dim,
        device=device,
    )
    model = MultiTaskPretrainModel(
        base_model=base_model,
        gene_out_dim=len(gene_cols),
        hidden_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
    ).to(device)

    train_loader = DataLoader(
        TabularGeneDataset(train_df, feature_cols, gene_cols),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=batch_collate_rows,
    )
    val_loader = DataLoader(
        TabularGeneDataset(val_df, feature_cols, gene_cols),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=batch_collate_rows,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = math.inf
    best_epoch = -1
    patience_count = 0
    history = []

    for epoch in range(cfg.max_epochs):
        tr = run_multitask_pretrain_epoch(model, train_loader, optimizer, feature_cols, cfg, device, epoch)
        va = run_multitask_pretrain_epoch(model, val_loader, None, feature_cols, cfg, device, epoch)
        row = {
            "epoch": epoch,
            "train_total_loss": tr["total_loss"],
            "train_contrastive_loss": tr["contrastive_loss"],
            "train_gene_loss": tr["gene_loss"],
            "val_total_loss": va["total_loss"],
            "val_contrastive_loss": va["contrastive_loss"],
            "val_gene_loss": va["gene_loss"],
        }
        history.append(row)
        print(
            f"[pretrain] epoch={epoch:03d} "
            f"train_total={tr['total_loss']:.6f} train_ctr={tr['contrastive_loss']:.6f} train_gene={tr['gene_loss']:.6f} "
            f"val_total={va['total_loss']:.6f} val_ctr={va['contrastive_loss']:.6f} val_gene={va['gene_loss']:.6f}"
        )

        if va["total_loss"] < best_val:
            best_val = va["total_loss"]
            best_epoch = epoch
            patience_count = 0
            save_pretrain_checkpoint(model, pretrain_dir, cfg, feature_cols, gene_cols)
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"[pretrain] early stopping at epoch={epoch}")
                break

    pd.DataFrame(history).to_csv(pretrain_dir / "pretrain_history.csv", index=False)
    with open(pretrain_dir / "pretrain_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "best_val_total_loss": best_val}, f, indent=2)


@torch.no_grad()
def compute_embeddings_df(
    extractor: PretrainedEmbeddingExtractor,
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int,
) -> np.ndarray:
    extractor.eval()
    x_all = df[feature_cols].reset_index(drop=True)
    chunks = []
    for start in range(0, len(x_all), batch_size):
        sub = x_all.iloc[start:start + batch_size]
        emb = extractor.extract(sub)
        chunks.append(emb.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run_head_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_count = 0
    for emb, y in loader:
        emb = emb.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(emb)
        loss = loss_fn(pred, y)
        if is_train:
            loss.backward()
            optimizer.step()
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_head(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_list, true_list = [], []
    for emb, y in loader:
        emb = emb.to(device)
        y = y.to(device)
        pred = model(emb)
        pred_list.append(pred.detach().cpu().numpy())
        true_list.append(y.detach().cpu().numpy())
    return np.concatenate(pred_list, axis=0), np.concatenate(true_list, axis=0)


def fit_fold(
    fold_id: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    gene_cols: list[str],
    cfg: Config,
    fold_dir: Path,
) -> dict:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    pretrain_dir = ensure_dir(fold_dir / "pretrain_ckpt")

    should_run_pretrain = True
    if cfg.resume_pretrained and cfg.skip_pretrain_if_exists and has_pretrain_checkpoint(pretrain_dir, cfg):
        should_run_pretrain = False

    if should_run_pretrain:
        print(f"[fold {fold_id}] pretraining multitask contrastive learner...")
        pretrain_one_fold(train_df, val_df, feature_cols, gene_cols, cfg, pretrain_dir, device)
    else:
        print(f"[fold {fold_id}] skip pretraining, reuse checkpoint: {pretrain_dir}")

    print(f"[fold {fold_id}] extracting pretrained embeddings...")
    extractor = PretrainedEmbeddingExtractor(feature_cols, str(pretrain_dir), cfg, device)

    train_emb = compute_embeddings_df(extractor, train_df, feature_cols, cfg.eval_batch_size)
    val_emb = compute_embeddings_df(extractor, val_df, feature_cols, cfg.eval_batch_size)
    test_emb = compute_embeddings_df(extractor, test_df, feature_cols, cfg.eval_batch_size)

    train_y = train_df[gene_cols].to_numpy(dtype=np.float32)
    val_y = val_df[gene_cols].to_numpy(dtype=np.float32)
    test_y = test_df[gene_cols].to_numpy(dtype=np.float32)

    train_loader = DataLoader(EmbeddingGeneDataset(train_emb, train_y), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=False)
    val_loader = DataLoader(EmbeddingGeneDataset(val_emb, val_y), batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)
    test_loader = DataLoader(EmbeddingGeneDataset(test_emb, test_y), batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

    gene_head = GeneHead(in_dim=train_emb.shape[1], hidden_dim=cfg.hidden_dim, out_dim=len(gene_cols), dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(gene_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = math.inf
    best_epoch = -1
    patience_count = 0
    best_ckpt_path = fold_dir / "best_gene_head.pt"
    history = []

    for epoch in range(cfg.max_epochs):
        train_loss = run_head_epoch(gene_head, train_loader, optimizer, device)
        val_loss = run_head_epoch(gene_head, val_loader, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[fold {fold_id}] epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": gene_head.state_dict(),
                    "embedding_dim": int(train_emb.shape[1]),
                    "gene_cols": gene_cols,
                },
                best_ckpt_path,
            )
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"[fold {fold_id}] early stopping at epoch={epoch}")
                break

    pd.DataFrame(history).to_csv(fold_dir / "gene_head_history.csv", index=False)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    gene_head.load_state_dict(ckpt["model_state_dict"])

    val_pred, val_true = predict_head(gene_head, val_loader, device)
    test_pred, test_true = predict_head(gene_head, test_loader, device)

    val_metrics = compute_gene_metrics(val_pred, val_true)
    test_metrics = compute_gene_metrics(test_pred, test_true)

    np.save(fold_dir / "train_emb.npy", train_emb)
    np.save(fold_dir / "val_emb.npy", val_emb)
    np.save(fold_dir / "test_emb.npy", test_emb)
    np.save(fold_dir / "val_pred.npy", val_pred)
    np.save(fold_dir / "val_true.npy", val_true)
    np.save(fold_dir / "test_pred.npy", test_pred)
    np.save(fold_dir / "test_true.npy", test_true)

    summary = {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "embedding_dim": int(train_emb.shape[1]),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    with open(fold_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    cfg_dict = load_yaml(args.config)
    cfg = build_config(cfg_dict)

    seed_everything(cfg.random_seed)
    mp.set_sharing_strategy("file_descriptor")
    output_dir = ensure_dir(cfg.output_dir)

    with open(output_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(output_dir / "config_raw.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

    joint_df, feature_cols, gene_cols = build_joint_dataframe(cfg)
    joint_df.to_csv(output_dir / "joint_aligned_dataset.csv", index=False)

    folds = build_leave_one_label_out_folds(
        joint_df,
        label_col=cfg.label_col,
        use_one_label_for_val=cfg.use_one_label_for_val,
        val_label_shift=cfg.val_label_shift,
    )

    results = []
    for fold_id, fold in enumerate(folds):
        if fold_id < cfg.start_fold:
            continue
        if cfg.end_fold is not None and fold_id > cfg.end_fold:
            continue

        test_label = fold["test_label"]
        val_label = fold["val_label"]
        train_labels = fold["train_labels"]

        train_df = joint_df[joint_df[cfg.label_col].isin(train_labels)].reset_index(drop=True)
        val_df = joint_df[joint_df[cfg.label_col] == val_label].reset_index(drop=True)
        test_df = joint_df[joint_df[cfg.label_col] == test_label].reset_index(drop=True)

        fold_dir = ensure_dir(output_dir / f"fold_{fold_id}_test{test_label}_val{val_label}")

        print("=" * 80)
        print(f"[fold {fold_id}] train_labels={train_labels} val_label={val_label} test_label={test_label}")
        print(f"[fold {fold_id}] n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)}")

        summary = fit_fold(
            fold_id=fold_id,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            gene_cols=gene_cols,
            cfg=cfg,
            fold_dir=fold_dir,
        )
        summary.update({"train_labels": train_labels, "val_label": val_label, "test_label": test_label})
        results.append(summary)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "cv_results.csv", index=False)

    agg = {
        "mean_test_mse": float(results_df["test_mse"].mean()),
        "std_test_mse": float(results_df["test_mse"].std(ddof=0)),
        "mean_test_mae": float(results_df["test_mae"].mean()),
        "std_test_mae": float(results_df["test_mae"].std(ddof=0)),
        "mean_test_mean_pcc": float(results_df["test_mean_pcc"].mean()),
        "std_test_mean_pcc": float(results_df["test_mean_pcc"].std(ddof=0)),
    }
    with open(output_dir / "cv_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print("[done] cross-validation finished")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
