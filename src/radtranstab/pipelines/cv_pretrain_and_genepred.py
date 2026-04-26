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
import torch.nn as nn
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader, Dataset

from hest.bench.st_dataset import H5PatchDataset, load_adata
from radtranstab.models.build import build_contrastive_learner, train


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

    # 추가
    resume_pretrained: bool = True
    skip_pretrain_if_exists: bool = True
    start_fold: int = 0
    end_fold: Optional[int] = None


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
            if isinstance(barcode, bytes):
                barcode_str = barcode.decode("utf-8")
            else:
                barcode_str = str(barcode)
            slide_barcodes.append(barcode_str)

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
    adata_df = load_adata(
        expr_path,
        genes=genes,
        barcodes=slide_barcodes,
        normalize=normalize_gene,
    )

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
                f"Feature columns mismatch in {sample_id}\n"
                f"Missing: {missing[:10]}\n"
                f"Extra: {extra[:10]}"
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
    feature_nan_rows = joint_df[feature_cols_ref].isna().any(axis=1).sum()
    gene_nan_rows = joint_df[gene_cols].isna().any(axis=1).sum()
    either_nan_rows = joint_df[feature_cols_ref + gene_cols].isna().any(axis=1).sum()
    print("rows with any feature NaN:", feature_nan_rows)
    print("rows with any gene NaN:", gene_nan_rows)
    print("rows with any feature/gene NaN:", either_nan_rows)
    top_nan = joint_df[feature_cols_ref + gene_cols].isna().sum().sort_values(ascending=False)
    print(top_nan.head(30))

    joint_df = joint_df.replace([np.inf, -np.inf], np.nan)
    keep_cols = feature_cols_ref + gene_cols
    joint_df = joint_df.dropna(subset=keep_cols).reset_index(drop=True)

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


def make_transtab_xy(df: pd.DataFrame, feature_cols: list[str], label_col: str):
    return df[feature_cols].copy(), df[label_col].copy()


def pretrain_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
    pretrain_dir: Path,
    device: str,
):
    model_pretrain, collate_fn = build_contrastive_learner(
        cat_cols=[],
        num_cols=feature_cols,
        bin_cols=[],
        supervised=cfg.supervised,
        num_partition=cfg.num_partition,
        overlap_ratio=cfg.overlap_ratio,
        device=device,
    )

    trainset = make_transtab_xy(train_df, feature_cols, cfg.label_col)
    valset = make_transtab_xy(val_df, feature_cols, cfg.label_col)

    train(
        model=model_pretrain,
        collate_fn=collate_fn,
        trainset=trainset,
        valset=valset,
        num_epoch=cfg.max_epochs,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        warmup_ratio=None,
        warmup_steps=None,
        eval_metric="val_loss",
        output_dir=str(pretrain_dir),
        num_workers=cfg.num_workers,
        ignore_duplicate_cols=cfg.ignore_duplicate_cols,
        eval_less_is_better=True,
        distributed=False,
        local_rank=0,
        rank=0,
        world_size=1,
        device=device,
    )


class PretrainedEmbeddingExtractor(nn.Module):
    def __init__(
        self,
        feature_cols: list[str],
        checkpoint_dir: str,
        cfg: Config,
        device: str,
    ) -> None:
        super().__init__()
        self.feature_cols = feature_cols
        self.device_name = device
        self.model, _ = build_contrastive_learner(
            cat_cols=[],
            num_cols=feature_cols,
            bin_cols=[],
            supervised=cfg.supervised,
            num_partition=cfg.num_partition,
            overlap_ratio=cfg.overlap_ratio,
            device=device,
            checkpoint=checkpoint_dir,
        )
        self.model.to(device)
        self.model.eval()

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device_name, dtype=torch.float32)
        arr = np.asarray(x)
        if arr.dtype == object:
            raise TypeError(f"Embedding candidate has object dtype: type={type(x)}")
        return torch.tensor(arr, dtype=torch.float32, device=self.device_name)

    def _pool(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim == 3:
            return tensor.mean(dim=1)
        if tensor.ndim > 3:
            return tensor.reshape(tensor.shape[0], -1)
        raise RuntimeError(f"Unsupported tensor ndim for embedding: {tensor.ndim}")

    @torch.no_grad()
    def extract(self, x_df: pd.DataFrame) -> torch.Tensor:
        encoded = self.model.input_encoder(x_df)

        if not isinstance(encoded, dict):
            raise RuntimeError(f"Unexpected input_encoder output: {type(encoded)}")

        # input_encoder output keys:
        #   - embedding
        #   - attention_mask
        encoded = self.model.cls_token(**encoded)
        z = self.model.encoder(**encoded)   # z: [B, T, H]

        if not isinstance(z, torch.Tensor):
            raise RuntimeError(f"Unexpected encoder output: {type(z)}")

        # CLS token embedding
        z = z[:, 0, :]   # [B, H]

        # optional: contrastive projection space를 쓰고 싶으면 아래 사용
        # if hasattr(self.model, "projection_head"):
        #     z = self.model.projection_head(z)

        return z.to(self.device_name, dtype=torch.float32)


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
def predict_head(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_list, true_list = [], []
    for emb, y in loader:
        emb = emb.to(device)
        y = y.to(device)
        pred = model(emb)
        pred_list.append(pred.detach().cpu().numpy())
        true_list.append(y.detach().cpu().numpy())
    return np.concatenate(pred_list, axis=0), np.concatenate(true_list, axis=0)


def has_pretrain_checkpoint(pretrain_dir: Path) -> bool:
    candidates = [
        pretrain_dir / "pytorch_model.bin",
        pretrain_dir / "model.safetensors",
        pretrain_dir / "config.json",
    ]
    return any(p.exists() for p in candidates)


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
    if cfg.resume_pretrained and cfg.skip_pretrain_if_exists and has_pretrain_checkpoint(pretrain_dir):
        should_run_pretrain = False

    if should_run_pretrain:
        print(f"[fold {fold_id}] pretraining contrastive learner...")
        pretrain_one_fold(train_df, val_df, feature_cols, cfg, pretrain_dir, device)
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

    train_dataset = EmbeddingGeneDataset(train_emb, train_y)
    val_dataset = EmbeddingGeneDataset(val_emb, val_y)
    test_dataset = EmbeddingGeneDataset(test_emb, test_y)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

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
