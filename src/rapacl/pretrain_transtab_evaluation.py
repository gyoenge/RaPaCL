# 평가 스크립트
# 체크포인트 로드 → embedding 추출 → UMAP/t-SNE 시각화 → label 기준 해석
"""
학습 때와 같은 방식으로 dataset load
checkpoint 자동 선택 또는 이름 지정
embedding 추출
KMeans 기반 clustering metric 계산
Silhouette
NMI
ARI
UMAP / t-SNE 시각화 저장
embeddings csv, coords csv, metrics json 저장
간단한 해석 문자열도 같이 저장
"""
"""
python -m src.rapacl.pretrain_transtab_evaluation \
  --config configs/pretrain_transtab/idc_allxenium.yaml \
  --mode eval
"""


from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None

import transtab
from transtab import constants
from transtab.modeling_transtab import TransTabForCL
from transtab.trainer_utils import TransTabCollatorForCL

from src.common.config import apply_cli_overrides, load_yaml, parse_common_args
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything


# =========================
# Dataset / model utilities
# =========================

def build_contrastive_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    projection_dim=128,
    num_partition=3,
    overlap_ratio=0.5,
    supervised=True,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation="relu",
    device="cuda:0",
    checkpoint=None,
    ignore_duplicate_cols=True,
    **kwargs,
):
    model = TransTabForCL(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        num_partition=num_partition,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        supervised=supervised,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        overlap_ratio=overlap_ratio,
        activation=activation,
        device=device,
    )

    if checkpoint is not None:
        model.load(checkpoint)

    collate_fn = TransTabCollatorForCL(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        overlap_ratio=overlap_ratio,
        num_partition=num_partition,
        ignore_duplicate_cols=ignore_duplicate_cols,
    )

    if checkpoint is not None:
        extractor_state_dir = os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        if os.path.exists(extractor_state_dir):
            collate_fn.feature_extractor.load(extractor_state_dir)

    return model, collate_fn


def save_column_info(
    run_dir: Path,
    categorical_columns: list[str],
    numerical_columns: list[str],
    binary_columns: list[str],
) -> None:
    info = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "binary_columns": binary_columns,
        "num_categorical": len(categorical_columns),
        "num_numerical": len(numerical_columns),
        "num_binary": len(binary_columns),
    }
    with open(run_dir / "column_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# =========================
# Checkpoint helpers
# =========================

def resolve_checkpoint_dir(checkpoint_root: str | Path, checkpoint_name: str | None = None) -> Path:
    checkpoint_root = Path(checkpoint_root)
    if checkpoint_name:
        checkpoint_dir = checkpoint_root / checkpoint_name
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
        return checkpoint_dir

    candidates = [p for p in checkpoint_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directory found in: {checkpoint_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# =========================
# Embedding extraction
# =========================

def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    return value


def _safe_detach_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported output type: {type(x)}")


def _pick_tensor_from_output(output: Any) -> torch.Tensor:
    """
    Heuristically pick the most useful embedding tensor from possibly different
    transtab forward output formats.
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        tensor_candidates = [x for x in output if isinstance(x, torch.Tensor)]
        if not tensor_candidates:
            raise ValueError("No tensor found in tuple/list output.")
        # Prefer 2D batch embeddings.
        for tensor in tensor_candidates:
            if tensor.ndim == 2:
                return tensor
        return tensor_candidates[0]

    if isinstance(output, dict):
        preferred_keys = [
            "cls_embedding",
            "embedding",
            "embeddings",
            "features",
            "hidden_states",
            "last_hidden_state",
            "logits",
        ]
        for key in preferred_keys:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value

        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value

    raise TypeError(f"Unsupported model output type: {type(output)}")


def _call_model_for_embeddings(model: torch.nn.Module, batch_x: Any) -> torch.Tensor:
    """
    Try several call patterns because TransTab versions may expose slightly
    different inference APIs.
    """
    candidate_calls = []

    if hasattr(model, "get_embeddings"):
        candidate_calls.append(lambda: model.get_embeddings(batch_x))
    if hasattr(model, "encode"):
        candidate_calls.append(lambda: model.encode(batch_x))
    if hasattr(model, "encoder"):
        candidate_calls.append(lambda: model.encoder(batch_x))

    candidate_calls.extend(
        [
            lambda: model(batch_x),
            lambda: model(batch_x, y=None),
            lambda: model(**batch_x) if isinstance(batch_x, dict) else (_raise_type_error()),
        ]
    )

    last_error: Exception | None = None
    for fn in candidate_calls:
        try:
            output = fn()
            return _pick_tensor_from_output(output)
        except Exception as e:  # pragma: no cover - version dependent fallback
            last_error = e
            continue

    raise RuntimeError(f"Failed to obtain embeddings from model. Last error: {last_error}")


def _raise_type_error():
    raise TypeError("Batch is not a dict, so model(**batch_x) cannot be used.")


def extract_embeddings(
    model: torch.nn.Module,
    x_df: pd.DataFrame,
    y: pd.Series | np.ndarray | list,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = np.asarray(y)

    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(x_df), batch_size):
            end = min(start + batch_size, len(x_df))
            batch_x = x_df.iloc[start:end].copy()
            batch_x = _to_device(batch_x, device)
            emb = _call_model_for_embeddings(model, batch_x)

            if emb.ndim > 2:
                emb = emb.reshape(emb.shape[0], -1)
            elif emb.ndim == 1:
                emb = emb.unsqueeze(0)

            all_embeddings.append(_safe_detach_to_numpy(emb))

    return np.concatenate(all_embeddings, axis=0), labels


# =========================
# Metrics / visualization
# =========================

def maybe_encode_labels(y: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    le = LabelEncoder()
    encoded = le.fit_transform(y)
    mapping = {int(i): str(label) for i, label in enumerate(le.classes_)}
    return encoded, mapping


def compute_clustering_metrics(embeddings: np.ndarray, y_true: np.ndarray) -> dict[str, float | int | None]:
    y_encoded, _ = maybe_encode_labels(y_true)
    num_classes = len(np.unique(y_encoded))

    metrics: dict[str, float | int | None] = {
        "n_samples": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "num_classes": int(num_classes),
    }

    if num_classes < 2 or len(embeddings) < 3:
        metrics.update({"silhouette": None, "nmi": None, "ari": None})
        return metrics

    pred = KMeans(n_clusters=num_classes, random_state=42, n_init=10).fit_predict(embeddings)

    metrics["silhouette"] = float(silhouette_score(embeddings, pred))
    metrics["nmi"] = float(normalized_mutual_info_score(y_encoded, pred))
    metrics["ari"] = float(adjusted_rand_score(y_encoded, pred))
    return metrics


def reduce_umap(embeddings: np.ndarray, random_state: int = 42, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    if umap is None:
        raise ImportError("umap-learn is not installed. Please `pip install umap-learn`.")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def reduce_tsne(embeddings: np.ndarray, random_state: int = 42, perplexity: float = 30.0) -> np.ndarray:
    perplexity = min(perplexity, max(5.0, len(embeddings) - 1.0))
    reducer = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(embeddings)


def plot_2d_scatter(coords: np.ndarray, y_true: np.ndarray, title: str, save_path: str | Path) -> None:
    y_encoded, mapping = maybe_encode_labels(y_true)
    plt.figure(figsize=(9, 7))

    unique_labels = np.unique(y_encoded)
    for label_id in unique_labels:
        mask = y_encoded == label_id
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.8,
            label=mapping[int(label_id)],
        )

    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend(markerscale=1.5, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def summarize_cluster_quality(metrics: dict[str, Any]) -> dict[str, str]:
    silhouette = metrics.get("silhouette")
    nmi = metrics.get("nmi")
    ari = metrics.get("ari")

    def bucket_score(v: float | None, kind: str) -> str:
        if v is None:
            return "not_available"
        if kind == "silhouette":
            if v >= 0.5:
                return "well_separated"
            if v >= 0.2:
                return "moderately_separated"
            if v >= 0.0:
                return "weakly_separated"
            return "poorly_separated"
        if v >= 0.8:
            return "very_strong_label_alignment"
        if v >= 0.5:
            return "moderate_label_alignment"
        if v >= 0.2:
            return "weak_label_alignment"
        return "poor_label_alignment"

    return {
        "silhouette_interpretation": bucket_score(silhouette, "silhouette"),
        "nmi_interpretation": bucket_score(nmi, "other"),
        "ari_interpretation": bucket_score(ari, "other"),
    }


# =========================
# Main
# =========================

def main() -> None:
    args = parse_common_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg["mode"] = args.mode

    eval_cfg = cfg.get("evaluation", {})
    seed = cfg.get("seed", 42)
    seed_everything(seed)

    log_dir = cfg["paths"]["log_dir"]
    timestamp, logger = setup_logger(log_dir, name="pretrain_transtab_evaluation")

    output_root = ensure_dir(cfg["paths"]["output_root"])
    run_dir = ensure_dir(output_root / f"eval_{timestamp}")
    save_yaml(cfg, run_dir / "config_eval.yaml")

    logger.info("Loaded config from: %s", args.config)
    logger.info("Execution mode: %s", args.mode)
    logger.info("Preparing dataset from: %s", cfg["paths"]["data_root"])

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data([
        f'{cfg["paths"]["data_root"]}'
    ])

    save_column_info(run_dir, cat_cols, num_cols, bin_cols)
    logger.info("Detected columns -> categorical=%d numerical=%d binary=%d", len(cat_cols), len(num_cols), len(bin_cols))

    checkpoint_dir = resolve_checkpoint_dir(
        cfg["paths"]["checkpoint_dir"],
        eval_cfg.get("checkpoint_name"),
    )
    logger.info("Using checkpoint dir: %s", checkpoint_dir)

    model_cfg = cfg["model"]
    runtime_device = cfg["runtime"].get("device", "cpu")
    if str(runtime_device).startswith("cuda") and not torch.cuda.is_available():
        runtime_device = "cpu"
    device = torch.device(runtime_device)

    try:
        model, collate_fn = build_contrastive_learner(
            categorical_columns=cat_cols,
            numerical_columns=num_cols,
            binary_columns=bin_cols,
            projection_dim=model_cfg.get("projection_dim", 128),
            num_partition=model_cfg.get("num_partition", 3),
            overlap_ratio=model_cfg.get("overlap_ratio", 0.5),
            supervised=model_cfg.get("supervised", True),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layer=model_cfg.get("num_layer", 2),
            num_attention_head=model_cfg.get("num_attention_head", 8),
            hidden_dropout_prob=model_cfg.get("hidden_dropout_prob", 0.0),
            ffn_dim=model_cfg.get("ffn_dim", 256),
            activation=model_cfg.get("activation", "relu"),
            device=str(device),
            checkpoint=str(checkpoint_dir),
            ignore_duplicate_cols=model_cfg.get("ignore_duplicate_cols", True),
        )
    except Exception:
        # Fallback to simpler public builder used in your training script.
        model, collate_fn = transtab.build_contrastive_learner(
            cat_cols=cat_cols,
            num_cols=num_cols,
            bin_cols=bin_cols,
            supervised=model_cfg.get("supervised", True),
            num_partition=model_cfg.get("num_partition", 3),
            overlap_ratio=model_cfg.get("overlap_ratio", 0.5),
        )
        if hasattr(model, "load"):
            model.load(str(checkpoint_dir))

    model = model.to(device)
    model.eval()

    split_name = eval_cfg.get("split", "test")
    split_map = {
        "all": allset,
        "train": trainset,
        "val": valset,
        "test": testset,
    }
    if split_name not in split_map:
        raise ValueError(f"Unsupported split: {split_name}. Choose from {list(split_map.keys())}")

    x_df, y = split_map[split_name]
    logger.info("Evaluation split: %s, num_samples=%d", split_name, len(x_df))

    batch_size = int(eval_cfg.get("batch_size", cfg.get("train", {}).get("eval_batch_size", 256)))
    embeddings, labels = extract_embeddings(
        model=model,
        x_df=x_df,
        y=y,
        batch_size=batch_size,
        device=device,
    )

    logger.info("Extracted embeddings: shape=%s", tuple(embeddings.shape))

    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df["label"] = labels
    embeddings_df.to_csv(run_dir / f"{split_name}_embeddings.csv", index=False)

    metrics = compute_clustering_metrics(embeddings, labels)
    metrics.update(summarize_cluster_quality(metrics))

    with open(run_dir / f"{split_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics to: %s", run_dir / f"{split_name}_metrics.json")
    logger.info("Metrics: %s", metrics)

    methods = eval_cfg.get("methods", ["umap", "tsne"])
    if isinstance(methods, str):
        methods = [methods]

    coords_summary: dict[str, str] = {}

    if "umap" in methods:
        umap_coords = reduce_umap(
            embeddings,
            random_state=seed,
            n_neighbors=int(eval_cfg.get("umap_n_neighbors", 15)),
            min_dist=float(eval_cfg.get("umap_min_dist", 0.1)),
        )
        plot_2d_scatter(
            umap_coords,
            labels,
            title=f"UMAP - {split_name}",
            save_path=run_dir / f"{split_name}_umap.png",
        )
        pd.DataFrame({"x": umap_coords[:, 0], "y": umap_coords[:, 1], "label": labels}).to_csv(
            run_dir / f"{split_name}_umap_coords.csv", index=False
        )
        coords_summary["umap_png"] = str(run_dir / f"{split_name}_umap.png")

    if "tsne" in methods:
        tsne_coords = reduce_tsne(
            embeddings,
            random_state=seed,
            perplexity=float(eval_cfg.get("tsne_perplexity", 30.0)),
        )
        plot_2d_scatter(
            tsne_coords,
            labels,
            title=f"t-SNE - {split_name}",
            save_path=run_dir / f"{split_name}_tsne.png",
        )
        pd.DataFrame({"x": tsne_coords[:, 0], "y": tsne_coords[:, 1], "label": labels}).to_csv(
            run_dir / f"{split_name}_tsne_coords.csv", index=False
        )
        coords_summary["tsne_png"] = str(run_dir / f"{split_name}_tsne.png")

    with open(run_dir / "artifacts.json", "w", encoding="utf-8") as f:
        json.dump(coords_summary, f, indent=2, ensure_ascii=False)

    logger.info("Evaluation finished. Outputs saved to: %s", run_dir)


if __name__ == "__main__":
    main()

