# rapacl/engines/evaluator.py
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.manifold import TSNE
from umap import UMAP

from rapacl.data.dataset import HestRadiomicsDataset, radiomics_collate_fn
from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES
from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import (
    set_seed,
    load_model_radiomics_from_full_checkpoint,
)
import rapacl.configs.default.train as train


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model_radiomics(device):
    model_radiomics = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=train.NUM_CLASS,
        hidden_dropout_prob=train.DROPOUT,
        projection_dim=train.PROJECTION_DIM,
        activation=train.ACTIVATION,
        ape_drop_rate=train.APE_DROP_RATE,
        device=device,
    )
    return model_radiomics.to(device)


def build_eval_dataset():
    dataset = HestRadiomicsDataset(
        radiomics_file=train.VAL_RADIOMCIS_FILE,
        root_dir=train.ROOT_DIR,
        label_col=train.LABEL_COL,
        id_col=train.ID_COL,
    )
    return dataset


def build_eval_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=train.BATCH_SIZE,
        shuffle=False,
        num_workers=train.NUM_WORKERS,
        pin_memory=True,
        collate_fn=radiomics_collate_fn,
        drop_last=False,
    )


def extract_projection_embeddings(model, loader, device):
    model.eval()

    proj_list = []
    label_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting projection embeddings")):

            if batch_idx == 0:
                print("[DEBUG] batch type:", type(batch))
                if isinstance(batch, dict):
                    print("[DEBUG] batch keys:", batch.keys())

            if isinstance(batch, dict):
                if "radiomics_features" in batch:
                    x = batch["radiomics_features"]
                elif "features" in batch:
                    x = batch["features"]
                elif "radiomics" in batch:
                    x = batch["radiomics"]
                elif "data" in batch:
                    x = batch["data"]
                elif "x" in batch:
                    x = batch["x"]
                else:
                    raise KeyError(f"Cannot find feature key. batch keys: {batch.keys()}")

                if "labels" in batch:
                    y = batch["labels"]
                elif "label" in batch:
                    y = batch["label"]
                elif "target" in batch:
                    y = batch["target"]
                elif "y" in batch:
                    y = batch["y"]
                else:
                    raise KeyError(f"Cannot find label key. batch keys: {batch.keys()}")

            else:
                x, y = batch

            # x는 반드시 pd.DataFrame이어야 함
            # TransTabForRadiomics.forward()가 pd.DataFrame만 받음
            if not hasattr(x, "columns"):
                raise TypeError(f"Expected x to be pd.DataFrame, got {type(x)}")

            # full 72 radiomics columns view만 사용
            feat = model.input_encoder(x)
            feat = model.contrastive_token(**feat)
            feat = model.cls_token(**feat)
            enc = model.encoder(**feat)

            # 중요: projection은 CLS가 아니라 contrastive token
            contrastive_token_emb = enc[:, 1, :]
            proj = model.projection_head(contrastive_token_emb)

            proj_list.append(proj.detach().cpu().numpy())

            if torch.is_tensor(y):
                label_list.append(y.detach().cpu().numpy())
            else:
                label_list.append(np.asarray(y))

    embeddings = np.concatenate(proj_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    return embeddings, labels


def extract_raw_radiomics_features(loader):
    raw_list = []
    label_list = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Extracting raw radiomics features")):
        x = batch["radiomics_features"]
        y = batch["labels"]

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)

        raw_list.append(x)
        label_list.append(y)

    raw_features = np.concatenate(raw_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    return raw_features, labels


def compute_clustering_metrics(embeddings, labels, num_classes):
    kmeans = KMeans(
        n_clusters=num_classes,
        random_state=train.SEED,
        n_init="auto",
    )
    cluster_ids = kmeans.fit_predict(embeddings)

    metrics = {
        "silhouette": float(silhouette_score(embeddings, labels)),
        "nmi": float(normalized_mutual_info_score(labels, cluster_ids)),
        "ari": float(adjusted_rand_score(labels, cluster_ids)),
    }

    return metrics, cluster_ids


def save_umap_plot(embeddings, labels, save_path, title):
    reducer = UMAP(n_components=2, random_state=train.SEED)
    z = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, s=8)
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return z


def save_tsne_plot(
    embeddings,
    labels,
    save_path,
    title,
    max_samples=8000,
):
    n = len(embeddings)

    if n > max_samples:
        rng = np.random.default_rng(train.SEED)
        idx = rng.choice(n, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    reducer = TSNE(
        n_components=2,
        random_state=train.SEED,
        init="pca",
        learning_rate="auto",
        perplexity=30,
    )

    z = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, s=8)
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return z


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_space_evaluation(
    space_name,
    embeddings,
    labels,
    save_dir,
):
    save_dir = ensure_dir(save_dir)

    print(f"===== {space_name} Evaluation =====")
    print(f"[INFO] embeddings shape: {embeddings.shape}")
    print(f"[INFO] labels shape: {labels.shape}")

    collapse_info = {
        "mean_abs": float(np.mean(np.abs(embeddings))),
        "std_mean": float(np.mean(np.std(embeddings, axis=0))),
        "global_std": float(np.std(embeddings)),
        "unique_rows_sampled": int(np.unique(embeddings[:5000], axis=0).shape[0]),
    }

    save_json(collapse_info, save_dir / "collapse_check.json")

    num_classes = len(np.unique(labels))

    metrics, cluster_ids = compute_clustering_metrics(
        embeddings=embeddings,
        labels=labels,
        num_classes=num_classes,
    )

    save_json(metrics, save_dir / "clustering_metrics.json")

    print("[INFO] clustering metrics:")
    print(f"  Silhouette: {metrics['silhouette']:.6f}")
    print(f"  NMI       : {metrics['nmi']:.6f}")
    print(f"  ARI       : {metrics['ari']:.6f}")

    save_umap_plot(
        embeddings=embeddings,
        labels=labels,
        save_path=save_dir / "umap_labels.png",
        title=f"UMAP of {space_name}",
    )

    save_umap_plot(
        embeddings=embeddings,
        labels=cluster_ids,
        save_path=save_dir / "umap_kmeans.png",
        title=f"UMAP of {space_name} with KMeans",
    )

    save_tsne_plot(
        embeddings=embeddings,
        labels=labels,
        save_path=save_dir / "tsne_labels.png",
        title=f"t-SNE of {space_name}",
    )

    np.save(save_dir / "embeddings.npy", embeddings)
    np.save(save_dir / "labels.npy", labels)

    print(f"[INFO] {space_name} artifacts saved to: {save_dir}")


def main():
    set_seed(train.SEED)

    device = torch.device(train.DEVICE)

    print(f"[INFO] device: {device}")
    print(f"[INFO] checkpoint path: {train.CHECKPOINT_PATH}")

    if train.CHECKPOINT_PATH is None:
        raise ValueError("constants.CHECKPOINT_PATH is None")

    model_radiomics = build_model_radiomics(device)
    print("[INFO] model built successfully")

    load_model_radiomics_from_full_checkpoint(
        model_radiomics=model_radiomics,
        checkpoint_path=train.CHECKPOINT_PATH,
        device=device,
        strict=False,
    )
    print("[INFO] checkpoint loaded successfully")

    dataset = build_eval_dataset()
    loader = build_eval_loader(dataset)

    # 1) Projection space
    projection_embeddings, projection_labels = extract_projection_embeddings(
        model=model_radiomics,
        loader=loader,
        device=device,
    )

    projection_save_dir = os.path.join(train.OUTPUT_DIR, "projection_space_eval")

    run_space_evaluation(
        space_name="Projection Space",
        embeddings=projection_embeddings,
        labels=projection_labels,
        save_dir=projection_save_dir,
    )

    # 2) Raw radiomics feature space
    raw_embeddings, raw_labels = extract_raw_radiomics_features(loader)

    raw_save_dir = os.path.join(train.OUTPUT_DIR, "raw_feature_space_eval")

    run_space_evaluation(
        space_name="Raw Radiomics Feature Space",
        embeddings=raw_embeddings,
        labels=raw_labels,
        save_dir=raw_save_dir,
    )


if __name__ == "__main__":
    main()
