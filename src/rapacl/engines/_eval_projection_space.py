# rapacl/engines/evaluator.py
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
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
import rapacl.engines.constants as constants


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model_radiomics(device):
    model_radiomics = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=constants.NUM_CLASS,
        hidden_dropout_prob=constants.DROPOUT,
        projection_dim=constants.PROJECTION_DIM,
        activation=constants.ACTIVATION,
        ape_drop_rate=constants.APE_DROP_RATE,
        device=device,
    )
    return model_radiomics.to(device)


def build_eval_dataset():
    dataset = HestRadiomicsDataset(
        radiomics_file=constants.VAL_RADIOMCIS_FILE,
        root_dir=constants.ROOT_DIR,
        label_col=constants.LABEL_COL,
        id_col=constants.ID_COL,
    )
    return dataset


def build_eval_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=False,
        num_workers=constants.NUM_WORKERS,
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


def compute_clustering_metrics(embeddings, labels, num_classes):
    kmeans = KMeans(
        n_clusters=num_classes,
        random_state=constants.SEED,
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
    reducer = UMAP(n_components=2, random_state=constants.SEED)
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
        rng = np.random.default_rng(constants.SEED)
        idx = rng.choice(n, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    reducer = TSNE(
        n_components=2,
        random_state=constants.SEED,
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


def run_projection_space_evaluation(
    embeddings,
    labels,
    save_dir,
):
    save_dir = ensure_dir(save_dir)

    print("===== Projection Space Evaluation =====")
    print(f"[INFO] embeddings shape: {embeddings.shape}")
    print(f"[INFO] labels shape: {labels.shape}")

    collapse_info = {
        "mean_abs": float(np.mean(np.abs(embeddings))),
        "std_mean": float(np.mean(np.std(embeddings, axis=0))),
        "global_std": float(np.std(embeddings)),
        "unique_rows_sampled": int(np.unique(embeddings[:5000], axis=0).shape[0]),
    }

    save_json(collapse_info, save_dir / "collapse_check.json")

    print("[INFO] collapse check:")
    for k, v in collapse_info.items():
        print(f"  {k}: {v}")

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
        title="UMAP of Projection Space",
    )

    save_umap_plot(
        embeddings=embeddings,
        labels=cluster_ids,
        save_path=save_dir / "umap_kmeans.png",
        title="UMAP of Projection Space with KMeans",
    )

    save_tsne_plot(
        embeddings=embeddings,
        labels=labels,
        save_path=save_dir / "tsne_labels.png",
        title="t-SNE of Projection Space",
    )

    np.save(save_dir / "projection_embeddings.npy", embeddings)
    np.save(save_dir / "labels.npy", labels)

    print(f"[INFO] evaluation artifacts saved to: {save_dir}")


def main():
    set_seed(constants.SEED)

    device = torch.device(constants.DEVICE)

    print(f"[INFO] device: {device}")
    print(f"[INFO] checkpoint path: {constants.CHECKPOINT_PATH}")

    if constants.CHECKPOINT_PATH is None:
        raise ValueError("constants.CHECKPOINT_PATH is None")

    model_radiomics = build_model_radiomics(device)
    print("[INFO] model built successfully")

    load_model_radiomics_from_full_checkpoint(
        model_radiomics=model_radiomics,
        checkpoint_path=constants.CHECKPOINT_PATH,
        device=device,
        strict=False,
    )
    print("[INFO] checkpoint loaded successfully")

    dataset = build_eval_dataset()
    loader = build_eval_loader(dataset)

    embeddings, labels = extract_projection_embeddings(
        model=model_radiomics,
        loader=loader,
        device=device,
    )

    save_dir = os.path.join(constants.OUTPUT_DIR, "projection_space_eval")

    run_projection_space_evaluation(
        embeddings=embeddings,
        labels=labels,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
