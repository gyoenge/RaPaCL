"""
Expected Results: 

detailed_eval/
  metrics.json
  umap_labels.png
  tsne_labels.png
  umap_kmeans.png
  representative_points.json
  representative_patches/
    class_0_center.png
    class_1_center.png
    ...
  radiomics_feature_sections/
    ngtdm_coarseness/
      q0_10.png
      q10_25.png
      ...
    glszm_largearealowgraylevelemphasis/
      ...
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.manifold import TSNE
from umap import UMAP

from src.common.utils import ensure_dir
from src.rapacl.transtab_custom import unwrap_dataset


def extract_embeddings(model, dataset, logger, batch_size=256):
    x, y = unwrap_dataset(dataset)
    model.eval()

    emb_list = []
    y_list = []

    for i in tqdm(
        range(0, len(x), batch_size),
        total=len(x) // batch_size + 1,
        desc="Extracting embeddings"
    ):
        bs_x = x.iloc[i:i+batch_size]
        bs_y = y.iloc[i:i+batch_size]

        with torch.no_grad():
            # 모델 구조에 맞게 수정 필요

            # just for output debugging ###########s
            # outputs = model.input_encoder(bs_x)
            # print("type(outputs):", type(outputs))
            # if isinstance(outputs, tuple):
            #     print("tuple len:", len(outputs))
            #     print("tuple[0] type:", type(outputs[0]))
            # elif isinstance(outputs, dict):
            #     print("dict keys:", outputs.keys())
            # else:
            #     print("outputs:", outputs)
            ########################################

            outputs = model.input_encoder(bs_x)

            if isinstance(outputs, dict):
                emb = outputs["embedding"]
            elif isinstance(outputs, tuple):
                emb = outputs[0]
            else:
                emb = outputs

            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            else:
                emb = np.asarray(emb)

            if emb.ndim == 0:
                raise ValueError(f"Embedding is scalar: {emb}")

            if emb.ndim == 1:
                emb = emb[None, :]
            
            if logger is not None and i == 0:
                logger.info("embedding shape: %s", emb.shape)

        emb_list.append(emb)
        y_list.append(bs_y.to_numpy())

    emb_all = np.concatenate(emb_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return emb_all, y_all

def compute_clustering_metrics(embeddings, labels, num_classes):
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    sil = silhouette_score(embeddings, labels)
    nmi = normalized_mutual_info_score(labels, cluster_ids)
    ari = adjusted_rand_score(labels, cluster_ids)

    return {
        "silhouette": sil,
        "nmi": nmi,
        "ari": ari,
        "cluster_ids": cluster_ids,
    }

def save_umap_plot(embeddings, labels, save_path, title="UMAP"):
    reducer = UMAP(n_components=2, random_state=42)
    z = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, s=8)
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return z

def save_tsne_plot(embeddings, labels, save_path, title="t-SNE"):
    reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    z = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, s=8)
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return z

def find_representative_points(z, labels):
    reps = {}
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        z_label = z[idx]
        center = z_label.mean(axis=0)
        d = ((z_label - center) ** 2).sum(axis=1)
        rep_idx = idx[np.argmin(d)]
        reps[int(label)] = int(rep_idx)
    return reps

def get_feature_quantile_sections(x_df, feature_name):
    values = x_df[feature_name].to_numpy()

    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    bounds = np.quantile(values, qs)

    sections = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i == len(bounds) - 2:
            idx = np.where((values >= lo) & (values <= hi))[0]
        else:
            idx = np.where((values >= lo) & (values < hi))[0]
        sections.append({
            "range": (qs[i], qs[i+1]),
            "indices": idx,
            "value_range": (lo, hi),
        })
    return sections

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_eval_detailed(
    model,
    allset,
    trainset,
    valset,
    testset,
    run_dir,
    logger,
    device,
    cfg,
):
    logger.info("Start detailed representation evaluation...")

    train_cfg = cfg["train"]

    x_all, y_all = unwrap_dataset(allset)
    num_class = len(np.unique(y_all))

    logger.info("Detailed eval - total samples: %d", len(y_all))
    logger.info("Detailed eval - num_class: %d", num_class)

    eval_dir = ensure_dir(run_dir / "detailed_eval")

    # 1) embedding extraction
    logger.info("Extracting Embeddings...")
    embeddings, labels = extract_embeddings(
        model=model,
        dataset=allset,
        logger=logger, 
        batch_size=train_cfg.get("eval_batch_size", 256),
    )

    logger.info("Embedding shape: %s", embeddings.shape)
    logger.info("Labels shape: %s", labels.shape)

    # 2) clustering metrics
    logger.info("Computin Clustering Metrics...")
    metrics, cluster_ids = compute_clustering_metrics(
        embeddings=embeddings,
        labels=labels,
        num_class=num_class,
    )

    logger.info("Silhouette: %.6f", metrics["silhouette"])
    logger.info("NMI       : %.6f", metrics["nmi"])
    logger.info("ARI       : %.6f", metrics["ari"])

    save_json(metrics, eval_dir / "clustering_metrics.json")

    # 3) UMAP / t-SNE
    logger.info("Saving UMAP & t-SNE plots...")

    z_umap = save_umap_plot(
        embeddings=embeddings,
        labels=labels,
        save_path=eval_dir / "umap_labels.png",
        title="UMAP of learned representations",
    )

    save_umap_plot(
        embeddings=embeddings,
        labels=cluster_ids,
        save_path=eval_dir / "umap_kmeans.png",
        title="UMAP with KMeans cluster ids",
    )

    z_tsne = save_tsne_plot(
        embeddings=embeddings,
        labels=labels,
        save_path=eval_dir / "tsne_labels.png",
        title="t-SNE of learned representations",
    )

    # 4) representative points
    logger.info("Finding Representative Points...")
    umap_reps = find_representative_points(z_umap, labels)
    tsne_reps = find_representative_points(z_tsne, labels)

    rep_info = {
        "umap_representatives": umap_reps,
        "tsne_representatives": tsne_reps,
    }
    save_json(rep_info, eval_dir / "representative_points.json")


    logger.info("All evaluation finished!")
    logger.info("Detailed evaluation artifacts saved to: %s", eval_dir)