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
from src.pretrain_transtab.transtab_custom import unwrap_dataset


def extract_projection_embeddings(model, dataset, batch_size=256, logger=None):
    x, y = unwrap_dataset(dataset)
    model.eval()

    proj_list = []
    y_list = []

    for i in tqdm(range(0, len(x), batch_size), desc="Extracting projection embeddings"):
        bs_x = x.iloc[i:i + batch_size]
        bs_y = y.iloc[i:i + batch_size]

        with torch.no_grad():
            inputs = model.input_encoder(bs_x)
            inputs = model.cls_token(**inputs)
            enc = model.encoder(**inputs)
            cls = enc[:, 0, :]
            proj = model.projection_head(cls)

            if logger is not None and i == 0:
                logger.info("encoder output shape: %s", tuple(enc.shape))
                logger.info("cls shape: %s", tuple(cls.shape))
                logger.info("projection shape: %s", tuple(proj.shape))

            proj = proj.detach().cpu().numpy()

        proj_list.append(proj)
        y_list.append(bs_y.to_numpy())

    proj_all = np.concatenate(proj_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return proj_all, y_all


def extract_raw_radiomics_features(dataset):
    x, y = unwrap_dataset(dataset)
    x_np = x.to_numpy(dtype=np.float32)
    y_np = y.to_numpy()
    return x, x_np, y_np


def compute_clustering_metrics(embeddings, labels, num_classes):
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    metrics = {
        "silhouette": float(silhouette_score(embeddings, labels)),
        "nmi": float(normalized_mutual_info_score(labels, cluster_ids)),
        "ari": float(adjusted_rand_score(labels, cluster_ids)),
    }
    return metrics, cluster_ids


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


def save_tsne_plot(embeddings, labels, save_path, title="t-SNE", max_samples=8000, random_state=42):
    n = len(embeddings)
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
    else:
        idx = np.arange(n)

    reducer = TSNE(
        n_components=2,
        random_state=random_state,
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

    return z, labels, idx


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


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_top_variance_features(x_df, top_k=5):
    variances = x_df.var(axis=0).sort_values(ascending=False)
    return variances.head(top_k)


def find_nearest_quantile_indices(x_df, feature_name, quantiles=(0.25, 0.5, 0.75)):
    values = x_df[feature_name].to_numpy()
    q_values = np.quantile(values, quantiles)

    result = {}
    for q, qv in zip(quantiles, q_values):
        idx = int(np.argmin(np.abs(values - qv)))
        result[str(q)] = {
            "index": idx,
            "quantile_value": float(qv),
            "actual_value": float(values[idx]),
        }
    return result


def save_feature_quantile_umap_plot(z_umap, feature_name, quantile_info, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], s=5, alpha=0.15)

    color_map = {
        "0.25": "red",
        "0.5": "green",
        "0.75": "blue",
    }

    for q_str, info in quantile_info.items():
        idx = info["index"]
        plt.scatter(
            z_umap[idx, 0],
            z_umap[idx, 1],
            s=120,
            c=color_map[q_str],
            label=f"{feature_name} q={q_str} (value={info['actual_value']:.4f})",
            edgecolors="black",
        )

    plt.title(f"UMAP representatives for {feature_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_space_evaluation(
    space_name,
    embeddings,
    labels,
    x_df,
    save_dir,
    logger,
    num_class,
):
    logger.info("===== [%s] evaluation start =====", space_name)
    logger.info("[%s] shape: %s", space_name, embeddings.shape)

    logger.info("------ [%s] collapse check ------", space_name)
    logger.info("[%s] mean abs   : %.8f", space_name, float(np.mean(np.abs(embeddings))))
    logger.info("[%s] std mean   : %.8f", space_name, float(np.mean(np.std(embeddings, axis=0))))
    logger.info("[%s] global std : %.8f", space_name, float(np.std(embeddings)))
    logger.info("[%s] unique rows(sampled): %d", space_name, np.unique(embeddings[:5000], axis=0).shape[0])
    logger.info("----------------------------------")

    metrics, cluster_ids = compute_clustering_metrics(
        embeddings=embeddings,
        labels=labels,
        num_classes=num_class,
    )
    save_json(metrics, save_dir / "clustering_metrics.json")

    logger.info("[%s] Silhouette: %.6f", space_name, metrics["silhouette"])
    logger.info("[%s] NMI       : %.6f", space_name, metrics["nmi"])
    logger.info("[%s] ARI       : %.6f", space_name, metrics["ari"])

    z_umap = save_umap_plot(
        embeddings=embeddings,
        labels=labels,
        save_path=save_dir / "umap_labels.png",
        title=f"UMAP of {space_name}",
    )

    save_umap_plot(
        embeddings=embeddings,
        labels=cluster_ids,
        save_path=save_dir / "umap_kmeans.png",
        title=f"UMAP of {space_name} with KMeans cluster ids",
    )

    z_tsne, labels_tsne, tsne_indices = save_tsne_plot(
        embeddings=embeddings,
        labels=labels,
        save_path=save_dir / "tsne_labels.png",
        title=f"t-SNE of {space_name}",
    )

    umap_reps = find_representative_points(z_umap, labels)
    tsne_reps_local = find_representative_points(z_tsne, labels_tsne)
    tsne_reps = {k: int(tsne_indices[v]) for k, v in tsne_reps_local.items()}

    rep_info = {
        "umap_representatives": umap_reps,
        "tsne_representatives": tsne_reps,
    }
    save_json(rep_info, save_dir / "representative_points.json")

    top_var_features = get_top_variance_features(x_df, top_k=5)
    top_var_dir = ensure_dir(save_dir / "top_variance_feature_umap")
    top_var_summary = {}

    for feature_name, var_value in top_var_features.items():
        quantile_info = find_nearest_quantile_indices(
            x_df=x_df,
            feature_name=feature_name,
            quantiles=(0.25, 0.5, 0.75),
        )

        save_feature_quantile_umap_plot(
            z_umap=z_umap,
            feature_name=feature_name,
            quantile_info=quantile_info,
            save_path=top_var_dir / f"{feature_name}_umap_quantiles.png",
        )

        top_var_summary[feature_name] = {
            "variance": float(var_value),
            "quantiles": quantile_info,
        }

    save_json(top_var_summary, save_dir / "top_variance_feature_quantiles.json")
    logger.info("===== [%s] evaluation done: %s =====", space_name, save_dir)


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

    # 1) projection embeddings
    logger.info("Extracting projection embeddings...")
    proj_embeddings, proj_labels = extract_projection_embeddings(
        model=model,
        dataset=allset,
        logger=logger,
        batch_size=train_cfg.get("eval_batch_size", 256),
    )

    run_space_evaluation(
        space_name="projection space",
        embeddings=proj_embeddings,
        labels=proj_labels,
        x_df=x_all,
        save_dir=eval_dir,
        logger=logger,
        num_class=num_class,
    )

    # 2) raw radiomics
    logger.info("Extracting raw radiomics features...")
    x_raw_df, raw_embeddings, raw_labels = extract_raw_radiomics_features(allset)

    raw_eval_dir = ensure_dir(eval_dir / "raw_radiomics")

    run_space_evaluation(
        space_name="raw radiomics",
        embeddings=raw_embeddings,
        labels=raw_labels,
        x_df=x_raw_df,
        save_dir=raw_eval_dir,
        logger=logger,
        num_class=num_class,
    )

    logger.info("All evaluation finished!")
    logger.info("Detailed evaluation artifacts saved to: %s", eval_dir)

    