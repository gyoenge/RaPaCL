"""
Expected Results: 

detailed_eval/
  clustering_metrics.json
  umap_labels.png
  tsne_labels.png
  umap_kmeans.png
  representative_points.json
  top_variance_feature_quantiles.json
  top_variance_feature_umap/
    original_ngtdm_coarseness_umap_quantiles.png
    original_glszm_largearealowgraylevelemphasis_umap_quantiles.png
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


from tqdm import tqdm
import numpy as np
import torch

def extract_embeddings(model, dataset, batch_size=256, logger=None):
    x, y = unwrap_dataset(dataset)
    model.eval()

    emb_list = []
    y_list = []

    for i in tqdm(range(0, len(x), batch_size), desc="Extracting embeddings"):
        bs_x = x.iloc[i:i+batch_size]
        bs_y = y.iloc[i:i+batch_size]

        with torch.no_grad():
            outputs = model.input_encoder(bs_x)

            if isinstance(outputs, dict):
                # use masked mean pooling
                emb = outputs["embedding"]          # (B, L, D)
                attn_mask = outputs["attention_mask"]  # (B, L)
                if logger is not None and i == 0:
                    logger.info("embedding shape: %s", tuple(emb.shape))
                    logger.info("attention_mask shape: %s", tuple(attn_mask.shape))
                mask = attn_mask.unsqueeze(-1).float()   # (B, L, 1)
                emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, D)

            elif isinstance(outputs, tuple):
                emb = outputs[0]
            else:
                emb = outputs

            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            else:
                emb = np.asarray(emb)

            if emb.ndim == 1:
                emb = emb[None, :]

            if emb.ndim != 2:
                raise ValueError(f"Expected pooled embedding to be 2D, got shape={emb.shape}")

        emb_list.append(emb)
        y_list.append(bs_y.to_numpy())

    emb_all = np.concatenate(emb_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return emb_all, y_all


"""
embedding shape: (256, 512, 128)은 의미가:
    256 = batch size
    512 = token 길이
    128 = hidden dim
즉 지금 embedding은 샘플당 하나의 벡터가 아니라, 샘플당 512개 토큰의 벡터 시퀀스이다. 

그래서 UMAP, t-SNE, clustering에 바로 넣으면 안 되고,
반드시 (batch, dim) 형태로 pooling 해야 한다. 

--> 가장 좋은 방법: output dict에 attention_mask도 있으니까, 
                masked mean pooling을 쓰는 게 제일 맞다. 
"""


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

    return z

# def save_tsne_plot(embeddings, labels, save_path, title="t-SNE"):
#     reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
#     z = reducer.fit_transform(embeddings)

#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, s=8)
#     plt.title(title)
#     plt.colorbar(scatter)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200)
#     plt.close()

#     return z

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


"""
추가 기능: 
    x_all에서 분산이 큰 column top 5 선택
    각 column에 대해 25%, 50%, 75% quantile 값에 가장 가까운 샘플 찾기
    그 샘플들의 UMAP 좌표를 강조 표시
    feature별로 그림 저장
즉, “label cluster”가 아니라 “radiomics feature 값 기준 representative point”를 UMAP 위에 올리는 것. 

각 feature마다:
    전체 embedding point는 연한 회색으로 찍고
    25% 지점 샘플은 빨간색
    50% 지점 샘플은 초록색
    75% 지점 샘플은 파란색
이렇게 표시 
"""

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


def save_feature_quantile_umap_plot(z_umap, x_df, feature_name, quantile_info, save_path):
    plt.figure(figsize=(8, 6))

    # background points
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

    logger.info("------ Check embedding collapse ------")
    logger.info("Embedding mean abs: %.8f", float(np.mean(np.abs(embeddings))))
    logger.info("Embedding std mean: %.8f", float(np.mean(np.std(embeddings, axis=0))))
    logger.info("Embedding global std: %.8f", float(np.std(embeddings)))
    logger.info("Unique rows (sampled): %d", np.unique(embeddings[:5000], axis=0).shape[0])
    logger.info("--------------------------------------")

    # 2) clustering metrics
    logger.info("Computing Clustering Metrics...")
    metrics, cluster_ids = compute_clustering_metrics(
        embeddings=embeddings,
        labels=labels,
        num_classes=num_class,
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

    # 5) top-variance radiomics features on UMAP
    logger.info("Visualizing top-variance feature representatives on UMAP...")

    top_var_features = get_top_variance_features(x_all, top_k=5)
    top_var_dir = ensure_dir(eval_dir / "top_variance_feature_umap")

    top_var_summary = {}

    for feature_name, var_value in top_var_features.items():
        quantile_info = find_nearest_quantile_indices(
            x_df=x_all,
            feature_name=feature_name,
            quantiles=(0.25, 0.5, 0.75),
        )

        save_feature_quantile_umap_plot(
            z_umap=z_umap,
            x_df=x_all,
            feature_name=feature_name,
            quantile_info=quantile_info,
            save_path=top_var_dir / f"{feature_name}_umap_quantiles.png",
        )

        top_var_summary[feature_name] = {
            "variance": float(var_value),
            "quantiles": quantile_info,
        }

    save_json(top_var_summary, eval_dir / "top_variance_feature_quantiles.json")


    logger.info("All evaluation finished!")
    logger.info("Detailed evaluation artifacts saved to: %s", eval_dir)