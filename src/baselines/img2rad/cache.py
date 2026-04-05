from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .dataset import STNetDataset


def safe_load_gene_names(gene_list_path: str) -> List[str]:
    with open(gene_list_path, "r", encoding="utf-8") as f:
        gene_info = json.load(f)

    if isinstance(gene_info, dict):
        if "genes" in gene_info:
            return gene_info["genes"]
        if "gene_names" in gene_info:
            return gene_info["gene_names"]

    if isinstance(gene_info, list):
        return gene_info

    raise ValueError(f"Unsupported gene list format: {gene_list_path}")


def _get_radiomics_logging_cfg(cfg: dict | None = None) -> dict:
    default_cfg = {
        "enabled": True,
        "log_ignored_by_name": False,
        "log_ignored_by_prefix": False,
        "log_non_numeric_columns": True,
        "log_nan_skips": True,
        "log_loaded_sample_summary": True,
        "log_final_summary": True,
        "max_logged_items": 20,
    }

    if cfg is None:
        return default_cfg

    data_cfg = cfg.get("data", {})
    logging_cfg = data_cfg.get("radiomics_logging", {})
    merged = default_cfg.copy()
    merged.update(logging_cfg)
    return merged


def load_sample_radiomics_parquet(
    parquet_path: str,
    key_column: str,
    ignore_columns: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    logger: logging.Logger | None = None,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    log_cfg = _get_radiomics_logging_cfg(cfg)
    max_items = int(log_cfg["max_logged_items"])

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Radiomics parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if key_column not in df.columns:
        raise ValueError(
            f"Key column '{key_column}' not found in parquet: {parquet_path}. "
            f"Available columns: {list(df.columns)}"
            )
            
    # 1) 제외 컬럼 제거
    ignore_columns = set(ignore_columns or [])
    ignore_prefixes = tuple(ignore_prefixes or [])
    ignore_columns.add(key_column)
    candidate_cols = [
        col for col in df.columns
        if col not in ignore_columns
        and not col.startswith(ignore_prefixes)
    ]   

    # 기존: numeric 전체 선택
    # 2) 숫자형 컬럼만 feature로 사용
    # numeric_cols = df[candidate_cols].select_dtypes(include=["number"]).columns.tolist()
    # feature_columns = numeric_cols
    # if not feature_columns:
    #     raise ValueError(
    #         f"No numeric feature columns found in parquet: {parquet_path}. "
    #         f"candidate_cols={candidate_cols[:20]}"
    #     )

    # 수정: radiomics prefix만 선택 
    valid_prefixes = (
        "original_",
        "squareroot_",
        "logarithm_",
        "wavelet-",
        "exponential_",
        "square_",
    )
    feature_columns = [
        col for col in df.columns
        if col.startswith(valid_prefixes)
    ]
    if not feature_columns:
        raise ValueError(f"No valid radiomics features found in {parquet_path}")

    # 디버깅용
    ignored_by_name = [col for col in df.columns if col in ignore_columns]
    ignored_by_prefix = [
        col for col in df.columns
        if col not in ignore_columns and col.startswith(ignore_prefixes)
    ]
    # non_numeric_cols = [col for col in candidate_cols if col not in numeric_cols]

    if logger is not None and log_cfg["enabled"]:
        if ignored_by_name and log_cfg["log_ignored_by_name"]:
            logger.info(
                "[RadiomicsParquet] ignored by exact name in %s: %s",
                parquet_path,
                ignored_by_name[:max_items],
            )

        if ignored_by_prefix and log_cfg["log_ignored_by_prefix"]:
            logger.info(
                "[RadiomicsParquet] ignored by prefix in %s: %s",
                parquet_path,
                ignored_by_prefix[:max_items],
            )

        if non_numeric_cols and log_cfg["log_non_numeric_columns"]:
            logger.warning(
                "[RadiomicsParquet] non-numeric columns excluded in %s: %s",
                parquet_path,
                non_numeric_cols[:max_items],
            )

    ### 

    df = df.copy()
    df[key_column] = df[key_column].astype(str)

    if df[key_column].duplicated().any():
        dup_keys = df.loc[df[key_column].duplicated(), key_column].unique().tolist()[:10]
        raise ValueError(
            f"Duplicated keys found in {parquet_path} for column '{key_column}': {dup_keys}"
        )

    return df, feature_columns


def load_samplewise_radiomics_targets(
    base_dataset: STNetDataset,
    radiomics_parquet_dir: str,
    logger: logging.Logger,
    key_column: str = "barcode",
    ignore_columns: list[str] | None = None,
    ignore_prefixes: list[str] | None = None,
    cfg: dict | None = None,
) -> Tuple[torch.Tensor, List[str], List[int]]:
    log_cfg = _get_radiomics_logging_cfg(cfg)
    max_items = int(log_cfg["max_logged_items"])

    unique_sample_ids = sorted({meta["sample_id"] for meta in base_dataset.patch_meta})
    logger.info("[RadiomicsParquet] unique samples in dataset = %d", len(unique_sample_ids))

    sample_feature_df_map: Dict[str, pd.DataFrame] = {}
    global_feature_names: list[str] | None = None

    for sample_id in unique_sample_ids:
        parquet_path = os.path.join(radiomics_parquet_dir, f"{sample_id}.parquet")
        df, feature_columns = load_sample_radiomics_parquet(
            parquet_path=parquet_path,
            key_column=key_column,
            ignore_columns=ignore_columns,
            ignore_prefixes=ignore_prefixes,
            logger=logger,
            cfg=cfg,
        )

        if global_feature_names is None:
            global_feature_names = feature_columns
        else:
            if feature_columns != global_feature_names:
                raise ValueError(
                    f"Feature columns mismatch for sample_id={sample_id}\n"
                    f"expected={global_feature_names[:10]} ... ({len(global_feature_names)} cols)\n"
                    f"got={feature_columns[:10]} ... ({len(feature_columns)} cols)"
                )

        sample_feature_df_map[sample_id] = df.set_index(key_column)

        if log_cfg["enabled"] and log_cfg["log_loaded_sample_summary"]:
            logger.info(
                "[RadiomicsParquet] loaded sample=%s rows=%d features=%d",
                sample_id,
                len(df),
                len(feature_columns),
            )

    assert global_feature_names is not None

    valid_indices = []
    all_features = []
    missing_rows = []

    for idx, meta in enumerate(base_dataset.patch_meta):
        sample_id = meta["sample_id"]
        barcode = str(meta["barcode"])

        if sample_id not in sample_feature_df_map:
            raise KeyError(f"sample_id missing in radiomics parquet map: {sample_id}")

        sample_df = sample_feature_df_map[sample_id]

        if barcode not in sample_df.index:
            missing_rows.append((idx, sample_id, barcode))
            continue

        # feat = sample_df.loc[barcode, global_feature_names].to_numpy(dtype=np.float32)

        row = sample_df.loc[barcode, global_feature_names]
        feat = pd.to_numeric(row, errors="coerce").to_numpy(dtype=np.float32)

        if np.isnan(feat).any():
            if log_cfg["enabled"] and log_cfg["log_nan_skips"]:
                logger.warning(
                    "[RadiomicsParquet] skip NaN row: sample_id=%s barcode=%s",
                    sample_id,
                    barcode,
                )
            continue

        if feat.ndim != 1:
            raise ValueError(
                f"Expected 1D feature vector, got shape={feat.shape} "
                f"for sample_id={sample_id}, barcode={barcode}"
            )

        all_features.append(feat)
        valid_indices.append(idx)

    if missing_rows:
        preview = missing_rows[:10]
        raise KeyError(
            f"{len(missing_rows)} barcodes from STNetDataset were not found in radiomics parquet. "
            f"Examples: {preview}"
        )

    rad_targets = torch.tensor(np.stack(all_features, axis=0), dtype=torch.float32)
    if log_cfg["enabled"] and log_cfg["log_final_summary"]:
        logger.info(
            "[RadiomicsParquet] assembled targets shape = %s",
            tuple(rad_targets.shape),
        )
        logger.info(
            "[RadiomicsParquet] kept %d / %d patches after NaN filtering",
            len(valid_indices),
            len(base_dataset.patch_meta),
        )

    return rad_targets, global_feature_names, valid_indices