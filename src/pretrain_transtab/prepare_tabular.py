"""
Prepare Custom Datasets from Radiomcis Feature Parquets, 
    for TransTab Pretraining.

Structure: 
radiomics_normed_tabular/
├── data_processed.csv
└── numerical_feature.txt 

data_processed.csv
,original_firstorder_mean,original_glcm_contrast,wavelet_h_glrlm_runentropy,target_label
0,0.182,1.54,3.21,1
1,0.093,0.88,2.74,0
2,0.201,1.12,3.05,1
--> use sample_id by number as target_label. 

numerical_feature.txt
original_firstorder_mean
original_glcm_contrast
wavelet_h_glrlm_runentropy
--> should be lowercase to avoid error. 

"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import pandas as pd
from src.rapacl.transtab_custom.dataset import load_single_data

from src.common.config import load_yaml
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare custom TransTab dataset from radiomics parquet files"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config file",
    )
    return parser.parse_args()


def is_feature_column(
    col: str,
    meta_columns: set[str],
    exclude_prefixes: tuple[str, ...],
    valid_prefixes: tuple[str, ...],
) -> bool:
    lowered = col.lower()
    return any(lowered.startswith(prefix) for prefix in valid_prefixes)

    # if lowered in meta_columns:
    #     return False
    # for prefix in exclude_prefixes:
    #     if lowered.startswith(prefix):
    #         return False
    # return True


def validate_feature_columns(
    feature_columns_ref: list[str] | None,
    feature_columns: list[str],
    sample_name: str,
) -> list[str]:
    if len(feature_columns) != len(set(feature_columns)):
        dupes = [c for c in set(feature_columns) if feature_columns.count(c) > 1]
        raise ValueError(f"Duplicate feature columns found in {sample_name}: {dupes[:10]}")

    if feature_columns_ref is None:
        return feature_columns

    ref_set = set(feature_columns_ref)
    cur_set = set(feature_columns)

    if ref_set != cur_set:
        missing_in_current = sorted(ref_set - cur_set)
        extra_in_current = sorted(cur_set - ref_set)
        raise ValueError(
            f"Feature columns mismatch in {sample_name}\n"
            f"Missing: {missing_in_current[:10]}\n"
            f"Extra: {extra_in_current[:10]}"
        )

    return feature_columns_ref


def prepare_transtab_dataset(cfg: dict[str, Any], logger) -> Path:
    radiomics_parquet_dir = cfg["paths"]["radiomics_parquet_dir"]
    output_dir = Path(cfg["paths"]["output_dir"])
    dataset_cfg = cfg["dataset"]

    ensure_dir(output_dir)

    meta_columns = {c.lower() for c in dataset_cfg.get("meta_columns", [])}
    exclude_prefixes = tuple(
        prefix.lower() for prefix in dataset_cfg.get("exclude_prefixes", [])
    )
    valid_prefixes = tuple(
        prefix.lower() for prefix in dataset_cfg.get("radiomics_valid_prefixes", [])
    )
    sample_label_map = dataset_cfg.get("sample_label_map", {})
    lowercase_columns = dataset_cfg.get("lowercase_columns", True)
    check_numeric = dataset_cfg.get("check_numeric", True)
    check_nan = dataset_cfg.get("check_nan", True)

    parquet_files = sorted(glob.glob(str(Path(radiomics_parquet_dir) / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in radiomics_parquet_dir: {radiomics_parquet_dir}"
        )

    logger.info("Found %d parquet files", len(parquet_files))
    logger.info("radiomics_parquet_dir: %s", radiomics_parquet_dir)
    logger.info("output_dir: %s", output_dir)

    all_dfs: list[pd.DataFrame] = []
    feature_columns_ref: list[str] | None = None

    for parquet_path in parquet_files:
        sample_name = Path(parquet_path).stem

        if sample_name not in sample_label_map:
            raise ValueError(f"Sample {sample_name} not found in sample_label_map")

        target_label = sample_label_map[sample_name]

        df = pd.read_parquet(parquet_path)

        if lowercase_columns:
            df.columns = [c.lower() for c in df.columns]

        feature_columns = [
            c for c in df.columns
            if is_feature_column(
                col=c,
                meta_columns=meta_columns,
                exclude_prefixes=exclude_prefixes,
                valid_prefixes=valid_prefixes,
            )
        ]

        feature_columns_ref = validate_feature_columns(
            feature_columns_ref=feature_columns_ref,
            feature_columns=feature_columns,
            sample_name=sample_name,
        )

        df_feat = df[feature_columns].copy()
        df_feat["target_label"] = target_label

        all_dfs.append(df_feat)

        logger.info(
            "[OK] %s: original_shape=%s -> feature_only_shape=%s",
            sample_name,
            df.shape,
            df_feat.shape,
        )

    merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    if feature_columns_ref is None:
        raise ValueError("No feature columns found.")

    if check_numeric:
        non_numeric_cols = [
            c for c in feature_columns_ref
            if not pd.api.types.is_numeric_dtype(merged_df[c])
        ]
        if non_numeric_cols:
            raise TypeError(
                f"Non-numeric columns found in feature columns: {non_numeric_cols[:20]}"
            )
        logger.info("All feature columns are numeric.")

    if check_nan:
        total_nan = merged_df[feature_columns_ref].isna().sum().sum()
        logger.info("Total NaN in feature columns: %d", total_nan)

    logger.info("Merged dataset shape: %s", merged_df.shape)
    logger.info("Number of numerical features: %d", len(feature_columns_ref))

    csv_path = output_dir / "data_processed.csv"
    txt_path = output_dir / "numerical_feature.txt"

    merged_df.to_csv(csv_path, index=True)

    with open(txt_path, "w", encoding="utf-8") as f:
        for col in feature_columns_ref:
            f.write(col + "\n")

    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved numerical feature list: %s", txt_path)

    return output_dir


def test_transtab_loading(output_dir: Path, cfg: dict[str, Any], logger) -> None:
    runtime_cfg = cfg.get("runtime", {})
    encode_cat = runtime_cfg.get("encode_cat", False)

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_single_data(
        dataname=str(output_dir),
        dataset_config=None,
        encode_cat=encode_cat,
        seed=cfg.get("seed", 42),
    )

    logger.info("TransTab load_single_data test completed.")
    logger.info("cat_cols: %s", cat_cols)
    logger.info("num_cols count: %d", len(num_cols))
    logger.info("bin_cols: %s", bin_cols)
    logger.info("train size: %d", len(trainset[0]))
    logger.info("val size: %d", len(valset[0]))
    logger.info("test size: %d", len(testset[0]))
    logger.info("allset size: %d", len(allset[0]))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed_everything(cfg.get("seed", 42))

    log_dir = cfg["paths"]["log_dir"]
    timestamp, logger = setup_logger(log_dir=log_dir, name="transtab_dataset")

    logger.info("Configuration loaded from: %s", args.config)
    logger.info("Configuration: %s", cfg)

    save_yaml(cfg, Path(log_dir) / f"config_{timestamp}.yaml")

    output_dir = prepare_transtab_dataset(cfg, logger)

    if cfg.get("runtime", {}).get("test_load_single_data", False):
        test_transtab_loading(output_dir, cfg, logger)


if __name__ == "__main__":
    main()
