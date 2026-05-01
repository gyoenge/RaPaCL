from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
from huggingface_hub import HfApi, login, snapshot_download

from src.common.config import apply_cli_overrides, load_yaml, parse_common_args
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml


def huggingface_checkin(hf_token: str, logger) -> None:
    login(token=hf_token)

    try:
        user_info = HfApi().whoami()
        logger.info("현재 로그인된 계정: %s", user_info["name"])
    except Exception as e:
        logger.warning("로그인 정보 확인 실패: %s", e)


def load_hest_metadata(metadata_uri: str) -> pd.DataFrame:
    return pd.read_csv(metadata_uri)


def filter_idc_samples(
    meta_df: pd.DataFrame,
    species: str = "Homo sapiens",
    oncotree_code: str = "IDC",
    exclude_tech: Sequence[str] | None = None,
) -> pd.DataFrame:
    filtered = meta_df.copy()

    if species:
        filtered = filtered[filtered["species"] == species]

    if exclude_tech:
        filtered = filtered[~filtered["st_technology"].isin(exclude_tech)]

    if oncotree_code:
        filtered = filtered[filtered["oncotree_code"] == oncotree_code]

    return filtered


def build_id_patterns(ids_to_query: Sequence[str]) -> list[str]:
    return [f"*{sample_id}[_.]**" for sample_id in ids_to_query]


def download_hest_idc(
    download_dir: str,
    metadata_uri: str,
    exclude_tech: Sequence[str],
    logger,
) -> None:
    meta_df = load_hest_metadata(metadata_uri)
    meta_df = filter_idc_samples(
        meta_df=meta_df,
        species="Homo sapiens",
        oncotree_code="IDC",
        exclude_tech=exclude_tech,
    )

    ids_to_query = meta_df["id"].astype(str).values.tolist()
    allow_patterns = build_id_patterns(ids_to_query)

    ensure_dir(download_dir)

    logger.info("HEST IDC 다운로드 시작")
    logger.info("download_dir=%s", download_dir)
    logger.info("num_ids=%d", len(ids_to_query))

    snapshot_download(
        repo_id="MahmoodLab/hest",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=allow_patterns,
    )

    logger.info("HEST IDC 다운로드 완료")
    logger.info("Downloaded IDs count: %d", len(ids_to_query))
    logger.info("Downloaded IDs: %s", ids_to_query)


def download_hest_bench_idc(
    download_dir: str,
    allow_patterns: Sequence[str],
    logger,
) -> None:
    ensure_dir(download_dir)

    logger.info("HEST-Bench IDC 다운로드 시작")
    logger.info("download_dir=%s", download_dir)
    logger.info("allow_patterns=%s", list(allow_patterns))

    snapshot_download(
        repo_id="MahmoodLab/hest-bench",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=list(allow_patterns),
    )

    logger.info("HEST-Bench IDC 다운로드 완료")


def main() -> None:
    load_dotenv()

    args = parse_common_args()
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    paths_cfg = cfg.setdefault("paths", {})
    hf_cfg = cfg.setdefault("huggingface", {})
    download_cfg = cfg.setdefault("download", {})

    log_dir = paths_cfg.get("log_dir", "logs/data")
    output_root = paths_cfg.get("output_root", "outputs/data")

    timestamp, logger = setup_logger(log_dir=log_dir, name="download_hest")
    run_root = ensure_dir(Path(output_root) / f"download_hest_{timestamp}")
    save_yaml(cfg, run_root / "resolved_config.yaml")

    hf_token = os.getenv(hf_cfg.get("token_env_key", "HF_TOKEN"))
    if not hf_token:
        raise ValueError(
            f"Hugging Face token not found. Set environment variable: {hf_cfg.get('token_env_key', 'HF_TOKEN')}"
        )

    huggingface_checkin(hf_token, logger)

    mode = download_cfg.get("mode", "both")
    logger.info("Download mode: %s", mode)

    if mode in {"hest", "both"}:
        download_hest_idc(
            download_dir=paths_cfg["hest_download_dir"],
            metadata_uri=download_cfg.get(
                "metadata_uri", "hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv"
            ),
            exclude_tech=download_cfg.get(
                "exclude_tech",
                ["Spatial Transcriptomics", "Visium HD", "Visium"],
            ),
            logger=logger,
        )

    if mode in {"bench", "both"}:
        download_hest_bench_idc(
            download_dir=paths_cfg["bench_download_dir"],
            allow_patterns=download_cfg.get("bench_allow_patterns", ["*IDC/*"]),
            logger=logger,
        )

    logger.info("모든 다운로드 작업 완료")


if __name__ == "__main__":
    main()