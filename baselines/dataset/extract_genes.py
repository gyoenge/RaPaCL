from __future__ import annotations

import json
from pathlib import Path

import scanpy as sc
from hest import get_k_genes

from src.common.config import apply_cli_overrides, load_yaml, parse_common_args
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml


def load_h5ad_list(adata_dir: str, sample_files: list[str], logger):
    ad_list = []
    for sample_file in sample_files:
        sample_path = Path(adata_dir) / sample_file
        if not sample_path.exists():
            raise FileNotFoundError(f"h5ad not found: {sample_path}")

        logger.info("Loading h5ad: %s", sample_path)
        ad = sc.read_h5ad(sample_path)
        ad_list.append(ad)

    return ad_list


def save_gene_list_json(save_path: Path, genes: list[str]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"genes": genes}, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_common_args()
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    paths_cfg = cfg.setdefault("paths", {})
    extract_cfg = cfg.setdefault("extract", {})

    log_dir = paths_cfg.get("log_dir", "logs/data")
    output_root = paths_cfg.get("output_root", "outputs/data")

    timestamp, logger = setup_logger(log_dir=log_dir, name="extract_genes")
    run_root = ensure_dir(Path(output_root) / f"extract_genes_{timestamp}")
    save_yaml(cfg, run_root / "resolved_config.yaml")

    adata_dir = paths_cfg["adata_dir"]
    sample_files = extract_cfg["sample_files"]
    k_values = extract_cfg.get("k_values", [50, 100, 250, 500])
    criteria_values = extract_cfg.get("criteria_values", ["var", "mean"])
    min_cells_pct = extract_cfg.get("min_cells_pct", 0.1)
    save_dir = Path(paths_cfg["gene_output_dir"])
    ensure_dir(save_dir)

    ad_list = load_h5ad_list(adata_dir, sample_files, logger)

    summary_lines = []

    for k in k_values:
        for criteria in criteria_values:
            logger.info(
                "Extracting genes | k=%d | criteria=%s | min_cells_pct=%.3f",
                k,
                criteria,
                min_cells_pct,
            )

            extracted_genes = get_k_genes(
                ad_list,
                k=k,
                criteria=criteria,
                min_cells_pct=min_cells_pct,
            )
            extracted_genes = list(extracted_genes)

            save_path = save_dir / f"{criteria}_{k}genes.json"
            save_gene_list_json(save_path, extracted_genes)

            logger.info("Saved gene list: %s", save_path)
            logger.info("Number of genes: %d", len(extracted_genes))

            summary_lines.append(
                {
                    "criteria": criteria,
                    "k": k,
                    "num_genes": len(extracted_genes),
                    "save_path": str(save_path),
                }
            )

    summary_path = run_root / "gene_extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_lines, f, ensure_ascii=False, indent=2)

    logger.info("Saved summary: %s", summary_path)
    logger.info("Gene extraction finished successfully")


if __name__ == "__main__":
    main()