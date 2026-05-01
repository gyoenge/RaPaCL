# For HEST-IDC-Radiomics Dataset

# from __future__ import annotations

import json
import os
from typing import Optional, Literal

import numpy as np
import pandas as pd
import torch
from hest.bench.st_dataset import H5PatchDataset, load_adata


class HestRadiomicsDataset(torch.utils.data.Dataset):
    """
    HEST-IDC-Radiomics Dataset

    Returns:
        {
            "image": Tensor[C, H, W],
            "gene": Tensor[num_genes],
            "radiomics": Tensor[num_features],
            "target_label": LongTensor scalar,
            "target_distribution": Tensor[num_classes],
            "barcode": str,
            "patch_idx": int,
            "sample_id": str,
        }
    """

    def __init__(
        self,
        bench_data_root: str,
        gene_list_path: str,
        feature_list_path: str,
        radiomics_dir: str = "radiomics_features",
        split_csv_path: Optional[str] = None,
        split_df: Optional[pd.DataFrame] = None,
        transforms=None,
        normalize_gene: bool = True,
        radiomics_fillna: float = 0.0,
        radiomics_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if (split_csv_path is None) == (split_df is None):
            raise ValueError("Provide exactly one of split_csv_path or split_df.")

        self.split_df = (
            split_df.reset_index(drop=True).copy()
            if split_df is not None
            else pd.read_csv(split_csv_path)
        )

        with open(gene_list_path, "r", encoding="utf-8") as f:
            gene_info = json.load(f)
        self.genes = gene_info["genes"]

        with open(feature_list_path, "r", encoding="utf-8") as f:
            self.feature_cols = [line.strip() for line in f if line.strip()]

        self.bench_data_root = bench_data_root
        self.radiomics_dir = radiomics_dir
        self.transforms = transforms
        self.normalize_gene = normalize_gene
        self.radiomics_fillna = radiomics_fillna
        self.radiomics_dtype = radiomics_dtype

        self.samples: list[dict] = []

        self._build_samples()

    @staticmethod
    def _normalize_barcode(barcode) -> str:
        barcode = str(barcode)

        if barcode.startswith("b'") and barcode.endswith("'"):
            barcode = barcode[2:-1]

        barcode = barcode.replace("-1", "")

        if "_" in barcode:
            barcode = barcode.split("_")[-1]

        return barcode

    def _build_samples(self) -> None:
        for _, row in self.split_df.iterrows():
            patches_h5_path = os.path.join(self.bench_data_root, row["patches_path"])
            expr_path = os.path.join(self.bench_data_root, row["expr_path"])

            sample_id = self._infer_sample_id(row, patches_h5_path)
            radiomics_path = os.path.join(
                self.bench_data_root,
                self.radiomics_dir,
                f"{sample_id}.parquet",
            )

            self._check_file(patches_h5_path, "Patch")
            self._check_file(expr_path, "Expr")
            self._check_file(radiomics_path, "Radiomics")

            patch_items = self._load_patches(patches_h5_path)
            barcodes = [item["barcode"] for item in patch_items]

            gene_df = load_adata(
                expr_path,
                genes=self.genes,
                barcodes=barcodes,
                normalize=self.normalize_gene,
            )

            radiomics_df = pd.read_parquet(radiomics_path)
            radiomics_df["barcode"] = radiomics_df["barcode"].apply(self._normalize_barcode)
            radiomics_df = radiomics_df.set_index("barcode")

            missing_features = [
                col for col in self.feature_cols if col not in radiomics_df.columns
            ]
            if missing_features:
                raise ValueError(
                    f"Missing radiomics feature columns in {radiomics_path}: "
                    f"{missing_features[:10]}"
                )

            for i, item in enumerate(patch_items):
                barcode = self._normalize_barcode(item["barcode"])

                if barcode not in radiomics_df.index:
                    # raise KeyError(
                    #     f"Barcode {barcode} not found in radiomics file: {radiomics_path}"
                    # )
                    continue

                rad_row = radiomics_df.loc[barcode]

                radiomics_values = (
                    rad_row[self.feature_cols]
                    .astype(float)
                    .fillna(self.radiomics_fillna)
                    .values
                )

                sample = {
                    "image": item["image"],
                    "gene": torch.tensor(gene_df.iloc[i].values, dtype=torch.float32),
                    "radiomics": torch.tensor(
                        radiomics_values,
                        dtype=self.radiomics_dtype,
                    ),
                    "barcode": barcode,
                    "patch_idx": int(rad_row["patch_idx"])
                    if "patch_idx" in rad_row.index
                    else int(item["patch_idx"]),
                    "sample_id": sample_id,
                }

                if "target_label" in rad_row.index:
                    sample["target_label"] = torch.tensor(
                        int(rad_row["target_label"]),
                        dtype=torch.long,
                    )

                if "target_distribution" in rad_row.index:
                    sample["target_distribution"] = self._parse_distribution(
                        rad_row["target_distribution"]
                    )

                self.samples.append(sample)

    def _load_patches(self, patches_h5_path: str) -> list[dict]:
        patch_dataset = H5PatchDataset(patches_h5_path)

        patch_items: list[dict] = []

        for i in range(len(patch_dataset)):
            chunk = patch_dataset[i]
            chunk_imgs = chunk["imgs"]
            chunk_barcodes = chunk["barcodes"]

            if isinstance(chunk_imgs, torch.Tensor):
                chunk_imgs = chunk_imgs.numpy()
            if isinstance(chunk_barcodes, torch.Tensor):
                chunk_barcodes = chunk_barcodes.numpy()

            if chunk_imgs.ndim == 3:
                chunk_imgs = np.expand_dims(chunk_imgs, axis=0)
                chunk_barcodes = [chunk_barcodes]

            for local_idx, (barcode, img) in enumerate(zip(chunk_barcodes, chunk_imgs)):
                barcode_str = self._to_str_barcode(barcode)

                patch_items.append(
                    {
                        "image": img,
                        "barcode": barcode_str,
                        "patch_idx": len(patch_items),
                    }
                )

        return patch_items

    def _parse_distribution(self, value) -> torch.Tensor:
        if isinstance(value, str):
            value = json.loads(value)

        if isinstance(value, dict):
            value = list(value.values())

        value = np.asarray(value, dtype=np.float32)
        return torch.tensor(value, dtype=torch.float32)

    def _infer_sample_id(self, row: pd.Series, patches_h5_path: str) -> str:
        if "sample_id" in row:
            return str(row["sample_id"])

        filename = os.path.basename(patches_h5_path)
        return os.path.splitext(filename)[0]

    @staticmethod
    def _to_str_barcode(barcode) -> str:
        if isinstance(barcode, bytes):
            return barcode.decode("utf-8")
        if isinstance(barcode, np.ndarray):
            barcode = barcode.item()
            if isinstance(barcode, bytes):
                return barcode.decode("utf-8")
        return str(barcode)

    @staticmethod
    def _check_file(path: str, name: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx].copy()

        img = sample["image"]

        if self.transforms is not None:
            img = self.transforms(img)

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(np.array(img))

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0

        sample["image"] = img
        return sample
    


"""
Usage example:

dataset = HestRadiomicsDataset(
    bench_data_root="/root/workspace/hest_data/eval/bench_data/IDC",
    split_csv_path="./splits/train.csv",
    gene_list_path="./var_250genes.json",
    feature_list_path="./feature_list.txt",
    radiomics_dir="radiomics_features",
)
"""

