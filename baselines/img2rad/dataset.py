from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hest.bench.st_dataset import H5PatchDataset, load_adata


def _load_gene_names(gene_list_path: str) -> list[str]:
    with open(gene_list_path, "r", encoding="utf-8") as f:
        gene_info = json.load(f)

    if isinstance(gene_info, dict):
        if "genes" in gene_info:
            return gene_info["genes"]
        if "gene_names" in gene_info:
            return gene_info["gene_names"]
        raise ValueError(f"Unsupported gene list dict keys in {gene_list_path}")

    if isinstance(gene_info, list):
        return gene_info

    raise ValueError(f"Unsupported gene list format in {gene_list_path}")


class STNetDataset(Dataset):
    def __init__(
        self,
        bench_data_root: str,
        gene_list_path: str,
        split_csv_path: Optional[str] = None,
        split_df: Optional[pd.DataFrame] = None,
        transforms_=None,
        normalize_expression: bool = True,
    ):
        assert (split_csv_path is not None) ^ (split_df is not None), (
            "Provide exactly one of split_csv_path or split_df."
        )

        if split_df is not None:
            self.split_df = split_df.reset_index(drop=True).copy()
        else:
            self.split_df = pd.read_csv(split_csv_path)

        self.genes = _load_gene_names(gene_list_path)
        self.transforms = transforms_
        self.normalize_expression = normalize_expression

        self.images: list[np.ndarray] = []
        self.targets: list[torch.Tensor] = []
        self.patch_meta: list[dict[str, str]] = []

        for _, row in self.split_df.iterrows():
            sample_id = str(row["sample_id"])
            patches_h5_path = os.path.join(bench_data_root, row["patches_path"])
            expr_path = os.path.join(bench_data_root, row["expr_path"])

            if not os.path.isfile(patches_h5_path):
                raise FileNotFoundError(f"Patch file not found: {patches_h5_path}")
            if not os.path.isfile(expr_path):
                raise FileNotFoundError(f"Expr file not found: {expr_path}")

            patch_dataset = H5PatchDataset(patches_h5_path)

            slide_imgs: list[np.ndarray] = []
            slide_barcodes: list[str] = []

            for i in range(len(patch_dataset)):
                item = patch_dataset[i]
                chunk_imgs = item["imgs"]
                chunk_barcodes = item["barcodes"]

                if isinstance(chunk_imgs, torch.Tensor):
                    chunk_imgs = chunk_imgs.numpy()
                if isinstance(chunk_barcodes, torch.Tensor):
                    chunk_barcodes = chunk_barcodes.numpy()

                if chunk_imgs.ndim == 3:
                    chunk_imgs = np.expand_dims(chunk_imgs, axis=0)
                    chunk_barcodes = [chunk_barcodes]

                for barcode, img in zip(chunk_barcodes, chunk_imgs):
                    barcode_str = (
                        barcode.decode("utf-8") if isinstance(barcode, bytes) else str(barcode)
                    )
                    slide_barcodes.append(barcode_str)
                    slide_imgs.append(img)

            adata_df = load_adata(
                expr_path,
                genes=self.genes,
                barcodes=slide_barcodes,
                normalize=self.normalize_expression,
            )

            for j in range(len(slide_imgs)):
                self.images.append(slide_imgs[j])
                self.targets.append(
                    torch.tensor(adata_df.iloc[j].values, dtype=torch.float32)
                )
                self.patch_meta.append(
                    {
                        "sample_id": sample_id,
                        "barcode": slide_barcodes[j],
                    }
                )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        target = self.targets[idx]

        if self.transforms:
            img = self.transforms(img)

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(np.array(img))

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0

        return img, target


class RadiomicsTargetDataset(Dataset):
    def __init__(self, base_dataset: STNetDataset, radiomics_targets: torch.Tensor):
        if len(base_dataset) != len(radiomics_targets):
            raise ValueError(
                f"Length mismatch: len(base_dataset)={len(base_dataset)}, "
                f"len(radiomics_targets)={len(radiomics_targets)}"
            )
        self.base_dataset = base_dataset
        self.radiomics_targets = radiomics_targets

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, _ = self.base_dataset[idx]
        target = self.radiomics_targets[idx]
        return img, target
    

class GeneWithRadiomicsDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        radiomics_features: torch.Tensor,
    ):
        if len(base_dataset) != len(radiomics_features):
            raise ValueError(
                f"Length mismatch: len(base_dataset)={len(base_dataset)}, "
                f"len(radiomics_features)={len(radiomics_features)}"
            )
        self.base_dataset = base_dataset
        self.radiomics_features = radiomics_features

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        img, gene_target = self.base_dataset[idx]
        raw_radiomics = self.radiomics_features[idx]
        return img, raw_radiomics, gene_target
