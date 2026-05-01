from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from hest.bench.st_dataset import H5PatchDataset, load_adata


class STNetDataset(torch.utils.data.Dataset):
    """
    STNet dataset for patch image -> gene expression prediction.

    Expected split columns:
      - patches_path
      - expr_path

    `gene_list_path` should contain JSON like:
      {"genes": ["geneA", "geneB", ...]}
    """

    def __init__(
        self,
        bench_data_root: str,
        gene_list_path: str,
        split_csv_path: Optional[str] = None,
        split_df: Optional[pd.DataFrame] = None,
        transforms=None,
    ) -> None:
        super().__init__()

        if (split_csv_path is None) == (split_df is None):
            raise ValueError("Provide exactly one of split_csv_path or split_df.")

        if split_df is not None:
            self.split_df = split_df.reset_index(drop=True).copy()
        else:
            self.split_df = pd.read_csv(split_csv_path)

        with open(gene_list_path, "r", encoding="utf-8") as f:
            gene_info = json.load(f)
        self.genes = gene_info["genes"]

        self.transforms = transforms
        self.images: list[np.ndarray] = []
        self.targets: list[torch.Tensor] = []

        self._build_samples(bench_data_root)

    def _build_samples(self, bench_data_root: str) -> None:
        for _, row in self.split_df.iterrows():
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

                for barcode, img in zip(chunk_barcodes, chunk_imgs):
                    if isinstance(barcode, bytes):
                        barcode_str = barcode.decode("utf-8")
                    else:
                        barcode_str = str(barcode)

                    slide_barcodes.append(barcode_str)
                    slide_imgs.append(img)

            adata_df = load_adata(
                expr_path,
                genes=self.genes,
                barcodes=slide_barcodes,
                normalize=True,
            )

            for j in range(len(slide_imgs)):
                self.images.append(slide_imgs[j])
                self.targets.append(
                    torch.tensor(adata_df.iloc[j].values, dtype=torch.float32)
                )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        target = self.targets[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(np.array(img))

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0

        return img, target
