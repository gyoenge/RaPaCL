import os
import json
from typing import Dict, Any, List, Optional

import pandas as pd
from torch.utils.data import Dataset

from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES


class HestRadiomicsDataset(Dataset):
    def __init__(
        self,
        radiomics_file: str,
        root_dir: Optional[str] = None,
        label_col: str = "target_label",
        id_col: str = "barcode",
    ):
        super().__init__()

        self.radiomics_file = radiomics_file
        self.root_dir = root_dir
        self.label_col = label_col
        self.id_col = id_col

        self.df = pd.read_parquet(radiomics_file)

        self.feature_cols = RADIOMICS_FEATURES_NAMES 

        missing_cols = [col for col in self.feature_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols[:10]}")

        if self.id_col not in self.df.columns:
            raise ValueError(f"Missing id column: {self.id_col}")

        if self.label_col not in self.df.columns:
            raise ValueError(f"Missing label column: {self.label_col}")

        self.radiomics_features_min_max = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image_id = str(row[self.id_col])

        radiomics_features_name = self.feature_cols
        radiomics_features = []

        for feature_name in radiomics_features_name:
            feature_value = row[feature_name]

            if pd.isna(feature_value):
                feature_value = 0.0

            # if self.normalize:
            #     feature_value = self._normalize_radiomics_feature(
            #         feature_name=feature_name,
            #         value=float(feature_value),
            #     )
            # else:
            #     feature_value = float(feature_value)
            feature_value = float(feature_value)

            radiomics_features.append(feature_value)

        label = int(row[self.label_col])

        return {
            "idx": idx,
            "id": image_id,
            "radiomics_features": radiomics_features,
            "radiomics_features_name": radiomics_features_name,
            "label": label,
        }


def radiomics_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [item["id"] for item in batch]
    idxes = [item["idx"] for item in batch]
    labels = [item["label"] for item in batch]

    radiomics_features = [item["radiomics_features"] for item in batch]
    radiomics_features_name = batch[0]["radiomics_features_name"]

    radiomics_features = pd.DataFrame(
        radiomics_features,
        columns=radiomics_features_name,
    )

    return {
        "idxes": idxes,
        "ids": ids,
        "radiomics_features": radiomics_features,
        "radiomics_features_name": radiomics_features_name,
        "labels": labels,
    }

