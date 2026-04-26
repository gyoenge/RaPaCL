import os
import json
from typing import Dict, Any, List

import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset


LABEL_TO_INDEX = {
    "SCC": 0,
    "LCC": 1,
    "ADC": 2,
    "NOS": 3,
}


class MyDataset(Dataset):
    def __init__(
        self,
        jsonl_file: str,
        hdf5_file: str,
        root_dir: str,
        is_train: bool = False,
    ):
        super().__init__()

        self.jsonl_file = jsonl_file
        self.hdf5_file = hdf5_file
        self.root_dir = root_dir
        self.is_train = is_train

        self.data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.hf = h5py.File(hdf5_file, "r")

        min_max_path = os.path.join(root_dir, "radiomics_features_min_max.json")
        with open(min_max_path, "r") as f:
            self.radiomics_features_min_max = json.load(f)

    def __len__(self):
        return len(self.data)

    def __del__(self):
        if hasattr(self, "hf"):
            try:
                self.hf.close()
            except Exception:
                pass

    def _normalize_radiomics_feature(self, feature_name: str, value: float) -> float:
        min_value, max_value = self.radiomics_features_min_max[feature_name]
        denom = max_value - min_value

        if denom == 0:
            return 0.0

        return (value - min_value) / denom

    def _encode_label(self, label: str) -> int:
        return LABEL_TO_INDEX.get(label, -1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        image_id = item["id"]

        radiomics_features_name = list(item["radiomics"].keys())
        radiomics_features = []

        for feature_name in radiomics_features_name:
            feature_value = item["radiomics"][feature_name]
            normalized_value = self._normalize_radiomics_feature(
                feature_name=feature_name,
                value=feature_value,
            )
            radiomics_features.append(normalized_value)

        label = self._encode_label(item["label"])

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

