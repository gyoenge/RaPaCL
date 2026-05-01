from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def compute_genewise_pcc(targets: np.ndarray, preds: np.ndarray):
    gene_pccs = []
    for i in range(targets.shape[1]):
        corr, _ = pearsonr(targets[:, i], preds[:, i])
        gene_pccs.append(0.0 if np.isnan(corr) else float(corr))

    mean_pcc = float(np.mean(gene_pccs))
    return mean_pcc, gene_pccs