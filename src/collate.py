"""Custom collate function: pads variable-length TOA sequences and builds masks."""

from __future__ import annotations

import torch
from typing import List


def collate_fn(batch: List[dict]) -> dict:
    """Collate a list of dataset items into a padded batch.

    Supports both standard (theta) and factorized (theta_global + theta_wn)
    modes, selected automatically by the presence of keys in the batch items.

    Returns
    -------
    dict with keys:
        features     – (B, L_max, n_feat) padded continuous features
        backend_id   – (B, L_max) padded integer
        mask         – (B, L_max) boolean, True = valid token
        seq_lens     – (B,)
        tspan_yr     – (B,)
    Standard mode also includes:
        theta        – (B, D)
    Factorized mode also includes:
        theta_global   – (B, 4)
        theta_wn       – (B, B_max, 3) padded per-backend WN params
        backend_active – (B, B_max) boolean, True = backend is present
    """
    B = len(batch)
    seq_lens = [item["seq_len"] for item in batch]
    L_max = max(seq_lens)

    # Continuous feature keys (order matters – must match model expectation)
    feat_keys = ["t_norm", "dt_prev", "r_over_sig", "log_sigma", "r_raw", "freq_norm"]
    n_feat = len(feat_keys)

    features = torch.zeros(B, L_max, n_feat)
    backend_id = torch.zeros(B, L_max, dtype=torch.long)
    mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["seq_len"]
        mask[i, :L] = True
        for j, key in enumerate(feat_keys):
            features[i, :L, j] = item["tokens"][key]
        backend_id[i, :L] = item["tokens"]["backend_id"]

    tspan_yr = torch.stack([item["tspan_yr"] for item in batch])

    result = {
        "features": features,
        "backend_id": backend_id,
        "mask": mask,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "tspan_yr": tspan_yr,
    }

    # Standard mode
    if "theta" in batch[0]:
        result["theta"] = torch.stack([item["theta"] for item in batch])

    # Factorized mode
    if "theta_global" in batch[0]:
        result["theta_global"] = torch.stack([item["theta_global"] for item in batch])
        result["theta_wn"] = torch.stack([item["theta_wn"] for item in batch])
        result["backend_active"] = torch.stack(
            [item["backend_active"] for item in batch]
        )

    return result
