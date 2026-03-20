"""Custom collate function: pads variable-length TOA sequences and builds masks."""

from __future__ import annotations

import torch
from typing import List


def collate_fn(batch: List[dict]) -> dict:
    """Collate a list of dataset items into a padded batch.

    Returns
    -------
    dict with keys:
        theta        – (B, 2)
        features     – (B, L_max, n_feat) padded continuous features
        backend_id   – (B, L_max) padded integer
        mask         – (B, L_max) boolean, True = valid token
        seq_lens     – (B,)
    """
    B = len(batch)
    seq_lens = [item["seq_len"] for item in batch]
    L_max = max(seq_lens)

    theta = torch.stack([item["theta"] for item in batch])

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

    return {
        "theta": theta,
        "features": features,
        "backend_id": backend_id,
        "mask": mask,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
    }
