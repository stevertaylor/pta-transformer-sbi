"""Tokenization: convert raw TOA-level data to input feature tensors."""

from __future__ import annotations

import torch
import numpy as np


def tokenize(
    t: np.ndarray,
    sigma: np.ndarray,
    residuals: np.ndarray,
    freq_mhz: np.ndarray | None = None,
    backend_id: np.ndarray | None = None,
) -> dict:
    """Convert arrays to a feature dict of torch tensors.

    Token features (per TOA):
        t_norm      – (t - t_min) / (t_max - t_min + eps)
        dt_prev     – normalised gap to previous TOA (0 for first)
        r_over_sig  – residual / sigma
        log_sigma   – log10(sigma)
        r_raw       – raw residual (for information, not always used)
        freq_norm   – (freq_mhz - 1400) / 1000  if provided, else 0
        backend_id  – integer, 0-indexed

    Returns dict of 1-D tensors, each of length N.
    """
    eps = 1e-10
    t = t.astype(np.float32)
    t_min, t_max = t.min(), t.max()
    tspan = t_max - t_min + eps
    t_norm = (t - t_min) / tspan

    dt = np.zeros_like(t)
    dt[1:] = np.diff(t) / tspan
    dt_prev = dt

    sigma32 = sigma.astype(np.float32)
    r32 = residuals.astype(np.float32)
    r_over_sig = r32 / (sigma32 + eps)
    log_sigma = np.log10(sigma32 + eps)

    if freq_mhz is not None:
        freq_norm = (freq_mhz.astype(np.float32) - 1400.0) / 1000.0
    else:
        freq_norm = np.zeros_like(t)

    if backend_id is None:
        backend_id = np.zeros(len(t), dtype=np.int64)

    return {
        "t_norm": torch.from_numpy(t_norm),
        "dt_prev": torch.from_numpy(dt_prev),
        "r_over_sig": torch.from_numpy(r_over_sig),
        "log_sigma": torch.from_numpy(log_sigma),
        "r_raw": torch.from_numpy(r32),
        "freq_norm": torch.from_numpy(freq_norm),
        "backend_id": torch.from_numpy(backend_id.astype(np.int64)),
    }


# Number of continuous token features (excluding backend_id which is categorical)
N_CONTINUOUS_FEATURES = 6  # t_norm, dt_prev, r_over_sig, log_sigma, r_raw, freq_norm
