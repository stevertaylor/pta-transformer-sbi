"""Tokenization: convert raw TOA-level data to input feature tensors."""

from __future__ import annotations

import torch
import numpy as np


# Canonical ordered list of continuous feature keys. Consumers (collate,
# evaluate, inference scripts) should import this to build the feature
# tensor in the correct order.
FEAT_KEYS = (
    "t_norm",
    "dt_prev",
    "r_over_sig",
    "log_sigma",
    "r_raw",
    "freq_norm",
    "log_f",
)

# Number of continuous token features (excluding backend_id which is categorical)
N_CONTINUOUS_FEATURES = len(FEAT_KEYS)


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
        r_over_sig  – residual / sigma (signed-log compressed)
        log_sigma   – log10(sigma)
        r_raw       – raw residual (for information, not always used)
        freq_norm   – (freq_mhz - 1400) / 1000  if provided, else 0
        log_f       – log10(freq_mhz / 1400) — chromatic coordinate. DM noise
                      amplitude scales as 10^(-2 * log_f) = (1400/f)^2, so this
                      is the natural coordinate for the encoder to extract DM
                      structure. Zero if freq_mhz not provided.
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
    # Signed-log transform: compresses extreme values while preserving sign.
    # Raw r_over_sig can reach 1e5 for strong red noise, overflowing float16.
    r_over_sig = np.sign(r_over_sig) * np.log1p(np.abs(r_over_sig))
    log_sigma = np.log10(sigma32 + eps)

    if freq_mhz is not None:
        f32 = freq_mhz.astype(np.float32)
        freq_norm = (f32 - 1400.0) / 1000.0
        # log10(f/1400): natural chromatic coordinate. Clamp to avoid log(0)
        # or negative arguments if a pathological input appears.
        log_f = np.log10(np.maximum(f32, 1.0) / 1400.0).astype(np.float32)
    else:
        freq_norm = np.zeros_like(t)
        log_f = np.zeros_like(t)

    if backend_id is None:
        backend_id = np.zeros(len(t), dtype=np.int64)

    return {
        "t_norm": torch.from_numpy(t_norm),
        "dt_prev": torch.from_numpy(dt_prev),
        "r_over_sig": torch.from_numpy(r_over_sig),
        "log_sigma": torch.from_numpy(log_sigma),
        "r_raw": torch.from_numpy(r32),
        "freq_norm": torch.from_numpy(freq_norm),
        "log_f": torch.from_numpy(log_f),
        "backend_id": torch.from_numpy(backend_id.astype(np.int64)),
    }
