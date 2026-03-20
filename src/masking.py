"""Structured masking / augmentation transforms for variable-length TOA sequences.

All transforms operate on numpy arrays and return *copies* (never modify in-place).
Each transform returns a boolean keep-mask over the original TOA indices.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def season_dropout(
    t: np.ndarray,
    rng: np.random.Generator,
    n_drop: int = 1,
    season_length_yr: float = 0.7,
) -> np.ndarray:
    """Drop all TOAs within *n_drop* randomly chosen seasonal windows.

    Returns boolean keep-mask of shape (N,).
    """
    keep = np.ones(len(t), dtype=bool)
    t_min, t_max = t.min(), t.max()
    tspan = t_max - t_min
    if tspan <= season_length_yr:
        return keep
    for _ in range(n_drop):
        hi = max(t_min, t_max - season_length_yr * 0.5)
        if hi <= t_min:
            break
        start = rng.uniform(t_min, hi)
        end = start + season_length_yr
        keep &= ~((t >= start) & (t <= end))
    return keep


def end_truncation(
    t: np.ndarray,
    rng: np.random.Generator,
    min_frac: float = 0.4,
    max_frac: float = 0.85,
) -> np.ndarray:
    """Keep only TOAs before a random cutoff, mimicking a shorter baseline."""
    t_min, t_max = t.min(), t.max()
    frac = rng.uniform(min_frac, max_frac)
    cutoff = t_min + frac * (t_max - t_min)
    return t <= cutoff


def cadence_thinning(
    t: np.ndarray,
    rng: np.random.Generator,
    keep_prob: float = 0.5,
) -> np.ndarray:
    """Randomly keep each TOA with probability *keep_prob*."""
    return rng.random(len(t)) < keep_prob


def apply_random_masking(
    t: np.ndarray,
    rng: np.random.Generator,
    severity: float = 0.3,
    min_remaining: int = 20,
) -> np.ndarray:
    """Apply a randomly chosen masking augmentation.

    Parameters
    ----------
    severity : float in [0, 1]
        Controls how aggressively data are removed.
        0 → no masking; 1 → heavy masking.
    min_remaining : int
        Hard lower bound on how many TOAs to keep.

    Returns boolean keep-mask.
    """
    if severity <= 0 or len(t) <= min_remaining:
        return np.ones(len(t), dtype=bool)

    # Choose augmentation type
    choice = rng.random()
    if choice < 0.35:
        n_drop = max(1, int(round(severity * 3)))
        keep = season_dropout(t, rng, n_drop=n_drop)
    elif choice < 0.65:
        max_frac = max(0.3, 1.0 - severity * 0.6)
        keep = end_truncation(t, rng, min_frac=0.3, max_frac=max_frac)
    elif choice < 0.85:
        kp = max(0.2, 1.0 - severity)
        keep = cadence_thinning(t, rng, keep_prob=kp)
    else:
        # compound: season dropout + thinning
        keep = season_dropout(t, rng, n_drop=1)
        kp = max(0.4, 1.0 - severity * 0.5)
        keep &= cadence_thinning(t, rng, keep_prob=kp)

    # Enforce minimum
    if keep.sum() < min_remaining:
        false_idx = np.where(~keep)[0]
        need = min_remaining - keep.sum()
        if len(false_idx) >= need:
            flip = rng.choice(false_idx, size=need, replace=False)
            keep[flip] = True
        else:
            keep[:] = True

    return keep
