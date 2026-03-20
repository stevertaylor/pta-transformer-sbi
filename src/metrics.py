"""Evaluation metrics: Hellinger distance, calibration / P-P, point errors."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def hellinger_distance_grid(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two 2-D probability grids (must be normalised).

    H(P,Q) = (1/sqrt(2)) * sqrt( sum( (sqrt(p) - sqrt(q))^2 * dA ) )

    Assumes grids share the same cell area.
    """
    sp = np.sqrt(np.maximum(p, 0))
    sq = np.sqrt(np.maximum(q, 0))
    # Normalise to unit integral
    sp = sp / (sp.sum() + 1e-30)
    sq = sq / (sq.sum() + 1e-30)
    return float(np.sqrt(0.5 * np.sum((sp - sq) ** 2)))


def calibration_percentiles(
    true_theta: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Compute percentile rank of true theta in posterior samples.

    Parameters
    ----------
    true_theta : (N, D) true parameter values
    samples    : (N, S, D) posterior samples

    Returns
    -------
    percentiles : (N, D) in [0, 1]
    """
    N, S, D = samples.shape
    percentiles = np.zeros((N, D))
    for d in range(D):
        for i in range(N):
            percentiles[i, d] = np.mean(samples[i, :, d] <= true_theta[i, d])
    return percentiles


def ks_statistic(percentiles: np.ndarray) -> float:
    """KS statistic: max deviation of empirical CDF from uniform diagonal."""
    n = len(percentiles)
    sorted_p = np.sort(percentiles)
    ecdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(ecdf - sorted_p)))


def point_estimate_error(
    true_theta: np.ndarray,
    estimated_theta: np.ndarray,
) -> np.ndarray:
    """Per-parameter absolute error. Returns shape (N, D)."""
    return np.abs(true_theta - estimated_theta)
