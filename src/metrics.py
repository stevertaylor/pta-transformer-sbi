"""Evaluation metrics: Hellinger distance, calibration / P-P, point errors."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def hellinger_distance_grid(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two nonneg density grids on a common uniform grid.

    Converts densities to PMFs (normalise to sum=1) then computes
    H(P,Q) = (1/sqrt(2)) * sqrt( sum( (sqrt(p_i) - sqrt(q_i))^2 ) )

    Returns 0 for identical distributions, up to 1 for disjoint support.
    """
    p_pmf = np.maximum(p, 0).astype(np.float64)
    q_pmf = np.maximum(q, 0).astype(np.float64)
    p_sum = p_pmf.sum()
    q_sum = q_pmf.sum()
    if p_sum <= 0 or q_sum <= 0:
        return 1.0
    p_pmf /= p_sum
    q_pmf /= q_sum
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_pmf) - np.sqrt(q_pmf)) ** 2)))


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
