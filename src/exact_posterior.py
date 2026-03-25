"""Exact Gaussian posterior on the 2-D parameter grid.

Because the signal model is linear in Fourier coefficients and the noise
is Gaussian, the likelihood p(r | θ, schedule) is a multivariate Gaussian
whose covariance depends on θ.  With a uniform prior the posterior is
proportional to the likelihood evaluated on a grid.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Optional
from scipy.linalg import solve_triangular

from .simulator import build_fourier_design_matrix, power_law_spectrum, SEC_PER_YR


def _log_likelihood_single(
    residuals: np.ndarray,
    sigma: np.ndarray,
    F: np.ndarray,
    tspan: float,
    n_modes: int,
    log10_A: float,
    gamma: float,
    jitter: float = 1e-20,
) -> float:
    """Exact log p(r | θ) for one (log10_A, gamma) pair.

    C = diag(σ²) + F Φ(θ) Fᵀ + jitter·I
    log p = -0.5 [ N log(2π) + log|C| + rᵀ C⁻¹ r ]
    """
    N = len(residuals)
    rho = power_law_spectrum(n_modes, tspan, log10_A, gamma)
    phi_diag = np.repeat(rho, 2).astype(np.float64)

    F64 = F.astype(np.float64)
    C = (
        np.diag(sigma.astype(np.float64) ** 2 + jitter)
        + F64 @ np.diag(phi_diag) @ F64.T
    )

    # Cholesky
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return -np.inf

    r64 = residuals.astype(np.float64)
    alpha = np.linalg.solve(L, r64)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * (N * np.log(2 * np.pi) + log_det + np.dot(alpha, alpha)))


def compute_log_likelihood_grid(
    residuals: np.ndarray,
    sigma: np.ndarray,
    F: np.ndarray,
    tspan: float,
    n_modes: int,
    log10_A_grid: np.ndarray,
    gamma_grid: np.ndarray,
    jitter: float = 1e-20,
) -> np.ndarray:
    """Evaluate log-likelihood on a 2-D grid using the Woodbury identity.

    Exploits the low-rank structure C = D + F Φ Fᵀ to decompose a
    (2·n_modes × 2·n_modes) matrix instead of (N × N) at each grid point,
    giving a large speedup when N >> 2·n_modes.

    Returns shape (len(log10_A_grid), len(gamma_grid)).
    """
    N = len(residuals)
    K = 2 * n_modes
    nA = len(log10_A_grid)
    nG = len(gamma_grid)

    # --- Precompute whitened quantities (independent of θ) ---
    d = sigma.astype(np.float64) ** 2 + jitter
    d_inv_sqrt = 1.0 / np.sqrt(d)
    F64 = F.astype(np.float64)
    G = F64 * d_inv_sqrt[:, None]  # D^{-1/2} F
    r_w = residuals.astype(np.float64) * d_inv_sqrt  # D^{-1/2} r

    GtG = G.T @ G  # (K, K)
    Gtr = G.T @ r_w  # (K,)
    rtr = float(np.dot(r_w, r_w))
    log_det_D = float(np.sum(np.log(d)))
    const_term = N * np.log(2.0 * np.pi)

    # --- Vectorise spectrum components over the grid ---
    fk = np.arange(1, n_modes + 1, dtype=np.float64) / tspan
    delta_f = 1.0 / tspan
    prefactor = SEC_PER_YR**2 / (12.0 * np.pi**2) * delta_f

    A2 = (10.0 ** log10_A_grid.astype(np.float64)) ** 2
    fk_pow = np.exp(np.outer(-gamma_grid.astype(np.float64), np.log(fk)))

    ll = np.full((nA, nG), -np.inf, dtype=np.float64)
    diag_idx = np.arange(K)

    for i in range(nA):
        for j in range(nG):
            rho = A2[i] * prefactor * fk_pow[j]
            phi_diag = np.repeat(rho, 2)

            M = GtG.copy()
            M[diag_idx, diag_idx] += 1.0 / phi_diag

            try:
                L = np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                continue

            log_det_Phi = np.sum(np.log(phi_diag))
            log_det_M = 2.0 * np.sum(np.log(np.diag(L)))
            log_det_C = log_det_D + log_det_Phi + log_det_M

            alpha = solve_triangular(L, Gtr, lower=True)
            quad = rtr - float(np.dot(alpha, alpha))

            ll[i, j] = -0.5 * (const_term + log_det_C + quad)

    return ll


@torch.no_grad()
def exact_posterior_grid(
    residuals: np.ndarray,
    sigma: np.ndarray,
    F: np.ndarray,
    tspan: float,
    n_modes: int,
    prior_bounds: dict,
    n_grid: int = 100,
    jitter: float = 1e-20,
) -> dict:
    """Compute the exact normalised posterior on a regular grid.

    Returns a dict with keys:
        log10_A_grid, gamma_grid, log_posterior, posterior,
        posterior_mean, posterior_cov, map_theta,
        marginal_log10_A, marginal_gamma
    """
    lo_A, hi_A = prior_bounds["log10_A_red"]
    lo_g, hi_g = prior_bounds["gamma_red"]

    log10_A_grid = np.linspace(lo_A, hi_A, n_grid, dtype=np.float64)
    gamma_grid = np.linspace(lo_g, hi_g, n_grid, dtype=np.float64)

    ll = compute_log_likelihood_grid(
        residuals, sigma, F, tspan, n_modes, log10_A_grid, gamma_grid, jitter
    )

    # Uniform prior → log posterior ∝ log likelihood (inside bounds)
    log_post = ll.copy()

    # Normalise via log-sum-exp
    dA = (hi_A - lo_A) / (n_grid - 1)
    dG = (hi_g - lo_g) / (n_grid - 1)
    log_max = np.max(log_post)
    log_norm = log_max + np.log(np.sum(np.exp(log_post - log_max)) * dA * dG)
    log_post_normed = log_post - log_norm
    post = np.exp(log_post_normed)

    # MAP
    idx = np.unravel_index(np.argmax(log_post_normed), log_post_normed.shape)
    map_theta = np.array([log10_A_grid[idx[0]], gamma_grid[idx[1]]])

    # Posterior mean & covariance
    AA, GG = np.meshgrid(log10_A_grid, gamma_grid, indexing="ij")
    mean_A = np.sum(AA * post) * dA * dG
    mean_G = np.sum(GG * post) * dA * dG
    var_A = np.sum((AA - mean_A) ** 2 * post) * dA * dG
    var_G = np.sum((GG - mean_G) ** 2 * post) * dA * dG
    cov_AG = np.sum((AA - mean_A) * (GG - mean_G) * post) * dA * dG

    # Marginals
    marginal_A = np.sum(post, axis=1) * dG
    marginal_G = np.sum(post, axis=0) * dA

    return {
        "log10_A_grid": log10_A_grid,
        "gamma_grid": gamma_grid,
        "log_posterior": log_post_normed,
        "posterior": post,
        "posterior_mean": np.array([mean_A, mean_G]),
        "posterior_cov": np.array([[var_A, cov_AG], [cov_AG, var_G]]),
        "map_theta": map_theta,
        "marginal_log10_A": marginal_A,
        "marginal_gamma": marginal_G,
    }
