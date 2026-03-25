"""Exact Gaussian posterior on a 2-D parameter grid.

Because the signal model is linear in Fourier coefficients and the noise
is Gaussian, the likelihood p(r | θ, schedule) is a multivariate Gaussian
whose covariance depends on θ.  With a uniform prior the posterior is
proportional to the likelihood evaluated on a grid.

Supports two modes:
  * **2-param** (backward-compat): grids over (log10_A_red, gamma_red) with
    D = diag(σ² + jitter).  Same fast Woodbury as v3.
  * **7-param conditional**: grids over (log10_A_red, gamma_red) while
    conditioning on the true values of the 5 nuisance parameters
    (log10_A_dm, gamma_dm, EFAC, log10_EQUAD, log10_ECORR).
    D_eff = diag(EFAC² σ² + EQUAD²) + F_dm Φ_dm F_dm^T + ECORR² U U^T
    is precomputed once, then Woodbury iterates over the red-noise part.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Optional
from scipy.linalg import solve_triangular, cho_factor, cho_solve

from .simulator import (
    build_fourier_design_matrix,
    build_dm_design_matrix,
    build_ecorr_matrix,
    power_law_spectrum,
    SEC_PER_YR,
)


# ------------------------------------------------------------------
# Reference (slow) single-point likelihood — kept for unit-test validation
# ------------------------------------------------------------------


def _log_likelihood_single(
    residuals: np.ndarray,
    sigma: np.ndarray,
    F: np.ndarray,
    tspan: float,
    n_modes: int,
    log10_A: float,
    gamma: float,
    jitter: float = 1e-20,
    F_dm: Optional[np.ndarray] = None,
    log10_A_dm: Optional[float] = None,
    gamma_dm: Optional[float] = None,
    efac: float = 1.0,
    equad: float = 0.0,
    ecorr: float = 0.0,
    epoch_id: Optional[np.ndarray] = None,
) -> float:
    """Exact log p(r | θ) for one parameter combination.

    C = D_eff + F Φ_red Fᵀ
    where D_eff = diag(EFAC²σ²+EQUAD²+jitter) [+ F_dm Φ_dm F_dm^T] [+ ECORR² U U^T]
    log p = -0.5 [ N log(2π) + log|C| + rᵀ C⁻¹ r ]
    """
    N = len(residuals)
    rho_red = power_law_spectrum(n_modes, tspan, log10_A, gamma)
    phi_red = np.repeat(rho_red, 2).astype(np.float64)

    F64 = F.astype(np.float64)
    d = (efac * sigma.astype(np.float64)) ** 2 + equad**2 + jitter
    C = np.diag(d) + F64 @ np.diag(phi_red) @ F64.T

    # DM contribution
    if F_dm is not None and log10_A_dm is not None and gamma_dm is not None:
        rho_dm = power_law_spectrum(n_modes, tspan, log10_A_dm, gamma_dm)
        phi_dm = np.repeat(rho_dm, 2).astype(np.float64)
        F_dm64 = F_dm.astype(np.float64)
        C += F_dm64 @ np.diag(phi_dm) @ F_dm64.T

    # ECORR contribution
    if ecorr > 0 and epoch_id is not None:
        U = build_ecorr_matrix(epoch_id).astype(np.float64)
        C += ecorr**2 * (U @ U.T)

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return -np.inf

    r64 = residuals.astype(np.float64)
    alpha = np.linalg.solve(L, r64)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return float(-0.5 * (N * np.log(2 * np.pi) + log_det + np.dot(alpha, alpha)))


# ------------------------------------------------------------------
# Woodbury-accelerated grid evaluation
# ------------------------------------------------------------------


def _build_d_eff(
    sigma: np.ndarray,
    jitter: float,
    F_dm: Optional[np.ndarray] = None,
    log10_A_dm: Optional[float] = None,
    gamma_dm: Optional[float] = None,
    tspan: Optional[float] = None,
    n_modes: Optional[int] = None,
    efac: float = 1.0,
    equad: float = 0.0,
    ecorr: float = 0.0,
    epoch_id: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the effective 'diagonal' noise matrix D_eff.

    D_eff = diag(EFAC²σ² + EQUAD² + jitter)
            [+ F_dm Φ_dm F_dm^T]  [+ ECORR² U U^T]

    Returns (N, N) dense symmetric matrix.
    """
    N = len(sigma)
    d = (efac * sigma.astype(np.float64)) ** 2 + equad**2 + jitter
    D_eff = np.diag(d)

    if F_dm is not None and log10_A_dm is not None:
        rho_dm = power_law_spectrum(n_modes, tspan, log10_A_dm, gamma_dm)
        phi_dm = np.repeat(rho_dm, 2).astype(np.float64)
        F_dm64 = F_dm.astype(np.float64)
        D_eff += F_dm64 @ np.diag(phi_dm) @ F_dm64.T

    if ecorr > 0 and epoch_id is not None:
        U = build_ecorr_matrix(epoch_id).astype(np.float64)
        D_eff += ecorr**2 * (U @ U.T)

    return D_eff


def compute_log_likelihood_grid(
    residuals: np.ndarray,
    sigma: np.ndarray,
    F: np.ndarray,
    tspan: float,
    n_modes: int,
    log10_A_grid: np.ndarray,
    gamma_grid: np.ndarray,
    jitter: float = 1e-20,
    *,
    F_dm: Optional[np.ndarray] = None,
    log10_A_dm: Optional[float] = None,
    gamma_dm: Optional[float] = None,
    efac: float = 1.0,
    equad: float = 0.0,
    ecorr: float = 0.0,
    epoch_id: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate log-likelihood on a 2-D grid over (log10_A_red, gamma_red).

    Uses the Woodbury identity with D_eff (which may be dense for the
    7-param model) as the base covariance.

    Returns shape (len(log10_A_grid), len(gamma_grid)).
    """
    N = len(residuals)
    K = 2 * n_modes
    nA = len(log10_A_grid)
    nG = len(gamma_grid)

    has_nuisance = (F_dm is not None) or (ecorr > 0) or (efac != 1.0) or (equad != 0.0)

    if has_nuisance:
        # --- Build dense D_eff and precompute its Cholesky ---
        D_eff = _build_d_eff(
            sigma,
            jitter,
            F_dm,
            log10_A_dm,
            gamma_dm,
            tspan,
            n_modes,
            efac,
            equad,
            ecorr,
            epoch_id,
        )
        L_eff = np.linalg.cholesky(D_eff)
        log_det_D_eff = 2.0 * np.sum(np.log(np.diag(L_eff)))

        # Whiten F and r through L_eff
        F64 = F.astype(np.float64)
        r64 = residuals.astype(np.float64)
        # solve L_eff @ X = F  and  L_eff @ y = r
        G = solve_triangular(L_eff, F64, lower=True)  # (N, K)
        r_w = solve_triangular(L_eff, r64, lower=True)  # (N,)
    else:
        # --- Original diagonal case (fast path, backward-compat) ---
        d = sigma.astype(np.float64) ** 2 + jitter
        d_inv_sqrt = 1.0 / np.sqrt(d)
        F64 = F.astype(np.float64)
        G = F64 * d_inv_sqrt[:, None]
        r_w = residuals.astype(np.float64) * d_inv_sqrt
        log_det_D_eff = float(np.sum(np.log(d)))

    GtG = G.T @ G  # (K, K)
    Gtr = G.T @ r_w  # (K,)
    rtr = float(np.dot(r_w, r_w))
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
            log_det_C = log_det_D_eff + log_det_Phi + log_det_M

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
    *,
    F_dm: Optional[np.ndarray] = None,
    theta_fixed: Optional[dict] = None,
) -> dict:
    """Compute the exact normalised posterior on a regular 2-D grid over
    (log10_A_red, gamma_red).

    Parameters
    ----------
    theta_fixed : dict, optional
        Fixed values for nuisance parameters when doing conditional inference.
        Keys may include: log10_A_dm, gamma_dm, EFAC, log10_EQUAD, log10_ECORR,
        epoch_id.

    Returns a dict with keys:
        log10_A_grid, gamma_grid, log_posterior, posterior,
        posterior_mean, posterior_cov, map_theta,
        marginal_log10_A, marginal_gamma
    """
    lo_A, hi_A = prior_bounds["log10_A_red"]
    lo_g, hi_g = prior_bounds["gamma_red"]

    log10_A_grid = np.linspace(lo_A, hi_A, n_grid, dtype=np.float64)
    gamma_grid = np.linspace(lo_g, hi_g, n_grid, dtype=np.float64)

    # Extract nuisance params from theta_fixed
    tf = theta_fixed or {}
    extra_kwargs = {}
    if "log10_A_dm" in tf and "gamma_dm" in tf:
        extra_kwargs["F_dm"] = F_dm
        extra_kwargs["log10_A_dm"] = float(tf["log10_A_dm"])
        extra_kwargs["gamma_dm"] = float(tf["gamma_dm"])
    if "EFAC" in tf:
        extra_kwargs["efac"] = float(tf["EFAC"])
    if "log10_EQUAD" in tf:
        extra_kwargs["equad"] = 10.0 ** float(tf["log10_EQUAD"])
    if "log10_ECORR" in tf:
        extra_kwargs["ecorr"] = 10.0 ** float(tf["log10_ECORR"])
    if "epoch_id" in tf:
        extra_kwargs["epoch_id"] = tf["epoch_id"]

    ll = compute_log_likelihood_grid(
        residuals,
        sigma,
        F,
        tspan,
        n_modes,
        log10_A_grid,
        gamma_grid,
        jitter,
        **extra_kwargs,
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
