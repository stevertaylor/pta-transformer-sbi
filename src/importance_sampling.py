"""Importance sampling (Dingo-IS) for amortized posterior correction.

Given a trained NPE model q(θ|x), importance sampling reweights flow
samples using the exact likelihood p(x|θ) to produce asymptotically
exact posteriors.  The effective sample size (ESS) measures how well
the learned proposal matches the true posterior.

    w_i = p(x|θ_i) π(θ_i) / q(θ_i|x),   θ_i ~ q(θ|x)

Reference: Dax et al. (2021), "Real-Time Gravitational Wave Science
with Neural Posterior Estimation" (Dingo-IS).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict

from .exact_posterior import _log_likelihood_single
from .simulator import SimulatedPulsar


# ------------------------------------------------------------------
# Log-likelihood evaluation
# ------------------------------------------------------------------


def log_likelihood(
    theta: np.ndarray,
    sim: SimulatedPulsar,
    jitter: float = 1e-20,
) -> float:
    """Evaluate exact log p(r | θ) for a single parameter vector.

    Supports:
      D=2: (log10_A_red, gamma_red)
      D=4: + (log10_A_dm, gamma_dm)
      D=7: + (EFAC, log10_EQUAD, log10_ECORR) — single backend
      D=4+3N (N>1): + per-backend (EFAC_b, log10_EQUAD_b, log10_ECORR_b)
    """
    D = len(theta)
    kwargs = dict(
        residuals=sim.residuals,
        sigma=sim.sigma,
        F=sim.F,
        tspan=sim.tspan,
        n_modes=sim.n_modes,
        log10_A=float(theta[0]),
        gamma=float(theta[1]),
        jitter=jitter,
    )
    if D >= 4:
        kwargs["F_dm"] = sim.F_dm
        kwargs["log10_A_dm"] = float(theta[2])
        kwargs["gamma_dm"] = float(theta[3])
    if D == 7:
        # Single-backend WN
        kwargs["efac"] = float(theta[4])
        kwargs["equad"] = 10.0 ** float(theta[5])
        kwargs["ecorr"] = 10.0 ** float(theta[6])
        kwargs["epoch_id"] = sim.epoch_id
    elif D > 7 and (D - 4) % 3 == 0:
        # Multi-backend WN: theta[4:] = [EFAC_0, log10_EQUAD_0, log10_ECORR_0, ...]
        n_backends = (D - 4) // 3
        wn_flat = theta[4:].reshape(n_backends, 3)
        kwargs["efac_per_backend"] = wn_flat[:, 0].astype(np.float64)
        kwargs["equad_per_backend"] = (10.0 ** wn_flat[:, 1]).astype(np.float64)
        kwargs["ecorr_per_backend"] = (10.0 ** wn_flat[:, 2]).astype(np.float64)
        kwargs["backend_id"] = sim.backend_id
        kwargs["epoch_id"] = sim.epoch_id
    elif D >= 6:
        # Fallback for D=5 or D=6 (partial WN)
        kwargs["efac"] = float(theta[4])
        kwargs["equad"] = 10.0 ** float(theta[5])
        if D >= 7:
            kwargs["ecorr"] = 10.0 ** float(theta[6])
            kwargs["epoch_id"] = sim.epoch_id
    return _log_likelihood_single(**kwargs)


def log_likelihood_batch(
    thetas: np.ndarray,
    sim: SimulatedPulsar,
    jitter: float = 1e-20,
) -> np.ndarray:
    """Evaluate exact log p(r | θ) for a batch of parameter vectors.

    Parameters
    ----------
    thetas : (N, D) array of parameter vectors
    sim : SimulatedPulsar
    jitter : float

    Returns
    -------
    (N,) array of log-likelihoods (float64)
    """
    N = thetas.shape[0]
    result = np.empty(N, dtype=np.float64)
    for i in range(N):
        result[i] = log_likelihood(thetas[i], sim, jitter)
    return result


# ------------------------------------------------------------------
# ESS and resampling
# ------------------------------------------------------------------


def effective_sample_size(log_weights: np.ndarray) -> float:
    """Compute ESS from unnormalized log importance weights.

        ESS = (Σ w)² / Σ w²

    Numerically stable via log-sum-exp.  Returns 0 if no finite weights.
    """
    valid = np.isfinite(log_weights)
    if valid.sum() == 0:
        return 0.0
    lw = log_weights[valid]
    lw_max = lw.max()
    w = np.exp(lw - lw_max)
    return float(w.sum() ** 2 / (w**2).sum())


def systematic_resample(
    samples: np.ndarray,
    weights: np.ndarray,
    n_resample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Systematic resampling from weighted particles.

    Parameters
    ----------
    samples : (N, D) particle positions
    weights : (N,) normalised weights (sum ≈ 1)
    n_resample : number of output particles
    rng : numpy random generator

    Returns
    -------
    (n_resample, D) equally-weighted resampled particles
    """
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # avoid floating-point overshoot
    u0 = rng.uniform(0, 1.0 / n_resample)
    u = u0 + np.arange(n_resample) / n_resample
    indices = np.searchsorted(cumsum, u)
    return samples[indices]


# ------------------------------------------------------------------
# Weighted calibration helpers
# ------------------------------------------------------------------


def weighted_percentile_rank(
    samples_1d: np.ndarray,
    weights: np.ndarray,
    true_val: float,
) -> float:
    """Weighted CDF evaluated at *true_val*: Σ w_i · 1[θ_i ≤ true]."""
    return float(np.sum(weights[samples_1d <= true_val]))


# ------------------------------------------------------------------
# Main IS entry point
# ------------------------------------------------------------------


@torch.no_grad()
def importance_sample(
    model,
    batch: dict,
    sim: SimulatedPulsar,
    prior,
    n_samples: int = 10_000,
    jitter: float = 1e-20,
) -> Dict:
    """Run importance sampling for one observation.

    1. Draw θ_1 … θ_N ~ q(θ|x)   (flow proposal)
    2. Evaluate log q(θ_i|x)       (flow log-prob)
    3. Evaluate log π(θ_i)         (prior)
    4. Evaluate log p(x|θ_i)       (exact likelihood — dominant cost)
    5. log w_i = log p(x|θ_i) + log π(θ_i) − log q(θ_i|x)

    For FactorizedNPEModel: IS is performed on the global flow with WN
    fixed at true values.  The likelihood uses the full 7-D theta (with true
    first-backend WN appended) to test the global posterior's calibration.

    Returns
    -------
    dict with keys:
        samples        (N, D)  proposal samples in original θ space
        log_weights    (N,)    unnormalized log IS weights
        weights        (N,)    self-normalized IS weights (sum = 1)
        ess            float   effective sample size
        ess_fraction   float   ESS / N
        n_valid        int     number of finite-weight samples
        log_likelihood (N,)    log p(x|θ)
        log_prior      (N,)    log π(θ)
        log_proposal   (N,)    log q(θ|x)
    """
    device = batch["features"].device
    is_factorized = hasattr(model, "global_flow")

    if is_factorized:
        # --- Factorized model: IS on global posterior ---
        # 1. Global samples
        global_samples, _ = model.sample_posterior(batch, n_samples)  # (1, S, 4)
        samples_t = global_samples[0]   # (S, 4)
        samples_np = samples_t.cpu().numpy().astype(np.float64)

        # 2. Global flow log-prob
        global_ctx, _ = model._get_contexts(batch)
        global_ctx_exp = global_ctx.expand(n_samples, -1)
        theta_norm = model._normalize_global(samples_t.to(device))
        log_probs_norm = model.global_flow.log_prob(theta_norm.float(), global_ctx_exp.float())
        log_probs = log_probs_norm - model.global_theta_std.log().sum()
        log_q_np = log_probs.cpu().numpy().astype(np.float64)

        # 3. Global prior log-prob
        log_prior_t = prior.global_prior.log_prob(samples_t.cpu())
        log_prior_np = log_prior_t.numpy().astype(np.float64)

        # 4. Exact likelihood with true WN params fixed (first backend)
        wn_true = sim.theta_wn[0] if sim.theta_wn is not None else np.zeros(3)
        # Build 7-D theta_full by appending true first-backend WN to each global sample
        wn_tile = np.tile(wn_true, (n_samples, 1))  # (S, 3)
        theta_full = np.concatenate([samples_np, wn_tile], axis=1)  # (S, 7)
        log_lik = log_likelihood_batch(theta_full, sim, jitter=jitter)

    else:
        # --- Non-factorized model ---
        # 1. Proposal samples from the flow
        samples_torch = model.sample_posterior(batch, n_samples)  # (1, N, D)
        samples_t = samples_torch[0]
        samples_np = samples_t.cpu().numpy().astype(np.float64)

        # 2. Flow log-prob (proposal density)
        log_q = model.log_prob_on_grid(batch, samples_t.to(device))
        log_q_np = log_q.cpu().numpy().astype(np.float64)

        # 3. Prior log-prob
        log_prior_np = prior.log_prob(samples_t.cpu()).numpy().astype(np.float64)

        # 4. Exact log-likelihood
        log_lik = log_likelihood_batch(samples_np, sim, jitter=jitter)

    # 5. Unnormalized log IS weights
    log_weights = log_lik + log_prior_np - log_q_np

    # 6. ESS and normalized weights
    valid = np.isfinite(log_weights)
    n_valid = int(valid.sum())
    ess = effective_sample_size(log_weights)

    if n_valid > 0:
        lw = np.where(valid, log_weights, -np.inf)
        lw_max = np.max(lw[valid])
        w = np.exp(lw - lw_max)
        w /= w.sum()
    else:
        w = np.zeros(n_samples)

    return {
        "samples": samples_np,
        "log_weights": log_weights,
        "weights": w,
        "ess": ess,
        "ess_fraction": ess / n_samples,
        "n_valid": n_valid,
        "log_likelihood": log_lik,
        "log_prior": log_prior_np,
        "log_proposal": log_q_np,
    }
