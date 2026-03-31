"""Tests for importance sampling module."""

import numpy as np
import pytest
import torch

from src.schedules import generate_schedule
from src.simulator import simulate_pulsar
from src.priors import UniformPrior
from src.importance_sampling import (
    log_likelihood,
    log_likelihood_batch,
    effective_sample_size,
    systematic_resample,
    weighted_percentile_rank,
)
from src.exact_posterior import _log_likelihood_single

PRIOR_7D = {
    "log10_A_red": [-17, -11],
    "gamma_red": [0.5, 6.5],
    "log10_A_dm": [-17, -11],
    "gamma_dm": [0.5, 6.5],
    "EFAC": [0.1, 10.0],
    "log10_EQUAD": [-8, -5],
    "log10_ECORR": [-8, -5],
}


def _make_sim_7d(seed=42):
    rng = np.random.default_rng(seed)
    sched = generate_schedule(
        rng, n_toa_min=40, n_toa_max=80, tspan_min_yr=5.0, tspan_max_yr=7.0
    )
    theta = np.array([-14.0, 3.0, -14.5, 2.5, 1.0, -7.0, -7.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)
    return sim, theta


# ---- log_likelihood matches reference ----


def test_log_likelihood_matches_reference():
    """log_likelihood wrapper must match _log_likelihood_single."""
    sim, theta = _make_sim_7d(seed=100)
    ll_wrapper = log_likelihood(theta, sim, jitter=1e-20)

    ll_ref = _log_likelihood_single(
        residuals=sim.residuals,
        sigma=sim.sigma,
        F=sim.F,
        tspan=sim.tspan,
        n_modes=sim.n_modes,
        log10_A=float(theta[0]),
        gamma=float(theta[1]),
        jitter=1e-20,
        F_dm=sim.F_dm,
        log10_A_dm=float(theta[2]),
        gamma_dm=float(theta[3]),
        efac=float(theta[4]),
        equad=10.0 ** float(theta[5]),
        ecorr=10.0 ** float(theta[6]),
        epoch_id=sim.epoch_id,
    )
    assert np.isfinite(ll_wrapper)
    assert ll_wrapper == pytest.approx(ll_ref, rel=1e-10)


def test_log_likelihood_batch_matches_loop():
    sim, theta = _make_sim_7d(seed=101)
    prior = UniformPrior(PRIOR_7D)
    rng = np.random.default_rng(55)
    thetas = prior.sample(20, rng=rng).numpy().astype(np.float64)
    batch_ll = log_likelihood_batch(thetas, sim, jitter=1e-20)
    for i in range(len(thetas)):
        ll_i = log_likelihood(thetas[i], sim, jitter=1e-20)
        assert batch_ll[i] == pytest.approx(ll_i, rel=1e-10)


# ---- ESS ----


def test_ess_uniform_weights():
    """Equal log-weights → ESS = N."""
    N = 1000
    log_w = np.zeros(N)
    ess = effective_sample_size(log_w)
    assert ess == pytest.approx(N, rel=1e-10)


def test_ess_degenerate():
    """One dominant weight → ESS ≈ 1."""
    log_w = np.full(1000, -1e10)
    log_w[0] = 0.0
    ess = effective_sample_size(log_w)
    assert ess == pytest.approx(1.0, abs=0.01)


def test_ess_all_invalid():
    log_w = np.full(100, -np.inf)
    assert effective_sample_size(log_w) == 0.0


def test_ess_partial_invalid():
    """Only finite weights should contribute."""
    log_w = np.array([0.0, 0.0, 0.0, -np.inf, -np.inf])
    ess = effective_sample_size(log_w)
    assert ess == pytest.approx(3.0, rel=1e-10)


# ---- Systematic resampling ----


def test_systematic_resample_shape():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(500, 3))
    weights = np.ones(500) / 500
    resampled = systematic_resample(samples, weights, 200, rng)
    assert resampled.shape == (200, 3)


def test_systematic_resample_degenerate():
    """If one weight dominates, all resamples should be that particle."""
    rng = np.random.default_rng(1)
    N, D = 100, 2
    samples = rng.normal(size=(N, D))
    weights = np.zeros(N)
    weights[42] = 1.0
    resampled = systematic_resample(samples, weights, 50, rng)
    for i in range(50):
        np.testing.assert_array_equal(resampled[i], samples[42])


# ---- Weighted percentile rank ----


def test_weighted_percentile_rank_uniform():
    """Uniform weights → standard percentile rank."""
    samples = np.arange(100, dtype=np.float64)
    weights = np.ones(100) / 100
    # true_val = 49.5 → 50% of samples ≤ 49.5
    pct = weighted_percentile_rank(samples, weights, 49.5)
    assert pct == pytest.approx(0.50, abs=0.01)


def test_weighted_percentile_rank_edges():
    samples = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, 0.3, 0.2])
    assert weighted_percentile_rank(samples, weights, 0.0) == pytest.approx(0.0)
    assert weighted_percentile_rank(samples, weights, 3.0) == pytest.approx(1.0)
    assert weighted_percentile_rank(samples, weights, 1.5) == pytest.approx(0.5)
