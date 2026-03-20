"""Tests for exact posterior evaluation."""

import numpy as np
import pytest
from src.schedules import generate_schedule
from src.simulator import simulate_pulsar
from src.exact_posterior import exact_posterior_grid


def test_exact_posterior_normalises():
    rng = np.random.default_rng(10)
    sched = generate_schedule(rng, n_toa_min=40, n_toa_max=80, tspan_min_yr=5.0, tspan_max_yr=7.0)
    theta = np.array([-1.5, 3.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    prior_bounds = {"log10_A_red": [-2.5, -0.5], "gamma_red": [0.5, 6.5]}
    result = exact_posterior_grid(
        sim.residuals, sim.sigma, sim.F, sim.tspan, sim.n_modes,
        prior_bounds, n_grid=40, jitter=1e-6,
    )
    post = result["posterior"]
    assert np.all(np.isfinite(post))
    assert np.all(post >= 0)

    # Integral should be approximately 1
    dA = (prior_bounds["log10_A_red"][1] - prior_bounds["log10_A_red"][0]) / 39
    dG = (prior_bounds["gamma_red"][1] - prior_bounds["gamma_red"][0]) / 39
    integral = np.sum(post) * dA * dG
    assert abs(integral - 1.0) < 0.1, f"Posterior integral = {integral}"


def test_exact_posterior_map_in_bounds():
    rng = np.random.default_rng(11)
    sched = generate_schedule(rng, n_toa_min=50, n_toa_max=100)
    theta = np.array([-1.0, 4.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    prior_bounds = {"log10_A_red": [-2.5, -0.5], "gamma_red": [0.5, 6.5]}
    result = exact_posterior_grid(
        sim.residuals, sim.sigma, sim.F, sim.tspan, sim.n_modes,
        prior_bounds, n_grid=40, jitter=1e-6,
    )
    m = result["map_theta"]
    assert prior_bounds["log10_A_red"][0] <= m[0] <= prior_bounds["log10_A_red"][1]
    assert prior_bounds["gamma_red"][0] <= m[1] <= prior_bounds["gamma_red"][1]


def test_exact_posterior_finite_likelihood():
    rng = np.random.default_rng(12)
    sched = generate_schedule(rng, n_toa_min=30, n_toa_max=50)
    theta = np.array([-2.0, 2.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=10, rng=rng)

    prior_bounds = {"log10_A_red": [-2.5, -0.5], "gamma_red": [0.5, 6.5]}
    result = exact_posterior_grid(
        sim.residuals, sim.sigma, sim.F, sim.tspan, sim.n_modes,
        prior_bounds, n_grid=30, jitter=1e-6,
    )
    # At least some grid points should have finite log posterior
    assert np.sum(np.isfinite(result["log_posterior"])) > 10
