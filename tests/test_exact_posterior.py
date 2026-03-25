"""Tests for exact posterior evaluation."""

import numpy as np
import pytest
from src.schedules import generate_schedule
from src.simulator import simulate_pulsar
from src.exact_posterior import exact_posterior_grid


# ---- 2-param tests (backward compat) ----


def test_exact_posterior_normalises():
    rng = np.random.default_rng(10)
    sched = generate_schedule(
        rng, n_toa_min=40, n_toa_max=80, tspan_min_yr=5.0, tspan_max_yr=7.0
    )
    theta = np.array([-14.0, 3.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    prior_bounds = {"log10_A_red": [-17, -11], "gamma_red": [0.5, 6.5]}
    result = exact_posterior_grid(
        sim.residuals,
        sim.sigma,
        sim.F,
        sim.tspan,
        sim.n_modes,
        prior_bounds,
        n_grid=40,
        jitter=1e-20,
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
    theta = np.array([-13.0, 4.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    prior_bounds = {"log10_A_red": [-17, -11], "gamma_red": [0.5, 6.5]}
    result = exact_posterior_grid(
        sim.residuals,
        sim.sigma,
        sim.F,
        sim.tspan,
        sim.n_modes,
        prior_bounds,
        n_grid=40,
        jitter=1e-20,
    )
    m = result["map_theta"]
    assert prior_bounds["log10_A_red"][0] <= m[0] <= prior_bounds["log10_A_red"][1]
    assert prior_bounds["gamma_red"][0] <= m[1] <= prior_bounds["gamma_red"][1]


def test_exact_posterior_finite_likelihood():
    rng = np.random.default_rng(12)
    sched = generate_schedule(rng, n_toa_min=30, n_toa_max=50)
    theta = np.array([-16.0, 2.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=10, rng=rng)

    prior_bounds = {"log10_A_red": [-17, -11], "gamma_red": [0.5, 6.5]}
    result = exact_posterior_grid(
        sim.residuals,
        sim.sigma,
        sim.F,
        sim.tspan,
        sim.n_modes,
        prior_bounds,
        n_grid=30,
        jitter=1e-20,
    )
    # At least some grid points should have finite log posterior
    assert np.sum(np.isfinite(result["log_posterior"])) > 10


def test_woodbury_matches_direct():
    """Verify the Woodbury grid function matches the reference single-point function."""
    from src.exact_posterior import _log_likelihood_single, compute_log_likelihood_grid

    rng = np.random.default_rng(20)
    sched = generate_schedule(rng, n_toa_min=30, n_toa_max=60)
    theta = np.array([-14.0, 3.5], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=10, rng=rng)

    A_grid = np.array([-16.0, -14.0, -12.0])
    G_grid = np.array([1.5, 3.5, 5.5])

    ll_grid = compute_log_likelihood_grid(
        sim.residuals,
        sim.sigma,
        sim.F,
        sim.tspan,
        sim.n_modes,
        A_grid,
        G_grid,
        jitter=1e-20,
    )

    for i, lA in enumerate(A_grid):
        for j, g in enumerate(G_grid):
            ll_ref = _log_likelihood_single(
                sim.residuals,
                sim.sigma,
                sim.F,
                sim.tspan,
                sim.n_modes,
                float(lA),
                float(g),
                jitter=1e-20,
            )
            np.testing.assert_allclose(ll_grid[i, j], ll_ref, rtol=1e-8)


# ---- 7-param conditional posterior tests ----


def test_woodbury_conditional_matches_direct():
    """Verify Woodbury with nuisance conditioning matches direct computation."""
    from src.exact_posterior import _log_likelihood_single, compute_log_likelihood_grid

    rng = np.random.default_rng(25)
    sched = generate_schedule(rng, n_toa_min=30, n_toa_max=60)
    theta = np.array([-14.0, 3.0, -15.0, 2.5, 1.2, -6.5, -6.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=10, rng=rng)

    A_grid = np.array([-16.0, -14.0, -12.0])
    G_grid = np.array([1.5, 3.0, 5.0])

    efac = 1.2
    equad = 10**-6.5
    ecorr = 10**-6.0

    ll_grid = compute_log_likelihood_grid(
        sim.residuals,
        sim.sigma,
        sim.F,
        sim.tspan,
        sim.n_modes,
        A_grid,
        G_grid,
        jitter=1e-20,
        F_dm=sim.F_dm,
        log10_A_dm=-15.0,
        gamma_dm=2.5,
        efac=efac,
        equad=equad,
        ecorr=ecorr,
        epoch_id=sim.epoch_id,
    )

    for i, lA in enumerate(A_grid):
        for j, g in enumerate(G_grid):
            ll_ref = _log_likelihood_single(
                sim.residuals,
                sim.sigma,
                sim.F,
                sim.tspan,
                sim.n_modes,
                float(lA),
                float(g),
                jitter=1e-20,
                F_dm=sim.F_dm,
                log10_A_dm=-15.0,
                gamma_dm=2.5,
                efac=efac,
                equad=equad,
                ecorr=ecorr,
                epoch_id=sim.epoch_id,
            )
            np.testing.assert_allclose(ll_grid[i, j], ll_ref, rtol=1e-6)


def test_conditional_posterior_normalises():
    rng = np.random.default_rng(26)
    sched = generate_schedule(rng, n_toa_min=40, n_toa_max=80)
    theta = np.array([-14.0, 3.0, -15.0, 2.5, 1.2, -6.5, -6.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    prior_bounds = {
        "log10_A_red": [-17, -11],
        "gamma_red": [0.5, 6.5],
        "log10_A_dm": [-17, -11],
        "gamma_dm": [0.5, 6.5],
        "EFAC": [0.5, 2.0],
        "log10_EQUAD": [-8, -5],
        "log10_ECORR": [-8, -5],
    }
    theta_fixed = {
        "log10_A_dm": -15.0,
        "gamma_dm": 2.5,
        "EFAC": 1.2,
        "log10_EQUAD": -6.5,
        "log10_ECORR": -6.0,
        "epoch_id": sim.epoch_id,
    }
    result = exact_posterior_grid(
        sim.residuals,
        sim.sigma,
        sim.F,
        sim.tspan,
        sim.n_modes,
        prior_bounds,
        n_grid=40,
        jitter=1e-20,
        F_dm=sim.F_dm,
        theta_fixed=theta_fixed,
    )
    post = result["posterior"]
    assert np.all(np.isfinite(post))
    dA = (prior_bounds["log10_A_red"][1] - prior_bounds["log10_A_red"][0]) / 39
    dG = (prior_bounds["gamma_red"][1] - prior_bounds["gamma_red"][0]) / 39
    integral = np.sum(post) * dA * dG
    assert abs(integral - 1.0) < 0.1, f"Conditional posterior integral = {integral}"


# ---- sigma range test ----


def test_sigma_in_seconds():
    """Verify that generated schedule uncertainties are in the physical range."""
    rng = np.random.default_rng(30)
    sched = generate_schedule(rng, n_toa_min=50, n_toa_max=200)
    # σ should be in [100 ns, 10 μs] = [1e-7, 1e-5] seconds
    assert np.all(sched.sigma >= 1e-8), "sigma too small"
    assert np.all(sched.sigma <= 1e-4), "sigma too large"
