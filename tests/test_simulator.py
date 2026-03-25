"""Tests for the simulator and schedule generator."""

import numpy as np
import pytest
from src.schedules import generate_schedule
from src.simulator import (
    simulate_pulsar,
    build_fourier_design_matrix,
    build_dm_design_matrix,
    build_ecorr_matrix,
    power_law_spectrum,
)


# ---- Schedule tests ----


def test_schedule_shape():
    rng = np.random.default_rng(0)
    sched = generate_schedule(rng, n_toa_min=50, n_toa_max=200)
    assert sched.n_toa >= 50
    assert sched.n_toa <= 250  # allow slight overshoot from padding
    assert sched.t.shape == sched.sigma.shape
    assert sched.t.shape == sched.freq_mhz.shape
    assert sched.t.shape == sched.backend_id.shape
    assert sched.t.shape == sched.epoch_id.shape
    assert np.all(np.diff(sched.t) >= 0), "times not sorted"


def test_schedule_variable_length():
    rng = np.random.default_rng(42)
    lengths = [generate_schedule(rng).n_toa for _ in range(20)]
    assert len(set(lengths)) > 1, "all schedules same length"


def test_schedule_epoch_structure():
    rng = np.random.default_rng(5)
    sched = generate_schedule(rng, n_toa_min=80, n_toa_max=300)
    # epoch_id should be contiguous 0..n_epoch-1
    assert sched.epoch_id.min() == 0
    assert sched.n_epoch == sched.epoch_id.max() + 1
    # Should have at least some multi-TOA epochs (ECORR structure)
    epoch_counts = np.bincount(sched.epoch_id)
    assert np.any(epoch_counts > 1), "no multi-TOA epochs found"


def test_schedule_multifreq_epochs():
    rng = np.random.default_rng(7)
    sched = generate_schedule(rng, n_toa_min=100, n_toa_max=300)
    # Check that multi-TOA epochs have different frequencies
    has_multifreq = False
    for eid in range(sched.n_epoch):
        mask = sched.epoch_id == eid
        if mask.sum() > 1:
            freqs = sched.freq_mhz[mask]
            if len(np.unique(freqs)) > 1:
                has_multifreq = True
                break
    assert has_multifreq, "no multi-frequency epochs found"


# ---- Fourier design matrix tests ----


def test_fourier_design_matrix_shape():
    t = np.linspace(0, 10, 100, dtype=np.float32)
    F = build_fourier_design_matrix(t, tspan=10.0, n_modes=20)
    assert F.shape == (100, 40)


def test_dm_design_matrix_shape():
    t = np.linspace(0, 10, 100, dtype=np.float32)
    F = build_fourier_design_matrix(t, tspan=10.0, n_modes=20)
    freq_mhz = np.full(100, 1400.0, dtype=np.float32)
    F_dm = build_dm_design_matrix(F, freq_mhz)
    assert F_dm.shape == F.shape


def test_dm_design_matrix_chromatic_scaling():
    t = np.linspace(0, 10, 3, dtype=np.float32)
    F = build_fourier_design_matrix(t, tspan=10.0, n_modes=2)
    freq_mhz = np.array([820.0, 1400.0, 2300.0], dtype=np.float32)
    F_dm = build_dm_design_matrix(F, freq_mhz)
    # At 1400 MHz, scaling = 1.0
    np.testing.assert_allclose(F_dm[1], F[1], rtol=1e-5)
    # At 820 MHz, scaling = (1400/820)^2 ≈ 2.914
    scale_820 = (1400.0 / 820.0) ** 2
    np.testing.assert_allclose(F_dm[0], F[0] * scale_820, rtol=1e-4)
    # At 2300 MHz, scaling = (1400/2300)^2 ≈ 0.370
    scale_2300 = (1400.0 / 2300.0) ** 2
    np.testing.assert_allclose(F_dm[2], F[2] * scale_2300, rtol=1e-4)


def test_ecorr_matrix_shape():
    epoch_id = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    U = build_ecorr_matrix(epoch_id)
    assert U.shape == (6, 3)
    np.testing.assert_array_equal(U[0], [1, 0, 0])
    np.testing.assert_array_equal(U[2], [0, 1, 0])
    np.testing.assert_array_equal(U[5], [0, 0, 1])


# ---- Power law spectrum tests ----


def test_power_law_spectrum():
    rho = power_law_spectrum(n_modes=20, tspan=10.0, log10_A=-14.0, gamma=3.0)
    assert rho.shape == (20,)
    assert np.all(rho > 0)
    # Power law: lower frequencies should have more power
    assert rho[0] > rho[-1]


# ---- 2-param simulator tests ----


def test_simulate_pulsar_2param_shape():
    rng = np.random.default_rng(1)
    sched = generate_schedule(rng)
    theta = np.array([-14.0, 3.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=rng)
    N = sched.n_toa
    assert sim.residuals.shape == (N,)
    assert sim.t.shape == (N,)
    assert sim.sigma.shape == (N,)
    assert sim.F.shape == (N, 40)
    assert sim.tspan > 0
    assert sim.theta.shape == (2,)
    assert sim.F_dm is None  # no DM in 2-param mode
    assert sim.epoch_id is not None


def test_simulate_pulsar_2param_finite():
    rng = np.random.default_rng(2)
    sched = generate_schedule(rng)
    theta = np.array([-13.0, 4.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=rng)
    assert np.all(np.isfinite(sim.residuals))


# ---- 7-param simulator tests ----


def test_simulate_pulsar_7param_shape():
    rng = np.random.default_rng(3)
    sched = generate_schedule(rng, n_toa_min=80, n_toa_max=200)
    theta = np.array([-14.0, 3.0, -15.0, 2.5, 1.2, -6.5, -6.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=rng)
    N = sched.n_toa
    assert sim.residuals.shape == (N,)
    assert sim.theta.shape == (7,)
    assert sim.F.shape == (N, 40)
    assert sim.F_dm is not None
    assert sim.F_dm.shape == (N, 40)
    assert sim.epoch_id is not None


def test_simulate_pulsar_7param_finite():
    rng = np.random.default_rng(4)
    sched = generate_schedule(rng)
    theta = np.array([-14.0, 3.0, -15.0, 2.5, 1.2, -6.5, -6.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=rng)
    assert np.all(np.isfinite(sim.residuals))


def test_efac_equad_effect():
    """EFAC > 1 and large EQUAD should increase residual variance."""
    rng_base = np.random.default_rng(10)
    sched = generate_schedule(rng_base, n_toa_min=200, n_toa_max=200)

    # Minimal white noise (EFAC=1, small EQUAD)
    theta_quiet = np.array([-17.0, 3.0, -17.0, 3.0, 1.0, -8.0, -8.0], dtype=np.float32)
    # Loud white noise (EFAC=2, large EQUAD)
    theta_loud = np.array([-17.0, 3.0, -17.0, 3.0, 2.0, -5.0, -8.0], dtype=np.float32)

    var_quiet = np.var(
        simulate_pulsar(theta_quiet, sched, rng=np.random.default_rng(10)).residuals
    )
    var_loud = np.var(
        simulate_pulsar(theta_loud, sched, rng=np.random.default_rng(10)).residuals
    )
    assert var_loud > var_quiet


def test_ecorr_correlation():
    """TOAs within same epoch should be correlated when ECORR is large."""
    rng = np.random.default_rng(20)
    sched = generate_schedule(rng, n_toa_min=100, n_toa_max=200)

    # Very large ECORR, tiny red/DM/white noise
    theta = np.array([-17.0, 3.0, -17.0, 3.0, 1.0, -8.0, -3.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=np.random.default_rng(20))

    # Within-epoch residual spread should be much smaller than between-epoch
    epoch_counts = np.bincount(sim.epoch_id)
    multi_epochs = np.where(epoch_counts > 1)[0]
    if len(multi_epochs) > 0:
        within_vars = []
        for e in multi_epochs[:10]:
            mask = sim.epoch_id == e
            r_epoch = sim.residuals[mask]
            within_vars.append(np.var(r_epoch))
        mean_within = np.mean(within_vars)
        total_var = np.var(sim.residuals)
        # Within-epoch variance should be small relative to total
        assert mean_within < total_var


# ---- sigma range test ----


def test_sigma_in_seconds():
    """Verify that generated schedule uncertainties are in the physical range."""
    rng = np.random.default_rng(30)
    sched = generate_schedule(rng, n_toa_min=50, n_toa_max=200)
    # σ should be in [100 ns, 10 μs] = [1e-7, 1e-5] seconds
    assert np.all(sched.sigma >= 1e-8), "sigma too small"
    assert np.all(sched.sigma <= 1e-4), "sigma too large"
