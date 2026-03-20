"""Tests for the simulator and schedule generator."""

import numpy as np
import pytest
from src.schedules import generate_schedule
from src.simulator import simulate_pulsar, build_fourier_design_matrix, power_law_spectrum


def test_schedule_shape():
    rng = np.random.default_rng(0)
    sched = generate_schedule(rng, n_toa_min=50, n_toa_max=200)
    assert sched.n_toa >= 50
    assert sched.n_toa <= 250  # allow slight overshoot from padding
    assert sched.t.shape == sched.sigma.shape
    assert sched.t.shape == sched.freq_mhz.shape
    assert sched.t.shape == sched.backend_id.shape
    assert np.all(np.diff(sched.t) >= 0), "times not sorted"


def test_schedule_variable_length():
    rng = np.random.default_rng(42)
    lengths = [generate_schedule(rng).n_toa for _ in range(20)]
    assert len(set(lengths)) > 1, "all schedules same length"


def test_fourier_design_matrix_shape():
    t = np.linspace(0, 10, 100, dtype=np.float32)
    F = build_fourier_design_matrix(t, tspan=10.0, n_modes=20)
    assert F.shape == (100, 40)


def test_power_law_spectrum():
    rho = power_law_spectrum(n_modes=20, tspan=10.0, log10_A=-1.5, gamma=3.0)
    assert rho.shape == (20,)
    assert np.all(rho > 0)
    # Power law: lower frequencies should have more power
    assert rho[0] > rho[-1]


def test_simulate_pulsar_shape():
    rng = np.random.default_rng(1)
    sched = generate_schedule(rng)
    theta = np.array([-1.5, 3.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=rng)
    N = sched.n_toa
    assert sim.residuals.shape == (N,)
    assert sim.t.shape == (N,)
    assert sim.sigma.shape == (N,)
    assert sim.F.shape == (N, 40)
    assert sim.tspan > 0


def test_simulate_pulsar_finite():
    rng = np.random.default_rng(2)
    sched = generate_schedule(rng)
    theta = np.array([-1.0, 4.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=rng)
    assert np.all(np.isfinite(sim.residuals))
