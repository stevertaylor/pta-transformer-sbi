"""Tests for tokenization and end-to-end simulation → tokenize → model round-trip."""

import numpy as np
import torch
import pytest

from src.schedules import generate_schedule
from src.simulator import simulate_pulsar, SEC_PER_YR, power_law_spectrum
from src.models.tokenization import tokenize, N_CONTINUOUS_FEATURES
from src.models.model_wrappers import build_model


# ---------- Tokenization unit tests ----------


def test_tokenize_shapes():
    rng = np.random.default_rng(0)
    sched = generate_schedule(rng, n_toa_min=50, n_toa_max=100)
    theta = np.array([-14.0, 3.5], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
    N = len(sim.t)

    assert tokens["t_norm"].shape == (N,)
    assert tokens["dt_prev"].shape == (N,)
    assert tokens["r_over_sig"].shape == (N,)
    assert tokens["log_sigma"].shape == (N,)
    assert tokens["r_raw"].shape == (N,)
    assert tokens["freq_norm"].shape == (N,)
    assert tokens["backend_id"].shape == (N,)


def test_tokenize_no_nans():
    rng = np.random.default_rng(1)
    sched = generate_schedule(rng)
    theta = np.array([-13.0, 4.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=rng)

    tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
    for key, val in tokens.items():
        assert torch.all(torch.isfinite(val)), f"NaN/Inf in token feature {key}"


def test_t_norm_bounded():
    rng = np.random.default_rng(2)
    sched = generate_schedule(rng)
    theta = np.array([-15.0, 2.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
    assert tokens["t_norm"].min() >= 0.0
    assert tokens["t_norm"].max() <= 1.0 + 1e-6


def test_log_sigma_physical_range():
    """log10(sigma) should be in [-7, -5] for physical seconds."""
    rng = np.random.default_rng(3)
    sched = generate_schedule(rng)
    theta = np.array([-14.0, 3.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
    # Should be approximately in [-7, -5] (100 ns to 10 μs)
    assert tokens["log_sigma"].min() >= -8.0
    assert tokens["log_sigma"].max() <= -4.0


# ---------- Physical-units consistency ----------


def test_enterprise_spectrum_convention():
    """Verify spectrum matches the enterprise power-law at a known point."""
    # log10_A = -14, gamma = 13/3, T = 10 yr, f_1 = 0.1 yr^{-1}
    rho = power_law_spectrum(n_modes=1, tspan=10.0, log10_A=-14.0, gamma=13.0 / 3.0)
    # Expected: A^2 * SEC_PER_YR^2 / (12 pi^2) * f_1^(-gamma) * delta_f
    A2 = 1e-28
    f1 = 0.1
    df = 0.1
    expected = A2 * SEC_PER_YR**2 / (12.0 * np.pi**2) * f1 ** (-13.0 / 3.0) * df
    np.testing.assert_allclose(float(rho[0]), expected, rtol=1e-5)


def test_residuals_in_seconds():
    """Residuals should be on a physically plausible scale (ns to ms)."""
    rng = np.random.default_rng(10)
    sched = generate_schedule(
        rng, n_toa_min=100, n_toa_max=200, tspan_min_yr=10.0, tspan_max_yr=12.0
    )
    # Strong red noise
    theta = np.array([-13.0, 4.0], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=20, rng=rng)

    rms = np.sqrt(np.mean(sim.residuals**2))
    # RMS should be in a plausible range (100 ns to 100 ms)
    assert rms > 1e-10, f"RMS too small: {rms}"
    assert rms < 1.0, f"RMS too large: {rms}"


# ---------- Round-trip: simulate → tokenize → model → finite loss ----------


@pytest.fixture
def round_trip_cfg():
    return {
        "prior": {
            "log10_A_red": [-17, -11],
            "gamma_red": [0.5, 6.5],
        },
        "model": {
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "context_dim": 16,
            "lstm_hidden": 32,
            "lstm_layers": 1,
            "flow_transforms": 2,
            "flow_hidden": 32,
            "flow_layers": 2,
            "flow_bins": 8,
            "use_rope": True,
            "use_aux_features": True,
        },
    }


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
def test_round_trip_finite_loss(round_trip_cfg, model_type):
    """Full pipeline: simulate physical data → tokenize → model forward → finite loss."""
    rng = np.random.default_rng(42)
    sched = generate_schedule(rng, n_toa_min=30, n_toa_max=80)
    theta = np.array([-14.0, 3.5], dtype=np.float32)
    sim = simulate_pulsar(theta, sched, n_modes=15, rng=rng)

    tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
    L = len(sim.t)

    feat_keys = ["t_norm", "dt_prev", "r_over_sig", "log_sigma", "r_raw", "freq_norm"]
    features = torch.stack([tokens[k] for k in feat_keys], dim=-1).unsqueeze(0)
    backend_id = tokens["backend_id"].unsqueeze(0)
    mask = torch.ones(1, L, dtype=torch.bool)
    tspan_yr = torch.tensor([float(sim.t.max() - sim.t.min())], dtype=torch.float32)

    batch = {
        "theta": torch.from_numpy(sim.theta).unsqueeze(0),
        "features": features,
        "backend_id": backend_id,
        "mask": mask,
        "seq_lens": torch.tensor([L]),
        "tspan_yr": tspan_yr,
    }

    model = build_model(model_type, round_trip_cfg)
    model.eval()
    with torch.no_grad():
        loss = model(batch)
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
