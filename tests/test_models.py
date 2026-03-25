"""Tests for transformer and LSTM model forward passes."""

import torch
import pytest
from src.models.model_wrappers import build_model


@pytest.fixture
def smoke_cfg():
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
        },
    }


@pytest.fixture
def v3_cfg():
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


@pytest.fixture
def v4_cfg():
    """7-parameter noise model config."""
    return {
        "prior": {
            "log10_A_red": [-17, -11],
            "gamma_red": [0.5, 6.5],
            "log10_A_dm": [-17, -11],
            "gamma_dm": [0.5, 6.5],
            "EFAC": [0.5, 2.0],
            "log10_EQUAD": [-8, -5],
            "log10_ECORR": [-8, -5],
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


def _make_dummy_batch(B=4, L_max=50, n_feat=6, theta_dim=2):
    """Create a random padded batch with variable lengths."""
    seq_lens = torch.randint(10, L_max, (B,))
    features = torch.randn(B, L_max, n_feat)
    backend_id = torch.zeros(B, L_max, dtype=torch.long)
    mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i in range(B):
        mask[i, : seq_lens[i]] = True
    theta = torch.randn(B, theta_dim)
    return {
        "theta": theta,
        "features": features,
        "backend_id": backend_id,
        "mask": mask,
        "seq_lens": seq_lens,
        "tspan_yr": torch.rand(B) * 10 + 5,
    }


# --- v1/v2 backward-compat tests (2-param) ---


def test_transformer_forward(smoke_cfg):
    model = build_model("transformer", smoke_cfg)
    batch = _make_dummy_batch()
    loss = model(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_lstm_forward(smoke_cfg):
    model = build_model("lstm", smoke_cfg)
    batch = _make_dummy_batch()
    loss = model(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_transformer_variable_length(smoke_cfg):
    model = build_model("transformer", smoke_cfg)
    batch1 = _make_dummy_batch(B=2, L_max=30)
    batch2 = _make_dummy_batch(B=2, L_max=80)
    loss1 = model(batch1)
    loss2 = model(batch2)
    assert torch.isfinite(loss1) and torch.isfinite(loss2)


def test_lstm_variable_length(smoke_cfg):
    model = build_model("lstm", smoke_cfg)
    batch1 = _make_dummy_batch(B=2, L_max=30)
    batch2 = _make_dummy_batch(B=2, L_max=80)
    loss1 = model(batch1)
    loss2 = model(batch2)
    assert torch.isfinite(loss1) and torch.isfinite(loss2)


def test_posterior_sampling(smoke_cfg):
    model = build_model("transformer", smoke_cfg)
    model.eval()
    batch = _make_dummy_batch(B=2)
    samples = model.sample_posterior(batch, n_samples=50)
    assert samples.shape == (2, 50, 2)
    assert torch.all(torch.isfinite(samples))


# --- v3 architecture tests (RoPE + aux features, 2-param) ---


def test_transformer_rope_forward(v3_cfg):
    model = build_model("transformer", v3_cfg)
    batch = _make_dummy_batch()
    loss = model(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_lstm_aux_forward(v3_cfg):
    model = build_model("lstm", v3_cfg)
    batch = _make_dummy_batch()
    loss = model(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_transformer_rope_variable_length(v3_cfg):
    model = build_model("transformer", v3_cfg)
    batch1 = _make_dummy_batch(B=2, L_max=30)
    batch2 = _make_dummy_batch(B=2, L_max=80)
    loss1 = model(batch1)
    loss2 = model(batch2)
    assert torch.isfinite(loss1) and torch.isfinite(loss2)


def test_rope_posterior_sampling(v3_cfg):
    model = build_model("transformer", v3_cfg)
    model.eval()
    batch = _make_dummy_batch(B=2)
    samples = model.sample_posterior(batch, n_samples=50)
    assert samples.shape == (2, 50, 2)
    assert torch.all(torch.isfinite(samples))


# --- v4 tests (7-param noise model) ---


def test_transformer_v4_forward(v4_cfg):
    model = build_model("transformer", v4_cfg)
    batch = _make_dummy_batch(theta_dim=7)
    loss = model(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_lstm_v4_forward(v4_cfg):
    model = build_model("lstm", v4_cfg)
    batch = _make_dummy_batch(theta_dim=7)
    loss = model(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_v4_posterior_sampling(v4_cfg):
    model = build_model("transformer", v4_cfg)
    model.eval()
    batch = _make_dummy_batch(B=2, theta_dim=7)
    samples = model.sample_posterior(batch, n_samples=50)
    assert samples.shape == (2, 50, 7)
    assert torch.all(torch.isfinite(samples))


def test_v4_variable_length(v4_cfg):
    model = build_model("transformer", v4_cfg)
    batch1 = _make_dummy_batch(B=2, L_max=30, theta_dim=7)
    batch2 = _make_dummy_batch(B=2, L_max=80, theta_dim=7)
    loss1 = model(batch1)
    loss2 = model(batch2)
    assert torch.isfinite(loss1) and torch.isfinite(loss2)


def test_v4_lstm_posterior_sampling(v4_cfg):
    model = build_model("lstm", v4_cfg)
    model.eval()
    batch = _make_dummy_batch(B=2, theta_dim=7)
    samples = model.sample_posterior(batch, n_samples=50)
    assert samples.shape == (2, 50, 7)
    assert torch.all(torch.isfinite(samples))
