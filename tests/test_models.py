"""Tests for transformer and LSTM model forward passes."""

import torch
import pytest
from src.models.model_wrappers import build_model


@pytest.fixture
def smoke_cfg():
    return {
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
        }
    }


def _make_dummy_batch(B=4, L_max=50, n_feat=6):
    """Create a random padded batch with variable lengths."""
    seq_lens = torch.randint(10, L_max, (B,))
    features = torch.randn(B, L_max, n_feat)
    backend_id = torch.zeros(B, L_max, dtype=torch.long)
    mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i in range(B):
        mask[i, : seq_lens[i]] = True
    theta = torch.randn(B, 2)
    return {
        "theta": theta,
        "features": features,
        "backend_id": backend_id,
        "mask": mask,
        "seq_lens": seq_lens,
    }


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
    # Two batches with different max lengths
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
