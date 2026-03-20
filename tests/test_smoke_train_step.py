"""Smoke test: one training step produces a finite loss."""

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


def _make_dummy_batch(B=8, L_max=40, n_feat=6):
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


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
def test_one_training_step(smoke_cfg, model_type):
    model = build_model(model_type, smoke_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    batch = _make_dummy_batch()
    loss = model(batch)
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Second step – loss should still be finite
    loss2 = model(batch)
    assert torch.isfinite(loss2), f"Loss after step not finite: {loss2.item()}"
