"""Model wrappers: encoder + posterior flow, with a unified API."""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoderModel
from .lstm_encoder import LSTMEncoderModel
from .posterior_flow import PosteriorFlow


class NPEModel(nn.Module):
    """Encoder + posterior flow for amortized neural posterior estimation.

    The encoder maps (features, backend_id, mask) → context.
    The flow maps (theta, context) → log_prob.
    """

    def __init__(self, encoder: nn.Module, flow: PosteriorFlow):
        super().__init__()
        self.encoder = encoder
        self.flow = flow

    def forward(self, batch: dict) -> torch.Tensor:
        """Compute negative log-prob loss for training.

        Returns scalar loss = -mean log q(theta | x).
        """
        context = self.encoder(
            batch["features"], batch["backend_id"], batch["mask"]
        )
        log_prob = self.flow.log_prob(batch["theta"], context)
        return -log_prob.mean()

    def get_context(self, batch: dict) -> torch.Tensor:
        return self.encoder(batch["features"], batch["backend_id"], batch["mask"])

    @torch.no_grad()
    def sample_posterior(self, batch: dict, n_samples: int = 1000) -> torch.Tensor:
        """Sample from learned posterior. Returns (B, n_samples, 2)."""
        context = self.get_context(batch)
        return self.flow.sample(context, n_samples)

    @torch.no_grad()
    def log_prob_on_grid(
        self, batch: dict, grid_points: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate log q(theta | x) on a grid.

        Parameters
        ----------
        batch : dict with a single example (B=1)
        grid_points : (G, 2) grid of theta values

        Returns
        -------
        log_probs : (G,)
        """
        context = self.get_context(batch)  # (1, context_dim)
        context_expanded = context.expand(len(grid_points), -1)  # (G, context_dim)
        return self.flow.log_prob(grid_points, context_expanded)


def build_model(model_type: str, cfg: dict) -> NPEModel:
    """Factory to create NPEModel from config."""
    mcfg = cfg["model"]
    context_dim = mcfg["context_dim"]

    if model_type == "transformer":
        encoder = TransformerEncoderModel(
            n_cont_features=6,
            d_model=mcfg["d_model"],
            nhead=mcfg["nhead"],
            num_layers=mcfg["num_layers"],
            dim_feedforward=mcfg["dim_feedforward"],
            dropout=mcfg["dropout"],
            context_dim=context_dim,
        )
    elif model_type == "lstm":
        encoder = LSTMEncoderModel(
            n_cont_features=6,
            d_model=mcfg["d_model"],
            lstm_hidden=mcfg["lstm_hidden"],
            lstm_layers=mcfg["lstm_layers"],
            dropout=mcfg["dropout"],
            context_dim=context_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    flow = PosteriorFlow(
        theta_dim=2,
        context_dim=context_dim,
        n_transforms=mcfg["flow_transforms"],
        hidden_features=mcfg["flow_hidden"],
    )

    return NPEModel(encoder, flow)
