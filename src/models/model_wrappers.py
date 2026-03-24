"""Model wrappers: encoder + posterior flow, with a unified API."""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoderModel
from .lstm_encoder import LSTMEncoderModel
from .posterior_flow import PosteriorFlow

N_AUX_FEATURES = 4  # log_n_toa, log_tspan, mean_log_sigma, std_log_sigma


class NPEModel(nn.Module):
    """Encoder + posterior flow for amortized neural posterior estimation.

    The encoder maps (features, backend_id, mask) → context.
    The flow maps (theta, context) → log_prob.
    When use_aux=True, 4 summary statistics are concatenated to the context.
    """

    def __init__(self, encoder: nn.Module, flow: PosteriorFlow, use_aux: bool = False):
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.use_aux = use_aux

    def _compute_aux(self, batch: dict) -> torch.Tensor:
        """Compute auxiliary summary features from the batch. Returns (B, 4)."""
        mask = batch["mask"]                          # (B, L)
        n_toa = mask.sum(1).float()                   # (B,)
        log_sigma = batch["features"][:, :, 3]        # (B, L)  — log_sigma channel
        mask_f = mask.float()
        n = mask_f.sum(1).clamp(min=1)
        mean_ls = (log_sigma * mask_f).sum(1) / n
        var_ls = ((log_sigma - mean_ls.unsqueeze(1)) ** 2 * mask_f).sum(1) / n
        std_ls = var_ls.sqrt()
        tspan = batch["tspan_yr"]                     # (B,)
        return torch.stack([
            torch.log(n_toa.clamp(min=1)),
            torch.log(tspan.clamp(min=1e-3)),
            mean_ls,
            std_ls,
        ], dim=-1)

    def _get_flow_context(self, batch: dict) -> torch.Tensor:
        context = self.encoder(
            batch["features"], batch["backend_id"], batch["mask"]
        )
        if self.use_aux:
            aux = self._compute_aux(batch)
            context = torch.cat([context, aux], dim=-1)
        return context

    def forward(self, batch: dict) -> torch.Tensor:
        """Compute negative log-prob loss for training.

        Returns scalar loss = -mean log q(theta | x).
        """
        context = self._get_flow_context(batch)
        log_prob = self.flow.log_prob(batch["theta"], context)
        return -log_prob.mean()

    def get_context(self, batch: dict) -> torch.Tensor:
        return self._get_flow_context(batch)

    @torch.no_grad()
    def sample_posterior(self, batch: dict, n_samples: int = 1000) -> torch.Tensor:
        """Sample from learned posterior. Returns (B, n_samples, 2)."""
        context = self._get_flow_context(batch)
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
        context = self._get_flow_context(batch)  # (1, context_dim)
        context_expanded = context.expand(len(grid_points), -1)  # (G, context_dim)
        return self.flow.log_prob(grid_points, context_expanded)


def build_model(model_type: str, cfg: dict) -> NPEModel:
    """Factory to create NPEModel from config."""
    mcfg = cfg["model"]
    context_dim = mcfg["context_dim"]
    use_aux = mcfg.get("use_aux_features", False)
    flow_context_dim = context_dim + (N_AUX_FEATURES if use_aux else 0)

    if model_type == "transformer":
        encoder = TransformerEncoderModel(
            n_cont_features=6,
            d_model=mcfg["d_model"],
            nhead=mcfg["nhead"],
            num_layers=mcfg["num_layers"],
            dim_feedforward=mcfg["dim_feedforward"],
            dropout=mcfg["dropout"],
            context_dim=context_dim,
            use_rope=mcfg.get("use_rope", False),
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
        context_dim=flow_context_dim,
        n_transforms=mcfg["flow_transforms"],
        hidden_features=mcfg["flow_hidden"],
        n_hidden_layers=mcfg.get("flow_layers", 2),
        n_bins=mcfg.get("flow_bins", 8),
    )

    return NPEModel(encoder, flow, use_aux=use_aux)
