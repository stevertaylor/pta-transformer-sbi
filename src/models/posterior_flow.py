"""Conditional normalizing flow posterior head: q(theta | context).

Uses Zuko's Neural Spline Flow (NSF) for a 2-D conditional density.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import zuko


class PosteriorFlow(nn.Module):
    """Thin wrapper around a Zuko NSF conditioned on context_dim."""

    def __init__(
        self,
        theta_dim: int = 2,
        context_dim: int = 64,
        n_transforms: int = 6,
        hidden_features: int = 128,
    ):
        super().__init__()
        self.flow = zuko.flows.NSF(
            features=theta_dim,
            context=context_dim,
            transforms=n_transforms,
            hidden_features=[hidden_features, hidden_features],
        )

    def log_prob(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Evaluate log q(theta | context).

        Parameters
        ----------
        theta   : (B, 2)
        context : (B, context_dim)

        Returns
        -------
        log_prob : (B,)
        """
        dist = self.flow(context)
        return dist.log_prob(theta)

    def sample(self, context: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """Draw samples from q(theta | context).

        Returns (B, n_samples, theta_dim).
        """
        dist = self.flow(context)
        # Zuko distribution .sample takes a sample shape
        return dist.sample((n_samples,)).permute(1, 0, 2)
