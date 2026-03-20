"""Uniform prior over the 2-D parameter vector theta = (log10_A_red, gamma_red).

All units are arbitrary simulator units – see README for details.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple


class UniformPrior:
    """Axis-aligned uniform prior on a 2-D box."""

    def __init__(self, bounds: dict):
        """
        Parameters
        ----------
        bounds : dict
            Must contain keys ``log10_A_red`` and ``gamma_red``, each mapping
            to a [low, high] list.
        """
        self.lo = torch.tensor(
            [bounds["log10_A_red"][0], bounds["gamma_red"][0]], dtype=torch.float32
        )
        self.hi = torch.tensor(
            [bounds["log10_A_red"][1], bounds["gamma_red"][1]], dtype=torch.float32
        )
        self.log_vol = torch.log(self.hi - self.lo).sum()

    # ------------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator | None = None) -> torch.Tensor:
        """Return (n, 2) tensor of prior samples."""
        if rng is None:
            u = torch.rand(n, 2)
        else:
            u = torch.from_numpy(rng.random((n, 2)).astype(np.float32))
        return self.lo + u * (self.hi - self.lo)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate log-prior (uniform), returns (n,)."""
        inside = ((theta >= self.lo) & (theta <= self.hi)).all(dim=-1)
        lp = torch.where(inside, -self.log_vol, torch.tensor(-float("inf")))
        return lp

    @property
    def dim(self) -> int:
        return 2

    @property
    def bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lo.numpy(), self.hi.numpy()
