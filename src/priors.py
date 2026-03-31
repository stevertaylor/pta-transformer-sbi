"""Uniform prior over the D-dimensional parameter vector theta.

The parameter dimension and bounds are read from the config dict whose
keys define the canonical parameter ordering.  Supports Sobol (quasi-random)
sampling for better coverage of high-dimensional prior spaces.

Also provides FactorizedPrior for models with separate global (red/DM)
and per-backend (EFAC/EQUAD/ECORR) parameter blocks.
"""

from __future__ import annotations

import torch
import numpy as np
from scipy.stats.qmc import Sobol
from typing import Tuple, List


class UniformPrior:
    """Axis-aligned uniform prior on a D-dimensional box."""

    def __init__(self, bounds: dict):
        """
        Parameters
        ----------
        bounds : dict
            Ordered mapping of parameter name → [low, high].
            Example::

                {"log10_A_red": [-17, -11],
                 "gamma_red": [0.5, 6.5],
                 "log10_A_dm": [-17, -11],
                 ...}
        """
        self.param_names: List[str] = list(bounds.keys())
        self.lo = torch.tensor(
            [bounds[k][0] for k in self.param_names], dtype=torch.float32
        )
        self.hi = torch.tensor(
            [bounds[k][1] for k in self.param_names], dtype=torch.float32
        )
        self.log_vol = torch.log(self.hi - self.lo).sum()

    # ------------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator | None = None) -> torch.Tensor:
        """Return (n, D) tensor of iid uniform prior samples."""
        D = len(self.param_names)
        if rng is None:
            u = torch.rand(n, D)
        else:
            u = torch.from_numpy(rng.random((n, D)).astype(np.float32))
        return self.lo + u * (self.hi - self.lo)

    def sample_sobol(self, n: int, seed: int = 42) -> torch.Tensor:
        """Return (n, D) tensor of scrambled Sobol quasi-random samples.

        Sobol sequences provide better coverage of the prior volume than
        iid sampling, with discrepancy O(log^d N / N) vs O(1/sqrt(N)).
        Scrambling preserves low-discrepancy while breaking lattice artifacts.
        """
        D = len(self.param_names)
        sampler = Sobol(d=D, scramble=True, seed=seed)
        u = torch.from_numpy(sampler.random(n).astype(np.float32))
        return self.lo + u * (self.hi - self.lo)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate log-prior (uniform), returns (n,)."""
        inside = ((theta >= self.lo) & (theta <= self.hi)).all(dim=-1)
        lp = torch.where(inside, -self.log_vol, torch.tensor(-float("inf")))
        return lp

    @property
    def dim(self) -> int:
        return len(self.param_names)

    @property
    def bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lo.numpy(), self.hi.numpy()


class FactorizedPrior:
    """Factorized prior: global params + shared per-backend white-noise params.

    Global: (log10_A_red, gamma_red, log10_A_dm, gamma_dm) — 4D
    Per-backend WN: (EFAC, log10_EQUAD, log10_ECORR) — 3D, same bounds for all backends
    """

    def __init__(
        self,
        global_bounds: dict,
        wn_bounds: dict,
        n_backends_max: int = 4,
    ):
        self.global_prior = UniformPrior(global_bounds)
        self.wn_prior = UniformPrior(wn_bounds)
        self.n_backends_max = n_backends_max

    @property
    def global_dim(self) -> int:
        return self.global_prior.dim

    @property
    def wn_dim(self) -> int:
        return self.wn_prior.dim

    def sample_global_sobol(self, n: int, seed: int = 42) -> torch.Tensor:
        """(n, global_dim) Sobol samples for global params."""
        return self.global_prior.sample_sobol(n, seed=seed)

    def sample_wn_sobol(
        self, n: int, backend_index: int = 0, seed: int = 42
    ) -> torch.Tensor:
        """(n, wn_dim) Sobol samples for one backend's WN params."""
        return self.wn_prior.sample_sobol(n, seed=seed + 1000 * (backend_index + 1))

    def sample_global(self, n: int, rng=None) -> torch.Tensor:
        return self.global_prior.sample(n, rng=rng)

    def sample_wn(self, n: int, rng=None) -> torch.Tensor:
        return self.wn_prior.sample(n, rng=rng)
