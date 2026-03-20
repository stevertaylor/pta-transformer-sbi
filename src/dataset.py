"""PyTorch Dataset and collate for on-the-fly simulation of pulsar data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional

from .schedules import generate_schedule, Schedule
from .simulator import simulate_pulsar, SimulatedPulsar
from .priors import UniformPrior
from .masking import apply_random_masking
from .models.tokenization import tokenize


class PulsarDataset(Dataset):
    """Generates (theta, token_features) pairs on the fly from RNG seeds.

    Each __getitem__ call:
    1. Seeds an RNG from (base_seed + index).
    2. Draws theta ~ prior.
    3. Generates a schedule.
    4. Simulates residuals.
    5. Optionally applies masking augmentations.
    6. Tokenizes.
    """

    def __init__(
        self,
        n_samples: int,
        prior: UniformPrior,
        data_cfg: dict,
        seed: int = 0,
        masking_severity: float = 0.0,
        augment: bool = False,
    ):
        self.n_samples = n_samples
        self.prior = prior
        self.data_cfg = data_cfg
        self.seed = seed
        self.masking_severity = masking_severity
        self.augment = augment

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        theta = self.prior.sample(1, rng=rng).squeeze(0).numpy()

        schedule = generate_schedule(
            rng,
            tspan_min_yr=self.data_cfg.get("tspan_min_yr", 5.0),
            tspan_max_yr=self.data_cfg.get("tspan_max_yr", 15.0),
            n_toa_min=self.data_cfg.get("n_toa_min", 80),
            n_toa_max=self.data_cfg.get("n_toa_max", 400),
        )

        n_modes = self.data_cfg.get("n_fourier_modes", 30)
        sim = simulate_pulsar(theta, schedule, n_modes=n_modes, rng=rng)

        # Apply masking augmentation during training
        sev = self.masking_severity
        if self.augment and sev > 0:
            # Random severity per sample
            sev_sample = rng.uniform(0, sev)
            keep = apply_random_masking(sim.t, rng, severity=sev_sample)
        else:
            keep = np.ones(len(sim.t), dtype=bool)

        t_k = sim.t[keep]
        sigma_k = sim.sigma[keep]
        r_k = sim.residuals[keep]
        freq_k = sim.freq_mhz[keep]
        backend_k = sim.backend_id[keep]

        tokens = tokenize(t_k, sigma_k, r_k, freq_k, backend_k)

        return {
            "theta": torch.from_numpy(sim.theta),
            "tokens": tokens,
            "seq_len": len(t_k),
        }


class FixedPulsarDataset(Dataset):
    """Pre-generated dataset for evaluation (stores full sim data for exact posterior)."""

    def __init__(
        self,
        n_samples: int,
        prior: UniformPrior,
        data_cfg: dict,
        seed: int = 100000,
    ):
        self.items: list[dict] = []
        for idx in range(n_samples):
            rng = np.random.default_rng(seed + idx)
            theta = prior.sample(1, rng=rng).squeeze(0).numpy()
            schedule = generate_schedule(
                rng,
                tspan_min_yr=data_cfg.get("tspan_min_yr", 5.0),
                tspan_max_yr=data_cfg.get("tspan_max_yr", 15.0),
                n_toa_min=data_cfg.get("n_toa_min", 80),
                n_toa_max=data_cfg.get("n_toa_max", 400),
            )
            n_modes = data_cfg.get("n_fourier_modes", 30)
            sim = simulate_pulsar(theta, schedule, n_modes=n_modes, rng=rng)
            tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
            self.items.append({
                "theta": torch.from_numpy(sim.theta),
                "tokens": tokens,
                "seq_len": len(sim.t),
                "sim": sim,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]
