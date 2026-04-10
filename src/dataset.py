"""PyTorch Dataset and collate for on-the-fly simulation of pulsar data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Union

from .schedules import generate_schedule, Schedule
from .simulator import simulate_pulsar, simulate_pulsar_factorized, SimulatedPulsar
from .priors import UniformPrior, FactorizedPrior
from .masking import apply_random_masking
from .models.tokenization import tokenize


def _post_process(
    sim: SimulatedPulsar,
    rng,
    masking_severity,
    augment,
    extra=None,
    deterministic_augment=False,
):
    """Apply masking, tokenize, and build output dict."""
    sev = masking_severity
    if augment and sev > 0:
        if deterministic_augment:
            # Reuse the per-sample seeded rng so val masking is reproducible
            # across epochs (same idx → same mask). rng has already been
            # consumed by schedule+simulation but its remaining state is
            # still deterministic per idx.
            aug_rng = rng
        else:
            # Non-deterministic RNG for training so masking varies across
            # epochs (the seeded rng would otherwise produce identical masks).
            aug_rng = np.random.default_rng()
        sev_sample = aug_rng.uniform(0, sev)
        keep = apply_random_masking(sim.t, aug_rng, severity=sev_sample)
    else:
        keep = np.ones(len(sim.t), dtype=bool)

    t_k = sim.t[keep]
    sigma_k = sim.sigma[keep]
    r_k = sim.residuals[keep]
    freq_k = sim.freq_mhz[keep]
    backend_k = sim.backend_id[keep]

    # Update backend_active if masking removed all tokens from a backend
    if extra and "backend_active" in extra:
        backend_active = extra["backend_active"]
        for b in range(len(backend_active)):
            if backend_active[b] and not np.any(backend_k == b):
                backend_active[b] = False

    tokens = tokenize(t_k, sigma_k, r_k, freq_k, backend_k)
    tspan_yr = float(t_k.max() - t_k.min()) if len(t_k) > 1 else 0.0

    result = {
        "tokens": tokens,
        "seq_len": len(t_k),
        "tspan_yr": torch.tensor(tspan_yr, dtype=torch.float32),
    }
    if extra:
        result.update(extra)
    return result


class PulsarDataset(Dataset):
    """Generates (theta, token_features) pairs on the fly from RNG seeds.

    Each __getitem__ call:
    1. Seeds an RNG from (base_seed + index).
    2. Draws theta from pre-generated Sobol array (or samples from prior).
    3. Generates a schedule.
    4. Simulates residuals.
    5. Optionally applies masking augmentations.
    6. Tokenizes.

    Supports both standard and factorized modes via the `factorized` flag.
    """

    def __init__(
        self,
        n_samples: int,
        prior: Union[UniformPrior, FactorizedPrior],
        data_cfg: dict,
        seed: int = 0,
        masking_severity: float = 0.0,
        augment: bool = False,
        use_sobol: bool = False,
        factorized: bool = False,
        reseed_per_epoch: bool = False,
        deterministic_augment: bool = False,
    ):
        self.n_samples = n_samples
        self.prior = prior
        self.data_cfg = data_cfg
        self.seed = seed
        self.masking_severity = masking_severity
        self.augment = augment
        self.factorized = factorized
        self.reseed_per_epoch = reseed_per_epoch
        self.deterministic_augment = deterministic_augment
        self._epoch = 0

        if factorized:
            if use_sobol:
                self._global_bank = prior.sample_global_sobol(
                    n_samples, seed=seed
                ).numpy()
                self._wn_banks = [
                    prior.sample_wn_sobol(n_samples, backend_index=b, seed=seed).numpy()
                    for b in range(prior.n_backends_max)
                ]
            else:
                self._global_bank = None
                self._wn_banks = None
            self._theta_bank = None
        else:
            if use_sobol:
                self._theta_bank = prior.sample_sobol(n_samples, seed=seed).numpy()
            else:
                self._theta_bank = None
            self._global_bank = None
            self._wn_banks = None

    def set_epoch(self, epoch: int) -> None:
        """Advance the dataset seed so each epoch draws fresh (θ, x) pairs.

        Has no effect when reseed_per_epoch=False or use_sobol=True (Sobol banks
        are pre-generated and cannot be reshuffled per epoch).
        """
        self._epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        if self.reseed_per_epoch:
            rng = np.random.default_rng(self.seed + idx + self._epoch * self.n_samples)
        else:
            rng = np.random.default_rng(self.seed + idx)

        schedule = generate_schedule(
            rng,
            tspan_min_yr=self.data_cfg.get("tspan_min_yr", 5.0),
            tspan_max_yr=self.data_cfg.get("tspan_max_yr", 15.0),
            n_toa_min=self.data_cfg.get("n_toa_min", 80),
            n_toa_max=self.data_cfg.get("n_toa_max", 400),
            n_backends_fixed=self.data_cfg.get("n_backends_fixed", None),
        )

        n_modes = self.data_cfg.get("n_fourier_modes", 30)

        if self.factorized:
            return self._getitem_factorized(idx, rng, schedule, n_modes)
        return self._getitem_standard(idx, rng, schedule, n_modes)

    def _getitem_standard(self, idx, rng, schedule, n_modes):
        if self._theta_bank is not None:
            theta = self._theta_bank[idx]
        else:
            theta = self.prior.sample(1, rng=rng).squeeze(0).numpy()

        n_backends_fixed = self.data_cfg.get("n_backends_fixed", None)
        if n_backends_fixed is not None and len(theta) > 7:
            # Monolithic multi-backend: split flat theta into global + per-backend WN
            # and simulate with per-backend white noise
            theta_global = theta[:4]
            theta_wn = theta[4:].reshape(n_backends_fixed, 3)
            sim = simulate_pulsar_factorized(
                theta_global, theta_wn, schedule, n_modes=n_modes, rng=rng,
            )
            # Return the flat theta for the monolithic flow
            sim.theta = theta.astype(np.float32)
        else:
            sim = simulate_pulsar(theta, schedule, n_modes=n_modes, rng=rng)

        return _post_process(
            sim,
            rng,
            self.masking_severity,
            self.augment,
            extra={"theta": torch.from_numpy(sim.theta)},
            deterministic_augment=self.deterministic_augment,
        )

    def _getitem_factorized(self, idx, rng, schedule, n_modes):
        n_backends = schedule.n_backends

        if self._global_bank is not None:
            theta_global = self._global_bank[idx]
            theta_wn = np.stack([self._wn_banks[b][idx] for b in range(n_backends)])
        else:
            theta_global = self.prior.sample_global(1, rng=rng).squeeze(0).numpy()
            theta_wn = np.stack(
                [
                    self.prior.sample_wn(1, rng=rng).squeeze(0).numpy()
                    for _ in range(n_backends)
                ]
            )

        sim = simulate_pulsar_factorized(
            theta_global,
            theta_wn,
            schedule,
            n_modes=n_modes,
            rng=rng,
        )

        # Pad to n_backends_max
        Bmax = self.prior.n_backends_max
        theta_wn_padded = torch.zeros(Bmax, 3)
        theta_wn_padded[:n_backends] = torch.from_numpy(sim.theta_wn)
        backend_active = torch.zeros(Bmax, dtype=torch.bool)
        backend_active[:n_backends] = True

        return _post_process(
            sim,
            rng,
            self.masking_severity,
            self.augment,
            extra={
                "theta_global": torch.from_numpy(sim.theta_global),
                "theta_wn": theta_wn_padded,
                "backend_active": backend_active,
            },
            deterministic_augment=self.deterministic_augment,
        )


class FixedPulsarDataset(Dataset):
    """Pre-generated dataset for evaluation (stores full sim data for exact posterior)."""

    def __init__(
        self,
        n_samples: int,
        prior: Union[UniformPrior, FactorizedPrior],
        data_cfg: dict,
        seed: int = 100000,
        factorized: bool = False,
    ):
        self.factorized = factorized
        self.items: list[dict] = []
        for idx in range(n_samples):
            rng = np.random.default_rng(seed + idx)
            schedule = generate_schedule(
                rng,
                tspan_min_yr=data_cfg.get("tspan_min_yr", 5.0),
                tspan_max_yr=data_cfg.get("tspan_max_yr", 15.0),
                n_toa_min=data_cfg.get("n_toa_min", 80),
                n_toa_max=data_cfg.get("n_toa_max", 400),
                n_backends_fixed=data_cfg.get("n_backends_fixed", None),
            )
            n_modes = data_cfg.get("n_fourier_modes", 30)

            if factorized:
                n_backends = schedule.n_backends
                theta_global = prior.sample_global(1, rng=rng).squeeze(0).numpy()
                theta_wn = np.stack(
                    [
                        prior.sample_wn(1, rng=rng).squeeze(0).numpy()
                        for _ in range(n_backends)
                    ]
                )
                sim = simulate_pulsar_factorized(
                    theta_global,
                    theta_wn,
                    schedule,
                    n_modes=n_modes,
                    rng=rng,
                )
                Bmax = prior.n_backends_max
                theta_wn_padded = torch.zeros(Bmax, 3)
                theta_wn_padded[:n_backends] = torch.from_numpy(sim.theta_wn)
                backend_active = torch.zeros(Bmax, dtype=torch.bool)
                backend_active[:n_backends] = True
                extra = {
                    "theta_global": torch.from_numpy(sim.theta_global),
                    "theta_wn": theta_wn_padded,
                    "backend_active": backend_active,
                }
            else:
                theta = prior.sample(1, rng=rng).squeeze(0).numpy()
                n_bf = data_cfg.get("n_backends_fixed", None)
                if n_bf is not None and len(theta) > 7:
                    theta_global = theta[:4]
                    theta_wn = theta[4:].reshape(n_bf, 3)
                    sim = simulate_pulsar_factorized(
                        theta_global, theta_wn, schedule,
                        n_modes=n_modes, rng=rng,
                    )
                    sim.theta = theta.astype(np.float32)
                else:
                    sim = simulate_pulsar(theta, schedule, n_modes=n_modes, rng=rng)
                extra = {"theta": torch.from_numpy(sim.theta)}

            tokens = tokenize(
                sim.t,
                sim.sigma,
                sim.residuals,
                sim.freq_mhz,
                sim.backend_id,
            )
            tspan_yr = float(sim.t.max() - sim.t.min()) if len(sim.t) > 1 else 0.0
            item = {
                "tokens": tokens,
                "seq_len": len(sim.t),
                "tspan_yr": torch.tensor(tspan_yr, dtype=torch.float32),
                "sim": sim,
            }
            item.update(extra)
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]
