"""Synthetic observing-schedule generator for a single pulsar.

Produces irregular, gappy, variable-length TOA schedules that mimic
realistic radio-pulsar timing programmes (seasonal windows, cadence
variations, missing seasons, variable baseline).

Each observing epoch contains 2–3 TOAs at different radio frequencies,
mimicking real multi-frequency PTA observations.  A per-epoch minimum of
2 is required for ECORR to be identifiable (single-TOA epochs are degenerate
between ECORR and EQUAD).  The epoch_id array
groups TOAs by observing session (needed for ECORR noise modelling).

Times are in years.  TOA uncertainties (sigma) are in seconds.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Schedule:
    """Container for one pulsar's observing schedule."""

    t: np.ndarray  # observation times in years, shape (N,)
    sigma: np.ndarray  # TOA uncertainty in seconds, shape (N,)
    freq_mhz: np.ndarray  # observing frequency in MHz, shape (N,)
    backend_id: np.ndarray  # integer backend label, shape (N,)
    epoch_id: np.ndarray  # integer epoch label (groups co-temporal TOAs), shape (N,)

    @property
    def n_toa(self) -> int:
        return len(self.t)

    @property
    def n_epoch(self) -> int:
        return int(self.epoch_id.max()) + 1 if len(self.epoch_id) > 0 else 0

    @property
    def n_backends(self) -> int:
        return int(self.backend_id.max()) + 1 if len(self.backend_id) > 0 else 0

    @property
    def tspan(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


def generate_schedule(
    rng: np.random.Generator,
    tspan_min_yr: float = 5.0,
    tspan_max_yr: float = 15.0,
    n_toa_min: int = 80,
    n_toa_max: int = 400,
    n_backends_fixed: Optional[int] = None,
) -> Schedule:
    """Create one random observing schedule with multi-frequency epochs.

    The schedule has:
    * a random total baseline between *tspan_min_yr* and *tspan_max_yr*
    * seasonal observing windows (≈8 months on, ≈4 months off each year)
    * irregular cadence within each season
    * randomly dropped seasons (~20 % chance each)
    * optional cadence thinning in some seasons
    * 2–3 TOAs per epoch at different frequencies (for ECORR structure)
    * heteroskedastic white-noise uncertainties
    * random backend labels
    """
    # ---- baseline ----
    tspan = rng.uniform(tspan_min_yr, tspan_max_yr)
    t_start = 0.0

    # ---- build seasonal epoch times ----
    n_years = int(np.ceil(tspan))
    epoch_time_list: list[np.ndarray] = []
    for yr in range(n_years):
        if rng.random() < 0.20:
            continue  # skip season
        season_start = t_start + yr + rng.uniform(0.0, 0.15)
        season_end = t_start + yr + rng.uniform(0.55, 0.80)
        season_end = min(season_end, t_start + tspan)
        if season_end <= season_start:
            continue
        mean_gap_yr = rng.uniform(2.0 / 52, 6.0 / 52)
        n_epochs_season = max(2, int((season_end - season_start) / mean_gap_yr))
        if rng.random() < 0.25:
            n_epochs_season = max(2, n_epochs_season // 2)
        epochs = np.sort(rng.uniform(season_start, season_end, size=n_epochs_season))
        epoch_time_list.append(epochs)

    if len(epoch_time_list) == 0:
        epoch_time_list.append(np.sort(rng.uniform(t_start, t_start + tspan, size=10)))

    epoch_times = np.sort(np.unique(np.concatenate(epoch_time_list)))

    # ---- target TOA count → adjust epoch count ----
    n_target = rng.integers(n_toa_min, n_toa_max + 1)
    mean_toa_per_epoch = 2.5  # average TOAs per epoch (2-3, uniform)
    n_epochs_target = max(5, int(n_target / mean_toa_per_epoch))
    if len(epoch_times) > n_epochs_target:
        idx = np.sort(rng.choice(len(epoch_times), size=n_epochs_target, replace=False))
        epoch_times = epoch_times[idx]

    # ---- generate TOAs per epoch ----
    n_backends = n_backends_fixed if n_backends_fixed is not None else rng.integers(1, 4)
    freq_choices = np.array([820.0, 1400.0, 2300.0])
    log_sig_low, log_sig_high = -7.0, -5.0  # seconds (100 ns to 10 μs)

    all_t: list[float] = []
    all_sigma: list[float] = []
    all_freq: list[float] = []
    all_backend: list[int] = []
    all_epoch_id: list[int] = []

    for eid, t_epoch in enumerate(epoch_times):
        n_toa_epoch = rng.integers(2, 4)  # 2–3 TOAs per epoch (min 2 for ECORR identifiability)
        n_freq = min(n_toa_epoch, len(freq_choices))
        freqs = rng.choice(freq_choices, size=n_freq, replace=False)
        # If more TOAs than unique freqs, allow repeats
        if n_toa_epoch > n_freq:
            extra = rng.choice(freq_choices, size=n_toa_epoch - n_freq)
            freqs = np.concatenate([freqs, extra])

        for j in range(n_toa_epoch):
            # Tiny time offset within epoch (up to ~30 min)
            dt_offset = rng.uniform(0, 0.5 / (365.25 * 24))
            all_t.append(t_epoch + dt_offset)
            all_sigma.append(10 ** rng.uniform(log_sig_low, log_sig_high))
            all_freq.append(float(freqs[j]))
            all_backend.append(int(rng.integers(0, n_backends)))
            all_epoch_id.append(eid)

    # ---- sort by time, keeping epoch assignments ----
    order = np.argsort(all_t)
    t_arr = np.array(all_t, dtype=np.float32)[order]
    sigma_arr = np.array(all_sigma, dtype=np.float32)[order]
    freq_arr = np.array(all_freq, dtype=np.float32)[order]
    backend_arr = np.array(all_backend, dtype=np.int64)[order]
    epoch_arr = np.array(all_epoch_id, dtype=np.int64)[order]

    # ---- trim to desired range ----
    n = len(t_arr)
    if n > n_toa_max:
        # Drop whole epochs (not individual TOAs) to preserve min-2 per epoch
        unique_eids = np.unique(epoch_arr)
        shuffled_eids = rng.permutation(unique_eids)
        keep_mask = np.zeros(n, dtype=bool)
        kept = 0
        for eid in shuffled_eids:
            eid_mask = epoch_arr == eid
            m = int(eid_mask.sum())
            if kept + m > n_toa_max:
                continue
            keep_mask |= eid_mask
            kept += m
            if kept == n_toa_max:
                break
        t_arr = t_arr[keep_mask]
        sigma_arr = sigma_arr[keep_mask]
        freq_arr = freq_arr[keep_mask]
        backend_arr = backend_arr[keep_mask]
        epoch_arr = epoch_arr[keep_mask]
        order = np.argsort(t_arr)
        t_arr = t_arr[order]
        sigma_arr = sigma_arr[order]
        freq_arr = freq_arr[order]
        backend_arr = backend_arr[order]
        epoch_arr = epoch_arr[order]
    elif n < n_toa_min:
        # Pad by adding extra 2-TOA epochs (min 2 per epoch for ECORR identifiability).
        # Round up to an even pad so the last epoch isn't a singleton; may slightly
        # overshoot n_toa_min (by at most 1 TOA).
        n_extra = n_toa_min - n
        n_extra_epochs = int(np.ceil(n_extra / 2))
        n_extra = n_extra_epochs * 2  # always even
        epoch_t = np.sort(
            rng.uniform(t_arr[0], t_arr[-1], size=n_extra_epochs)
        ).astype(np.float32)
        next_eid = epoch_arr.max() + 1 if len(epoch_arr) > 0 else 0
        extra_t_list, extra_epoch_list = [], []
        for k in range(n_extra_epochs):
            for _ in range(2):
                dt_offset = rng.uniform(0, 0.5 / (365.25 * 24))
                extra_t_list.append(epoch_t[k] + dt_offset)
                extra_epoch_list.append(next_eid + k)
        extra_t = np.array(extra_t_list, dtype=np.float32)
        extra_sigma = (
            10 ** rng.uniform(log_sig_low, log_sig_high, size=n_extra)
        ).astype(np.float32)
        extra_freq = rng.choice(freq_choices, size=n_extra).astype(np.float32)
        extra_backend = rng.integers(0, n_backends, size=n_extra).astype(np.int64)
        extra_epoch = np.array(extra_epoch_list, dtype=np.int64)

        t_arr = np.concatenate([t_arr, extra_t])
        sigma_arr = np.concatenate([sigma_arr, extra_sigma])
        freq_arr = np.concatenate([freq_arr, extra_freq])
        backend_arr = np.concatenate([backend_arr, extra_backend])
        epoch_arr = np.concatenate([epoch_arr, extra_epoch])

        order = np.argsort(t_arr)
        t_arr = t_arr[order]
        sigma_arr = sigma_arr[order]
        freq_arr = freq_arr[order]
        backend_arr = backend_arr[order]
        epoch_arr = epoch_arr[order]

    # Re-label epoch IDs to be contiguous 0..n_epoch-1
    _, epoch_arr = np.unique(epoch_arr, return_inverse=True)

    return Schedule(
        t=t_arr,
        sigma=sigma_arr,
        freq_mhz=freq_arr,
        backend_id=backend_arr,
        epoch_id=epoch_arr.astype(np.int64),
    )
