"""Synthetic observing-schedule generator for a single pulsar.

Produces irregular, gappy, variable-length TOA schedules that mimic
realistic radio-pulsar timing programmes (seasonal windows, cadence
variations, missing seasons, variable baseline).

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
    sigma: np.ndarray  # TOA uncertainty in arbitrary units, shape (N,)
    freq_mhz: np.ndarray  # observing frequency in MHz, shape (N,)
    backend_id: np.ndarray  # integer backend label, shape (N,)

    @property
    def n_toa(self) -> int:
        return len(self.t)

    @property
    def tspan(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


def generate_schedule(
    rng: np.random.Generator,
    tspan_min_yr: float = 5.0,
    tspan_max_yr: float = 15.0,
    n_toa_min: int = 80,
    n_toa_max: int = 400,
) -> Schedule:
    """Create one random observing schedule.

    The schedule has:
    * a random total baseline between *tspan_min_yr* and *tspan_max_yr*
    * seasonal observing windows (≈8 months on, ≈4 months off each year)
    * irregular cadence within each season
    * randomly dropped seasons (~20 % chance each)
    * optional cadence thinning in some seasons
    * heteroskedastic white-noise uncertainties
    * random backend labels and observing frequencies
    """
    # ---- baseline & start epoch ----
    tspan = rng.uniform(tspan_min_yr, tspan_max_yr)
    t_start = 0.0

    # ---- build seasonal windows ----
    n_years = int(np.ceil(tspan))
    times_list: list[np.ndarray] = []
    for yr in range(n_years):
        # ~20 % chance to skip a whole season
        if rng.random() < 0.20:
            continue
        season_start = t_start + yr + rng.uniform(0.0, 0.15)
        season_end = t_start + yr + rng.uniform(0.55, 0.80)
        season_end = min(season_end, t_start + tspan)
        if season_end <= season_start:
            continue
        # cadence within season: mean gap 2-6 weeks
        mean_gap_yr = rng.uniform(2.0 / 52, 6.0 / 52)
        n_obs_season = max(2, int((season_end - season_start) / mean_gap_yr))
        # cadence thinning: sometimes only keep a fraction
        if rng.random() < 0.25:
            n_obs_season = max(2, n_obs_season // 2)
        obs = np.sort(rng.uniform(season_start, season_end, size=n_obs_season))
        times_list.append(obs)

    if len(times_list) == 0:
        # fallback: at least a few observations
        times_list.append(np.sort(rng.uniform(t_start, t_start + tspan, size=10)))

    t_all = np.concatenate(times_list)
    t_all = np.sort(np.unique(t_all))

    # ---- trim to desired count range ----
    n_target = rng.integers(n_toa_min, n_toa_max + 1)
    if len(t_all) > n_target:
        idx = np.sort(rng.choice(len(t_all), size=n_target, replace=False))
        t_all = t_all[idx]
    elif len(t_all) < n_toa_min:
        # pad with extra random observations
        extra = np.sort(rng.uniform(t_all[0], t_all[-1], size=n_toa_min - len(t_all)))
        t_all = np.sort(np.concatenate([t_all, extra]))

    n = len(t_all)

    # ---- uncertainties (heteroskedastic, log-uniform) ----
    log_sig_low, log_sig_high = -7.0, -5.0  # seconds (100 ns to 10 μs)
    sigma = 10 ** rng.uniform(log_sig_low, log_sig_high, size=n).astype(np.float32)

    # ---- observing frequency ----
    freq_choices = np.array([820.0, 1400.0, 2300.0])
    freq_mhz = rng.choice(freq_choices, size=n).astype(np.float32)

    # ---- backend id ----
    n_backends = rng.integers(1, 4)
    backend_id = rng.integers(0, n_backends, size=n).astype(np.int64)

    return Schedule(
        t=t_all.astype(np.float32),
        sigma=sigma,
        freq_mhz=freq_mhz,
        backend_id=backend_id,
    )
