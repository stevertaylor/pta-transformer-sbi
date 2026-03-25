"""Single-pulsar noise simulator using a Fourier design-matrix approach.

Generates residuals  r = red + dm + white + ecorr  given a parameter vector
theta and an observing schedule.

Supports two parameter-vector formats:
  2-D:  theta = (log10_A_red, gamma_red)           — red noise only
  7-D:  theta = (log10_A_red, gamma_red,
                  log10_A_dm, gamma_dm,
                  EFAC, log10_EQUAD, log10_ECORR)  — full noise model

Uses the standard PTA power-law parameterisation (enterprise convention).
Times are in years; residuals, uncertainties, and Fourier-coefficient
variances are in seconds / seconds².
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .schedules import Schedule

SEC_PER_YR = 365.25 * 86400  # seconds per Julian year

# Canonical parameter ordering for the 7-D model
PARAM_NAMES_7D = [
    "log10_A_red",
    "gamma_red",
    "log10_A_dm",
    "gamma_dm",
    "EFAC",
    "log10_EQUAD",
    "log10_ECORR",
]

DM_FREQ_REF = 1400.0  # MHz – reference frequency for chromatic DM scaling


@dataclass
class SimulatedPulsar:
    """Everything needed for training and exact-likelihood evaluation."""

    theta: np.ndarray  # (D,) parameter vector
    t: np.ndarray  # (N,) observation times in years
    sigma: np.ndarray  # (N,) *raw* white-noise uncertainties (before EFAC/EQUAD)
    residuals: np.ndarray  # (N,) observed residuals
    freq_mhz: np.ndarray  # (N,) observing frequency MHz
    backend_id: np.ndarray  # (N,) integer backend label
    epoch_id: np.ndarray  # (N,) integer epoch label (for ECORR grouping)
    F: np.ndarray  # (N, 2*n_modes) Fourier design matrix
    tspan: float  # time span in years
    n_modes: int  # number of Fourier frequency modes
    F_dm: Optional[np.ndarray] = None  # (N, 2*n_modes) chromatic DM design matrix


def build_fourier_design_matrix(
    t: np.ndarray, tspan: float, n_modes: int
) -> np.ndarray:
    """Build the (N, 2*n_modes) Fourier design matrix.

    Columns are  [cos(2π f_1 t), sin(2π f_1 t), cos(2π f_2 t), ...].
    Frequencies  f_k = k / T  for k = 1 .. n_modes.
    """
    N = len(t)
    F = np.zeros((N, 2 * n_modes), dtype=np.float32)
    for k in range(1, n_modes + 1):
        fk = k / tspan  # cycles per year
        phase = 2.0 * np.pi * fk * t
        F[:, 2 * (k - 1)] = np.cos(phase)
        F[:, 2 * (k - 1) + 1] = np.sin(phase)
    return F


def build_dm_design_matrix(
    F: np.ndarray, freq_mhz: np.ndarray, freq_ref: float = DM_FREQ_REF
) -> np.ndarray:
    """Build the chromatic DM design matrix.

    F_dm[i, :] = F[i, :] * (freq_ref / freq_mhz[i])^2

    DM variations produce a timing delay ∝ 1/f_obs², so the Fourier design
    matrix is scaled by a frequency-dependent chromatic factor.
    """
    K_dm = (freq_ref / freq_mhz) ** 2  # (N,) chromatic weights
    return F * K_dm[:, None].astype(np.float32)


def build_ecorr_matrix(epoch_id: np.ndarray) -> np.ndarray:
    """Build the (N_toa, N_epoch) indicator matrix U for ECORR grouping.

    U[i, e] = 1  if TOA i belongs to epoch e, else 0.
    """
    n_toa = len(epoch_id)
    n_epoch = int(epoch_id.max()) + 1
    U = np.zeros((n_toa, n_epoch), dtype=np.float32)
    U[np.arange(n_toa), epoch_id] = 1.0
    return U


def power_law_spectrum(
    n_modes: int,
    tspan: float,
    log10_A: float,
    gamma: float,
    f_ref: float = 1.0,
) -> np.ndarray:
    """Per-mode Fourier-coefficient variance ρ_k for k = 1 .. n_modes.

    Uses the standard PTA power-law parameterisation (enterprise convention):

        ρ_k = (A² / 12π²) · SEC_PER_YR² · (f_k / f_ref)^(−γ) · Δf

    where f_k = k / T in yr⁻¹, Δf = 1 / T, and the SEC_PER_YR² factor
    converts from the (dimensionless A, yr⁻¹ frequencies) representation
    to variance in seconds².

    Returns shape (n_modes,) in units of s².
    """
    A = 10.0**log10_A
    delta_f = 1.0 / tspan
    fk = np.arange(1, n_modes + 1, dtype=np.float64) / tspan
    rho = (
        (A**2 * SEC_PER_YR**2 / (12.0 * np.pi**2)) * (fk / f_ref) ** (-gamma) * delta_f
    )
    return rho.astype(np.float32)


def simulate_pulsar(
    theta: np.ndarray,
    schedule: Schedule,
    n_modes: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> SimulatedPulsar:
    """Simulate residuals for one pulsar given theta and a schedule.

    Supports 2-D theta (red noise only) and 7-D theta (full noise model).
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_dim = len(theta)

    # --- parse parameters ---
    log10_A_red = float(theta[0])
    gamma_red = float(theta[1])

    if theta_dim >= 4:
        log10_A_dm = float(theta[2])
        gamma_dm = float(theta[3])
    else:
        log10_A_dm, gamma_dm = None, None

    if theta_dim >= 6:
        efac = float(theta[4])
        equad = 10.0 ** float(theta[5])
    else:
        efac = 1.0
        equad = 0.0

    if theta_dim >= 7:
        ecorr = 10.0 ** float(theta[6])
    else:
        ecorr = 0.0

    t = schedule.t.copy()
    sigma = schedule.sigma.copy()
    tspan = float(t[-1] - t[0])
    if tspan <= 0:
        tspan = 1.0  # safety
    N = len(t)

    # --- achromatic red noise ---
    F = build_fourier_design_matrix(t, tspan, n_modes)
    rho_red = power_law_spectrum(n_modes, tspan, log10_A_red, gamma_red)
    phi_red = np.repeat(rho_red, 2)  # (2*n_modes,)
    a_red = rng.normal(size=2 * n_modes).astype(np.float32) * np.sqrt(phi_red)
    red_noise = F @ a_red  # (N,)

    # --- DM noise (chromatic) ---
    F_dm = None
    dm_noise = np.zeros(N, dtype=np.float32)
    if log10_A_dm is not None:
        F_dm = build_dm_design_matrix(F, schedule.freq_mhz)
        rho_dm = power_law_spectrum(n_modes, tspan, log10_A_dm, gamma_dm)
        phi_dm = np.repeat(rho_dm, 2)
        a_dm = rng.normal(size=2 * n_modes).astype(np.float32) * np.sqrt(phi_dm)
        dm_noise = F_dm @ a_dm  # (N,)

    # --- white noise: EFAC * sigma ⊕ EQUAD ---
    sigma_eff = np.sqrt((efac * sigma) ** 2 + equad**2).astype(np.float32)
    white_noise = rng.normal(size=N).astype(np.float32) * sigma_eff

    # --- ECORR (epoch-correlated jitter) ---
    ecorr_noise = np.zeros(N, dtype=np.float32)
    if ecorr > 0 and hasattr(schedule, "epoch_id") and schedule.epoch_id is not None:
        unique_epochs = np.unique(schedule.epoch_id)
        for e in unique_epochs:
            mask_e = schedule.epoch_id == e
            j_e = rng.normal() * ecorr
            ecorr_noise[mask_e] = j_e

    residuals = red_noise + dm_noise + white_noise + ecorr_noise

    return SimulatedPulsar(
        theta=np.asarray(theta, dtype=np.float32),
        t=t,
        sigma=sigma,
        residuals=residuals,
        freq_mhz=schedule.freq_mhz.copy(),
        backend_id=schedule.backend_id.copy(),
        epoch_id=schedule.epoch_id.copy(),
        F=F,
        tspan=tspan,
        n_modes=n_modes,
        F_dm=F_dm,
    )
