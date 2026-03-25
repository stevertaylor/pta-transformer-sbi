"""Single-pulsar red-noise simulator using a Fourier design-matrix approach.

Generates residuals  r_i = red_noise_i + white_noise_i  given a parameter
vector  theta = (log10_A_red, gamma_red)  and an observing schedule.

Uses the standard PTA power-law parameterisation (enterprise convention).
Times are in years; residuals, uncertainties, and Fourier-coefficient
variances are in seconds / seconds².
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .schedules import Schedule

SEC_PER_YR = 365.25 * 86400  # seconds per Julian year


@dataclass
class SimulatedPulsar:
    """Everything needed for training and exact-likelihood evaluation."""

    theta: np.ndarray  # (2,) – [log10_A_red, gamma_red]
    t: np.ndarray  # (N,) observation times in years
    sigma: np.ndarray  # (N,) white-noise uncertainties
    residuals: np.ndarray  # (N,) observed residuals
    freq_mhz: np.ndarray  # (N,) observing frequency MHz
    backend_id: np.ndarray  # (N,) integer backend label
    F: np.ndarray  # (N, 2*n_modes) Fourier design matrix
    tspan: float  # time span in years
    n_modes: int  # number of Fourier frequency modes


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
    """Simulate residuals for one pulsar given theta and a schedule."""
    if rng is None:
        rng = np.random.default_rng()

    log10_A, gamma = float(theta[0]), float(theta[1])
    t = schedule.t.copy()
    sigma = schedule.sigma.copy()
    tspan = float(t[-1] - t[0])
    if tspan <= 0:
        tspan = 1.0  # safety

    F = build_fourier_design_matrix(t, tspan, n_modes)
    rho = power_law_spectrum(n_modes, tspan, log10_A, gamma)

    # Phi = diag(rho_1, rho_1, rho_2, rho_2, ...)
    phi_diag = np.repeat(rho, 2)  # (2*n_modes,)

    # Draw Fourier coefficients
    a = rng.normal(size=2 * n_modes).astype(np.float32) * np.sqrt(phi_diag)

    red_noise = F @ a  # (N,)
    white_noise = rng.normal(size=len(t)).astype(np.float32) * sigma
    residuals = red_noise + white_noise

    return SimulatedPulsar(
        theta=np.array([log10_A, gamma], dtype=np.float32),
        t=t,
        sigma=sigma,
        residuals=residuals,
        freq_mhz=schedule.freq_mhz,
        backend_id=schedule.backend_id,
        F=F,
        tspan=tspan,
        n_modes=n_modes,
    )
