"""Generate v7a corner-plot examples showing the full 10-parameter monolithic posterior.

Parameters:  (log10_A_red, gamma_red, log10_A_dm, gamma_dm,
              EFAC_0, log10_EQUAD_0, log10_ECORR_0,
              EFAC_1, log10_EQUAD_1, log10_ECORR_1)

IS correction is applied:
  log w = log p(x|θ) + log π(θ) - log q(θ|x)

Usage:
    conda run -n pta-transformer-sbi python3 -m src.make_corner_examples_v7a
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, ensure_dir
from src.priors import UniformPrior
from src.simulator import simulate_pulsar
from src.schedules import generate_schedule
from src.models.model_wrappers import build_model
from src.models.tokenization import tokenize, FEAT_KEYS
from src.importance_sampling import (
    log_likelihood_batch,
    effective_sample_size,
    systematic_resample,
)

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT = "outputs/v7a/transformer/best_model.pt"
CONFIG = "configs/transformer_v7a.yaml"
OUT_DIR = "outputs/v7a/eval/corner_examples"
N_SAMPLES = 100_000  # flow proposal samples
N_RESAMPLE = 5_000   # IS resampled particles for display
N_EXAMPLES = 10
JITTER = 1e-20

PARAM_NAMES = [
    "log10_A_red",
    "gamma_red",
    "log10_A_dm",
    "gamma_dm",
    "EFAC_0",
    "log10_EQUAD_0",
    "log10_ECORR_0",
    "EFAC_1",
    "log10_EQUAD_1",
    "log10_ECORR_1",
]

PARAM_LABELS = [
    r"$\log_{10} A_\mathrm{red}$",
    r"$\gamma_\mathrm{red}$",
    r"$\log_{10} A_\mathrm{DM}$",
    r"$\gamma_\mathrm{DM}$",
    r"EFAC$_0$",
    r"$\log_{10}$ EQUAD$_0$",
    r"$\log_{10}$ ECORR$_0$",
    r"EFAC$_1$",
    r"$\log_{10}$ EQUAD$_1$",
    r"$\log_{10}$ ECORR$_1$",
]

# Fixed test cases: 10D theta vectors
# [log10_A_red, gamma_red, log10_A_dm, gamma_dm,
#  EFAC_0, log10_EQUAD_0, log10_ECORR_0,
#  EFAC_1, log10_EQUAD_1, log10_ECORR_1]
FIXED_CASES = [
    # Corners of the red-noise space (mid WN for both backends)
    ([-16.0, 2.0, -14.5, 3.5, 1.0, -6.5, -6.5, 1.0, -6.5, -6.5], "weak-red flat-spec"),
    ([-12.0, 2.0, -14.5, 3.5, 1.0, -6.5, -6.5, 1.0, -6.5, -6.5], "strong-red flat-spec"),
    ([-16.0, 5.5, -14.5, 3.5, 1.0, -6.5, -6.5, 1.0, -6.5, -6.5], "weak-red steep-spec"),
    ([-12.0, 5.5, -14.5, 3.5, 1.0, -6.5, -6.5, 1.0, -6.5, -6.5], "strong-red steep-spec"),
    # Strong DM
    ([-14.0, 3.5, -12.0, 4.0, 1.0, -6.5, -6.5, 1.0, -6.5, -6.5], "strong-DM"),
    # Asymmetric WN: backend 0 noisy, backend 1 quiet
    ([-14.0, 3.5, -14.5, 3.5, 1.8, -5.5, -6.0, 0.6, -7.5, -7.5], "asym-WN"),
    # Both backends noisy
    ([-14.0, 3.5, -14.5, 3.5, 1.8, -5.5, -5.5, 1.7, -5.5, -5.5], "high-WN"),
    # Mid-prior random examples
    None,
    None,
    None,
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_batch(sim, device="cpu"):
    tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
    features = torch.stack([tokens[k] for k in FEAT_KEYS], dim=-1).unsqueeze(0)
    L = len(sim.t)
    return {
        "features": features.to(device),
        "backend_id": tokens["backend_id"].unsqueeze(0).to(device),
        "mask": torch.ones(1, L, dtype=torch.bool, device=device),
        "tspan_yr": torch.tensor(
            [float(sim.t.max() - sim.t.min())], dtype=torch.float32, device=device
        ),
    }


@torch.no_grad()
def monolithic_samples_and_is(model, batch, sim, prior, n_samples, jitter, device):
    """Sample from monolithic 10D flow and compute IS weights.

    Returns
    -------
    flat_samples  (N, 10)  flow proposal samples
    is_samples    (M, 10)  IS-resampled particles
    ess           float
    ess_frac      float
    """
    samples = model.sample_posterior(batch, n_samples=n_samples)  # (1, N, 10)
    samples = samples[0]  # (N, 10)
    flat_np = samples.cpu().numpy().astype(np.float64)

    # log q(θ|x) via flow
    context = model._get_flow_context(batch)  # (1, ctx_dim)
    ctx_exp = context.expand(n_samples, -1)
    theta_norm = model._normalize_theta(samples)
    with torch.autocast(device.type, enabled=False):
        log_q_norm = model.flow.log_prob(theta_norm.float(), ctx_exp.float())
    log_q = (log_q_norm - model.theta_std.log().sum()).cpu().numpy().astype(np.float64)

    # log π(θ) — uniform prior
    log_pi = prior.log_prob(samples.cpu()).numpy().astype(np.float64)

    # log p(x|θ) — exact likelihood (handles per-backend WN via D=10)
    log_lik = log_likelihood_batch(flat_np, sim, jitter=jitter)

    # IS weights
    log_w = log_lik + log_pi - log_q
    ess = effective_sample_size(log_w)

    valid = np.isfinite(log_w)
    if valid.sum() > 0:
        lw = np.where(valid, log_w, -np.inf)
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
    else:
        w = np.ones(n_samples) / n_samples

    rng = np.random.default_rng(42)
    is_samples = systematic_resample(flat_np, w, N_RESAMPLE, rng)

    return flat_np, is_samples, ess, ess / n_samples


def make_corner(
    flat_np,
    is_samples,
    true_theta,
    label,
    example_idx,
    prior_bounds,
    out_path,
    ess=0.0,
    ess_frac=0.0,
):
    """Plot a 10×10 corner with flow (blue) and IS-corrected (red) samples."""
    fig = plt.figure(figsize=(20, 20))

    ranges = [tuple(prior_bounds[k]) for k in PARAM_NAMES]

    common_kw = dict(
        labels=PARAM_LABELS,
        range=ranges,
        show_titles=False,
        title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 8},
        tick_kwargs={"labelsize": 5},
        plot_contours=True,
        smooth=1.0,
        bins=35,
        no_fill_contours=False,
        fig=fig,
    )

    # Flow samples (blue)
    corner.corner(
        flat_np,
        color="royalblue",
        alpha=0.4,
        contourf_kwargs={"alpha": 0.25},
        contour_kwargs={"linewidths": 0.6},
        hist_kwargs={"alpha": 0.4},
        **common_kw,
    )

    # IS-corrected samples (red overlay)
    corner.corner(
        is_samples,
        color="crimson",
        alpha=0.7,
        contourf_kwargs={"alpha": 0.0},
        contour_kwargs={"linewidths": 1.0},
        hist_kwargs={"alpha": 0.0, "histtype": "step", "linewidth": 1.0},
        **common_kw,
    )

    # Truth
    corner.overplot_lines(fig, true_theta, color="seagreen", lw=1.0)
    corner.overplot_points(
        fig, true_theta[None], marker="*", color="seagreen", ms=5, zorder=5
    )

    # Title
    title = (
        f"Example {example_idx:02d}  |  {label}  |  "
        f"ESS={ess:.0f} ({ess_frac * 100:.1f}%)\n"
        f"A_red={true_theta[0]:.2f}, γ_red={true_theta[1]:.2f}, "
        f"A_dm={true_theta[2]:.2f}, γ_dm={true_theta[3]:.2f}\n"
        f"EFAC₀={true_theta[4]:.2f}, EQ₀={true_theta[5]:.2f}, "
        f"EC₀={true_theta[6]:.2f}  |  "
        f"EFAC₁={true_theta[7]:.2f}, EQ₁={true_theta[8]:.2f}, "
        f"EC₁={true_theta[9]:.2f}"
    )

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elems = [
        Patch(facecolor="royalblue", alpha=0.5, label=r"Flow $q(\theta|x)$"),
        Line2D([0], [0], color="crimson", lw=1.5, label="IS-corrected"),
        Line2D([0], [0], color="seagreen", lw=1.2, ls="-", label="Truth"),
    ]
    fig.legend(
        handles=legend_elems, loc="upper right", fontsize=9, bbox_to_anchor=(0.98, 0.98)
    )

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    cfg = load_config(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    prior_cfg = cfg["prior"]
    prior = UniformPrior(prior_cfg)

    # Load model
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_model(ckpt["model_type"], ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt['model_type']} from {CHECKPOINT}")

    out_dir = ensure_dir(OUT_DIR)
    data_cfg = cfg["data"]
    n_backends = data_cfg.get("n_backends_fixed", 2)

    for idx in range(N_EXAMPLES):
        case = FIXED_CASES[idx]
        rng = np.random.default_rng(1000 + idx)

        if case is None:
            theta = prior.sample(1, rng=rng).squeeze(0).numpy()
            label = "random"
        else:
            theta_list, label = case
            theta = np.array(theta_list, dtype=np.float32)

        # Generate schedule with 2 fixed backends and simulate
        sched = generate_schedule(
            rng,
            tspan_min_yr=data_cfg.get("tspan_min_yr", 5.0),
            tspan_max_yr=data_cfg.get("tspan_max_yr", 15.0),
            n_toa_min=data_cfg.get("n_toa_min", 80),
            n_toa_max=data_cfg.get("n_toa_max", 400),
            n_backends_fixed=n_backends,
        )
        sim = simulate_pulsar(
            theta, sched, n_modes=data_cfg.get("n_fourier_modes", 30), rng=rng
        )

        true_theta = sim.theta  # (10,)
        batch = make_batch(sim, device=str(device))

        print(
            f"\nExample {idx:02d} [{label}]  "
            f"N_TOA={len(sim.t)}, T_span={sim.tspan:.1f} yr"
        )
        print(f"  θ = {sim.theta.round(3)}")

        flat_np, is_np, ess, ess_frac = monolithic_samples_and_is(
            model, batch, sim, prior, N_SAMPLES, JITTER, device
        )

        print(f"  ESS = {ess:.0f} / {N_SAMPLES} ({ess_frac * 100:.1f}%)")

        out_path = os.path.join(out_dir, f"corner_example_{idx:02d}.png")
        make_corner(
            flat_np,
            is_np,
            true_theta,
            label,
            idx,
            prior_cfg,
            out_path,
            ess=ess,
            ess_frac=ess_frac,
        )

    print(f"\nDone. Corner plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
