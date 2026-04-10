"""Generate v4e corner-plot examples showing the full 7-parameter monolithic posterior.

Parameters:  (log10_A_red, gamma_red, log10_A_dm, gamma_dm, EFAC, log10_EQUAD, log10_ECORR)

IS correction is applied:
  log w = log p(x|θ) + log π(θ) - log q(θ|x)

Usage:
    conda run -n pta-transformer-sbi python3 -m src.make_corner_examples_v4e
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

CHECKPOINT = "outputs/v4e/transformer/best_model.pt"
CONFIG = "configs/transformer_v4e.yaml"
OUT_DIR = "outputs/v4e/eval/corner_examples"
N_SAMPLES = 20_000  # flow proposal samples
N_RESAMPLE = 2_000  # IS resampled particles for display
N_EXAMPLES = 10
JITTER = 1e-20

PARAM_LABELS = [
    r"$\log_{10} A_\mathrm{red}$",
    r"$\gamma_\mathrm{red}$",
    r"$\log_{10} A_\mathrm{DM}$",
    r"$\gamma_\mathrm{DM}$",
    "EFAC",
    r"$\log_{10}$ EQUAD",
    r"$\log_{10}$ ECORR",
]

PARAM_NAMES = [
    "log10_A_red",
    "gamma_red",
    "log10_A_dm",
    "gamma_dm",
    "EFAC",
    "log10_EQUAD",
    "log10_ECORR",
]

# Fixed test cases: 7D theta vectors
# [log10_A_red, gamma_red, log10_A_dm, gamma_dm, EFAC, log10_EQUAD, log10_ECORR]
FIXED_CASES = [
    # Corners of the red-noise space
    ([-16.0, 2.0, -14.5, 3.5, 1.0, -6.5, -6.5], "weak-red flat-spec"),
    ([-12.0, 2.0, -14.5, 3.5, 1.0, -6.5, -6.5], "strong-red flat-spec"),
    ([-16.0, 5.5, -14.5, 3.5, 1.0, -6.5, -6.5], "weak-red steep-spec"),
    ([-12.0, 5.5, -14.5, 3.5, 1.0, -6.5, -6.5], "strong-red steep-spec"),
    # Strong DM
    ([-14.0, 3.5, -12.0, 4.0, 1.0, -6.5, -6.5], "strong-DM"),
    # High WN (EFAC near upper prior bound)
    ([-14.0, 3.5, -14.5, 3.5, 1.8, -5.5, -6.0], "high-WN"),
    # Low WN (EFAC near lower prior bound)
    ([-14.0, 3.5, -14.5, 3.5, 0.6, -7.5, -7.5], "low-WN"),
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
    """Sample from monolithic 7D flow and compute IS weights.

    Returns
    -------
    flat_samples  (N, 7)   flow proposal samples
    is_samples    (M, 7)   IS-resampled particles
    ess           float
    ess_frac      float
    """
    # Sample from 7D flow
    samples = model.sample_posterior(batch, n_samples=n_samples)  # (1, N, 7)
    samples = samples[0]  # (N, 7)
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

    # log p(x|θ) — exact likelihood
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
    """Plot a 7×7 corner with flow (blue) and IS-corrected (red) samples."""
    fig = plt.figure(figsize=(14, 14))

    ranges = [tuple(prior_bounds[k]) for k in PARAM_NAMES]

    common_kw = dict(
        labels=PARAM_LABELS,
        range=ranges,
        show_titles=False,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
        tick_kwargs={"labelsize": 7},
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
        contour_kwargs={"linewidths": 0.8},
        hist_kwargs={"alpha": 0.4},
        **common_kw,
    )

    # IS-corrected samples (red overlay)
    corner.corner(
        is_samples,
        color="crimson",
        alpha=0.7,
        contourf_kwargs={"alpha": 0.0},
        contour_kwargs={"linewidths": 1.2},
        hist_kwargs={"alpha": 0.0, "histtype": "step", "linewidth": 1.2},
        **common_kw,
    )

    # Truth
    corner.overplot_lines(fig, true_theta, color="seagreen", lw=1.2)
    corner.overplot_points(
        fig, true_theta[None], marker="*", color="seagreen", ms=6, zorder=5
    )

    # Title
    a_red = true_theta[0]
    g_red = true_theta[1]
    a_dm = true_theta[2]
    g_dm = true_theta[3]
    efac = true_theta[4]
    equad = true_theta[5]
    ecorr = true_theta[6]
    title = (
        f"Example {example_idx:02d}  |  {label}  |  ESS={ess:.0f} ({ess_frac*100:.1f}%)\n"
        f"A_red={a_red:.2f}, γ_red={g_red:.2f},  "
        f"A_dm={a_dm:.2f}, γ_dm={g_dm:.2f},  "
        f"EFAC={efac:.2f}, EQUAD={equad:.2f}, ECORR={ecorr:.2f}"
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

    # Build prior (monolithic UniformPrior)
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

    for idx in range(N_EXAMPLES):
        case = FIXED_CASES[idx]
        rng = np.random.default_rng(1000 + idx)

        if case is None:
            # Random draw from prior
            theta = prior.sample(1, rng=rng).squeeze(0).numpy()
            label = "random"
        else:
            theta_list, label = case
            theta = np.array(theta_list, dtype=np.float32)

        # Generate schedule and simulate
        sched = generate_schedule(
            rng,
            tspan_min_yr=data_cfg.get("tspan_min_yr", 5.0),
            tspan_max_yr=data_cfg.get("tspan_max_yr", 15.0),
            n_toa_min=data_cfg.get("n_toa_min", 80),
            n_toa_max=data_cfg.get("n_toa_max", 400),
        )
        sim = simulate_pulsar(
            theta, sched, n_modes=data_cfg.get("n_fourier_modes", 30), rng=rng
        )

        true_theta = sim.theta  # (7,)
        batch = make_batch(sim, device=str(device))

        print(
            f"\nExample {idx:02d} [{label}]  "
            f"N_TOA={len(sim.t)}, T_span={sim.tspan:.1f} yr"
        )
        print(f"  θ = {sim.theta.round(3)}")

        flat_np, is_np, ess, ess_frac = monolithic_samples_and_is(
            model, batch, sim, prior, N_SAMPLES, JITTER, device
        )

        print(f"  ESS = {ess:.0f} / {N_SAMPLES} ({ess_frac*100:.1f}%)")

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
