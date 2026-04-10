"""Generate v5 corner-plot examples showing the full 7-parameter factorized structure.

Global flow:  (log10_A_red, gamma_red, log10_A_dm, gamma_dm)
WN flow:      (EFAC, log10_EQUAD, log10_ECORR)  — per backend, shared network

Joint IS correction is applied:
  log w = log p(x|θ_global, θ_wn) + log π(θ_global) + log π(θ_wn)
          - log q_global(θ_global|x) - log q_wn(θ_wn|c_wn)

Usage:
    conda run -n pta-transformer-sbi python3 -m src.make_corner_examples_v5
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
from src.priors import FactorizedPrior
from src.simulator import simulate_pulsar_factorized
from src.schedules import generate_schedule
from src.models.model_wrappers import build_model
from src.models.tokenization import tokenize, FEAT_KEYS
from src.importance_sampling import (
    log_likelihood_batch,
    effective_sample_size,
    systematic_resample,
)

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT = "outputs/v5/transformer/best_model.pt"
CONFIG     = "configs/transformer_v5.yaml"
OUT_DIR    = "outputs/v5/eval/corner_examples"
N_SAMPLES  = 20_000   # flow proposal samples
N_RESAMPLE = 2_000    # IS resampled particles for display
N_EXAMPLES = 10
JITTER     = 1e-20

PARAM_LABELS = [
    r"$\log_{10} A_\mathrm{red}$",
    r"$\gamma_\mathrm{red}$",
    r"$\log_{10} A_\mathrm{DM}$",
    r"$\gamma_\mathrm{DM}$",
    "EFAC",
    r"$\log_{10}$ EQUAD",
    r"$\log_{10}$ ECORR",
]

# Fixed test cases: (theta_global, theta_wn_b0, label)
# theta_global = [log10_A_red, gamma_red, log10_A_dm, gamma_dm]
# theta_wn     = [EFAC, log10_EQUAD, log10_ECORR]
FIXED_CASES = [
    # Corners of the red-noise space
    ([-16.0, 2.0, -14.5, 3.5], [1.0, -6.5, -6.5], "weak-red flat-spec"),
    ([-12.0, 2.0, -14.5, 3.5], [1.0, -6.5, -6.5], "strong-red flat-spec"),
    ([-16.0, 5.5, -14.5, 3.5], [1.0, -6.5, -6.5], "weak-red steep-spec"),
    ([-12.0, 5.5, -14.5, 3.5], [1.0, -6.5, -6.5], "strong-red steep-spec"),
    # Strong DM
    ([-14.0, 3.5, -12.0, 4.0], [1.0, -6.5, -6.5], "strong-DM"),
    # Noisy WN (high EFAC + high EQUAD)
    ([-14.0, 3.5, -14.5, 3.5], [3.0, -5.5, -6.0], "high-WN"),
    # Quiet WN (low EFAC + low EQUAD/ECORR)
    ([-14.0, 3.5, -14.5, 3.5], [0.5, -7.5, -7.5], "low-WN"),
    # Mid-prior random examples
    None, None, None,
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_batch(sim, device="cpu"):
    tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
    features = torch.stack([tokens[k] for k in FEAT_KEYS], dim=-1).unsqueeze(0)
    L = len(sim.t)
    return {
        "features":   features.to(device),
        "backend_id": tokens["backend_id"].unsqueeze(0).to(device),
        "mask":       torch.ones(1, L, dtype=torch.bool, device=device),
        "tspan_yr":   torch.tensor([float(sim.t.max() - sim.t.min())],
                                   dtype=torch.float32, device=device),
    }


@torch.no_grad()
def joint_samples_and_is(model, batch, sim, prior, n_samples, jitter, device):
    """Sample from both flows and compute joint IS weights.

    Returns
    -------
    flat_samples  (N, 7)   flow proposal samples
    flat_is       (M, 7)   IS-resampled particles (M = N_RESAMPLE)
    ess           float
    ess_frac      float
    """
    global_ctx, wn_ctx = model._get_contexts(batch)
    B = global_ctx.shape[0]

    # -- Global samples (N, 4) --
    gs_norm = model.global_flow.sample(global_ctx, n_samples)   # (1, N, 4)
    gs = model._denormalize_global(gs_norm)[0]                  # (N, 4)

    # -- WN samples for backend 0 (N, 3) --
    ctx_b0 = wn_ctx[:, 0, :]                                    # (1, wn_ctx_dim)
    ws_norm = model.wn_flow.sample(ctx_b0, n_samples)           # (1, N, 3)
    ws = model._denormalize_wn(ws_norm)[0]                      # (N, 3)

    # -- log q_global --
    ctx_g_exp = global_ctx.expand(n_samples, -1)
    tg_norm = model._normalize_global(gs)
    lq_g = model.global_flow.log_prob(tg_norm.float(), ctx_g_exp.float())
    lq_g -= model.global_theta_std.log().sum()                  # Jacobian correction

    # -- log q_wn --
    ctx_w_exp = ctx_b0.expand(n_samples, -1)
    tw_norm = model._normalize_wn(ws)
    lq_w = model.wn_flow.log_prob(tw_norm.float(), ctx_w_exp.float())
    lq_w -= model.wn_theta_std.log().sum()

    log_q = (lq_g + lq_w).cpu().numpy().astype(np.float64)

    # -- flat 7D samples --
    flat_np = torch.cat([gs, ws], dim=-1).cpu().numpy().astype(np.float64)

    # -- log prior --
    log_pi_g = prior.global_prior.log_prob(gs.cpu()).numpy().astype(np.float64)
    log_pi_w = prior.wn_prior.log_prob(ws.cpu()).numpy().astype(np.float64)
    log_pi = log_pi_g + log_pi_w

    # -- exact likelihood (7D: global + first-backend WN) --
    log_lik = log_likelihood_batch(flat_np, sim, jitter=jitter)

    # -- IS weights --
    log_w = log_lik + log_pi - log_q
    ess = effective_sample_size(log_w)

    valid = np.isfinite(log_w)
    if valid.sum() > 0:
        lw = np.where(valid, log_w, -np.inf)
        lw -= lw.max()
        w = np.exp(lw);  w /= w.sum()
    else:
        w = np.ones(n_samples) / n_samples

    rng = np.random.default_rng(42)
    is_samples = systematic_resample(flat_np, w, N_RESAMPLE, rng)

    return flat_np, is_samples, ess, ess / n_samples


def make_corner(flat_np, is_samples, true_theta, label, example_idx,
                prior_bounds_flat, out_path, ess=0.0, ess_frac=0.0):
    """Plot a 7×7 corner with flow (blue) and IS-corrected (red) samples."""
    fig = plt.figure(figsize=(14, 14))

    # Build ranges from prior bounds
    ranges = [tuple(prior_bounds_flat[k]) for k in [
        "log10_A_red", "gamma_red", "log10_A_dm", "gamma_dm",
        "EFAC", "log10_EQUAD", "log10_ECORR",
    ]]

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
    corner.corner(flat_np, color="royalblue", alpha=0.4,
                  contourf_kwargs={"alpha": 0.25},
                  contour_kwargs={"linewidths": 0.8},
                  hist_kwargs={"alpha": 0.4},
                  **common_kw)

    # IS-corrected samples (red overlay)
    corner.corner(is_samples, color="crimson", alpha=0.7,
                  contourf_kwargs={"alpha": 0.0},
                  contour_kwargs={"linewidths": 1.2},
                  hist_kwargs={"alpha": 0.0, "histtype": "step", "linewidth": 1.2},
                  **common_kw)

    # Truth lines
    corner.overplot_lines(fig, true_theta, color="seagreen", lw=1.2)
    corner.overplot_points(fig, true_theta[None], marker="*", color="seagreen",
                           ms=6, zorder=5)

    # Title
    a_red  = true_theta[0]; g_red = true_theta[1]
    a_dm   = true_theta[2]; g_dm  = true_theta[3]
    efac   = true_theta[4]; equad = true_theta[5]; ecorr = true_theta[6]
    title  = (f"Example {example_idx:02d}  |  {label}  |  ESS={ess:.0f} ({ess_frac*100:.1f}%)\n"
              f"A_red={a_red:.2f}, γ_red={g_red:.2f},  "
              f"A_dm={a_dm:.2f}, γ_dm={g_dm:.2f},  "
              f"EFAC={efac:.2f}, EQUAD={equad:.2f}, ECORR={ecorr:.2f}")

    # Add legend manually
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elems = [
        Patch(facecolor="royalblue", alpha=0.5, label="Flow q(θ|x)"),
        Line2D([0], [0], color="crimson", lw=1.5, label="IS-corrected"),
        Line2D([0], [0], color="seagreen", lw=1.2, ls="-", label="Truth"),
    ]
    fig.legend(handles=legend_elems, loc="upper right", fontsize=9,
               bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg    = load_config(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build prior
    prior = FactorizedPrior(
        cfg["prior"]["global"], cfg["prior"]["white_noise"],
        n_backends_max=cfg["model"].get("n_backends_max", 4),
    )

    # Flat prior bounds for corner ranges
    prior_bounds_flat = {**cfg["prior"]["global"], **cfg["prior"]["white_noise"]}

    # Load model
    ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_model(ckpt["model_type"], ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt['model_type']} from {CHECKPOINT}")

    out_dir = ensure_dir(OUT_DIR)
    data_cfg = cfg["data"]

    for idx in range(N_EXAMPLES):
        case = FIXED_CASES[idx]
        rng  = np.random.default_rng(1000 + idx)

        if case is None:
            # Random draw
            theta_g  = prior.sample_global(1, rng=rng).squeeze(0).numpy()
            theta_wn = prior.sample_wn(1, rng=rng).numpy()          # (1, 3)
            label    = "random"
        else:
            theta_g_list, theta_wn_list, label = case
            theta_g  = np.array(theta_g_list, dtype=np.float32)
            theta_wn = np.array([theta_wn_list], dtype=np.float32)  # (1, 3)

        # Generate pulsar, then collapse to single backend so IS is exact
        # (log_likelihood_batch uses scalar WN; multi-backend requires per-backend extension)
        sched = generate_schedule(
            rng,
            tspan_min_yr=data_cfg.get("tspan_min_yr", 5.0),
            tspan_max_yr=data_cfg.get("tspan_max_yr", 15.0),
            n_toa_min=data_cfg.get("n_toa_min", 80),
            n_toa_max=data_cfg.get("n_toa_max", 400),
        )
        # Force single backend so IS likelihood is correct
        sched.backend_id[:] = 0
        sim = simulate_pulsar_factorized(theta_g, theta_wn, sched,
                                         n_modes=30, rng=rng)

        true_theta = np.concatenate([sim.theta_global, sim.theta_wn[0]])  # (7,)
        batch      = make_batch(sim, device=str(device))

        print(f"\nExample {idx:02d} [{label}]  "
              f"N_TOA={len(sim.t)}, T_span={sim.tspan:.1f} yr")
        print(f"  θ_global = {sim.theta_global.round(2)}")
        print(f"  θ_wn     = {sim.theta_wn[0].round(3)}")

        with torch.no_grad():
            flat_np, is_np, ess, ess_frac = joint_samples_and_is(
                model, batch, sim, prior, N_SAMPLES, JITTER, device
            )

        print(f"  ESS = {ess:.0f} / {N_SAMPLES} ({ess_frac*100:.1f}%)")

        out_path = os.path.join(out_dir, f"corner_example_{idx:02d}.png")
        make_corner(flat_np, is_np, true_theta, label, idx,
                    prior_bounds_flat, out_path, ess=ess, ess_frac=ess_frac)

    print(f"\nDone. Corner plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
