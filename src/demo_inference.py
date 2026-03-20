"""Demo inference: load a checkpoint, simulate one example, compare posteriors.

Usage:
    python -m src.demo_inference --checkpoint outputs/smoke/transformer/best_model.pt
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import load_config, set_seed, get_device
from .priors import UniformPrior
from .schedules import generate_schedule
from .simulator import simulate_pulsar
from .models.tokenization import tokenize
from .exact_posterior import exact_posterior_grid
from .models.model_wrappers import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Demo: single-example posterior comparison")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--output", type=str, default="demo_inference.png")
    p.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda/mps)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_type = ckpt["model_type"]

    set_seed(args.seed)
    prior = UniformPrior(cfg["prior"])

    # Simulate one example
    rng = np.random.default_rng(args.seed)
    theta = prior.sample(1, rng=rng).squeeze(0).numpy()
    schedule = generate_schedule(rng)
    n_modes = cfg["data"].get("n_fourier_modes", 30)
    sim = simulate_pulsar(theta, schedule, n_modes=n_modes, rng=rng)

    print(f"True theta: log10_A={theta[0]:.3f}, gamma={theta[1]:.3f}")
    print(f"N_TOA: {len(sim.t)}, T_span: {sim.tspan:.2f} yr")

    # Exact posterior
    exact = exact_posterior_grid(
        sim.residuals, sim.sigma, sim.F, sim.tspan, sim.n_modes,
        cfg["prior"], n_grid=80, jitter=cfg["data"].get("jitter", 1e-6),
    )
    print(f"Exact MAP: log10_A={exact['map_theta'][0]:.3f}, gamma={exact['map_theta'][1]:.3f}")

    # Load model
    model = build_model(model_type, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Prepare batch
    tokens = tokenize(sim.t, sim.sigma, sim.residuals, sim.freq_mhz, sim.backend_id)
    feat_keys = ["t_norm", "dt_prev", "r_over_sig", "log_sigma", "r_raw", "freq_norm"]
    features = torch.stack([tokens[k] for k in feat_keys], dim=-1).unsqueeze(0).to(device)
    backend_id = tokens["backend_id"].unsqueeze(0).to(device)
    mask = torch.ones(1, len(sim.t), dtype=torch.bool, device=device)
    batch = {"theta": torch.from_numpy(sim.theta).unsqueeze(0).to(device),
             "features": features, "backend_id": backend_id, "mask": mask}

    # Sample learned posterior
    samples = model.sample_posterior(batch, n_samples=5000)  # (1, 5000, 2)
    samples_np = samples[0].cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Exact posterior
    AG, GG = np.meshgrid(exact["log10_A_grid"], exact["gamma_grid"], indexing="ij")
    axes[0].contourf(AG, GG, exact["posterior"], levels=30, cmap="Blues")
    axes[0].plot(theta[0], theta[1], "r*", ms=14)
    axes[0].set_title("Exact posterior")
    axes[0].set_xlabel("log10_A_red")
    axes[0].set_ylabel("gamma_red")

    # Learned posterior (histogram of samples)
    axes[1].hist2d(samples_np[:, 0], samples_np[:, 1], bins=40, cmap="Blues",
                   range=[[exact["log10_A_grid"][0], exact["log10_A_grid"][-1]],
                          [exact["gamma_grid"][0], exact["gamma_grid"][-1]]])
    axes[1].plot(theta[0], theta[1], "r*", ms=14)
    axes[1].set_title(f"Learned posterior ({model_type})")
    axes[1].set_xlabel("log10_A_red")
    axes[1].set_ylabel("gamma_red")

    # Residuals time series
    axes[2].errorbar(sim.t, sim.residuals, yerr=sim.sigma, fmt=".", ms=3, alpha=0.5)
    axes[2].set_xlabel("Time (yr)")
    axes[2].set_ylabel("Residual")
    axes[2].set_title(f"TOAs (N={len(sim.t)})")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Figure saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
