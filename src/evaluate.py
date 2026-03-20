"""Evaluation script: exact-posterior comparison, calibration, robustness.

Usage:
    python -m src.evaluate --config configs/smoke.yaml \
        --checkpoint outputs/smoke/transformer/best_model.pt \
        --baseline-checkpoint outputs/smoke/lstm/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .utils import load_config, set_seed, get_device, ensure_dir
from .priors import UniformPrior
from .dataset import FixedPulsarDataset
from .collate import collate_fn
from .models.model_wrappers import build_model
from .exact_posterior import exact_posterior_grid
from .masking import apply_random_masking
from .models.tokenization import tokenize
from .metrics import hellinger_distance_grid, calibration_percentiles, ks_statistic, point_estimate_error
from .plots import plot_posterior_comparison, plot_pp, plot_robustness


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NPE models")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Transformer checkpoint")
    p.add_argument("--baseline-checkpoint", type=str, default=None, help="LSTM checkpoint")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda/mps)")
    return p.parse_args()


def load_model(ckpt_path: str, cfg: dict, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt["model_type"]
    model = build_model(model_type, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_type


def _make_batch_single(sim_item, mask_keep=None, device="cpu"):
    """Create a single-example batch dict from a FixedPulsarDataset item."""
    sim = sim_item["sim"]
    if mask_keep is not None:
        t = sim.t[mask_keep]
        sigma = sim.sigma[mask_keep]
        r = sim.residuals[mask_keep]
        freq = sim.freq_mhz[mask_keep]
        be = sim.backend_id[mask_keep]
    else:
        t, sigma, r = sim.t, sim.sigma, sim.residuals
        freq, be = sim.freq_mhz, sim.backend_id

    tokens = tokenize(t, sigma, r, freq, be)
    L = len(t)

    feat_keys = ["t_norm", "dt_prev", "r_over_sig", "log_sigma", "r_raw", "freq_norm"]
    features = torch.stack([tokens[k] for k in feat_keys], dim=-1).unsqueeze(0)
    backend_id = tokens["backend_id"].unsqueeze(0)
    mask = torch.ones(1, L, dtype=torch.bool)

    return {
        "theta": sim_item["theta"].unsqueeze(0).to(device),
        "features": features.to(device),
        "backend_id": backend_id.to(device),
        "mask": mask.to(device),
    }


def evaluate_model(
    model, dataset, cfg, device, n_exact, n_grid, n_posterior_samples,
    masking_severity=0.0, seed=999,
):
    """Run full evaluation for one model at one masking level."""
    prior_bounds = cfg["prior"]
    jitter = cfg["data"].get("jitter", 1e-6)

    hellinger_dists = []
    all_true = []
    all_samples = []
    all_means = []
    exact_posts_out = []
    learned_posts_out = []

    rng = np.random.default_rng(seed)

    n_eval = min(n_exact, len(dataset))

    for i in tqdm(range(n_eval), desc="Evaluating", leave=False):
        item = dataset[i]
        sim = item["sim"]

        # Apply masking
        if masking_severity > 0:
            keep = apply_random_masking(sim.t, rng, severity=masking_severity)
        else:
            keep = None

        batch = _make_batch_single(item, mask_keep=keep, device=device)

        # Exact posterior
        exact = exact_posterior_grid(
            sim.residuals, sim.sigma, sim.F, sim.tspan, sim.n_modes,
            prior_bounds, n_grid=n_grid, jitter=jitter,
        )

        # Learned posterior on grid
        A_grid = torch.from_numpy(exact["log10_A_grid"].astype(np.float32))
        G_grid = torch.from_numpy(exact["gamma_grid"].astype(np.float32))
        AA, GG = torch.meshgrid(A_grid, G_grid, indexing="ij")
        grid_pts = torch.stack([AA.reshape(-1), GG.reshape(-1)], dim=-1).to(device)

        log_probs = model.log_prob_on_grid(batch, grid_pts)
        log_probs_np = log_probs.cpu().numpy().reshape(n_grid, n_grid).astype(np.float64)

        # Normalise learned posterior on grid
        dA = float(A_grid[1] - A_grid[0])
        dG = float(G_grid[1] - G_grid[0])
        log_max = np.max(log_probs_np)
        log_norm = log_max + np.log(np.sum(np.exp(log_probs_np - log_max)) * dA * dG)
        learned_post = np.exp(log_probs_np - log_norm)

        h = hellinger_distance_grid(exact["posterior"], learned_post)
        hellinger_dists.append(h)

        # Samples for calibration
        samples = model.sample_posterior(batch, n_posterior_samples)  # (1, S, 2)
        all_true.append(sim.theta)
        all_samples.append(samples[0].cpu().numpy())

        # Posterior mean from samples
        all_means.append(samples[0].cpu().numpy().mean(axis=0))

        if i < 4:
            exact_posts_out.append(exact)
            learned_posts_out.append(learned_post)

    true_arr = np.array(all_true)
    samples_arr = np.array(all_samples)
    means_arr = np.array(all_means)

    percentiles = calibration_percentiles(true_arr, samples_arr)
    ks0 = ks_statistic(percentiles[:, 0])
    ks1 = ks_statistic(percentiles[:, 1])

    pe = point_estimate_error(true_arr, means_arr)

    return {
        "hellinger": float(np.mean(hellinger_dists)),
        "hellinger_std": float(np.std(hellinger_dists)),
        "ks_log10A": ks0,
        "ks_gamma": ks1,
        "ks_mean": float(np.mean([ks0, ks1])),
        "point_error": float(np.mean(pe)),
        "point_error_A": float(np.mean(pe[:, 0])),
        "point_error_gamma": float(np.mean(pe[:, 1])),
        "percentiles": percentiles,
        "exact_posts": exact_posts_out,
        "learned_posts": learned_posts_out,
        "true_thetas": true_arr[:4] if len(true_arr) >= 4 else true_arr,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    out_dir = args.output_dir or cfg["output_dir"]
    eval_dir = ensure_dir(os.path.join(out_dir, "eval"))

    ecfg = cfg["eval"]
    prior = UniformPrior(cfg["prior"])

    # Test dataset
    test_ds = FixedPulsarDataset(
        n_samples=cfg["data"]["test_samples"],
        prior=prior,
        data_cfg=cfg["data"],
        seed=cfg["training"]["seed"] + 2_000_000,
    )

    # Load models
    model_t, type_t = load_model(args.checkpoint, cfg, device)
    print(f"Loaded {type_t} from {args.checkpoint}")

    models = {type_t: model_t}
    if args.baseline_checkpoint:
        model_b, type_b = load_model(args.baseline_checkpoint, cfg, device)
        print(f"Loaded {type_b} from {args.baseline_checkpoint}")
        models[type_b] = model_b

    masking_levels = ecfg.get("masking_levels", [0.0, 0.3, 0.6])
    all_results = {}

    for name, model in models.items():
        print(f"\n=== Evaluating {name} ===")
        model_results = {}
        for sev in masking_levels:
            print(f"  Masking severity: {sev}")
            res = evaluate_model(
                model, test_ds, cfg, device,
                n_exact=ecfg["n_test_exact"],
                n_grid=ecfg["n_grid"],
                n_posterior_samples=ecfg["n_posterior_samples"],
                masking_severity=sev,
            )
            model_results[sev] = res

            # Posterior comparison plots (no-mask only)
            if sev == 0.0:
                for k in range(min(4, len(res["exact_posts"]))):
                    ep = res["exact_posts"][k]
                    plot_posterior_comparison(
                        ep["log10_A_grid"], ep["gamma_grid"],
                        ep["posterior"], res["learned_posts"][k],
                        res["true_thetas"][k],
                        os.path.join(eval_dir, f"posterior_{name}_{k}.png"),
                        title=f"{name} – test example {k}",
                    )

                # P-P plot
                plot_pp(
                    res["percentiles"],
                    ["log10_A_red", "gamma_red"],
                    os.path.join(eval_dir, f"pp_{name}.png"),
                    ks_stats=[res["ks_log10A"], res["ks_gamma"]],
                    title=f"P-P plot – {name}",
                )

        all_results[name] = model_results

    # Robustness plot
    if len(models) == 2:
        names = list(models.keys())
        plot_robustness(
            masking_levels,
            all_results[names[0]],
            all_results[names[1]],
            os.path.join(eval_dir, "robustness.png"),
        )

    # Save summary metrics
    summary = {}
    for name in all_results:
        summary[name] = {}
        for sev in masking_levels:
            r = all_results[name][sev]
            summary[name][str(sev)] = {
                "hellinger": r["hellinger"],
                "hellinger_std": r["hellinger_std"],
                "ks_log10A": r["ks_log10A"],
                "ks_gamma": r["ks_gamma"],
                "ks_mean": r["ks_mean"],
                "point_error": r["point_error"],
                "point_error_A": r["point_error_A"],
                "point_error_gamma": r["point_error_gamma"],
            }

    with open(os.path.join(eval_dir, "eval_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {eval_dir}/")
    print("\nSummary:")
    for name in summary:
        print(f"\n  {name}:")
        for sev in summary[name]:
            s = summary[name][sev]
            print(f"    mask={sev}: H={s['hellinger']:.4f} KS={s['ks_mean']:.4f} PE={s['point_error']:.4f}")


if __name__ == "__main__":
    main()
