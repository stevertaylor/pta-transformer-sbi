"""Evaluation script: calibration, exact-posterior comparison, robustness.

For 2-param models: exact 2-D posterior comparison (Hellinger) + P-P calibration.
For 7-param models: P-P calibration is the primary metric.  Optional 2-D
conditional posterior comparison over (A_red, gamma_red) with nuisance params
fixed at truth.

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
from .metrics import (
    hellinger_distance_grid,
    calibration_percentiles,
    ks_statistic,
    point_estimate_error,
)
from .plots import plot_posterior_comparison, plot_pp, plot_robustness
from .importance_sampling import importance_sample, weighted_percentile_rank


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NPE models")
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--checkpoint", type=str, required=True, help="Transformer checkpoint"
    )
    p.add_argument(
        "--baseline-checkpoint", type=str, default=None, help="LSTM checkpoint"
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--device", type=str, default=None, help="Force device (cpu/cuda/mps)"
    )
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
    tspan_yr = float(t.max() - t.min()) if L > 1 else 0.0

    feat_keys = ["t_norm", "dt_prev", "r_over_sig", "log_sigma", "r_raw", "freq_norm"]
    features = torch.stack([tokens[k] for k in feat_keys], dim=-1).unsqueeze(0)
    backend_id = tokens["backend_id"].unsqueeze(0)
    mask = torch.ones(1, L, dtype=torch.bool)

    return {
        "theta": sim_item["theta"].unsqueeze(0).to(device),
        "features": features.to(device),
        "backend_id": backend_id.to(device),
        "mask": mask.to(device),
        "tspan_yr": torch.tensor([tspan_yr], dtype=torch.float32).to(device),
    }


def _build_theta_fixed(sim, prior_param_names):
    """Build a theta_fixed dict from a SimulatedPulsar for conditional evaluation.

    Fixes all nuisance parameters (everything except log10_A_red and gamma_red)
    at their true values.
    """
    tf = {}
    for i, name in enumerate(prior_param_names):
        if name in ("log10_A_red", "gamma_red"):
            continue
        tf[name] = float(sim.theta[i])
    if sim.epoch_id is not None:
        tf["epoch_id"] = sim.epoch_id
    return tf


def evaluate_model(
    model,
    dataset,
    cfg,
    device,
    n_exact,
    n_grid,
    n_posterior_samples,
    masking_severity=0.0,
    seed=999,
):
    """Run full evaluation for one model at one masking level."""
    prior_bounds = cfg["prior"]
    param_names = list(prior_bounds.keys())
    theta_dim = len(param_names)
    jitter = cfg["data"].get("jitter", 1e-20)

    # Decide whether to compute exact 2-D grid posteriors
    do_exact_grid = True  # always possible: 2D grid over (A_red, gamma_red)

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

        # --- Exact posterior (2-D grid over A_red, gamma_red) ---
        exact = None
        if do_exact_grid:
            theta_fixed = None
            if theta_dim > 2:
                theta_fixed = _build_theta_fixed(sim, param_names)
            exact = exact_posterior_grid(
                sim.residuals,
                sim.sigma,
                sim.F,
                sim.tspan,
                sim.n_modes,
                prior_bounds,
                n_grid=n_grid,
                jitter=jitter,
                F_dm=sim.F_dm,
                theta_fixed=theta_fixed,
            )

        # --- Hellinger on 2-D grid ---
        if exact is not None:
            A_grid = torch.from_numpy(exact["log10_A_grid"].astype(np.float32))
            G_grid = torch.from_numpy(exact["gamma_grid"].astype(np.float32))
            AA, GG = torch.meshgrid(A_grid, G_grid, indexing="ij")

            if theta_dim == 2:
                grid_pts = torch.stack([AA.reshape(-1), GG.reshape(-1)], dim=-1).to(
                    device
                )
            else:
                # For 7-D model: build full grid by tiling true nuisance values
                n_pts = AA.numel()
                grid_full = torch.zeros(n_pts, theta_dim)
                grid_full[:, 0] = AA.reshape(-1)
                grid_full[:, 1] = GG.reshape(-1)
                for k, name in enumerate(param_names):
                    if name in ("log10_A_red", "gamma_red"):
                        continue
                    grid_full[:, k] = float(sim.theta[k])
                grid_pts = grid_full.to(device)

            log_probs = model.log_prob_on_grid(batch, grid_pts)
            log_probs_np = (
                log_probs.cpu().numpy().reshape(n_grid, n_grid).astype(np.float64)
            )
            dA = float(A_grid[1] - A_grid[0])
            dG = float(G_grid[1] - G_grid[0])
            log_max = np.max(log_probs_np)
            log_norm = log_max + np.log(
                np.sum(np.exp(log_probs_np - log_max)) * dA * dG
            )
            learned_post = np.exp(log_probs_np - log_norm)

            h = hellinger_distance_grid(exact["posterior"], learned_post)
            hellinger_dists.append(h)

            if i < 4:
                exact_posts_out.append(exact)
                learned_posts_out.append(learned_post)

        # --- Samples for calibration (all D dimensions) ---
        samples = model.sample_posterior(batch, n_posterior_samples)  # (1, S, D)
        all_true.append(sim.theta)
        all_samples.append(samples[0].cpu().numpy())
        all_means.append(samples[0].cpu().numpy().mean(axis=0))

    true_arr = np.array(all_true)  # (n_eval, D)
    samples_arr = np.array(all_samples)  # (n_eval, S, D)
    means_arr = np.array(all_means)  # (n_eval, D)

    percentiles = calibration_percentiles(true_arr, samples_arr)
    ks_per_param = [ks_statistic(percentiles[:, d]) for d in range(theta_dim)]
    pe = point_estimate_error(true_arr, means_arr)

    result = {
        "theta_dim": theta_dim,
        "param_names": param_names,
        "ks_per_param": {name: ks_per_param[d] for d, name in enumerate(param_names)},
        "ks_mean": float(np.mean(ks_per_param)),
        "point_error": float(np.mean(pe)),
        "point_error_per_param": {
            name: float(np.mean(pe[:, d])) for d, name in enumerate(param_names)
        },
        "percentiles": percentiles,
        "true_thetas": true_arr[:4] if len(true_arr) >= 4 else true_arr,
    }

    # Backward-compat keys for 2-param and Hellinger
    if len(hellinger_dists) > 0:
        result["hellinger"] = float(np.mean(hellinger_dists))
        result["hellinger_std"] = float(np.std(hellinger_dists))
    else:
        result["hellinger"] = float("nan")
        result["hellinger_std"] = float("nan")

    result["exact_posts"] = exact_posts_out
    result["learned_posts"] = learned_posts_out

    # Legacy keys (always present for backward compat)
    result["ks_log10A"] = ks_per_param[0] if theta_dim >= 1 else float("nan")
    result["ks_gamma"] = ks_per_param[1] if theta_dim >= 2 else float("nan")
    result["point_error_A"] = (
        float(np.mean(pe[:, 0])) if theta_dim >= 1 else float("nan")
    )
    result["point_error_gamma"] = (
        float(np.mean(pe[:, 1])) if theta_dim >= 2 else float("nan")
    )

    return result


def evaluate_is(
    model,
    dataset,
    cfg,
    device,
    n_is_examples: int = 50,
    n_is_samples: int = 10_000,
    seed: int = 777,
):
    """Run importance-sampling evaluation: ESS and IS-corrected calibration.

    Only evaluated on unmasked data (masking_severity=0).
    """
    prior = UniformPrior(cfg["prior"])
    param_names = list(cfg["prior"].keys())
    D = len(param_names)
    jitter = cfg["data"].get("jitter", 1e-20)

    ess_list = []
    ess_frac_list = []
    percentiles_is = []

    n_eval = min(n_is_examples, len(dataset))

    for i in tqdm(range(n_eval), desc="IS evaluation", leave=False):
        item = dataset[i]
        sim = item["sim"]
        batch = _make_batch_single(item, device=device)

        result = importance_sample(
            model, batch, sim, prior, n_samples=n_is_samples, jitter=jitter
        )

        ess_list.append(result["ess"])
        ess_frac_list.append(result["ess_fraction"])

        # IS-corrected percentile ranks for P-P calibration
        true_theta = sim.theta
        pctls = np.zeros(D)
        for d in range(D):
            pctls[d] = weighted_percentile_rank(
                result["samples"][:, d], result["weights"], true_theta[d]
            )
        percentiles_is.append(pctls)

    percentiles_is = np.array(percentiles_is)  # (n_eval, D)
    ks_is = [ks_statistic(percentiles_is[:, d]) for d in range(D)]

    return {
        "ess_mean": float(np.mean(ess_list)),
        "ess_std": float(np.std(ess_list)),
        "ess_fraction_mean": float(np.mean(ess_frac_list)),
        "ess_fraction_std": float(np.std(ess_frac_list)),
        "ess_per_example": [float(e) for e in ess_list],
        "ess_fraction_per_example": [float(e) for e in ess_frac_list],
        "ks_is_per_param": {
            name: float(ks_is[d]) for d, name in enumerate(param_names)
        },
        "ks_is_mean": float(np.mean(ks_is)),
        "percentiles_is": percentiles_is,
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
    param_names = list(cfg["prior"].keys())

    for name, model in models.items():
        print(f"\n=== Evaluating {name} ===")
        model_results = {}
        for sev in masking_levels:
            print(f"  Masking severity: {sev}")
            res = evaluate_model(
                model,
                test_ds,
                cfg,
                device,
                n_exact=ecfg["n_test_exact"],
                n_grid=ecfg["n_grid"],
                n_posterior_samples=ecfg["n_posterior_samples"],
                masking_severity=sev,
            )
            model_results[sev] = res

            # Posterior comparison plots (no-mask only, if exact grid available)
            if sev == 0.0 and len(res.get("exact_posts", [])) > 0:
                model_eval_dir = ensure_dir(os.path.join(eval_dir, name))
                for k in range(min(4, len(res["exact_posts"]))):
                    ep = res["exact_posts"][k]
                    plot_posterior_comparison(
                        ep["log10_A_grid"],
                        ep["gamma_grid"],
                        ep["posterior"],
                        res["learned_posts"][k],
                        res["true_thetas"][k],
                        os.path.join(model_eval_dir, f"posterior_{k}.png"),
                        title=f"{name} – test example {k}",
                    )

                # P-P plot (all parameters)
                ks_stats = [res["ks_per_param"].get(p, 0.0) for p in param_names]
                plot_pp(
                    res["percentiles"],
                    param_names,
                    os.path.join(model_eval_dir, "pp.png"),
                    ks_stats=ks_stats,
                    title=f"P-P plot – {name}",
                )

        all_results[name] = model_results

    # --- Importance sampling evaluation (unmasked data only) ---
    is_cfg = ecfg.get("importance_sampling", {})
    is_results = {}
    if is_cfg.get("enabled", False):
        n_is_examples = is_cfg.get("n_examples", 50)
        n_is_samples = is_cfg.get("n_samples", 10_000)
        for name, model in models.items():
            print(f"\n=== IS evaluation: {name} ===")
            is_res = evaluate_is(
                model,
                test_ds,
                cfg,
                device,
                n_is_examples=n_is_examples,
                n_is_samples=n_is_samples,
            )
            is_results[name] = is_res
            print(
                f"  ESS: {is_res['ess_mean']:.1f} ± {is_res['ess_std']:.1f}  "
                f"({is_res['ess_fraction_mean']:.3f} ± {is_res['ess_fraction_std']:.3f})"
            )
            print(f"  IS-corrected KS: {is_res['ks_is_mean']:.4f}")

            # IS-corrected P-P plot
            model_eval_dir = ensure_dir(os.path.join(eval_dir, name))
            ks_is_stats = [is_res["ks_is_per_param"].get(p, 0.0) for p in param_names]
            plot_pp(
                is_res["percentiles_is"],
                param_names,
                os.path.join(model_eval_dir, "pp_is.png"),
                ks_stats=ks_is_stats,
                title=f"P-P plot (IS-corrected) – {name}",
            )

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
            entry = {
                "hellinger": r["hellinger"],
                "hellinger_std": r["hellinger_std"],
                "ks_mean": r["ks_mean"],
                "point_error": r["point_error"],
            }
            # Per-param KS and point error
            for pn in param_names:
                entry[f"ks_{pn}"] = r["ks_per_param"].get(pn, float("nan"))
                entry[f"pe_{pn}"] = r["point_error_per_param"].get(pn, float("nan"))
            summary[name][str(sev)] = entry

    # Include IS results in summary
    for name in is_results:
        ir = is_results[name]
        if name not in summary:
            summary[name] = {}
        summary[name]["importance_sampling"] = {
            "ess_mean": ir["ess_mean"],
            "ess_std": ir["ess_std"],
            "ess_fraction_mean": ir["ess_fraction_mean"],
            "ess_fraction_std": ir["ess_fraction_std"],
            "ks_is_mean": ir["ks_is_mean"],
            "ks_is_per_param": ir["ks_is_per_param"],
            "ess_per_example": ir["ess_per_example"],
        }

    with open(os.path.join(eval_dir, "eval_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {eval_dir}/")
    print("\nSummary:")
    for name in summary:
        print(f"\n  {name}:")
        for sev in summary[name]:
            s = summary[name][sev]
            print(
                f"    mask={sev}: H={s['hellinger']:.4f} KS={s['ks_mean']:.4f} PE={s['point_error']:.4f}"
            )


if __name__ == "__main__":
    main()
