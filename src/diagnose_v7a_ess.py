"""Diagnose v7a/v7b collapsed-ESS cases: identify parameter regimes that cause IS failure.

Reconstructs the 50 IS test examples, pairs them with ESS from eval_metrics.json,
prints a sorted diagnostic table, and generates corner plots for the worst cases.

Usage:
    conda run -n pta-transformer-sbi python3 -m src.diagnose_v7a_ess [v7a|v7b]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, ensure_dir
from src.priors import UniformPrior
from src.dataset import FixedPulsarDataset
from src.models.model_wrappers import build_model
from src.make_corner_examples_v7a import (
    make_batch,
    monolithic_samples_and_is,
    make_corner,
    PARAM_NAMES,
    JITTER,
)

# ── Config paths per run ──────────────────────────────────────────────────────

RUN_PATHS = {
    "v7a": {
        "config": "configs/transformer_v7a.yaml",
        "checkpoint": "outputs/v7a/transformer/best_model.pt",
        "eval_metrics": "outputs/v7a/eval/eval/eval_metrics.json",
        "out_dir": "outputs/v7a/eval/ess_diagnostics",
    },
    "v7b": {
        "config": "configs/transformer_v7b.yaml",
        "checkpoint": "outputs/v7b/transformer/best_model.pt",
        "eval_metrics": "outputs/v7b/eval/eval_metrics.json",
        "out_dir": "outputs/v7b/eval/ess_diagnostics",
    },
}

N_CORNER_WORST = 3       # generate corner plots for this many worst cases
N_CORNER_SAMPLES = 100_000
N_RESAMPLE = 5_000
ESS_LOW_THRESHOLD = 50   # "collapsed" group
ESS_HIGH_THRESHOLD = 500  # "healthy" group

# Short names for table display
SHORT_NAMES = [
    "A_red", "g_red", "A_dm", "g_dm",
    "EF0", "EQ0", "EC0", "EF1", "EQ1", "EC1",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", default="v7a", choices=list(RUN_PATHS.keys()))
    args = parser.parse_args()
    paths = RUN_PATHS[args.run]
    config_path = paths["config"]
    checkpoint_path = paths["checkpoint"]
    eval_metrics_path = paths["eval_metrics"]
    out_dir_path = paths["out_dir"]
    print(f"=== Diagnostics for run: {args.run} ===")

    cfg = load_config(config_path)

    # ── Load ESS values from eval ─────────────────────────────────────────
    with open(eval_metrics_path) as f:
        metrics = json.load(f)
    ess_list = metrics["transformer"]["importance_sampling"]["ess_per_example"]
    n_examples = len(ess_list)
    print(f"Loaded {n_examples} ESS values from {eval_metrics_path}")

    # ── Reconstruct test examples ─────────────────────────────────────────
    prior = UniformPrior(cfg["prior"])
    seed = cfg["training"]["seed"] + 2_000_000
    print(f"Reconstructing {n_examples} test examples (seed={seed})...")
    test_ds = FixedPulsarDataset(
        n_samples=n_examples,
        prior=prior,
        data_cfg=cfg["data"],
        seed=seed,
        factorized=False,
    )

    # Extract theta for each example
    thetas = np.array([test_ds[i]["sim"].theta for i in range(n_examples)])  # (50, 10)

    # ── Sort by ESS and print table ───────────────────────────────────────
    ess_arr = np.array(ess_list)
    order = np.argsort(ess_arr)

    header = f"{'Idx':>4s}  {'ESS':>8s}  " + "  ".join(f"{n:>7s}" for n in SHORT_NAMES)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for rank, idx in enumerate(order):
        ess = ess_arr[idx]
        theta = thetas[idx]
        marker = "**" if ess < ESS_LOW_THRESHOLD else "  "
        vals = "  ".join(f"{v:7.3f}" for v in theta)
        print(f"{idx:4d}  {ess:8.1f}  {vals}  {marker}")
    print(sep)

    # ── Statistical comparison: low vs high ESS groups ────────────────────
    low_mask = ess_arr < ESS_LOW_THRESHOLD
    high_mask = ess_arr > ESS_HIGH_THRESHOLD
    n_low = low_mask.sum()
    n_high = high_mask.sum()
    print(f"\nGroup sizes: low ESS (<{ESS_LOW_THRESHOLD}): {n_low}, "
          f"high ESS (>{ESS_HIGH_THRESHOLD}): {n_high}")

    if n_low > 0 and n_high > 0:
        low_thetas = thetas[low_mask]
        high_thetas = thetas[high_mask]

        print(f"\n{'Parameter':<18s}  {'Low mean':>9s} {'Low std':>8s}  "
              f"{'High mean':>9s} {'High std':>8s}  {'Delta/σ':>7s}  Flag")
        print("-" * 85)
        for d, name in enumerate(PARAM_NAMES):
            lm, ls = low_thetas[:, d].mean(), low_thetas[:, d].std()
            hm, hs = high_thetas[:, d].mean(), high_thetas[:, d].std()
            pooled_std = np.sqrt((ls**2 + hs**2) / 2) if (ls + hs) > 0 else 1.0
            delta_sigma = abs(lm - hm) / pooled_std if pooled_std > 0 else 0.0
            flag = " <== ***" if delta_sigma > 1.0 else (" <== *" if delta_sigma > 0.7 else "")
            print(f"{name:<18s}  {lm:9.3f} {ls:8.3f}  {hm:9.3f} {hs:8.3f}  "
                  f"{delta_sigma:7.2f}  {flag}")

    # ── Summary stats ─────────────────────────────────────────────────────
    print(f"\nESS summary: mean={ess_arr.mean():.1f}, median={np.median(ess_arr):.1f}, "
          f"min={ess_arr.min():.1f}, max={ess_arr.max():.1f}")
    print(f"ESS < 10: {(ess_arr < 10).sum()}, ESS < 50: {(ess_arr < 50).sum()}, "
          f"ESS > 500: {(ess_arr > 500).sum()}, ESS > 1000: {(ess_arr > 1000).sum()}")

    # ── Corner plots for worst cases ──────────────────────────────────────
    worst_indices = order[:N_CORNER_WORST]
    print(f"\nGenerating corner plots for {N_CORNER_WORST} worst ESS cases: {worst_indices.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_type"], ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    out_dir = ensure_dir(out_dir_path)
    prior_cfg = cfg["prior"]

    for rank, idx in enumerate(worst_indices):
        item = test_ds[idx]
        sim = item["sim"]
        batch = make_batch(sim, device=str(device))
        ess = ess_arr[idx]

        print(f"\n[Rank {rank}] Example {idx}, ESS={ess:.1f}")
        print(f"  theta = {sim.theta.round(3)}")
        print(f"  N_TOA={len(sim.t)}, T_span={sim.tspan:.1f} yr")

        flat_np, is_np, ess_calc, ess_frac = monolithic_samples_and_is(
            model, batch, sim, prior, N_CORNER_SAMPLES, JITTER, device
        )
        print(f"  ESS (recomputed) = {ess_calc:.0f} / {N_CORNER_SAMPLES} ({ess_frac * 100:.1f}%)")

        label = f"worst-{rank}_idx{idx}_ESS{ess:.0f}"
        out_path = os.path.join(out_dir, f"corner_worst_{rank:02d}_idx{idx:02d}.png")
        make_corner(
            flat_np, is_np, sim.theta, label, idx, prior_cfg, out_path,
            ess=ess_calc, ess_frac=ess_frac,
        )

    print(f"\nDone. Diagnostics saved to {out_dir}/")


if __name__ == "__main__":
    main()
