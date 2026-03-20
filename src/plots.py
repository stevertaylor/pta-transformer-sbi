"""Plotting helpers for posterior comparison, calibration, and robustness."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: str,
    title: str = "Training curves",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Neg. log-prob loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_posterior_comparison(
    log10_A_grid: np.ndarray,
    gamma_grid: np.ndarray,
    exact_post: np.ndarray,
    learned_post: np.ndarray,
    true_theta: np.ndarray,
    save_path: str,
    title: str = "",
) -> None:
    """Side-by-side contour plots of exact vs learned posterior."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    AG, GG = np.meshgrid(log10_A_grid, gamma_grid, indexing="ij")

    for ax, post, label in zip(axes, [exact_post, learned_post], ["Exact", "Learned"]):
        ax.contourf(AG, GG, post, levels=30, cmap="Blues")
        ax.plot(true_theta[0], true_theta[1], "r*", ms=12, label="true")
        ax.set_xlabel("log10_A_red")
        ax.set_ylabel("gamma_red")
        ax.set_title(f"{label} posterior")
        ax.legend(loc="upper right")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pp(
    percentiles: np.ndarray,
    param_names: list[str],
    save_path: str,
    ks_stats: Optional[list[float]] = None,
    title: str = "P-P plot",
) -> None:
    """P-P / calibration plot: empirical CDF vs uniform diagonal."""
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.linspace(0, 1, 200)

    for d, name in enumerate(param_names):
        sorted_p = np.sort(percentiles[:, d])
        ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        label = name
        if ks_stats is not None:
            label += f" (KS={ks_stats[d]:.3f})"
        ax.plot(sorted_p, ecdf, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax.set_xlabel("Credible level")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_robustness(
    masking_levels: list[float],
    metrics_transformer: dict,
    metrics_lstm: dict,
    save_path: str,
) -> None:
    """Bar / line chart: metric vs masking severity for both models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metric_keys = ["hellinger", "ks_mean", "point_error"]
    metric_labels = ["Hellinger distance", "Mean KS stat", "Mean point error"]

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        vals_t = [metrics_transformer[m][key] for m in masking_levels]
        vals_l = [metrics_lstm[m][key] for m in masking_levels]
        ax.plot(masking_levels, vals_t, "o-", label="Transformer")
        ax.plot(masking_levels, vals_l, "s--", label="LSTM")
        ax.set_xlabel("Masking severity")
        ax.set_ylabel(label)
        ax.legend()

    fig.suptitle("Robustness under structured masking", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
