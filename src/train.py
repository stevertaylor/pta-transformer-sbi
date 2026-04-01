"""Training script for transformer / LSTM + normalizing flow NPE.

Usage:
    python -m src.train --config configs/smoke.yaml --model transformer
    python -m src.train --config configs/smoke.yaml --model lstm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import load_config, set_seed, get_device, ensure_dir
from .priors import UniformPrior, FactorizedPrior
from .dataset import PulsarDataset
from .collate import collate_fn
from .models.model_wrappers import build_model, FactorizedNPEModel
from .plots import plot_training_curves


class TeeLogger:
    """Duplicate stdout/stderr to a log file (append mode)."""

    def __init__(self, log_path: str):
        self._file = open(log_path, "a")
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NPE model")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--model", type=str, required=True, choices=["transformer", "lstm"])
    p.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    p.add_argument(
        "--device", type=str, default=None, help="Force device (cpu/cuda/mps)"
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (appends). Duplicates stdout to file.",
    )
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device, grad_clip, scaler=None):
    model.train()
    total_loss = 0.0
    total_global = 0.0
    total_wn = 0.0
    n = 0
    use_amp = scaler is not None
    is_factorized = isinstance(model, FactorizedNPEModel)
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.autocast(device.type, enabled=use_amp):
            loss = model(batch)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        B = batch["mask"].shape[0]
        total_loss += loss.item() * B
        if is_factorized:
            total_global += model._last_global_loss * B
            total_wn += model._last_wn_loss * B
        n += B
    avg = total_loss / max(n, 1)
    if is_factorized:
        return avg, total_global / max(n, 1), total_wn / max(n, 1)
    return avg


@torch.no_grad()
def validate(model, loader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_global = 0.0
    total_wn = 0.0
    n = 0
    is_factorized = isinstance(model, FactorizedNPEModel)
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.autocast(device.type, enabled=use_amp):
            loss = model(batch)
        B = batch["mask"].shape[0]
        total_loss += loss.item() * B
        if is_factorized:
            total_global += model._last_global_loss * B
            total_wn += model._last_wn_loss * B
        n += B
    avg = total_loss / max(n, 1)
    if is_factorized:
        return avg, total_global / max(n, 1), total_wn / max(n, 1)
    return avg


def _run_training(args, cfg):
    """Core training logic, called inside optional TeeLogger context."""
    tcfg = cfg["training"]
    set_seed(tcfg["seed"])

    out_dir = args.output_dir or cfg["output_dir"]
    out_dir = os.path.join(out_dir, args.model)
    ensure_dir(out_dir)

    # Banner matching v4d log style
    version = os.path.basename(cfg.get("output_dir", "unknown"))
    print(f"=== {version} {args.model.capitalize()} training ===")

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")

    prior_cfg = cfg["prior"]
    factorized = cfg.get("model", {}).get("factorized", False)
    if factorized:
        n_backends_max = cfg["model"].get("n_backends_max", 4)
        prior = FactorizedPrior(
            prior_cfg["global"],
            prior_cfg["white_noise"],
            n_backends_max=n_backends_max,
        )
    else:
        prior = UniformPrior(prior_cfg)

    # Datasets
    data_cfg = cfg.get("data", {})
    use_sobol = data_cfg.get("use_sobol", False)
    reseed_per_epoch = data_cfg.get("reseed_per_epoch", False)
    train_ds = PulsarDataset(
        n_samples=cfg["data"]["train_samples"],
        prior=prior,
        data_cfg=cfg["data"],
        seed=tcfg["seed"],
        masking_severity=0.5,
        augment=True,
        use_sobol=use_sobol,
        factorized=factorized,
        reseed_per_epoch=reseed_per_epoch,
    )
    val_ds = PulsarDataset(
        n_samples=cfg["data"]["val_samples"],
        prior=prior,
        data_cfg=cfg["data"],
        seed=tcfg["seed"] + 1_000_000,
        masking_severity=0.0,
        augment=False,
        use_sobol=use_sobol,
        factorized=factorized,
    )

    num_workers = tcfg.get("num_workers", 0)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Model
    model = build_model(args.model, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, parameters: {n_params:,}")

    flow_weight_decay = tcfg.get("flow_weight_decay", None)
    if isinstance(model, FactorizedNPEModel) and flow_weight_decay is not None:
        flow_param_ids = {
            id(p)
            for submod in [model.global_flow, model.wn_flow]
            for p in submod.parameters()
        }
        encoder_params = [p for p in model.parameters() if id(p) not in flow_param_ids]
        flow_params = [p for p in model.parameters() if id(p) in flow_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "weight_decay": tcfg["weight_decay"]},
                {"params": flow_params, "weight_decay": flow_weight_decay},
            ],
            lr=tcfg["lr"],
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
        )

    # LR scheduler: optional linear warmup + cosine annealing
    warmup_fraction = tcfg.get("warmup_fraction", 0.0)
    warmup_epochs = (
        max(1, int(warmup_fraction * tcfg["epochs"])) if warmup_fraction > 0 else 0
    )
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg["epochs"] - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        print(
            f"LR schedule: {warmup_epochs} warmup + {tcfg['epochs'] - warmup_epochs} cosine"
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg["epochs"]
        )

    # AMP (only beneficial on CUDA)
    use_amp = device.type == "cuda" and tcfg.get("use_amp", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using automatic mixed precision (AMP)")

    # Training loop
    ema_alpha = tcfg.get("ema_alpha", 0.0)
    best_val = float("inf")
    best_ema_ckpt = float("inf")
    best_ema_global = float("inf")
    best_ema_wn = float("inf")
    patience_counter = 0
    patience_wn_counter = 0
    ema_val = ema_global_s = ema_wn_s = None
    train_losses, val_losses = [], []

    is_factorized = isinstance(model, FactorizedNPEModel)
    train_global_losses, train_wn_losses = [], []
    val_global_losses, val_wn_losses = [], []

    if ema_alpha > 0:
        print(f"EMA smoothing: alpha={ema_alpha:.3f} (~{1/ema_alpha:.0f}-epoch window)")

    def _ema(prev, cur):
        return cur if (prev is None or ema_alpha == 0.0) else ema_alpha * cur + (1.0 - ema_alpha) * prev

    for epoch in range(1, tcfg["epochs"] + 1):
        train_ds.set_epoch(epoch)
        t0 = time.time()
        train_result = train_one_epoch(
            model, train_loader, optimizer, device, tcfg["grad_clip"], scaler
        )
        val_result = validate(model, val_loader, device, use_amp)
        scheduler.step()
        dt = time.time() - t0

        if is_factorized:
            tl, tg, tw = train_result
            vl, vg, vw = val_result
            train_global_losses.append(tg)
            train_wn_losses.append(tw)
            val_global_losses.append(vg)
            val_wn_losses.append(vw)
        else:
            tl = train_result
            vl = val_result

        train_losses.append(tl)
        val_losses.append(vl)

        # EMA update
        ema_val = _ema(ema_val, vl)
        if is_factorized:
            ema_global_s = _ema(ema_global_s, vg)
            ema_wn_s = _ema(ema_wn_s, vw)
            ema_ckpt = ema_global_s + ema_wn_s  # unit-weight composite for checkpoint
        else:
            ema_ckpt = ema_val

        # Checkpoint: save when smoothed composite improves
        improved = ema_ckpt < best_ema_ckpt
        if improved:
            best_ema_ckpt = ema_ckpt
            best_val = vl
            ckpt_path = os.path.join(out_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": vl,
                    "config": cfg,
                    "model_type": args.model,
                },
                ckpt_path,
            )

        # Per-component patience (factorized) or combined (non-factorized)
        if is_factorized:
            if ema_global_s < best_ema_global:
                best_ema_global = ema_global_s
                patience_counter = 0
            else:
                patience_counter += 1
            if ema_wn_s < best_ema_wn:
                best_ema_wn = ema_wn_s
                patience_wn_counter = 0
            else:
                patience_wn_counter += 1
        else:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1

        lr_now = optimizer.param_groups[0]["lr"]
        if is_factorized:
            print(
                f"Epoch {epoch:3d}/{tcfg['epochs']}  train={tl:.4f} (g={tg:.4f} w={tw:.4f})  "
                f"val={vl:.4f} (g={vg:.4f} w={vw:.4f})  ckpt={ema_ckpt:.4f}  "
                f"lr={lr_now:.2e}  [{dt:.1f}s]  [pg={patience_counter} pw={patience_wn_counter}]"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{tcfg['epochs']}  train={tl:.4f}  val={vl:.4f}  lr={lr_now:.2e}  [{dt:.1f}s]"
            )

        # Early stopping: factorized requires both components to plateau
        if is_factorized:
            if patience_counter >= tcfg["patience"] and patience_wn_counter >= tcfg["patience"]:
                print(f"Early stopping at epoch {epoch} (pg={patience_counter} pw={patience_wn_counter})")
                break
        else:
            if patience_counter >= tcfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save final artifacts
    plot_training_curves(
        train_losses,
        val_losses,
        os.path.join(out_dir, "training_curves.png"),
        title=f"{args.model} training curves",
    )

    metrics = {
        "model_type": args.model,
        "best_val_loss": best_val,
        "final_epoch": len(train_losses),
        "n_params": n_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    if is_factorized:
        metrics["train_global_losses"] = train_global_losses
        metrics["train_wn_losses"] = train_wn_losses
        metrics["val_global_losses"] = val_global_losses
        metrics["val_wn_losses"] = val_wn_losses
    with open(os.path.join(out_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Best val loss: {best_val:.4f}. Checkpoint: {ckpt_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.log_file:
        ensure_dir(os.path.dirname(args.log_file) or ".")
        with TeeLogger(args.log_file):
            _run_training(args, cfg)
    else:
        _run_training(args, cfg)


if __name__ == "__main__":
    main()
