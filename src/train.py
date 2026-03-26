"""Training script for transformer / LSTM + normalizing flow NPE.

Usage:
    python -m src.train --config configs/smoke.yaml --model transformer
    python -m src.train --config configs/smoke.yaml --model lstm
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import load_config, set_seed, get_device, ensure_dir
from .priors import UniformPrior
from .dataset import PulsarDataset
from .collate import collate_fn
from .models.model_wrappers import build_model
from .plots import plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NPE model")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--model", type=str, required=True, choices=["transformer", "lstm"])
    p.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    p.add_argument(
        "--device", type=str, default=None, help="Force device (cpu/cuda/mps)"
    )
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device, grad_clip, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0
    use_amp = scaler is not None
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
        total_loss += loss.item() * batch["theta"].shape[0]
        n += batch["theta"].shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.autocast(device.type, enabled=use_amp):
            loss = model(batch)
        total_loss += loss.item() * batch["theta"].shape[0]
        n += batch["theta"].shape[0]
    return total_loss / max(n, 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    set_seed(tcfg["seed"])

    out_dir = args.output_dir or cfg["output_dir"]
    out_dir = os.path.join(out_dir, args.model)
    ensure_dir(out_dir)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")

    prior = UniformPrior(cfg["prior"])

    # Datasets
    use_sobol = cfg.get("data", {}).get("use_sobol", False)
    train_ds = PulsarDataset(
        n_samples=cfg["data"]["train_samples"],
        prior=prior,
        data_cfg=cfg["data"],
        seed=tcfg["seed"],
        masking_severity=0.5,
        augment=True,
        use_sobol=use_sobol,
    )
    val_ds = PulsarDataset(
        n_samples=cfg["data"]["val_samples"],
        prior=prior,
        data_cfg=cfg["data"],
        seed=tcfg["seed"] + 1_000_000,
        masking_severity=0.0,
        augment=False,
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
    best_val = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        tl = train_one_epoch(
            model, train_loader, optimizer, device, tcfg["grad_clip"], scaler
        )
        vl = validate(model, val_loader, device, use_amp)
        scheduler.step()
        dt = time.time() - t0

        train_losses.append(tl)
        val_losses.append(vl)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']}  train={tl:.4f}  val={vl:.4f}  lr={lr_now:.2e}  [{dt:.1f}s]"
        )

        if vl < best_val:
            best_val = vl
            patience_counter = 0
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
        else:
            patience_counter += 1
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
    with open(os.path.join(out_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Best val loss: {best_val:.4f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
