"""One-off script to reconstruct last_checkpoint.pt from best_model.pt + log data.

The training run was killed at epoch 10 before the resume feature existed.
This reconstructs the EMA/patience state from the logged values so --resume works.
"""

import torch

ckpt = torch.load("outputs/v5/transformer/best_model.pt", map_location="cpu", weights_only=False)

# Epochs 1-10 from the new run (tail of v5_train.log)
train_losses = [4.0256, 2.9447, 1.9823, 1.5023, 1.2480, 1.0563, 0.9086, 0.7737, 0.6642, 0.5818]
val_losses =   [3.4682, 2.4380, 1.7818, 1.2521, 1.0312, 0.8514, 1.0319, 0.8003, 0.8182, 0.7219]
train_g = [3.2393, 2.3391, 1.4621, 1.0629, 0.8566, 0.6970, 0.5739, 0.4609, 0.3690, 0.3035]
train_w = [2.6210, 2.0184, 1.7339, 1.4649, 1.3046, 1.1975, 1.1156, 1.0425, 0.9841, 0.9276]
val_g =   [2.7541, 1.8789, 1.2228, 0.8263, 0.6119, 0.4788, 0.7184, 0.4524, 0.5107, 0.4163]
val_w =   [2.3802, 1.8639, 1.8635, 1.4193, 1.3974, 1.2419, 1.0452, 1.1598, 1.0248, 1.0187]

# Reconstruct EMA values (alpha=0.15)
alpha = 0.15
ema_val = ema_global_s = ema_wn_s = None
best_ema_ckpt = best_ema_global = best_ema_wn = float("inf")
patience_counter = patience_wn_counter = 0
best_val = float("inf")

for i in range(10):
    vl, vg, vw = val_losses[i], val_g[i], val_w[i]
    ema_val = vl if ema_val is None else alpha * vl + (1 - alpha) * ema_val
    ema_global_s = vg if ema_global_s is None else alpha * vg + (1 - alpha) * ema_global_s
    ema_wn_s = vw if ema_wn_s is None else alpha * vw + (1 - alpha) * ema_wn_s
    ema_ckpt = ema_global_s + ema_wn_s

    if ema_ckpt < best_ema_ckpt:
        best_ema_ckpt = ema_ckpt
        best_val = vl

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

print(f"Reconstructed state at epoch 10:")
print(f"  ema_ckpt={ema_ckpt:.4f}  best_ema_ckpt={best_ema_ckpt:.4f}")
print(f"  ema_global_s={ema_global_s:.4f}  ema_wn_s={ema_wn_s:.4f}")
print(f"  pg={patience_counter}  pw={patience_wn_counter}")

last_ckpt = {
    "epoch": 10,
    "model_state_dict": ckpt["model_state_dict"],
    "optimizer_state_dict": ckpt["optimizer_state_dict"],
    "best_val": best_val,
    "best_ema_ckpt": best_ema_ckpt,
    "best_ema_global": best_ema_global,
    "best_ema_wn": best_ema_wn,
    "patience_counter": patience_counter,
    "patience_wn_counter": patience_wn_counter,
    "ema_val": ema_val,
    "ema_global_s": ema_global_s,
    "ema_wn_s": ema_wn_s,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_global_losses": train_g,
    "train_wn_losses": train_w,
    "val_global_losses": val_g,
    "val_wn_losses": val_w,
    "config": ckpt["config"],
    "model_type": ckpt["model_type"],
}
torch.save(last_ckpt, "outputs/v5/transformer/last_checkpoint.pt")
print("Saved outputs/v5/transformer/last_checkpoint.pt")
