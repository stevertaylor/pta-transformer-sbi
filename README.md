# PTA Transformer SBI

Toy proof-of-principle: **transformer-based neural posterior estimation for pulsar timing data**.

## What is this?

This project answers the question:

> *Does a transformer encoder over irregular, gappy TOA-level pulsar timing data provide a useful embedding for subsequent neural posterior estimation (NPE), compared to a simpler LSTM baseline?*

We implement a complete toy pipeline for a **single-pulsar red-noise inference** problem using simulation-based inference (SBI). Two encoder architectures (Transformer, LSTM) are paired with the **same** conditional normalizing flow posterior head, trained on simulated data, and compared against an exact analytic posterior.

## Scientific toy problem

**Infer a 2-D parameter vector** for one pulsar:

```
θ = (log10_A_red, gamma_red)
```

- `log10_A_red`: amplitude-like scale parameter for red noise (arbitrary simulator units)
- `gamma_red`: spectral slope of the red-noise power law

The simulator generates irregular, variable-length, gappy observation schedules and produces TOA-level residuals. The model input is **TOA-level tokens** (not a fixed-length handcrafted spectrum), and evaluation includes comparison against an **exact Gaussian posterior** computed on a 2-D grid.

### What is simplified vs. a real PTA analysis

| Aspect | This toy | Real PTA |
|--------|----------|----------|
| Number of pulsars | 1 | 20–100 |
| Parameters | 2 (red noise) | Hundreds (DM, timing model, GWB, ...) |
| Timing model | None (residuals given) | Full multi-parameter fit |
| Units | Physical (seconds, yr⁻¹, strain) | Physical (seconds, Hz, strain) |
| White noise | Diagonal, known σ | EFAC/EQUAD/ECORR |
| Likelihood | Exact Gaussian | Same form but much larger |
| Schedule | Synthetic seasonal | Real observatory logs |

### Units and scaling

All quantities use the **standard PTA / enterprise convention**:
- Times are in years
- TOA uncertainties σ are in seconds, log-uniform in [10⁻⁷, 10⁻⁵] (100 ns – 10 μs)
- Red-noise amplitude A_red = 10^(log10_A_red), with log10_A_red ∈ [−17, −11]
- Spectral index gamma_red ∈ [0.5, 6.5]
- Reference frequency f_ref = 1.0 yr⁻¹
- Per-mode variance: ρ_k = (A² / 12π²) · yr² · (f_k/f_ref)^(−γ) · Δf   (in s²)
- Covariance: C = diag(σ²) + F·Φ(θ)·Fᵀ + jitter·I  with jitter = 10⁻²⁰

## Installation

```bash
conda create -n pta-sbi python=3.11 -y
conda activate pta-sbi
pip install numpy scipy matplotlib pyyaml tqdm torch zuko pytest
```

## Quick start — smoke run (CPU, ~30 seconds)

```bash
# Train transformer
python -m src.train --config configs/smoke.yaml --model transformer --device cpu

# Train LSTM
python -m src.train --config configs/smoke.yaml --model lstm --device cpu

# Evaluate both (comparison plots, metrics, robustness)
python -m src.evaluate --config configs/smoke.yaml \
    --checkpoint outputs/smoke/transformer/best_model.pt \
    --baseline-checkpoint outputs/smoke/lstm/best_model.pt \
    --device cpu

# Demo: single-example posterior comparison
python -m src.demo_inference \
    --checkpoint outputs/smoke/transformer/best_model.pt \
    --output outputs/smoke/demo_inference.png \
    --device cpu
```

## Full run (GPU recommended)

```bash
# Train transformer (default config)
python -m src.train --config configs/transformer.yaml --model transformer

# Train LSTM (default config)
python -m src.train --config configs/lstm.yaml --model lstm

# Evaluate
python -m src.evaluate --config configs/transformer.yaml \
    --checkpoint outputs/transformer/transformer/best_model.pt \
    --baseline-checkpoint outputs/lstm/lstm/best_model.pt
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Project structure

```
├── configs/
│   ├── smoke.yaml          # Fast CPU smoke test config
│   ├── transformer.yaml    # Full transformer config
│   └── lstm.yaml           # Full LSTM config
├── src/
│   ├── priors.py           # Uniform prior over θ
│   ├── schedules.py        # Synthetic observing schedule generator
│   ├── simulator.py        # Fourier-basis red-noise simulator
│   ├── exact_posterior.py  # Exact Gaussian posterior on 2-D grid
│   ├── masking.py          # Structured masking augmentations
│   ├── dataset.py          # PyTorch datasets (on-the-fly + fixed)
│   ├── collate.py          # Padding collate function
│   ├── metrics.py          # Hellinger, calibration, point-error
│   ├── plots.py            # All plotting helpers
│   ├── utils.py            # Config loading, seeding, device
│   ├── train.py            # Training script (CLI)
│   ├── evaluate.py         # Evaluation script (CLI)
│   ├── demo_inference.py   # Single-example demo (CLI)
│   └── models/
│       ├── tokenization.py        # TOA → token features
│       ├── transformer_encoder.py # Transformer + CLS + time embed
│       ├── lstm_encoder.py        # LSTM baseline encoder
│       ├── posterior_flow.py      # Zuko NSF conditional flow
│       └── model_wrappers.py      # NPEModel = encoder + flow
├── tests/
│   ├── test_simulator.py
│   ├── test_exact_posterior.py
│   ├── test_models.py
│   └── test_smoke_train_step.py
├── outputs/                # Generated checkpoints, plots, metrics
└── requirements.txt
```

## Output plots

| Plot | Description |
|------|-------------|
| `training_curves.png` | Train/val neg-log-prob loss per epoch |
| `posterior_*.png` | Side-by-side exact vs learned 2-D posterior contours |
| `pp_*.png` | P-P calibration plot with KS statistics |
| `robustness.png` | Hellinger / KS / point-error vs masking severity for both models |
| `demo_inference.png` | Single-example: exact posterior, learned posterior, TOA time series |

## Interpretation guide

**In-distribution (no masking):** Both models should learn posteriors that roughly match the exact posterior. With only a smoke run (3 epochs, 2k samples), the posteriors will be diffuse but show the right structure.

**Under structured masking / truncation:** If the transformer matches the LSTM in-distribution but is clearly better under structured masking / truncation, that is evidence the attention-based approach is worth exploring further for real PTA data with irregular, gappy schedules.

**With the smoke run (3 epochs):** Both models are severely undertrained so the comparison is not conclusive. A full run with ~20k samples and 40 epochs should show clearer differentiation.

## Key design choices

- **TOA-level tokens**: Each observation is a token with features (normalized time, gap, residual/σ, log σ, frequency, backend). No fixed-length spectrum.
- **Learned continuous-time embedding**: An MLP maps (t_norm, log1p(dt_prev)) → d_model, added to each token embedding. No fixed sinusoidal positional encoding.
- **CLS token**: A learnable summary token prepended to the sequence; its transformer output is the sequence embedding.
- **Shared posterior head**: Both encoders feed into the same Zuko NSF conditional normalizing flow, ensuring a fair comparison.
- **Structured masking augmentations**: Season dropout, end truncation, cadence thinning — not just iid random dropout.
