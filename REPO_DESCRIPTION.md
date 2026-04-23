# PTA Transformer SBI — Complete Repository Description

## 1. Project Overview

This repository implements a **toy proof-of-principle** for using a **transformer-based neural posterior estimation (NPE)** pipeline to perform Bayesian inference on **pulsar timing array (PTA) data**. The central research question is:

> Does a transformer encoder applied directly to irregular, gappy, TOA-level (Time of Arrival) pulsar timing data provide a useful embedding for subsequent neural posterior estimation, compared to a simpler LSTM baseline?

The project constructs a complete **simulation-based inference (SBI)** pipeline for **single-pulsar red-noise inference**, going from synthetic data generation through model training to quantitative evaluation against analytic Bayesian posteriors. Two encoder architectures — a **Transformer** and an **LSTM** — are paired with the **same** conditional normalizing flow posterior head and evaluated head-to-head.

### Scientific Context

Pulsar Timing Arrays (PTAs) use networks of millisecond pulsars as galactic-scale gravitational wave detectors. Detecting deviations in the Times of Arrival (TOAs) of radio pulses requires modeling correlated noise processes (red noise, DM variations) and white noise (EFAC, EQUAD, ECORR). Traditional Bayesian analyses use MCMC/nested sampling on a per-pulsar basis, which is computationally expensive. This project explores whether **amortized neural inference** — training a neural network to produce posteriors for *any* input in one forward pass — can replace or accelerate this process.

### What Is Simplified vs. Real PTA Analysis

| Aspect | This Toy Problem | Real PTA |
|--------|------------------|----------|
| Number of pulsars | 1 | 20–100 |
| Parameters | 7 (4 global + 3 WN per backend) | Hundreds (DM, timing model, GWB, ...) |
| Timing model | None (residuals given) | Full multi-parameter fit |
| White noise | EFAC/EQUAD/ECORR per backend | Same |
| Likelihood | Exact Gaussian | Same form but much larger matrices |
| Schedule | Synthetic seasonal, multi-backend | Real observatory logs |

---

## 2. Tech Stack

- **Language**: Python 3.11
- **Deep learning**: PyTorch ≥ 2.0
- **Normalizing flows**: Zuko ≥ 1.0 (Neural Spline Flows)
- **Scientific computing**: NumPy, SciPy
- **Visualization**: Matplotlib
- **Testing**: pytest
- **Hardware**: Developed and trained on NVIDIA RTX 4090 (CUDA); CPU smoke tests included
- **Quasi-random sampling**: SciPy Sobol sequences for better prior coverage

---

## 3. Repository Structure

```
├── configs/                    # YAML experiment configurations (all versions)
│   ├── smoke*.yaml             # Fast CPU smoke tests (~30 sec)
│   ├── transformer*.yaml       # Full transformer GPU configs
│   └── lstm*.yaml              # Full LSTM GPU configs
├── src/                        # Core source code
│   ├── priors.py               # UniformPrior + FactorizedPrior classes
│   ├── schedules.py            # Synthetic observing schedule generator
│   ├── simulator.py            # Fourier-basis noise simulator (standard + factorized)
│   ├── exact_posterior.py      # Exact Gaussian posterior on 2-D grid (Woodbury)
│   ├── masking.py              # Structured masking augmentations
│   ├── dataset.py              # PyTorch datasets (on-the-fly + fixed; epoch reseeding)
│   ├── collate.py              # Padding collate function for variable-length TOAs
│   ├── metrics.py              # Hellinger distance, calibration, point-error
│   ├── plots.py                # Plotting helpers (posterior, calibration, robustness)
│   ├── importance_sampling.py  # Dingo-IS posterior correction
│   ├── utils.py                # Config loading, seeding, device selection
│   ├── train.py                # Main training script (CLI)
│   ├── evaluate.py             # Evaluation script (CLI)
│   ├── demo_inference.py       # Single-example demo (CLI)
│   └── models/
│       ├── tokenization.py         # TOA → token features (7 continuous + backend ID)
│       ├── transformer_encoder.py  # Transformer + RoPE + multiple pooling strategies
│       ├── lstm_encoder.py         # LSTM baseline encoder
│       ├── posterior_flow.py       # Zuko NSF conditional flow wrapper
│       └── model_wrappers.py       # NPEModel + FactorizedNPEModel + build_model factory
├── tests/                      # Comprehensive test suite
│   ├── test_simulator.py       # Simulator correctness tests
│   ├── test_exact_posterior.py # Exact posterior validation
│   ├── test_factorized.py      # Factorized model pipeline tests
│   ├── test_models.py          # Model architecture tests
│   ├── test_smoke_train_step.py # End-to-end training step test
│   ├── test_tokenization.py    # Tokenization correctness
│   └── test_importance_sampling.py # IS pipeline tests
├── outputs/                    # All training outputs, checkpoints, logs, metrics
├── tutorials/                  # 5-part deep-dive notebook series
├── tutorial_sbi_framework.ipynb # Single-notebook overview
├── scripts/
│   └── reconstruct_checkpoint.py # One-off script to reconstruct interrupted checkpoint
├── README.md
└── requirements.txt
```

---

## 4. The Physics / Signal Model

### 4.1 Observing Schedule Generation (`schedules.py`)

Each simulated pulsar has a synthetic observing schedule with:
- **Random total baseline**: 5–15 years
- **Seasonal observing windows**: ~8 months on, ~4 months off per year
- **Irregular cadence** within seasons (variable gap between observations)
- **Randomly dropped seasons** (~20% chance each) — mimics real telescope downtime
- **1–3 TOAs per epoch** at different radio frequencies (820, 1400, 2300 MHz) — mimics real multi-frequency PTA observations
- **Multiple backends** (1–3): Each TOA is assigned to a random receiver backend
- **Heteroskedastic white noise**: TOA uncertainties drawn log-uniform in [100 ns, 10 μs]
- **Total TOA count**: 80–400 per pulsar (configurable)

The `Schedule` dataclass stores `(t, sigma, freq_mhz, backend_id, epoch_id)` arrays, where `epoch_id` groups co-temporal TOAs (needed for ECORR modeling).

### 4.2 Prior Distributions (`priors.py`)

Two prior classes support different model versions:

**`UniformPrior`** (v1–v4): Flat D-dimensional box prior over all parameters jointly. Supports both iid sampling and **Sobol quasi-random sampling** for better high-D coverage (discrepancy O(log^d N / N) vs. O(1/√N)).

**`FactorizedPrior`** (v5+): Separates parameters into two blocks:
- **Global**: `(log10_A_red, gamma_red, log10_A_dm, gamma_dm)` — 4D red/DM noise
- **Per-backend WN**: `(EFAC, log10_EQUAD, log10_ECORR)` — 3D, independently sampled per backend

Prior bounds for the v5+ factorized version:
| Parameter | Range | Description |
|-----------|-------|-------------|
| log10_A_red | [−18, −11] | Red noise amplitude (strain) |
| gamma_red | [1.0, 7.0] | Red noise spectral index |
| log10_A_dm | [−18, −11] | DM noise amplitude |
| gamma_dm | [1.0, 7.0] | DM noise spectral index |
| EFAC | [0.5, 2.0] | White noise scale factor |
| log10_EQUAD | [−8, −5] | Additional white noise (seconds) |
| log10_ECORR | [−8, −5] | Epoch-correlated jitter (seconds) |

### 4.3 Noise Simulator (`simulator.py`)

Uses the standard **PTA/enterprise convention** for power-law processes:

**Fourier design matrix**: F[i, k] contains sin/cos pairs at frequencies f_k = k/T for k = 1…n_modes, where T is the time span.

**Power-law spectrum** (enterprise convention):
$$\rho_k = \frac{A^2}{12\pi^2} \cdot \text{SEC\_PER\_YR}^2 \cdot \left(\frac{f_k}{f_\text{ref}}\right)^{-\gamma} \cdot \Delta f$$

where A = 10^{log10\_A}, f_ref = 1 yr⁻¹, Δf = 1/T. Units are in seconds².

**Noise components**:
1. **Red noise** (achromatic): r_red = F · a_red, where a_red ~ N(0, diag(ρ_red))
2. **DM noise** (chromatic): r_dm = F_dm · a_dm, where F_dm = F · (f_ref/f_obs)². DM variations produce timing delays ∝ 1/f_obs².
3. **White noise**: r_wn ~ N(0, (EFAC·σ)² + EQUAD²) per-TOA independently
4. **ECORR** (epoch-correlated jitter): r_ecorr = ECORR · j_e, one random draw per observing epoch, applied to all TOAs in that epoch

Total residual: r = r_red + r_dm + r_wn + r_ecorr

Two simulator functions:
- `simulate_pulsar()` — supports 2-D (red only) and 7-D (full) theta vectors; single set of WN params
- `simulate_pulsar_factorized()` — per-backend white noise parameters; used by v5+ factorized models

### 4.4 Exact Posterior (`exact_posterior.py`)

Because the signal model is **linear in Fourier coefficients** and the noise is **Gaussian**, the likelihood is a multivariate Gaussian whose covariance depends on θ. With a uniform prior, the posterior is proportional to the likelihood on a grid.

The code evaluates **exact log-likelihood on a 2-D grid** over (log10_A_red, gamma_red) using the **Woodbury identity** for numerical efficiency:

**2-param mode** (v1–v3): C = diag(σ²+jitter) + F·Φ·F^T. Woodbury avoids inverting the N×N matrix directly.

**7-param conditional mode** (v4+): Conditions on true values of nuisance parameters (DM, WN) and grids over the 2-D red-noise marginal:
$$D_\text{eff} = \text{diag}(\text{EFAC}^2 \sigma^2 + \text{EQUAD}^2) + F_\text{dm} \Phi_\text{dm} F_\text{dm}^T + \text{ECORR}^2 U U^T$$

Then applies Woodbury over the red-noise part with D_eff as the (possibly dense) base.

---

## 5. The Neural Architecture

### 5.1 Tokenization (`tokenization.py`)

Each TOA becomes a **token** with 7 continuous features plus a categorical backend ID:

| Feature | Formula | Purpose |
|---------|---------|---------|
| t_norm | (t − t_min) / T_span | Normalized time |
| dt_prev | Δt / T_span | Gap to previous TOA |
| r_over_sig | sign(r/σ) · ln(1 + |r/σ|) | Signed-log compressed SNR |
| log_sigma | log₁₀(σ) | Uncertainty scale |
| r_raw | residual in seconds | Raw residual |
| freq_norm | (f − 1400) / 1000 | Normalized frequency |
| log_f | log₁₀(f / 1400) | Chromatic coordinate for DM |
| backend_id | integer 0–3 | Categorical (embedded separately) |

The signed-log transform on r/σ is critical: raw r/σ can reach 10⁵ for strong red noise, which would overflow float16 under AMP training.

### 5.2 Transformer Encoder (`transformer_encoder.py`)

**Modern path** (use_rope=True, used in v3+):

1. **Token embedding**: Backend ID → Embedding(4, 8), concatenated with 7 continuous features → 15-d. Then MLP: Linear(15 → d_model) → GELU → Linear(d_model → d_model).

2. **Rotary Position Embeddings (RoPE)**: Continuous-time RoPE using `positions = t_norm × 512`. Inverse frequencies: `inv_freq = 10000^(−2i/d_k)`. Applied to Q and K in attention. This is critical for encoding the irregular temporal structure of PTA data.

3. **Pre-norm transformer blocks** (×num_layers): LayerNorm → RoPE Multi-Head Self-Attention → Residual → LayerNorm → FFN (d_model → 4·d_model → d_model, GELU) → Residual.

4. **Sequence-to-vector pooling**:
   - `CLSQueryPooling`: Learned query vector + multi-head cross-attention to extract a single context vector.
   - `BackendQueryPooling` (v5+): Per-backend cross-attention — a shared learned query attends only over tokens from each backend, producing backend-specific context vectors.

5. **Context projection**: LayerNorm → Linear(d_model → context_dim).

**Full-scale configuration** (v5): d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, context_dim=128.

### 5.3 LSTM Encoder (`lstm_encoder.py`)

Bidirectional LSTM baseline:
1. Same token embedding MLP as the transformer.
2. Multi-layer bidirectional LSTM (hidden=256, layers=3).
3. Attention pooling over the hidden states.
4. Linear projection to context_dim.

For factorized mode, the LSTM has a `forward_factorized()` method that produces both global context (from pooling over all tokens) and per-backend context (pooling over backend-masked tokens).

### 5.4 Normalizing Flow Posterior Head (`posterior_flow.py`)

Thin wrapper around **Zuko's Neural Spline Flow (NSF)**:
- **Rational-quadratic spline** coupling transforms
- Conditioned on the encoder's context vector
- Input theta is **pre-normalized** to approximately [−1, 1] using prior bounds (mean/std normalization) so values stay within spline domain
- **Explicit float32 enforcement** under AMP — spline flows are numerically unstable in float16

### 5.5 Model Wrappers (`model_wrappers.py`)

**`NPEModel`** (v1–v4): Standard encoder + flow. Methods: `forward()` (training loss), `sample_posterior()` (inference), `log_prob_on_grid()` (IS evaluation). Optional auxiliary features (4-d: log N_TOA, log T_span, mean/std of log σ).

**`FactorizedNPEModel`** (v5+): Dual-flow architecture:
- **Global flow** (4-D): Conditioned on encoder's global context + auxiliary features
- **WN flow** (3-D, shared weights): Conditioned on [global_ctx || backend_ctx || backend_aux]. Applied independently to each active backend.
- **Optional context bottlenecks**: LayerNorm → Linear → GELU to compress context before flows
- **Context dropout**: Applied during training for regularization
- **Loss**: `−log q_global(θ_global|c) − w · mean_b[log q_wn(θ_wn_b|c_b)]`

---

## 6. Training Pipeline (`train.py`)

### 6.1 Data Generation

- **On-the-fly simulation**: Each `__getitem__` seeds an RNG from `(base_seed + idx [+ epoch * n_samples])`, draws theta, generates a schedule, simulates residuals, applies masking, and tokenizes. No data stored to disk.
- **Epoch reseeding** (v5+): Each epoch sees fresh (θ, schedule, noise) triples, preventing the flow from memorizing fixed training pairs.
- **Sobol vs. random sampling**: v4+ optionally uses Sobol quasi-random sequences for better prior coverage.
- **Structured masking augmentations** (v5): Season dropout, end truncation, cadence thinning applied with configurable severity during training. Severity is sampled uniformly in [0, masking_severity] per example.

### 6.2 Optimization

- **Optimizer**: AdamW with per-group weight decay
  - v5+: Flow parameters get 4× higher weight decay (2×10⁻³) vs. encoder (5×10⁻⁴) — flows are more prone to overfitting
- **LR schedule**: Linear warmup (5–10% of epochs) + cosine annealing to zero
- **Gradient clipping**: By norm (1.0–5.0)
- **AMP**: Automatic mixed precision (float16) with explicit float32 for flow computations
- **EMA smoothing** (v5+): Exponential moving average of validation loss (α=0.15, ~7-epoch window) for smoother early stopping decisions
- **Checkpointing**: Best model (by EMA checkpoint metric) + resumable last checkpoint
- **Early stopping**: Per-component patience tracking for factorized models (separate patience counters for global and WN flows)

### 6.3 Training Scale

Full runs: 500,000 training samples, 50,000 validation, batch size 256, up to 200 epochs. Each epoch takes ~275–330 seconds on RTX 4090. Models range from 5.8M to 8.4M parameters.

---

## 7. Evaluation Pipeline (`evaluate.py`)

### 7.1 Metrics

1. **Hellinger distance**: Between exact 2-D posterior grid and learned posterior (evaluated via flow log-prob on the same grid). Measures distributional overlap. H=0 means identical, H=1 means no overlap.

2. **P-P (probability-probability) calibration**: For N test examples, compute the percentile rank of the true θ in the posterior samples. If the posterior is well-calibrated, these percentile ranks should be uniform. Quantified by KS statistic (max deviation from the diagonal).

3. **Point estimate error**: Absolute error between true θ and posterior mean.

4. **Robustness under masking**: All metrics evaluated at multiple masking severity levels (0.0, 0.15, 0.3, 0.45, 0.6) to test degradation under data loss.

### 7.2 Importance Sampling (`importance_sampling.py`)

Implements **Dingo-IS** (Dax et al. 2021): reweights flow proposal samples using exact likelihood to produce asymptotically exact posteriors.

$$w_i = \frac{p(x|\theta_i) \cdot \pi(\theta_i)}{q(\theta_i|x)}, \quad \theta_i \sim q(\theta|x)$$

Effective Sample Size (ESS) measures how well the learned proposal matches the true posterior. Higher ESS means the flow proposal is closer to the true posterior. Reports ESS, ESS fraction, and IS-corrected KS statistics.

For factorized models, IS is performed on the **global flow only**, with WN parameters fixed at truth — testing the global posterior's calibration independently.

---

## 8. Version History and Experimental Progression

### v1 (Initial Prototype)
- **Architecture**: 2-D NPE (log10_A_red, gamma_red only)
- **Units**: Arbitrary (not physical PTA convention)
- **Purpose**: Proof of concept — can a neural network learn to estimate noise parameters from TOA data?

### v2 (Physical Units)
- **Architecture**: Same 2-D NPE
- **Key change**: Adopted enterprise/PTA convention for units (times in years, residuals in seconds, standard power-law parameterization)
- **Config**: `transformer.yaml`, `lstm.yaml` — 50K training samples, 100 epochs
- **Model**: d_model=128, nhead=4, num_layers=4, flow_transforms=6, flow_hidden=128
- **Prior**: log10_A_red ∈ [−17, −11], gamma_red ∈ [0.5, 6.5]
- **Status**: Baseline established

### v3 (RoPE + Attention Pooling + Deeper Flow)
- **Key architectural changes**:
  - Rotary Position Embeddings (RoPE) — critical for irregular temporal data
  - Pre-norm transformer blocks (instead of post-norm)
  - CLS-query attention pooling (instead of mean/CLS token)
  - Auxiliary summary features (log N_TOA, log T_span, mean/std log σ)
  - Deeper/wider flow (10 transforms, 256 hidden, 3 layers, 16 bins)
  - Wider encoder (d_model=256, nhead=8, num_layers=6)
- **Training improvements**: AMP, warmup, larger batch (256), num_workers=4
- **Scale**: 200K training samples, 200 epochs max. LSTM gets 3-layer BiLSTM with 256 hidden.
- **Results** (from eval metrics):
  - **Transformer**: H=0.0075, KS=0.108, PE=0.663 at masking=0.0 | 6,745,421 params, 59 epochs, best val −1.053
  - **LSTM**: H=0.0089, KS=0.065, PE=0.690 at masking=0.0 | 6,312,524 params, 48 epochs, best val −0.980
  - Transformer beats LSTM on Hellinger; LSTM has better KS calibration (surprising)
  - Both show graceful degradation under masking
- **Conclusion**: RoPE and attention pooling provide clear benefit. Time to expand to full 7-D problem.

### v4 (7-Parameter Model — First Full Noise Model)
- **Key change**: Expanded from 2-D to full **7-D parameter vector** (log10_A_red, gamma_red, log10_A_dm, gamma_dm, EFAC, log10_EQUAD, log10_ECORR)
- **Architecture**: Same v3 encoder but flow now outputs 7-D density
- **Prior**: Added DM noise + white noise parameters
- **Scale**: 500K training samples, Sobol quasi-random prior sampling
- **Sub-variants explored**:
  - **v4** (original): context_dim from v3. Transformer 7.36M params, 51 epochs, best val −1.303
  - **v4b**: Reduced context_dim experiment. Transformer 6.16M params, 71 epochs, best val −1.570
  - **v4c**: Another context_dim variant. Transformer 6.65M params, 70 epochs, best val −1.551
  - **v4d**: Strongest flow (context_dim=128 reverted, flow_transforms=10, flow_hidden=256). Transformer 7.63M params, 76 epochs, best val −1.565
- **Results** (v4d, best variant):
  - **Transformer**: H=0.0035, KS=0.082, PE=0.665 at masking=0.0
  - **LSTM**: H=0.0069, KS=0.072, PE=0.663 at masking=0.0 (best val −1.004, 63 epochs)
  - Transformer has sub-1% Hellinger; both handle masking well
  - Importance sampling added: evaluates flow quality via ESS
- **Key observation**: Transformer overfits much more than LSTM in 7-D (train-val gap grows faster). EFAC calibration is poor — the 7-D monolithic flow struggles with the shared-vs-per-backend parameter structure.
- **Conclusion**: Need architectural changes to address EFAC miscalibration. Motivates v5 factorization.

### v5 (Factorized Amortized Inference — Major Architecture Change)
- **Key innovation**: Factorized the 7-D posterior into:
  - **4-D global flow** (log10_A_red, gamma_red, log10_A_dm, gamma_dm)
  - **3-D per-backend WN flow** with shared weights (EFAC, log10_EQUAD, log10_ECORR) — one flow applied independently per backend
- **New architectural components**:
  - `BackendQueryPooling`: Per-backend cross-attention over only that backend's tokens
  - WN context bottleneck: Compresses [global_ctx || backend_ctx || backend_aux] before WN flow
  - Context dropout (0.2) for regularization
  - Epoch reseeding: Fresh data every epoch
  - Per-group weight decay: 2×10⁻³ for flow, 5×10⁻⁴ for encoder
  - EMA-smoothed validation loss for early stopping
  - Structured masking augmentations during training (severity=0.5)
- **Training details**:
  - Smaller flows than v4: 6×128×2×8 for both global and WN (intentional, to test factorization benefit without excess capacity)
  - wn_loss_weight=0.3 (reduced WN gradient contribution)
  - grad_clip=1.0 (tighter than v4's 5.0)
  - 5,810,412 parameters. Trained for 165 epochs (early stopped).
- **Results**:
  - Best val loss: 0.7016 (positive! worse than v4's negative val losses in absolute terms, because the loss function changed with factorization)
  - **Evaluation**: H=0.0183, KS=0.176, PE=0.971 at masking=0.0 | WN KS=0.080, WN PE=0.371
  - **Importance sampling**: ESS=1746 ± 2710 (ESS fraction=0.175), KS_IS=0.227
  - Masking robustness: Very stable across masking levels (H≈0.018, KS≈0.15–0.18)
- **Problem identified**: **Regression from v4** on global posterior quality:
  - v4d Hellinger 0.0035 vs. v5 Hellinger 0.0183 (5× worse)
  - v4d KS 0.082 vs. v5 KS 0.176 (2× worse)
  - Root cause analysis: The bottleneck layers, reduced flow capacity, context dropout, and training masking all conspired to constrain the model too aggressively. The global flow was **underfitting**, not overfitting.
  - WN calibration was genuinely improved (KS=0.080) — the factorization works for its intended purpose.

### v5_f12 (Improved v5 — Larger Global Flow)
- Run with a different configuration applied to the same factorized architecture (6,686,124 parameters — slightly larger global flow). Specific config details coded into the checkpoint reconstruction script.
- **Trained for 187 epochs** (early stopped).
- **Results** (improved):
  - Best val loss: −0.0507 (significant improvement over v5's 0.7016)
  - H=0.0170, KS=0.111, PE=0.948 at masking=0.0
  - IS: ESS=2383 ± 3029 (ESS fraction=0.238), KS_IS=0.203
  - Masking robustness: H≈0.017 and KS≈0.09–0.10 across all masking levels — very robust
- **Conclusion**: Increasing global flow capacity within the factorized architecture helps. But still not matching v4d's Hellinger of 0.0035. This motivates v6 — a systematic remediation.

### v6 (Phase 1 Remediation — Full Capacity Restoration)
- **Diagnosis**: v5's regression was caused by multiple simultaneous regularization changes (bottlenecks, context dropout, smaller flow, training masking, higher encoder dropout). v6 reverses them one at a time.
- **Changes from v5**:
  1. Removed context bottlenecks entirely
  2. Restored global flow to v4-level capacity (10×256×3×16 — matching v4c/v4d)
  3. Increased WN flow (8×192×2×12)
  4. wn_loss_weight: 1.0 (full gradient to both heads)
  5. context_dropout: 0.0 (model is underfitting)
  6. use_sobol: true (better prior coverage)
  7. No training masking (removes distribution shift)
  8. Encoder dropout: 0.05 (reverted from v5's 0.1)
- **Model**: 8,422,080 parameters (largest variant)
- **Training** (in progress at time of log capture — 9 epochs shown):
  - By epoch 9: train=-3.285, val=5.377, significant overfitting already
  - The **train-val gap grew explosively** — by epoch 9, an ~8.66 nats gap
  - This proved that removing *all* regularization simultaneously while also increasing WN flow capacity was catastrophic
- **Conclusion**: Aggressive overfitting. Need to reintroduce targeted regularization.

### v6b (Targeted Fix — Keep Global Flow, Restore Regularization)
- **Surgical changes from v6**:
  1. Reverted WN flow to v5 size (6×128×2×8) — WN was never the bottleneck
  2. Restored encoder dropout to 0.1 — a 6-layer transformer needs it
  3. **Kept** large global flow (10×256×3×16) — this IS the key fix
  4. **Kept** no bottlenecks, no context dropout, no masking, Sobol on
- **Model**: 7,910,934 parameters
- **Training** (in progress — 25 epochs captured in log):
  - By epoch 25: train=-5.167, val=8.042
  - Still showing significant overfitting (train-val gap ~13 nats by epoch 25)
  - Patience counters: pg=17, pw=23 — both flows stalling on validation
  - EMA checkpoint metric: 7.228 at epoch 25
- **Status**: The v6b experiment appears to still be overfitting, though the log captures only the first 25 of 200 epochs. The learning rate was still in the early cosine phase.

### v6c (Additional Variant)
- **Model**: 7,729,420 parameters
- **Training** (only 2 epochs captured in log):
  - Epoch 1: train=5.458, val=4.791
  - Epoch 2: train=3.031, val=3.505
- **Status**: Very early, insufficient data to evaluate

### v7 Series — Scaling, Chain-Rule Factorization, and the HEAD-Era Ablation Plan

The v7 arc set out to scale beyond v5/v6 and crystallized into a systematic ablation that uncovered a fundamental eval-code issue and eventually found a new champion (v7e_cap_half). It runs in three phases: monolithic-scaling (v7a, v7b, v7b_ecorr), chain-rule factorization retries (v7c/v7c2/v7c3), and the clean HEAD-era ablation plan (v7d0, v7d0_v5exact, v7d, v7e_cap_half, v7e_cap).

#### Metric-Drift Discovery (critical caveat)

Partway through v7, a rewrite of `src/evaluate.py` and `src/importance_sampling.py` changed two things:

1. **Hellinger formula bug fix** — the v5-era code used a normalized `√p` comparison that systematically under-reported Hellinger by ~1-2 orders of magnitude. v5's reported H=0.018 is truly on the order of 0.4-0.5 under the corrected metric.
2. **Importance sampling 4D → full 10D joint** — v5-era IS only evaluated the 4D global parameters. The HEAD eval uses the full 10D joint (4 global + 3 WN × 2 backends). Full 10D IS is a much harder metric, so ESS fractions drop accordingly.

**Consequence:** v3–v7b eval metrics are NOT directly comparable to v7d–v7e eval metrics. The "v5 achieved ESS=17.5%" and "v4d achieved H=0.0035" headline numbers are eval-code artifacts of the older pipeline. Under HEAD eval, the realistic v5-class baseline ESS is ~1.4% (verified: v7d0_v5exact reproduces this from scratch). See `memory/project_metric_drift.md`.

#### Phase 1 — Monolithic 10D Scaling

**v7a** (monolithic 10D baseline). Same architecture as v4e but expanded to theta_dim=10 (4 global + 2×3 WN, N_backends_fixed=2). Factorized=false. Under pre-fix eval: H=0.017, ESS_frac=8.2%, KS_IS=0.17. Diagnosed ESS<10 tail on ~20% of cases at high log10_ECORR — motivates v7b scaling.

**v7b** (scaled monolithic). Same as v7a but flow 16 transforms × 384 hidden × 24 bins, 1M training samples, lower LR. Marginal-KS improved (0.076 mask=0) but structural ~15% ESS-collapse tail did NOT close — capacity scaling alone can't solve the WN↔red-noise coupling pathology at prior-edge regions. Under pre-fix eval: H=0.464, ESS_frac=9.5%.

**v7b_ecorr** (ECORR-prior probe). Variant sweeping ECORR prior; confirmed the structural tail lives in the log10_ECORR_1 loud region independent of prior width. H=0.484, ESS_frac=7.2% (pre-fix).

**Takeaway from Phase 1:** Monolithic scaling does not close the WN-coupling tail. Factorization needs revisiting, this time with a principled coupling structure.

#### Phase 2 — Chain-Rule Factorized Retries (all failed, diagnostic)

The v5 failure mode ("global collapsed under 48D bottleneck") was re-interpreted as "independence `q(θ_g|x)·∏_b q(wn_b|x)` drops the critical WN↔red-noise coupling." v7c family introduced the correct chain-rule factorization `q(θ_g|x) · ∏_b q(wn_b | x, θ_g)` via teacher-forced θ_g pass-through into the WN context.

- **v7c** — chain-rule + 16× global flow capacity + 96D bottleneck + no masking. By ep30: 37-nat train/val gap, WN flow catastrophic overfit. The 96D bottleneck (inherited from v6c) plus the teacher-forced θ_g created a new memorization pathway (the WN flow locks onto exact θ_g labels).
- **v7c2** — reverted bottleneck to v5's 48D/32D, added θ_g teacher noise σ=0.05. Early-stopped ep63 at val=5.55. σ=0.05 too small; WN val climbed monotonically.
- **v7c3** — more aggressive θ_g regularization (higher noise). Early-stopped ep63 at val=5.16. Still catastrophic.

**Takeaway from Phase 2:** Chain-rule with teacher-forced θ_g + any of the v6/v7c training-recipe deltas (removed masking, larger flow, higher wn_loss_weight, or 96D bottleneck) causes WN-flow memorization. The confounding is severe; a clean single-knob test is needed.

#### Phase 3 — HEAD-Era Ablation Plan (disciplined single-knob sweep)

Each step tests ONE change versus a known-good baseline.

**v7d0** (data-regime regression test). Attempted "chain-rule + v5 data regime" but tripped the 3-nat kill criterion at ep10 (gap 5.84 nats). Triggered the data-regime diagnostic.

**v7d0_v5exact** (Step 1 — data-regime anchor). Reverted four knobs to v5 values: `train_samples 1M → 500k`, `use_sobol true → false`, removed `n_backends_fixed=2`, prior `[-17,-11] → [-18,-11]`. Independence factorization, no chain-rule. **HEAD-era baseline: ESS=142 (1.42%), KS_IS=0.124, H(mask=0)=0.512.** Reproduces v5's checkpoint ESS of 150 on the same HEAD eval — confirms v5's "headline ESS=17.5%" was eval-code, not a training regression. See `memory/project_metric_drift.md`.

**v7d** (Step 2 — chain-rule isolation). Only change from v7d0_v5exact: `chain_rule: false → true` with θ_g teacher noise σ=0.05. **Result: ESS=96 (0.96%), a 32% regression vs anchor.** Best val loss improved (0.82 vs 0.96) but ESS dropped — the factorized loss improved by sharpening incorrect conditionals. DM sector (log10_A_dm, γ_dm) degraded most. **Verdict: chain-rule with teacher-forced θ_g is actively harmful on this problem**; the failure mode is mechanistic, not a σ-tuning issue. See `memory/project_v7d_chain_rule_verdict.md`.

**v7e_cap_half** (Step 3a — intermediate capacity scaling). Only change from v7d0_v5exact: scale global flow `6×128×2×8 → 8×256×2×12` (~165k → ~1.0M params). All v5 regularizers preserved. Independence factorization. **Result: ESS=279 (2.79%), +96% over anchor. Median ESS 76 vs anchor 22. ESS>200: 15/50 vs 8/50.** Best val 0.087 vs anchor 0.96 — training-loss improvement translated cleanly to IS quality. **The v5 sizing was under-parameterized for the 4D global posterior; ~1M params is the right default.** Current HEAD-era champion. See `memory/project_v7e_cap_half_verdict.md`.

**v7e_cap** (Step 3a — full capacity). Scale further: `10×256×3×16` (~2.6M params, 2.5× v7e_cap_half). Trained to ep199. Best_model.pt saved ~ep40-50 (composite `ema_global + ema_wn` selection); pg climbed to 102 by ep177 while WN kept resetting pw. **Result: ESS=285 mean (ESS median regressed 76 → 40, ESS>200 count regressed 15 → 9).** Mean tied because a single outlier case (ESS=5436) and a few large ones propped it up while the bulk regressed. **Verdict: saturation (slight regression on distribution). v7e_cap_half is the capacity sweet spot** — 2.6M overfits the 500k-sample regime faster than it converges. See `memory/project_v7e_cap_verdict.md`.

### Summary Table of All Full-Scale Runs

⚠️ **Pre-v7c (v3 through v7b_ecorr) rows use the v5-era eval pipeline** (buggy Hellinger + 4D-only IS). They are NOT directly comparable to v7d+ rows, which use corrected HEAD eval (full 10D joint IS, fixed Hellinger formula). See Metric-Drift Discovery above.

| Version | Params | Best Val | Epochs | H (mask=0) | KS (mask=0) | ESS_frac | Eval era | Key finding |
|---------|--------|----------|--------|------------|-------------|----------|----------|-------------|
| v3 Transformer | 6.75M | −1.053 | 59 | 0.0075 | 0.108 | — | v5-era | RoPE + wide encoder baseline |
| v3 LSTM | 6.31M | −0.980 | 48 | 0.0089 | 0.065 | — | v5-era | Surprisingly good calibration |
| v4d Transformer | 7.63M | −1.565 | 76 | 0.0035 | 0.082 | — | v5-era | Headline "best Hellinger" was eval artifact |
| v4d LSTM | 6.93M | −1.004 | 63 | 0.0069 | 0.072 | — | v5-era | Solid baseline |
| v5 Transformer | 5.81M | 0.702 | 165 | 0.0183 | 0.176 | 17.5%* | v5-era | Headline regression also eval-code artifact |
| v5_f12 Transformer | 6.69M | −0.051 | 187 | 0.0170 | 0.111 | 23.8%* | v5-era | Slightly larger global flow |
| v6 Transformer | 8.42M | — | 9+ | — | — | — | — | Catastrophic overfitting, aborted |
| v6b Transformer | 7.91M | — | 25+ | — | — | — | — | Still overfitting, abandoned |
| v7a Transformer | — | — | — | 0.0170 | 0.072 | 8.2%* | pre-fix | Monolithic 10D; structural tail at high ECORR |
| v7b Transformer | — | — | — | 0.464 | 0.076 | 9.5%* | mixed | Scaled monolithic; tail NOT closed |
| v7b_ecorr Transformer | — | — | — | 0.484 | 0.082 | 7.2%* | mixed | ECORR-prior probe |
| v7c / v7c2 / v7c3 | — | 5.2-5.6 | 30-63 | — | — | — | — | Chain-rule + confounds → catastrophic overfit |
| v7d0_v5exact | — | 0.958 | 166 | 0.512 | 0.156 | **1.42%** | HEAD | Independence baseline — true v5-class ESS |
| v7d (chain-rule) | — | 0.818 | 117 | 0.502 | 0.161 | 0.96% | HEAD | Chain-rule+σ regressed 32% vs anchor |
| **v7e_cap_half** | — | 0.087 | 169 | **0.470** | 0.106 | **2.79%** | HEAD | **CHAMPION** (+96% ESS vs anchor) |
| v7e_cap (2.6M global) | — | ~0.93 | 199 | 0.489 | 0.143 | 2.85%† | HEAD | Saturation; median ESS 76→40 |

\* Pre-fix / mixed eval — not comparable to HEAD rows.
† Mean tied with v7e_cap_half due to a single outlier (ESS=5436); median dropped 76→40, ESS>200 count dropped 15→9.

---

## 9. Key Design Decisions and Lessons Learned

### 9.1 Things That Worked

1. **RoPE for irregular time series**: Rotary position embeddings elegantly handle the non-uniform temporal spacing of PTA data by injecting continuous-time information directly into the attention mechanism. This was a clear improvement over additive position embeddings.

2. **Pre-norm transformer blocks**: LayerNorm before (not after) attention and FFN improved training stability with deeper models, consistent with the literature (e.g., GPT-2 and beyond).

3. **CLS-query attention pooling**: A learned query vector attending over the encoder output was superior to mean pooling or fixed CLS tokens, because it lets the model learn *what* information to extract.

4. **Theta normalization**: Normalizing the target parameters to approximately [−1, 1] using prior bounds before passing to the spline flow was essential for numerical stability with NSF.

5. **Explicit float32 for flows under AMP**: Spline flows are numerically unstable in float16. Forcing float32 for the flow forward pass while keeping the encoder in float16 was critical.

6. **Signed-log transform on r/σ**: The raw signal-to-noise ratio can span 5+ orders of magnitude; the signed-log compression `sign(x)·ln(1+|x|)` keeps values in a reasonable range.

7. **Factorized architecture for WN calibration**: The per-backend white-noise flow with BackendQueryPooling genuinely improved WN parameter calibration (KS=0.080 vs. poor EFAC calibration in v4).

8. **Epoch reseeding**: Drawing fresh (θ, schedule, noise) triples each epoch prevents the flow from memorizing training pairs and acts as a strong regularizer.

9. **Sobol quasi-random sampling**: Better coverage of the prior volume than iid random sampling, especially important in 7+ dimensions.

### 9.2 Things That Went Wrong / Lessons

1. **v5 regression**: Adding factorization + bottlenecks + context dropout + masking + smaller flows + tighter grad clip simultaneously made it impossible to attribute effects. The global flow was severely underfitting due to insufficient capacity (6×128×2×8 vs. v4's 10×256×3×16).

2. **v6 catastrophic overfitting**: Removing *all* regularization simultaneously while increasing capacity caused explosive train-val gap (8.66 nats by epoch 9). The 6-layer transformer genuinely needs dropout≈0.1.

3. **The overfitting-underfitting tension**: The central challenge emerged — the global flow needs v4-level capacity to be expressive enough, but the full model needs regularization to generalize. v6b attempted the minimal fix (big flow + encoder dropout restored) but still showed overfitting at 25 epochs.

4. **wn_loss_weight sensitivity**: v5 used wn_loss_weight=0.3, which may have starved gradient signal to the WN head. v6/v6b used 1.0 (equal weighting).

5. **Training masking creates distribution shift**: v5 applied masking at train time (severity=0.5) but evaluated with lower masking, creating a train/val distribution mismatch. v6+ correctly disabled training masking.

---

## 10. Testing Infrastructure

The test suite validates each component:

- **`test_simulator.py`**: Tests that simulated residuals have correct statistical properties (variance scales with amplitude, spectrum shape, ECORR grouping).

- **`test_exact_posterior.py`**: Validates the Woodbury-accelerated grid evaluator against the reference (slow) direct-inverse implementation. Ensures the analytic posterior is correctly normalized and the MAP is near the true parameters.

- **`test_factorized.py`**: End-to-end test of the factorized pipeline: FactorizedPrior sampling, factorized simulation, dataset creation, collation, model forward pass, and loss computation.

- **`test_models.py`**: Tests both Transformer and LSTM encoders in standard and factorized modes. Verifies output shapes, gradient flow, and that different model types produce different outputs.

- **`test_smoke_train_step.py`**: End-to-end training step test — creates a minimal dataset, instantiates a model, runs one forward/backward pass, and verifies the loss decreases.

- **`test_tokenization.py`**: Verifies tokenization produces correct shapes, value ranges, and that the signed-log transform behaves as expected.

- **`test_importance_sampling.py`**: Tests the IS pipeline: log-likelihood evaluation, ESS computation, systematic resampling, and the full importance_sample() function.

---

## 11. Tutorial Materials

### Main Overview Notebook (`tutorial_sbi_framework.ipynb`)
Single-notebook walkthrough of the entire pipeline from simulation to amortized inference.

### Five-Part Deep-Dive Series (`tutorials/`)
1. **Synthetic Data Generation**: Schedules, FactorizedPrior, Fourier red/DM noise, power-law spectrum, sensitivity analysis
2. **Data Pipeline**: Tokenization, masking augmentations, factorized PulsarDataset, epoch reseeding, collation
3. **Model Architecture**: Token MLP, RoPE, pre-norm blocks, BackendQueryPooling, WN bottleneck, context dropout, dual flows
4. **Training**: v5 config walkthrough, factorized NPE loss, LR scheduling, per-group weight decay, gradient clipping, early stopping
5. **Evaluation**: Exact posteriors (Woodbury identity), Hellinger distance, P-P calibration, KS statistics, robustness analysis

---

## 12. Current Status and Open Questions

### Status (as of v7e_cap eval, 2026-04-23)

- **v7e_cap_half is the HEAD-era champion** on the corrected eval pipeline: ESS=279 (2.79%), median ESS=76, KS_IS=0.101, H(mask=0)=0.470. ~1M global-flow params (8×256×2×12) is the capacity sweet spot under v5's data + regularization regime. See `memory/project_v7e_cap_half_verdict.md`.
- **The pre-v7c "best Hellinger ever" claim for v4d (H=0.0035) was an eval-code artifact** (buggy Hellinger normalization + 4D-only IS). Under HEAD eval, the realistic v5-class ESS is ~1.4%, not 17.5%. See `memory/project_metric_drift.md`.
- **Chain-rule factorization with teacher-forced θ_g was actively harmful** (v7d ESS=96 vs anchor 142, −32%). The failure mode is mechanistic (the WN flow learns to cross-reference x against θ_g labels), not a noise-σ tuning issue.
- **Step 3a (capacity scaling) is closed.** v7e_cap (2.6M params, 2.5× v7e_cap_half) saturates on mean and regresses on median (76→40) — the larger global flow overfits the 500k-sample regime faster than it converges, and best_model.pt locks in at ep40-50.
- The fundamental ESS gap remains: even the champion sits at ~3% ESS, far from typical NPE benchmarks. Capacity has been diagnosed and resolved as one limiter; the next limiters are likely structural (flow family, factorization choice, data scale).

### Recent ablation arc (HEAD-era Step 1-3a)

| Step | Run | Single-knob change | ESS_frac | Verdict |
|------|-----|--------------------|----------|---------|
| 0 (anchor) | v7d0_v5exact | v5-equivalent data regime | 1.42% | Baseline confirmed |
| 1 (data regime) | v7d0_v5exact vs v7d0 | Reverted Sobol+n_fixed+1M+wider prior | — | v7d0 broke; v5 regime is the right anchor |
| 2 (chain-rule) | v7d | + chain_rule + θ_g σ=0.05 | 0.96% | FAIL (-32%) |
| 3a-half (capacity) | v7e_cap_half | + global flow 6× scaling | **2.79%** | **WIN (+96%)** |
| 3a-full (capacity) | v7e_cap | + global flow 16× scaling | 2.85% mean / 40 median | Saturation/regression |

### Open Questions

1. **Is the next limiter the global flow's *dependency-graph expressiveness* (coupling-NSF vs autoregressive)?** v7e_cap_half's coupling-NSF on a 4D posterior with known curved/banana ridges may be limited by the 2/2 split structure. Single-knob test: swap to MAF (cleaner hypothesis test) or autoregressive-NSF (strict superset).
2. **Does v7e_cap_half generalize to N=3+ backends?** This is the original payoff of choosing factorization. Untested.
3. **Is the WN flow now the bottleneck?** With global at 1M params and WN still at 165k, the architecture is lopsided. Single-knob: scale WN flow to ~1M.
4. **Would v7e_cap unlock at 1-2M training samples?** The 2.6M-param overfit was diagnosed as data-bound, not capacity-bound per se.
5. **Multi-pulsar PTA scaling**: per-backend WN factorization was designed for this; still untested.

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **TOA** | Time of Arrival — the fundamental measurement in pulsar timing |
| **PTA** | Pulsar Timing Array — network of pulsars used as GW detectors |
| **NPE** | Neural Posterior Estimation — a form of SBI using normalizing flows |
| **SBI** | Simulation-Based Inference — Bayesian inference without explicit likelihood |
| **NSF** | Neural Spline Flow — normalizing flow using rational-quadratic splines |
| **RoPE** | Rotary Position Embeddings — position encoding via rotation of Q/K pairs |
| **EFAC** | Error Factor — multiplicative white noise scaling |
| **EQUAD** | Error in Quadrature — additive white noise in quadrature with σ |
| **ECORR** | Epoch-Correlated Error — jitter common to all TOAs in one epoch |
| **DM** | Dispersion Measure — chromatic (frequency-dependent) signal delay |
| **Woodbury identity** | Matrix identity for efficient inversion: (A + UCV)⁻¹ |
| **AMP** | Automatic Mixed Precision — float16 compute with float32 accumulation |
| **EMA** | Exponential Moving Average — smoothed statistic for early stopping |
| **ESS** | Effective Sample Size — measure of importance sampling efficiency |
| **Dingo-IS** | Importance sampling correction for amortized posterior proposals |
| **Hellinger distance** | Statistical distance between probability distributions, H ∈ [0,1] |
| **KS statistic** | Kolmogorov-Smirnov statistic — max deviation of CDF from reference |
| **P-P plot** | Probability-Probability plot — calibration diagnostic for posteriors |
| **Sobol sequence** | Low-discrepancy quasi-random sequence for better prior coverage |
