# Plan: PTA Transformer SBI — Performance Assessment & Remediation (Revised)

## TL;DR — Correction of Previous Diagnosis

The previous version of this plan blamed the v4c→v5 regression primarily on simulator physics (missing polynomial subtraction, only 3 frequencies, red/DM conflation). **That diagnosis was incorrect.** v4c achieved excellent results (H=0.005, KS=0.109) using the _exact same simulator, the same 3 frequencies, and no polynomial subtraction_. The physics hasn't changed between versions — the architecture has.

The v5 regression (H=0.018, KS=0.176, val loss +0.70 vs v4c's −1.55) is overwhelmingly caused by **compounding architectural regressions** introduced alongside the factorization: crushed flow capacity, aggressive context bottlenecks, damped WN loss, and added regularization noise. Fixing the simulator first — as the original plan proposed — would be solving the wrong problem.

---

## What Actually Changed: v4c → v5

| Setting | v4c (works) | v5 (broken) | Impact |
|---|---|---|---|
| **Flow architecture** | Single 7D: 10 transforms × 256h × 3L × 16 bins | Two flows: 6×128×2×8 each | **~75% capacity cut** |
| **Context → flow** | 132D direct to flow | 132D→48D bottleneck (global), 132D→32D (WN) | **64–76% info loss** |
| **WN loss weight** | 1.0 (implicit, joint) | 0.3 | **WN starved of gradient** |
| **Context dropout** | None | 0.2 | Extra noise on bottlenecked signal |
| **EFAC prior** | [0.1, 10.0] | [0.5, 2.0] | 80% range narrowed |
| **A_red prior** | [-17, -11] | [-18, -11] | Wider by 1 decade |
| **γ prior** | [0.5, 6.5] | [1.0, 7.0] | Shifted |
| **Encoder dropout** | 0.05 | 0.1 | 2× encoder noise |
| **Sobol sampling** | Yes | No | Worse prior coverage |
| **Training masking** | None | 50% severity | Distribution shift |

Multiple capacity/information cuts applied simultaneously, with no ablation to isolate effects.

## The Telling Asymmetry

v5's **white-noise calibration is actually BETTER than v4c's**:

| Parameter | v4c KS | v5 KS | Winner |
|---|---|---|---|
| EFAC | 0.168 | **0.053** | v5 by 3× |
| log10_EQUAD | 0.166 | **0.070** | v5 by 2× |
| log10_ECORR | 0.083 | 0.119 | v4c (marginal) |

But v5's **global parameter inference collapsed**:

| Parameter | v4c KS | v5 KS | Regression |
|---|---|---|---|
| log10_A_red | **0.072** | 0.261 | 3.6× worse |
| gamma_red | 0.109 | 0.107 | ~same |
| log10_A_dm | **0.119** | 0.240 | 2× worse |
| gamma_dm | **0.049** | 0.094 | 1.9× worse |

This means **the factorized WN flow works well**, but the global flow was crippled. The shared encoder must serve two heads, and the global flow got the worst of it: smaller flow, tighter bottleneck (relative to problem complexity), and lost access to the joint parameter correlations that the 7D flow handled automatically.

---

## Detailed Assessment by Question

### 1. Flow Capacity — ROOT CAUSE OF REGRESSION

v4c's single 7D flow: 10 NSF transforms with 256-wide 3-layer conditioner nets and 16 spline bins. This is a powerful density estimator.

v5's global 4D flow: 6 transforms, 128-wide, 2-layer, 8 bins — fed through a 48D bottleneck. Despite having fewer output dimensions (4 vs 7), this flow is dramatically less expressive. The global posterior for (A_red, γ_red, A_dm, γ_dm) has strong non-linear correlations (power-law degeneracies, red/DM coupling), and 6×128×2 with 8 bins simply can't capture them.

The val loss being **positive** (+0.70) means the flow is barely better than the prior for many examples. This is a capacity/optimization failure, not a physics failure.

**Fix**: Increase global flow to at least 8–10 transforms, 192–256 hidden, 2–3 layers, 12–16 bins. The 4D output is simpler than 7D, but the conditioning relationship is harder (the flow must extract all red/DM information from a compressed context rather than a joint context).

### 2. Context Bottlenecks — ROOT CAUSE OF REGRESSION

The global context path: encoder→128D→concat aux→132D→Linear→ReLU→Linear→**48D**→flow. This is a 64% dimensionality reduction before the flow even sees the data. The WN path is even worse: 132D→**32D**.

These bottlenecks were added to "reduce overfitting," but the positive val loss shows the model is **underfitting**, not overfitting. The bottlenecks are killing signal, not noise.

**Fix**: Remove bottlenecks entirely (set `global_context_proj_dim: null`, `wn_context_proj_dim: null`) or widen to ≥96D. The model needs more information flow, not less.

### 3. WN Loss Weight — CONTRIBUTING CAUSE

With `wn_loss_weight: 0.3`, the encoder receives only 23% of its gradient signal from the WN head (0.3 / 1.3). The WN flow itself still works well (see KS scores above), suggesting it's efficient at extracting WN information. But the low weight means the encoder is optimized primarily for the global flow — which then can't use WN information because it has no access to it (one-directional conditioning).

**Fix**: Set `wn_loss_weight` to 0.7–1.0. The WN flow is already learning well; giving it more gradient weight won't hurt and will force the encoder to produce representations useful for both tasks.

### 4. Missing Global←WN Conditioning — REAL BUT SECONDARY

The asymmetric conditioning (WN sees global context, but global doesn't see WN) matters because red-noise amplitude estimation requires knowing the white-noise level to resolve the total-power degeneracy.

**However**: v4c proves this isn't fundamentally necessary — a joint 7D flow learns these correlations implicitly. The factorized architecture _creates the need_ for explicit cross-conditioning that the joint approach handled automatically.

**Fix**: If keeping the factorized architecture, add stop-gradient WN encoder context to the global flow input. But this is secondary to fixing flow capacity and bottlenecks — even with perfect WN information, a 6×128×2 flow through a 48D bottleneck can't represent the posterior well.

### 5. Polynomial Timing-Model Subtraction — IMPORTANT FOR REALISM, NOT FOR REGRESSION

**Status**: Not implemented. The simulator produces raw residuals with no timing-model subtraction.

**Revised assessment**: This is a real gap for eventual deployment on real data, but **it is not causing the v4c→v5 regression** (v4c succeeded without it). The network can learn to use all Fourier modes — it just learns a slightly different mapping than what real post-fit data would present.

**When to fix**: After the architecture is performing at v4c-level parity on the current simulator. Then add polynomial subtraction + G-matrix projection to make the training distribution realistic. This prevents a re-training cost later but doesn't unblock the current bottleneck.

### 6. Radio Frequency Coverage — IMPORTANT FOR REALISM, NOT FOR REGRESSION

**Status**: 3 frequencies at [820, 1400, 2300] MHz.

**Revised assessment**: v4c achieves KS(A_dm)=0.119 and KS(γ_dm)=0.049 with these same 3 frequencies. The chromatic leverage is limited but _sufficient_ for the current problem size. More frequencies will help — especially for scaling to harder problems — but are not the cause of the current failure.

**When to fix**: Alongside polynomial subtraction, as part of a "simulator realism" phase after architecture is fixed.

### 7. Red/DM Amplitude Conflation — REAL, BUT ARCHITECTURE-AMPLIFIED

**Per-parameter v5 KS scores at mask=0.0**:

| Parameter | v5 KS | v4c KS | Delta |
|---|---|---|---|
| `log10_A_red` | **0.261** | 0.072 | 3.6× worse |
| `log10_A_dm` | **0.240** | 0.119 | 2.0× worse |
| `gamma_red` | 0.107 | 0.109 | ~same |
| `gamma_dm` | 0.094 | 0.049 | 1.9× worse |
| `EFAC` | 0.053 | 0.168 | 3.2× **better** |
| `log10_EQUAD` | 0.070 | 0.166 | 2.4× **better** |
| `log10_ECORR` | 0.119 | 0.083 | 1.4× worse |

The red/DM conflation is a **real physical degeneracy** with only 3 frequencies, but v4c shows it's manageable with sufficient model capacity. v5's architecture amplifies this degeneracy into a failure because the 48D-bottlenecked 6×128×2 flow can't represent the correlated (A_red, A_dm) posterior surface.

### 8. Per-Epoch vs Per-TOA Tokens — GOOD IDEA, LOW PRIORITY

Per-epoch tokens would reduce sequence length ~2×, enable explicit chromatic differencing, and improve ECORR estimation. But v4c achieves good results with per-TOA tokens at 80–400 TOAs, so this isn't blocking current performance.

**Recommended epoch-token features** (for when this is implemented):
- Mean time of epoch; duration gap to previous epoch
- Per-frequency: (residual, sigma) at each observed frequency → structured vector
- Epoch-level derived: DM-like difference estimator (r_low − r_high) / (1/f_low² − 1/f_high²)
- Number of TOAs in epoch; which backends observed
- Within-epoch scatter (ECORR diagnostic)

**When to implement**: After architecture fixes, as a bridge to scaling beyond ~1000 TOAs. It becomes essential at 10k+ TOAs where O(n²) attention is prohibitive.

### 9. Training Data Volume — NOT THE BOTTLENECK

500k samples with epoch reseeding is adequate. The positive val loss is an underfitting signal, not a data-hunger signal.

### 10. Scaling to 10k–100k TOAs — PREMATURE

Fix the architecture first. Scaling amplifies every problem.

---

## Failure Modes the Original Plan Missed

### A. No Ablation Between Architecture Changes

v4c→v5 changed at least 8 things simultaneously (factorization, flow capacity, bottlenecks, loss weight, dropout, priors, masking, Sobol). With no ablation study, it's impossible to know which change helped (the factorization for WN likely helped), which was neutral, and which was catastrophic (bottlenecks + capacity cuts likely were). Every future version should change one thing at a time.

### B. The Model Is Underfitting, Not Overfitting

The context bottlenecks, context dropout, and small flow capacity were all motivated by fear of overfitting. But the positive val loss (+0.70) and the ~1.6 nat train-val gap prove the opposite: the model can't even fit the training distribution well. The regularization should be stripped back until underfitting is resolved.

### C. Prior Range Changes Compound the Problem

EFAC was narrowed from [0.1, 10.0] to [0.5, 2.0] — a 20× reduction in range that actually _helps_ the WN flow (less to cover), but A_red was widened from [-17, -11] to [-18, -11], making the global flow's job harder. The spectral index ranges also shifted: γ from [0.5, 6.5] to [1.0, 7.0]. These prior changes interact with every other change in ways that are hard to predict without ablations.

### D. Sobol → Random Sampling

v4c used Sobol (quasi-random) sampling for parameter coverage. v5 switched to random. For a 4D parameter space, Sobol provides significantly better coverage — particularly important for the extremes of the prior where the model is most likely to fail. This could be contributing to poor performance in degenerate corners of parameter space.

### E. Training Masking Distribution Shift

v5 adds 50% masking severity during training but only 15% during validation and 0% during one of the eval masking levels. The encoder sees heavily masked data during training but clean data at test time. While this was intended to improve robustness, it may cause an encoder distribution shift that degrades clean-data performance. v4c had no masking at all and achieved better results.

---

## Revised Phased Remediation

### Phase 1: Fix the Architecture (addresses root causes 1–3)

The goal is to achieve v4c-level performance with the factorized architecture. Changes should be made **one at a time**, with smoke-test evaluation after each:

1. **Remove context bottlenecks**: Set `global_context_proj_dim: null`, `wn_context_proj_dim: null`. The model is underfitting — it needs more information, not less.
2. **Increase global flow capacity**: 10 transforms, 256 hidden, 3 layers, 16 bins (match v4c's flow proportionally).
3. **Increase WN flow capacity**: 8 transforms, 192 hidden, 2 layers, 12 bins.
4. **Set `wn_loss_weight: 1.0`**: Let both heads train at full strength.
5. **Remove context dropout**: Set to 0.0. The model is underfitting.
6. **Revert Sobol sampling**: Set `use_sobol: true` for better prior coverage.
7. **Verification**: Val loss should go clearly negative (target: < −1.0). Hellinger ≤ 0.008, global KS < 0.12, WN KS should remain good.

### Phase 2: Ablate and Optimize (after Phase 1 achieves parity)

8. Re-introduce regularization _one piece at a time_ with evaluation after each, only if overfitting is observed:
   - Context dropout (try 0.05, then 0.1)
   - Mild context projection (try 96D, never below 64D)
   - Training masking (try 0.2 severity, then 0.3)
9. Ablate prior ranges: test whether [-17, -11] vs [-18, -11] matters; whether EFAC [0.5, 2.0] vs [0.1, 10.0] matters for the WN flow.
10. Add global←WN conditioning (stop-gradient WN encoder context concatenated to global flow input), evaluate whether it helps beyond what restored capacity provides.
11. **Verification**: Each change in isolation, compared to Phase 1 baseline. Keep only changes that improve or maintain metrics.

### Phase 3: Simulator Realism (addresses long-term validity)

These are not urgent for the current regression but are essential before applying the model to real data:

12. Add quadratic polynomial fit-and-removal in the simulator. Project the Fourier design matrix onto the timing-model complement (G-matrix) so the exact posterior remains consistent.
13. Expand frequency coverage to 5–8 channels spanning ~300–3000 MHz (e.g., [327, 430, 820, 1150, 1400, 1900, 2300]).
14. Add a WN summary token to the global flow context (empirical whitened-residual statistics per frequency band).
15. **Verification**: Exact posteriors show improved DM/red decorrelation. A_red and A_dm KS improve further. Model performance should not degrade (it should slightly improve due to more realistic data distribution).

### Phase 4: Epoch Tokenization and Scaling

16. Implement per-epoch token aggregation (sub-epoch MLP/attention → single epoch embedding).
17. Include within-epoch chromatic diagnostics (DM estimator, scatter) as epoch features.
18. Scale TOA range to 400–5000 with epoch tokens.
19. If parity achieved, push to 10k+ TOAs with hierarchical or sparse attention.
20. **Verification**: Sequence length ~50% shorter at same TOA count. KS < 0.10 for all parameters.

---

## Key Files to Modify

**Phase 1 (architecture fixes — modify config, potentially nothing in model code):**
- `configs/transformer_v6.yaml` — new config with fixed capacity, no bottlenecks, full loss weight
- `configs/smoke_v6.yaml` — corresponding smoke test

**Phase 2 (ablation — small model code changes):**
- `src/models/model_wrappers.py` — add global←WN conditioning path if Phase 2 shows it helps

**Phase 3 (simulator realism):**
- `src/simulator.py` — add polynomial subtraction in `simulate_pulsar()` and `simulate_pulsar_factorized()`
- `src/schedules.py` — expand `freq_choices` to 5–8 bands
- `src/exact_posterior.py` — update for polynomial-projected Fourier basis

**Phase 4 (scaling):**
- `src/models/tokenization.py` — epoch-level aggregation option
- `src/models/transformer_encoder.py` — sub-epoch aggregation layer
- `src/dataset.py`, `src/collate.py` — plumb epoch-token option

---

## What the Original Plan Got Right

- The asymmetric WN→global conditioning is a real issue (points 4/5)
- Epoch tokens are the right approach for scaling (point 8)
- Scaling to 10k+ TOAs is premature (point 3/10)
- Data volume is not the bottleneck (point 9)
- Polynomial subtraction and frequency expansion are genuinely needed for realism (points 5/6)

## What the Original Plan Got Wrong

- **Root cause**: The regression was attributed to simulator physics. The actual root cause is architectural: crushed flow capacity, aggressive bottlenecks, damped loss weight, and excess regularization — all introduced alongside the factorization.
- **Phasing**: "Fix simulator physics first" is backwards. The model can't even fit the current training distribution (positive val loss). Improving the data won't help until the model can learn from it.
- **Red/DM conflation**: Attributed to insufficient frequencies (3). But v4c achieves A_dm KS=0.119 with the same 3 frequencies. The conflation in v5 is architecture-amplified, not physics-limited.
- **Missing the underfitting signal**: The positive val loss (+0.70) means the model is capacity-starved, yet the original plan didn't mention flow capacity at all. Every regularization mechanism (bottlenecks, context dropout, masking, reduced flow size) was making underfitting worse.
- **No ablation discipline**: The original plan proposed more simultaneous changes. Each phase should test one change at a time.
