# Tutorial Notebooks

A five-part series walking through the full SBI pipeline for pulsar timing red-noise inference — from synthetic data generation through to evaluation.

## Notebook Index

| # | Notebook | Topics |
|---|---------|--------|
| 1 | [Synthetic Data Generation](01_synthetic_data.ipynb) | Observing schedules, `FactorizedPrior` (4D global + 3D WN per backend), Fourier-basis red/DM noise, power-law spectrum (enterprise convention), sensitivity across the prior volume |
| 2 | [The Data Pipeline](02_data_pipeline.ipynb) | Tokenization (6 features + signed-log transform), masking augmentations (season dropout, end truncation, cadence thinning), factorized `PulsarDataset` with epoch reseeding, collation and padding |
| 3 | [Model Architecture](03_model_architecture.ipynb) | Token MLP, RoPE, pre-norm transformer blocks, `BackendQueryPooling`, WN context bottleneck, context dropout, `FactorizedNPEModel`, dual NSF flows (4D global + 3D WN), LSTM comparison |
| 4 | [Training](04_training.ipynb) | v5 YAML config, factorized NPE loss, `wn_loss_weight`, LR scheduling (warmup + cosine), per-group weight decay for flows, gradient clipping, epoch reseeding, early stopping |
| 5 | [Evaluation](05_evaluation.ipynb) | Exact Bayesian posterior (Woodbury identity, red-noise marginal), Hellinger distance, P-P calibration, KS statistics, point estimate error, robustness under structured masking |

## Prerequisites

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Run notebooks from the `tutorials/` directory so that the relative import path (`../`) resolves correctly.

## See Also

- [Overview tutorial](../tutorial_sbi_framework.ipynb) — single-notebook walkthrough of the entire pipeline
- [README](../README.md) — project overview, architecture diagram, and quick-start commands
