# Tutorial Notebooks

A five-part series walking through the full SBI pipeline for pulsar timing red-noise inference — from synthetic data generation through to evaluation.

## Notebook Index

| # | Notebook | Topics |
|---|---------|--------|
| 1 | [Synthetic Data Generation](01_synthetic_data.ipynb) | Observing schedules, priors (IID vs Sobol), Fourier-basis red noise, power-law spectrum (enterprise convention), sensitivity across the prior volume |
| 2 | [The Data Pipeline](02_data_pipeline.ipynb) | Tokenization (6 features + signed-log transform), masking augmentations (season dropout, end truncation, cadence thinning), `PulsarDataset` / `FixedPulsarDataset`, collation and padding |
| 3 | [Model Architecture](03_model_architecture.ipynb) | Token MLP, rotary position embeddings (RoPE), pre-norm transformer blocks, CLS-query attention pooling, auxiliary features, neural spline flow (NSF), `NPEModel` wrapper, LSTM comparison |
| 4 | [Training](04_training.ipynb) | YAML configuration, NPE loss function, learning rate scheduling (warmup + cosine), automatic mixed precision, gradient clipping, early stopping, checkpointing |
| 5 | [Evaluation](05_evaluation.ipynb) | Exact Bayesian posterior (Woodbury identity), Hellinger distance, P-P calibration plots, KS statistics, point estimate error, robustness under structured masking |

## Prerequisites

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

Run notebooks from the `tutorials/` directory so that the relative import path (`../`) resolves correctly.

## See Also

- [Overview tutorial](../tutorial_sbi_framework.ipynb) — single-notebook walkthrough of the entire pipeline
- [README](../README.md) — project overview, architecture diagram, and quick-start commands
