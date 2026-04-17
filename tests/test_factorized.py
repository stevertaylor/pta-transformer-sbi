"""Tests for v5 factorized amortization architecture."""

import numpy as np
import torch
import pytest

from src.schedules import generate_schedule
from src.simulator import simulate_pulsar_factorized
from src.priors import UniformPrior, FactorizedPrior
from src.dataset import PulsarDataset, FixedPulsarDataset
from src.collate import collate_fn
from src.models.model_wrappers import build_model, FactorizedNPEModel
from src.models.tokenization import N_CONTINUOUS_FEATURES


# ---- Fixtures ----


@pytest.fixture
def v5_cfg():
    return {
        "prior": {
            "global": {
                "log10_A_red": [-17, -11],
                "gamma_red": [0.5, 6.5],
                "log10_A_dm": [-17, -11],
                "gamma_dm": [0.5, 6.5],
            },
            "white_noise": {
                "EFAC": [0.1, 10.0],
                "log10_EQUAD": [-8, -5],
                "log10_ECORR": [-8, -5],
            },
        },
        "model": {
            "factorized": True,
            "n_backends_max": 4,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "context_dim": 16,
            "use_rope": True,
            "use_aux_features": True,
            "lstm_hidden": 32,
            "lstm_layers": 1,
            "global_flow_transforms": 2,
            "global_flow_hidden": 32,
            "global_flow_layers": 2,
            "global_flow_bins": 8,
            "wn_flow_transforms": 2,
            "wn_flow_hidden": 32,
            "wn_flow_layers": 2,
            "wn_flow_bins": 8,
        },
    }


@pytest.fixture
def factorized_prior():
    return FactorizedPrior(
        global_bounds={
            "log10_A_red": [-17, -11],
            "gamma_red": [0.5, 6.5],
            "log10_A_dm": [-17, -11],
            "gamma_dm": [0.5, 6.5],
        },
        wn_bounds={
            "EFAC": [0.1, 10.0],
            "log10_EQUAD": [-8, -5],
            "log10_ECORR": [-8, -5],
        },
        n_backends_max=4,
    )


def _make_factorized_batch(B=4, L_max=50, n_feat=N_CONTINUOUS_FEATURES, n_backends_max=4):
    """Create a dummy factorized batch."""
    seq_lens = torch.randint(10, L_max, (B,))
    features = torch.randn(B, L_max, n_feat)
    # Assign random backend IDs (0, 1, or 2)
    backend_id = torch.randint(0, 3, (B, L_max))
    mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i in range(B):
        mask[i, : seq_lens[i]] = True

    theta_global = torch.randn(B, 4)
    theta_wn = torch.randn(B, n_backends_max, 3)
    backend_active = torch.zeros(B, n_backends_max, dtype=torch.bool)
    # Each sample: activate 1-3 backends
    for i in range(B):
        nb = torch.randint(1, 4, (1,)).item()
        backend_active[i, :nb] = True

    return {
        "features": features,
        "backend_id": backend_id,
        "mask": mask,
        "seq_lens": seq_lens,
        "tspan_yr": torch.rand(B) * 10 + 5,
        "theta_global": theta_global,
        "theta_wn": theta_wn,
        "backend_active": backend_active,
    }


# ---- Schedule n_backends property ----


def test_schedule_n_backends():
    rng = np.random.default_rng(42)
    sched = generate_schedule(rng)
    assert sched.n_backends >= 1
    assert sched.n_backends <= 3


# ---- Factorized simulator ----


def test_simulate_factorized_shape():
    rng = np.random.default_rng(100)
    sched = generate_schedule(rng, n_toa_min=80, n_toa_max=200)
    theta_global = np.array([-14.0, 3.0, -15.0, 2.5], dtype=np.float32)
    n_backends = sched.n_backends
    theta_wn = np.array([[1.2, -6.5, -6.0]] * n_backends, dtype=np.float32)
    sim = simulate_pulsar_factorized(theta_global, theta_wn, sched, n_modes=20, rng=rng)
    N = sched.n_toa
    assert sim.residuals.shape == (N,)
    assert sim.theta_global.shape == (4,)
    assert sim.theta_wn.shape == (n_backends, 3)
    assert sim.n_backends == n_backends
    assert sim.theta.shape == (4 + 3 * n_backends,)
    assert sim.F_dm is not None
    assert np.all(np.isfinite(sim.residuals))


def test_simulate_factorized_per_backend_wn():
    """Different EFAC per backend should affect per-backend residual variance."""
    rng_base = np.random.default_rng(200)
    sched = generate_schedule(rng_base, n_toa_min=200, n_toa_max=300)
    n_backends = sched.n_backends
    if n_backends < 2:
        pytest.skip("Need at least 2 backends")

    theta_global = np.array([-17.0, 3.0, -17.0, 3.0], dtype=np.float32)
    # Backend 0: small EFAC, Backend 1: large EFAC
    theta_wn = np.zeros((n_backends, 3), dtype=np.float32)
    theta_wn[:, 0] = 0.5  # small EFAC
    theta_wn[:, 1] = -8.0  # tiny EQUAD
    theta_wn[:, 2] = -8.0  # tiny ECORR
    theta_wn[1, 0] = 5.0  # large EFAC for backend 1

    sim = simulate_pulsar_factorized(
        theta_global,
        theta_wn,
        sched,
        n_modes=20,
        rng=np.random.default_rng(200),
    )
    assert np.all(np.isfinite(sim.residuals))

    # Backend 1 residuals should have more variance
    mask0 = sched.backend_id == 0
    mask1 = sched.backend_id == 1
    if mask0.sum() > 5 and mask1.sum() > 5:
        var0 = np.var(sim.residuals[mask0])
        var1 = np.var(sim.residuals[mask1])
        assert var1 > var0, "Large EFAC backend should have more variance"


# ---- FactorizedPrior ----


def test_factorized_prior_dims(factorized_prior):
    assert factorized_prior.global_dim == 4
    assert factorized_prior.wn_dim == 3
    assert factorized_prior.n_backends_max == 4


def test_factorized_prior_sampling(factorized_prior):
    rng = np.random.default_rng(42)
    g = factorized_prior.sample_global(100, rng=rng)
    assert g.shape == (100, 4)
    w = factorized_prior.sample_wn(50, rng=rng)
    assert w.shape == (50, 3)


def test_factorized_prior_sobol(factorized_prior):
    g = factorized_prior.sample_global_sobol(1000, seed=0)
    assert g.shape == (1000, 4)
    w = factorized_prior.sample_wn_sobol(1000, backend_index=0, seed=0)
    assert w.shape == (1000, 3)
    # Check within bounds
    lo_g, hi_g = factorized_prior.global_prior.bounds_array
    assert (g.numpy() >= lo_g).all() and (g.numpy() <= hi_g).all()
    lo_w, hi_w = factorized_prior.wn_prior.bounds_array
    assert (w.numpy() >= lo_w).all() and (w.numpy() <= hi_w).all()


# ---- FactorizedNPEModel ----


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
def test_factorized_model_forward(v5_cfg, model_type):
    model = build_model(model_type, v5_cfg)
    assert isinstance(model, FactorizedNPEModel)
    batch = _make_factorized_batch()
    loss = model(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
def test_factorized_model_backward(v5_cfg, model_type):
    model = build_model(model_type, v5_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    batch = _make_factorized_batch()
    loss = model(batch)
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss2 = model(batch)
    assert torch.isfinite(loss2)


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
def test_factorized_model_sampling(v5_cfg, model_type):
    model = build_model(model_type, v5_cfg)
    model.eval()
    batch = _make_factorized_batch(B=2)
    global_samples, wn_samples = model.sample_posterior(batch, n_samples=50)
    assert global_samples.shape == (2, 50, 4)
    assert wn_samples.shape == (2, 4, 50, 3)  # n_backends_max=4
    assert torch.all(torch.isfinite(global_samples))
    assert torch.all(torch.isfinite(wn_samples))


def test_factorized_model_sample_flat(v5_cfg):
    model = build_model("transformer", v5_cfg)
    model.eval()
    batch = _make_factorized_batch(B=2)
    flat = model.sample_posterior_flat(batch, n_samples=50)
    assert flat.shape == (2, 50, 4 + 3 * 4)  # 4 global + 3*4 WN
    assert torch.all(torch.isfinite(flat))


@pytest.mark.parametrize("model_type", ["transformer", "lstm"])
def test_factorized_variable_length(v5_cfg, model_type):
    model = build_model(model_type, v5_cfg)
    batch1 = _make_factorized_batch(B=2, L_max=30)
    batch2 = _make_factorized_batch(B=2, L_max=80)
    loss1 = model(batch1)
    loss2 = model(batch2)
    assert torch.isfinite(loss1) and torch.isfinite(loss2)


def test_factorized_no_active_backends(v5_cfg):
    """Loss should still be finite when no WN backends are active."""
    model = build_model("transformer", v5_cfg)
    batch = _make_factorized_batch(B=2)
    batch["backend_active"][:] = False
    loss = model(batch)
    assert torch.isfinite(loss)


# ---- Dataset + Collate integration ----


def test_factorized_dataset(factorized_prior):
    data_cfg = {
        "n_toa_min": 60,
        "n_toa_max": 200,
        "tspan_min_yr": 5.0,
        "tspan_max_yr": 10.0,
        "n_fourier_modes": 20,
    }
    ds = PulsarDataset(
        n_samples=10,
        prior=factorized_prior,
        data_cfg=data_cfg,
        seed=42,
        factorized=True,
    )
    item = ds[0]
    assert "theta_global" in item
    assert "theta_wn" in item
    assert "backend_active" in item
    assert item["theta_global"].shape == (4,)
    assert item["theta_wn"].shape == (4, 3)  # n_backends_max=4
    assert "theta" not in item


def test_factorized_dataset_sobol(factorized_prior):
    data_cfg = {
        "n_toa_min": 60,
        "n_toa_max": 200,
        "tspan_min_yr": 5.0,
        "tspan_max_yr": 10.0,
        "n_fourier_modes": 20,
    }
    ds = PulsarDataset(
        n_samples=100,
        prior=factorized_prior,
        data_cfg=data_cfg,
        seed=42,
        use_sobol=True,
        factorized=True,
    )
    item = ds[0]
    assert "theta_global" in item
    assert item["theta_global"].shape == (4,)


def test_factorized_collate(factorized_prior):
    data_cfg = {
        "n_toa_min": 60,
        "n_toa_max": 200,
        "tspan_min_yr": 5.0,
        "tspan_max_yr": 10.0,
        "n_fourier_modes": 20,
    }
    ds = PulsarDataset(
        n_samples=8,
        prior=factorized_prior,
        data_cfg=data_cfg,
        seed=42,
        factorized=True,
    )
    items = [ds[i] for i in range(8)]
    batch = collate_fn(items)
    assert "theta_global" in batch
    assert "theta_wn" in batch
    assert "backend_active" in batch
    assert batch["theta_global"].shape[0] == 8
    assert batch["theta_wn"].shape == (8, 4, 3)
    assert batch["backend_active"].shape == (8, 4)
    assert batch["mask"].shape[0] == 8


def test_factorized_fixed_dataset(factorized_prior):
    data_cfg = {
        "n_toa_min": 60,
        "n_toa_max": 200,
        "tspan_min_yr": 5.0,
        "tspan_max_yr": 10.0,
        "n_fourier_modes": 20,
    }
    ds = FixedPulsarDataset(
        n_samples=3,
        prior=factorized_prior,
        data_cfg=data_cfg,
        seed=42,
        factorized=True,
    )
    item = ds[0]
    assert "theta_global" in item
    assert "theta_wn" in item
    assert "sim" in item
    assert item["sim"].theta_global is not None


# ---- End-to-end smoke training step ----


def test_e2e_factorized_train_step(v5_cfg, factorized_prior):
    """End-to-end: dataset → collate → model forward/backward."""
    data_cfg = {
        "n_toa_min": 60,
        "n_toa_max": 200,
        "tspan_min_yr": 5.0,
        "tspan_max_yr": 10.0,
        "n_fourier_modes": 20,
    }
    ds = PulsarDataset(
        n_samples=16,
        prior=factorized_prior,
        data_cfg=data_cfg,
        seed=42,
        factorized=True,
    )
    items = [ds[i] for i in range(8)]
    batch = collate_fn(items)

    model = build_model("transformer", v5_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    loss = model(batch)
    assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss2 = model(batch)
    assert torch.isfinite(loss2), f"Loss after step not finite: {loss2.item()}"


# ---- Backward compat: standard model still works ----


def test_standard_model_still_works():
    """Verify non-factorized configs still produce correct models."""
    cfg = {
        "prior": {
            "log10_A_red": [-17, -11],
            "gamma_red": [0.5, 6.5],
        },
        "model": {
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "context_dim": 16,
            "lstm_hidden": 32,
            "lstm_layers": 1,
            "flow_transforms": 2,
            "flow_hidden": 32,
        },
    }
    model = build_model("transformer", cfg)
    assert not isinstance(model, FactorizedNPEModel)
    seq_lens = torch.randint(10, 50, (4,))
    mask = torch.zeros(4, 50, dtype=torch.bool)
    for i in range(4):
        mask[i, : seq_lens[i]] = True
    batch = {
        "theta": torch.randn(4, 2),
        "features": torch.randn(4, 50, N_CONTINUOUS_FEATURES),
        "backend_id": torch.zeros(4, 50, dtype=torch.long),
        "mask": mask,
        "seq_lens": seq_lens,
        "tspan_yr": torch.rand(4) * 10 + 5,
    }
    loss = model(batch)
    assert torch.isfinite(loss)


# ---- Autoregressive wiring checks (v7c) ----


def test_factorized_wn_context_includes_theta_g(v5_cfg):
    """Autoregressive model stores theta_g_dim and can forward/backward."""
    model = build_model("transformer", v5_cfg)
    assert model.theta_g_dim == 4
    batch = _make_factorized_batch(B=2)
    loss = model(batch)
    assert torch.isfinite(loss)


def test_factorized_wn_conditional_on_theta_g(v5_cfg):
    """Changing teacher-forced θ_g must change the WN log-prob term."""
    torch.manual_seed(0)
    model = build_model("transformer", v5_cfg)
    model.eval()
    batch = _make_factorized_batch(B=2)
    batch_a = dict(batch)
    batch_b = dict(batch)
    batch_b["theta_global"] = batch["theta_global"] + 5.0
    with torch.no_grad():
        model(batch_a)
        lw_a = model._last_wn_loss
        model(batch_b)
        lw_b = model._last_wn_loss
    assert abs(lw_a - lw_b) > 1e-4, (
        f"WN loss unchanged under θ_g change: {lw_a} vs {lw_b} — "
        "autoregressive conditioning is not wired in."
    )


def test_factorized_sample_joint_alignment(v5_cfg):
    """Joint samples return paired (θ_g_i, θ_wn_b,i) draws."""
    model = build_model("transformer", v5_cfg)
    model.eval()
    batch = _make_factorized_batch(B=1)
    gs, ws = model.sample_posterior(batch, n_samples=128)
    assert gs.shape == (1, 128, 4)
    assert ws.shape == (1, 4, 128, 3)
    assert torch.all(torch.isfinite(gs)) and torch.all(torch.isfinite(ws))


def test_v7c_config_builds():
    """The v7c config builds a valid autoregressive factorized model."""
    import yaml
    import os

    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "transformer_v7c.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model("transformer", cfg)
    assert isinstance(model, FactorizedNPEModel)
    assert model.theta_g_dim == 4
    batch = _make_factorized_batch(B=2, n_backends_max=cfg["model"]["n_backends_max"])
    loss = model(batch)
    assert torch.isfinite(loss)


def test_v7c2_config_builds_with_noise():
    """v7c2 config wires teacher-signal noise into the model."""
    import yaml
    import os

    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "transformer_v7c2.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model("transformer", cfg)
    assert isinstance(model, FactorizedNPEModel)
    assert model.theta_g_noise_std == pytest.approx(cfg["model"]["theta_g_noise"])
    assert model.theta_g_noise_std > 0.0
    batch = _make_factorized_batch(B=2, n_backends_max=cfg["model"]["n_backends_max"])
    loss = model(batch)
    assert torch.isfinite(loss)


def test_theta_g_noise_train_only(v5_cfg):
    """Noise perturbs WN loss in training mode and is a no-op at eval."""
    cfg = {**v5_cfg}
    cfg["model"] = {**v5_cfg["model"], "theta_g_noise": 0.1}
    torch.manual_seed(0)
    model = build_model("transformer", cfg)
    assert model.theta_g_noise_std == pytest.approx(0.1)
    batch = _make_factorized_batch(B=4)

    # Eval mode: two forward passes must be identical (noise is disabled).
    model.eval()
    torch.manual_seed(1)
    with torch.no_grad():
        model(batch)
        lw_eval_a = model._last_wn_loss
        torch.manual_seed(2)
        model(batch)
        lw_eval_b = model._last_wn_loss
    assert lw_eval_a == lw_eval_b, (
        f"Eval-mode noise must be a no-op: {lw_eval_a} vs {lw_eval_b}"
    )

    # Train mode: two passes with different seeds draw different noise and
    # must produce different WN losses on the same batch.
    model.train()
    torch.manual_seed(1)
    loss1 = model(batch)
    lw_train_a = model._last_wn_loss
    torch.manual_seed(2)
    loss2 = model(batch)
    lw_train_b = model._last_wn_loss
    assert torch.isfinite(loss1) and torch.isfinite(loss2)
    assert abs(lw_train_a - lw_train_b) > 1e-6, (
        f"Train-mode noise must perturb WN loss: {lw_train_a} vs {lw_train_b}"
    )


def test_theta_g_noise_zero_is_deterministic(v5_cfg):
    """With noise=0 (default), train and eval give identical WN losses."""
    torch.manual_seed(0)
    model = build_model("transformer", v5_cfg)
    assert model.theta_g_noise_std == 0.0
    batch = _make_factorized_batch(B=2)
    model.train()
    with torch.no_grad():
        model(batch)
        lw_train = model._last_wn_loss
    model.eval()
    with torch.no_grad():
        model(batch)
        lw_eval = model._last_wn_loss
    assert lw_train == lw_eval


# ---- chain_rule flag (v5 independence vs v7c chain-rule) ----


def test_chain_rule_default_true(v5_cfg):
    """Default builds keep v7c chain-rule behavior (backward compat)."""
    model = build_model("transformer", v5_cfg)
    assert model.chain_rule is True


def test_chain_rule_false_wn_flow_is_smaller(v5_cfg):
    """chain_rule=False drops θ_g from the WN flow context (strictly fewer params)."""
    cfg_cr = {**v5_cfg, "model": {**v5_cfg["model"], "chain_rule": True}}
    cfg_ind = {**v5_cfg, "model": {**v5_cfg["model"], "chain_rule": False}}
    m_cr = build_model("transformer", cfg_cr)
    m_ind = build_model("transformer", cfg_ind)
    n_cr = sum(p.numel() for p in m_cr.wn_flow.parameters())
    n_ind = sum(p.numel() for p in m_ind.wn_flow.parameters())
    assert n_ind < n_cr, (
        f"Independence WN flow should have fewer params than chain-rule: {n_ind} vs {n_cr}"
    )


def test_chain_rule_false_forward_and_sample(v5_cfg):
    """chain_rule=False: forward + sample_posterior run and return finite values."""
    cfg = {**v5_cfg, "model": {**v5_cfg["model"], "chain_rule": False}}
    model = build_model("transformer", cfg)
    batch = _make_factorized_batch(B=2)
    loss = model(batch)
    assert torch.isfinite(loss)
    model.eval()
    gs, ws = model.sample_posterior(batch, n_samples=32)
    assert gs.shape == (2, 32, 4)
    assert ws.shape == (2, 4, 32, 3)
    assert torch.all(torch.isfinite(gs)) and torch.all(torch.isfinite(ws))


def test_chain_rule_false_wn_loss_independent_of_theta_g(v5_cfg):
    """Under independence, WN loss must NOT change when θ_g is perturbed."""
    cfg = {**v5_cfg, "model": {**v5_cfg["model"], "chain_rule": False}}
    torch.manual_seed(0)
    model = build_model("transformer", cfg)
    model.eval()
    batch = _make_factorized_batch(B=2)
    batch_b = dict(batch)
    batch_b["theta_global"] = batch["theta_global"] + 5.0
    with torch.no_grad():
        model(batch)
        lw_a = model._last_wn_loss
        model(batch_b)
        lw_b = model._last_wn_loss
    assert lw_a == lw_b, (
        f"Independence WN loss must be invariant to θ_g: {lw_a} vs {lw_b}"
    )
