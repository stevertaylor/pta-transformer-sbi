"""Model wrappers: encoder + posterior flow, with a unified API."""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoderModel
from .lstm_encoder import LSTMEncoderModel
from .posterior_flow import PosteriorFlow
from .tokenization import N_CONTINUOUS_FEATURES

N_AUX_FEATURES = 4  # log_n_toa, log_tspan, mean_log_sigma, std_log_sigma
N_BACKEND_AUX_FEATURES = 3  # log_n_toa_b, mean_log_sigma_b, std_log_sigma_b


class NPEModel(nn.Module):
    """Encoder + posterior flow for amortized neural posterior estimation.

    The encoder maps (features, backend_id, mask) → context.
    The flow maps (theta, context) → log_prob.
    When use_aux=True, 4 summary statistics are concatenated to the context.

    Theta is normalized to approximately [-1, 1] before being passed to the
    flow so that values stay well within the NSF spline domain.
    """

    def __init__(
        self,
        encoder: nn.Module,
        flow: PosteriorFlow,
        use_aux: bool = False,
        theta_mean: torch.Tensor | None = None,
        theta_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.use_aux = use_aux
        # Theta normalization buffers (set from prior bounds in build_model)
        theta_dim = flow.flow.features if hasattr(flow.flow, "features") else 2
        if theta_mean is None:
            theta_mean = torch.zeros(theta_dim)
        if theta_std is None:
            theta_std = torch.ones(theta_dim)
        self.register_buffer("theta_mean", theta_mean)
        self.register_buffer("theta_std", theta_std)

    def _normalize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.theta_mean) / self.theta_std

    def _denormalize_theta(self, theta_norm: torch.Tensor) -> torch.Tensor:
        return theta_norm * self.theta_std + self.theta_mean

    def _compute_aux(self, batch: dict) -> torch.Tensor:
        """Compute auxiliary summary features from the batch. Returns (B, 4)."""
        mask = batch["mask"]  # (B, L)
        n_toa = mask.sum(1).float()  # (B,)
        log_sigma = batch["features"][:, :, 3]  # (B, L)  — log_sigma channel
        mask_f = mask.float()
        n = mask_f.sum(1).clamp(min=1)
        mean_ls = (log_sigma * mask_f).sum(1) / n
        var_ls = ((log_sigma - mean_ls.unsqueeze(1)) ** 2 * mask_f).sum(1) / n
        std_ls = (var_ls + 1e-8).sqrt()
        tspan = batch["tspan_yr"]  # (B,)
        return torch.stack(
            [
                torch.log(n_toa.clamp(min=1)),
                torch.log(tspan.clamp(min=1e-3)),
                mean_ls,
                std_ls,
            ],
            dim=-1,
        )

    def _get_flow_context(self, batch: dict) -> torch.Tensor:
        context = self.encoder(batch["features"], batch["backend_id"], batch["mask"])
        if self.use_aux:
            aux = self._compute_aux(batch)
            context = torch.cat([context, aux], dim=-1)
        return context

    def forward(self, batch: dict) -> torch.Tensor:
        """Compute negative log-prob loss for training.

        Returns scalar loss = -mean log q(theta | x).
        """
        context = self._get_flow_context(batch)
        theta_norm = self._normalize_theta(batch["theta"])
        # Force flow to float32 even under AMP autocast — spline flows
        # are numerically unstable in float16.
        with torch.autocast(context.device.type, enabled=False):
            log_prob = self.flow.log_prob(theta_norm.float(), context.float())
        return -log_prob.mean()

    def get_context(self, batch: dict) -> torch.Tensor:
        return self._get_flow_context(batch)

    @torch.no_grad()
    def sample_posterior(self, batch: dict, n_samples: int = 1000) -> torch.Tensor:
        """Sample from learned posterior. Returns (B, n_samples, theta_dim)."""
        context = self._get_flow_context(batch)
        samples_norm = self.flow.sample(context, n_samples)
        return self._denormalize_theta(samples_norm)

    @torch.no_grad()
    def log_prob_on_grid(self, batch: dict, grid_points: torch.Tensor) -> torch.Tensor:
        """Evaluate log q(theta | x) on a grid.

        Parameters
        ----------
        batch : dict with a single example (B=1)
        grid_points : (G, theta_dim) grid of theta values

        Returns
        -------
        log_probs : (G,) — in the *original* theta space (includes Jacobian correction)
        """
        context = self._get_flow_context(batch)  # (1, context_dim)
        context_expanded = context.expand(len(grid_points), -1)  # (G, context_dim)
        grid_norm = self._normalize_theta(grid_points)
        log_probs_norm = self.flow.log_prob(grid_norm, context_expanded)
        # Jacobian correction: log|det(d theta_norm / d theta)| = -sum(log(std))
        log_probs = log_probs_norm - self.theta_std.log().sum()
        return log_probs


def build_model(model_type: str, cfg: dict) -> nn.Module:
    """Factory to create NPEModel or FactorizedNPEModel from config."""
    mcfg = cfg["model"]
    factorized = mcfg.get("factorized", False)

    if factorized:
        return _build_factorized_model(model_type, cfg)
    return _build_standard_model(model_type, cfg)


def _build_standard_model(model_type: str, cfg: dict) -> NPEModel:
    """Build standard (non-factorized) NPEModel."""
    mcfg = cfg["model"]
    context_dim = mcfg["context_dim"]
    use_aux = mcfg.get("use_aux_features", False)
    flow_context_dim = context_dim + (N_AUX_FEATURES if use_aux else 0)

    # Compute theta dimension and normalization from prior bounds
    prior_cfg = cfg.get("prior", {})
    param_names = list(prior_cfg.keys())
    theta_dim = len(param_names)
    lo = torch.tensor([prior_cfg[k][0] for k in param_names], dtype=torch.float32)
    hi = torch.tensor([prior_cfg[k][1] for k in param_names], dtype=torch.float32)
    theta_mean = (lo + hi) / 2
    theta_std = (hi - lo) / 2

    if model_type == "transformer":
        encoder = TransformerEncoderModel(
            n_cont_features=N_CONTINUOUS_FEATURES,
            d_model=mcfg["d_model"],
            nhead=mcfg["nhead"],
            num_layers=mcfg["num_layers"],
            dim_feedforward=mcfg["dim_feedforward"],
            dropout=mcfg["dropout"],
            context_dim=context_dim,
            use_rope=mcfg.get("use_rope", False),
        )
    elif model_type == "lstm":
        encoder = LSTMEncoderModel(
            n_cont_features=N_CONTINUOUS_FEATURES,
            d_model=mcfg["d_model"],
            lstm_hidden=mcfg["lstm_hidden"],
            lstm_layers=mcfg["lstm_layers"],
            dropout=mcfg["dropout"],
            context_dim=context_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    flow = PosteriorFlow(
        theta_dim=theta_dim,
        context_dim=flow_context_dim,
        n_transforms=mcfg["flow_transforms"],
        hidden_features=mcfg["flow_hidden"],
        n_hidden_layers=mcfg.get("flow_layers", 2),
        n_bins=mcfg.get("flow_bins", 8),
    )

    return NPEModel(
        encoder, flow, use_aux=use_aux, theta_mean=theta_mean, theta_std=theta_std
    )


# ======================================================================
# Factorized NPE model: global flow + per-backend white-noise flow
# ======================================================================


class FactorizedNPEModel(nn.Module):
    """Autoregressive factorized amortized NPE.

    Posterior decomposition:
        q(θ | x) = q_global(θ_g | x) · ∏_b q_wn(θ_wn_b | x, θ_g)

    The per-backend WN flow is conditioned on the global parameters θ_g
    (teacher-forced at train time, auto-regressively drawn at inference),
    which captures the WN ↔ red-noise coupling that an independence
    factorization drops. Loss is the negative log joint under this chain
    rule; no independence assumption is made.
    """

    def __init__(
        self,
        encoder: nn.Module,
        global_flow: PosteriorFlow,
        wn_flow: PosteriorFlow,
        use_aux: bool = False,
        global_theta_mean: torch.Tensor | None = None,
        global_theta_std: torch.Tensor | None = None,
        wn_theta_mean: torch.Tensor | None = None,
        wn_theta_std: torch.Tensor | None = None,
        n_backends_max: int = 4,
        context_dropout: float = 0.0,
        wn_loss_weight: float = 1.0,
        global_context_raw_dim: int | None = None,
        global_context_proj_dim: int | None = None,
        wn_context_raw_dim: int | None = None,
        wn_context_proj_dim: int | None = None,
        theta_g_dim: int = 4,
        theta_g_noise_std: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.global_flow = global_flow
        self.wn_flow = wn_flow
        self.use_aux = use_aux
        self.n_backends_max = n_backends_max
        self.wn_loss_weight = wn_loss_weight
        self.theta_g_dim = theta_g_dim
        # Small Gaussian noise on teacher-forced θ_g (normalized space) in the
        # WN branch only. Training-only, closes the gap to q_g(·|x) at inference
        # and prevents the WN flow from collapsing onto exact θ_g coordinates.
        self.theta_g_noise_std = theta_g_noise_std
        self.ctx_dropout = nn.Dropout(context_dropout) if context_dropout > 0 else None

        # Optional bottleneck to compress global context before the flow
        if global_context_proj_dim is not None and global_context_raw_dim is not None:
            self.global_ctx_bottleneck = nn.Sequential(
                nn.LayerNorm(global_context_raw_dim),
                nn.Linear(global_context_raw_dim, global_context_proj_dim),
                nn.GELU(),
            )
        else:
            self.global_ctx_bottleneck = None

        # Optional bottleneck to compress WN context before the flow
        if wn_context_proj_dim is not None and wn_context_raw_dim is not None:
            self.wn_ctx_bottleneck = nn.Sequential(
                nn.LayerNorm(wn_context_raw_dim),
                nn.Linear(wn_context_raw_dim, wn_context_proj_dim),
                nn.GELU(),
            )
        else:
            self.wn_ctx_bottleneck = None

        if global_theta_mean is None:
            global_theta_mean = torch.zeros(4)
        if global_theta_std is None:
            global_theta_std = torch.ones(4)
        if wn_theta_mean is None:
            wn_theta_mean = torch.zeros(3)
        if wn_theta_std is None:
            wn_theta_std = torch.ones(3)

        self.register_buffer("global_theta_mean", global_theta_mean)
        self.register_buffer("global_theta_std", global_theta_std)
        self.register_buffer("wn_theta_mean", wn_theta_mean)
        self.register_buffer("wn_theta_std", wn_theta_std)

    def _normalize_global(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.global_theta_mean) / self.global_theta_std

    def _denormalize_global(self, theta_norm: torch.Tensor) -> torch.Tensor:
        return theta_norm * self.global_theta_std + self.global_theta_mean

    def _normalize_wn(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.wn_theta_mean) / self.wn_theta_std

    def _denormalize_wn(self, theta_norm: torch.Tensor) -> torch.Tensor:
        return theta_norm * self.wn_theta_std + self.wn_theta_mean

    def _compute_global_aux(self, batch: dict) -> torch.Tensor:
        """Global auxiliary features. Returns (B, 4)."""
        mask = batch["mask"]
        n_toa = mask.sum(1).float()
        log_sigma = batch["features"][:, :, 3]
        mask_f = mask.float()
        n = mask_f.sum(1).clamp(min=1)
        mean_ls = (log_sigma * mask_f).sum(1) / n
        var_ls = ((log_sigma - mean_ls.unsqueeze(1)) ** 2 * mask_f).sum(1) / n
        std_ls = (var_ls + 1e-8).sqrt()
        tspan = batch["tspan_yr"]
        return torch.stack(
            [
                torch.log(n_toa.clamp(min=1)),
                torch.log(tspan.clamp(min=1e-3)),
                mean_ls,
                std_ls,
            ],
            dim=-1,
        )

    def _compute_backend_aux(self, batch: dict) -> torch.Tensor:
        """Per-backend auxiliary features. Returns (B, n_backends_max, 3)."""
        mask = batch["mask"]
        backend_id = batch["backend_id"]
        log_sigma = batch["features"][:, :, 3]
        B = mask.shape[0]

        aux_list = []
        for b in range(self.n_backends_max):
            b_mask = mask & (backend_id == b)
            b_mask_f = b_mask.float()
            n_b = b_mask_f.sum(1).clamp(min=1)  # (B,)
            mean_ls_b = (log_sigma * b_mask_f).sum(1) / n_b
            var_ls_b = ((log_sigma - mean_ls_b.unsqueeze(1)) ** 2 * b_mask_f).sum(
                1
            ) / n_b
            std_ls_b = (var_ls_b + 1e-8).sqrt()
            aux_b = torch.stack([torch.log(n_b), mean_ls_b, std_ls_b], dim=-1)
            aux_list.append(aux_b)

        return torch.stack(aux_list, dim=1)  # (B, Bmax, 3)

    def _get_contexts(self, batch: dict):
        """Compute global and per-backend flow contexts.

        Returns
        -------
        global_ctx  : (B, global_flow_context_dim)
        wn_ctx      : (B, n_backends_max, wn_flow_context_dim)
        """
        global_ctx, backend_ctx = self.encoder.forward_factorized(
            batch["features"],
            batch["backend_id"],
            batch["mask"],
        )
        if self.use_aux:
            global_aux = self._compute_global_aux(batch)
            global_ctx = torch.cat([global_ctx, global_aux], dim=-1)
            backend_aux = self._compute_backend_aux(batch)
            # WN context: [global_ctx broadcast, backend_ctx, backend_aux]
            global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, self.n_backends_max, -1)
            wn_ctx = torch.cat([global_ctx_exp, backend_ctx, backend_aux], dim=-1)
        else:
            global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, self.n_backends_max, -1)
            wn_ctx = torch.cat([global_ctx_exp, backend_ctx], dim=-1)
        # Bottleneck: compress global context to reduce overfitting
        if self.global_ctx_bottleneck is not None:
            global_ctx = self.global_ctx_bottleneck(global_ctx)
        # Bottleneck: compress WN context to reduce overfitting
        if self.wn_ctx_bottleneck is not None:
            wn_ctx = self.wn_ctx_bottleneck(wn_ctx)
        return global_ctx, wn_ctx

    def forward(self, batch: dict) -> torch.Tensor:
        """Compute autoregressive factorized negative log-prob loss.

        log q(θ|x) = log q_global(θ_g|x) + Σ_b log q_wn(θ_wn_b | x, θ_g)

        θ_g is teacher-forced from the simulator ground truth.
        Returns: loss = -E[log q_global] - w * E_active[log q_wn]
        """
        global_ctx, wn_ctx = self._get_contexts(batch)

        # Context dropout for regularization (train only, no-op at eval)
        if self.ctx_dropout is not None:
            global_ctx = self.ctx_dropout(global_ctx)
            wn_ctx = self.ctx_dropout(wn_ctx)

        # Global flow loss
        theta_global = batch["theta_global"]  # (B, 4)
        tg_norm = self._normalize_global(theta_global)
        with torch.autocast(global_ctx.device.type, enabled=False):
            global_lp = self.global_flow.log_prob(tg_norm.float(), global_ctx.float())
        global_loss = -global_lp.mean()

        # Per-backend WN flow loss: condition WN context on teacher-forced θ_g.
        # At training only, optionally perturb the teacher signal with small
        # Gaussian noise — keeps the chain-rule loss an unbiased estimator of
        # the smoothed conditional and stops the flow locking onto exact
        # θ_g coordinates (see v7c catastrophic overfit).
        tg_for_wn = tg_norm
        if self.training and self.theta_g_noise_std > 0.0:
            tg_for_wn = tg_norm + self.theta_g_noise_std * torch.randn_like(tg_norm)
        tg_exp = tg_for_wn.unsqueeze(1).expand(-1, self.n_backends_max, -1)  # (B, Bmax, 4)
        wn_ctx_ar = torch.cat([wn_ctx, tg_exp], dim=-1)  # (B, Bmax, wn_ctx+4)

        theta_wn = batch["theta_wn"]  # (B, Bmax, 3)
        backend_active = batch["backend_active"]  # (B, Bmax) bool
        active_mask = backend_active
        n_active = active_mask.sum()
        if n_active > 0:
            tw_norm = self._normalize_wn(theta_wn)  # (B, Bmax, 3)
            tw_flat = tw_norm[active_mask]  # (n_active, 3)
            wn_ctx_flat = wn_ctx_ar[active_mask]  # (n_active, wn_ctx+4)
            with torch.autocast(wn_ctx_flat.device.type, enabled=False):
                wn_lp = self.wn_flow.log_prob(tw_flat.float(), wn_ctx_flat.float())
            wn_loss = -wn_lp.mean()
        else:
            wn_loss = torch.tensor(0.0, device=global_ctx.device)

        # Cache component losses for logging
        self._last_global_loss = global_loss.item()
        self._last_wn_loss = wn_loss.item()

        return global_loss + self.wn_loss_weight * wn_loss

    @torch.no_grad()
    def sample_posterior(self, batch: dict, n_samples: int = 1000):
        """Autoregressively sample from q(θ|x).

        Draw θ_g ~ q_global(·|x), then per backend draw θ_wn_b ~ q_wn(·|x, θ_g)
        so the i-th global sample and the i-th WN sample form a joint draw.

        Returns
        -------
        global_samples : (B, n_samples, 4)
        wn_samples     : (B, n_backends_max, n_samples, 3) — index i aligned
                         with global_samples[:, i] (same joint draw).
        """
        global_ctx, wn_ctx = self._get_contexts(batch)
        B = global_ctx.shape[0]
        Bmax = self.n_backends_max
        S = n_samples

        # Global samples
        gs_norm = self.global_flow.sample(global_ctx, S)  # (B, S, 4)
        global_samples = self._denormalize_global(gs_norm)

        # Expand θ_g and wn_ctx to (B, Bmax, S, .) then cat + flatten for batched flow call
        gs_norm_exp = gs_norm.unsqueeze(1).expand(B, Bmax, S, -1)  # (B, Bmax, S, 4)
        wn_ctx_exp = wn_ctx.unsqueeze(2).expand(B, Bmax, S, -1)  # (B, Bmax, S, wctx)
        wn_ctx_ar = torch.cat([wn_ctx_exp, gs_norm_exp], dim=-1)
        wn_ctx_flat = wn_ctx_ar.reshape(B * Bmax * S, -1)
        ws_norm_flat = self.wn_flow.sample(wn_ctx_flat, 1).squeeze(1)  # (B*Bmax*S, 3)
        ws_norm = ws_norm_flat.reshape(B, Bmax, S, 3)
        wn_samples = self._denormalize_wn(ws_norm)

        return global_samples, wn_samples

    @torch.no_grad()
    def sample_posterior_flat(self, batch: dict, n_samples: int = 1000) -> torch.Tensor:
        """Sample and concatenate to flat theta for backward compat.

        Returns (B, n_samples, 4 + 3*n_backends_max).
        """
        global_samples, wn_samples = self.sample_posterior(batch, n_samples)
        B, Bmax, S, _ = wn_samples.shape
        wn_flat = wn_samples.permute(0, 2, 1, 3).reshape(B, S, Bmax * 3)
        return torch.cat([global_samples, wn_flat], dim=-1)


def _build_factorized_model(model_type: str, cfg: dict) -> FactorizedNPEModel:
    """Build FactorizedNPEModel from config."""
    mcfg = cfg["model"]
    context_dim = mcfg["context_dim"]
    use_aux = mcfg.get("use_aux_features", False)
    n_backends_max = mcfg.get("n_backends_max", 4)

    # Global flow context dim
    global_flow_ctx_raw = context_dim + (N_AUX_FEATURES if use_aux else 0)
    global_ctx_proj_dim = mcfg.get("global_context_proj_dim", None)
    global_flow_ctx = global_ctx_proj_dim if global_ctx_proj_dim else global_flow_ctx_raw

    # Prior bounds for normalization (needed before we size the WN flow)
    global_cfg = cfg.get("prior", {}).get("global", {})
    wn_cfg = cfg.get("prior", {}).get("white_noise", {})

    g_names = list(global_cfg.keys())
    g_lo = torch.tensor([global_cfg[k][0] for k in g_names], dtype=torch.float32)
    g_hi = torch.tensor([global_cfg[k][1] for k in g_names], dtype=torch.float32)
    global_theta_mean = (g_lo + g_hi) / 2
    global_theta_std = (g_hi - g_lo) / 2

    # WN flow context dim: encoder-derived context (optionally bottlenecked)
    # plus θ_g (autoregressive conditioning, pass-through — not bottlenecked).
    theta_g_dim = len(g_names)
    wn_flow_ctx_raw = (
        global_flow_ctx_raw + context_dim + (N_BACKEND_AUX_FEATURES if use_aux else 0)
    )
    wn_ctx_proj_dim = mcfg.get("wn_context_proj_dim", None)
    wn_flow_ctx = (wn_ctx_proj_dim if wn_ctx_proj_dim else wn_flow_ctx_raw) + theta_g_dim

    w_names = list(wn_cfg.keys())
    w_lo = torch.tensor([wn_cfg[k][0] for k in w_names], dtype=torch.float32)
    w_hi = torch.tensor([wn_cfg[k][1] for k in w_names], dtype=torch.float32)
    wn_theta_mean = (w_lo + w_hi) / 2
    wn_theta_std = (w_hi - w_lo) / 2

    # Encoder
    if model_type == "transformer":
        encoder = TransformerEncoderModel(
            n_cont_features=N_CONTINUOUS_FEATURES,
            d_model=mcfg["d_model"],
            nhead=mcfg["nhead"],
            num_layers=mcfg["num_layers"],
            dim_feedforward=mcfg["dim_feedforward"],
            dropout=mcfg["dropout"],
            context_dim=context_dim,
            use_rope=mcfg.get("use_rope", False),
            factorized=True,
            n_backends_max=n_backends_max,
        )
    elif model_type == "lstm":
        encoder = LSTMEncoderModel(
            n_cont_features=N_CONTINUOUS_FEATURES,
            d_model=mcfg["d_model"],
            lstm_hidden=mcfg["lstm_hidden"],
            lstm_layers=mcfg["lstm_layers"],
            dropout=mcfg["dropout"],
            context_dim=context_dim,
            factorized=True,
            n_backends_max=n_backends_max,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    global_flow = PosteriorFlow(
        theta_dim=len(g_names),
        context_dim=global_flow_ctx,
        n_transforms=mcfg.get("global_flow_transforms", 8),
        hidden_features=mcfg.get("global_flow_hidden", 192),
        n_hidden_layers=mcfg.get("global_flow_layers", 3),
        n_bins=mcfg.get("global_flow_bins", 16),
    )

    wn_flow = PosteriorFlow(
        theta_dim=len(w_names),
        context_dim=wn_flow_ctx,
        n_transforms=mcfg.get("wn_flow_transforms", 6),
        hidden_features=mcfg.get("wn_flow_hidden", 128),
        n_hidden_layers=mcfg.get("wn_flow_layers", 2),
        n_bins=mcfg.get("wn_flow_bins", 8),
    )

    return FactorizedNPEModel(
        encoder=encoder,
        global_flow=global_flow,
        wn_flow=wn_flow,
        use_aux=use_aux,
        global_theta_mean=global_theta_mean,
        global_theta_std=global_theta_std,
        wn_theta_mean=wn_theta_mean,
        wn_theta_std=wn_theta_std,
        n_backends_max=n_backends_max,
        context_dropout=mcfg.get("context_dropout", 0.0),
        wn_loss_weight=mcfg.get("wn_loss_weight", 1.0),
        global_context_raw_dim=global_flow_ctx_raw if global_ctx_proj_dim else None,
        global_context_proj_dim=global_ctx_proj_dim,
        wn_context_raw_dim=wn_flow_ctx_raw if wn_ctx_proj_dim else None,
        wn_context_proj_dim=wn_ctx_proj_dim,
        theta_g_dim=theta_g_dim,
        theta_g_noise_std=mcfg.get("theta_g_noise", 0.0),
    )
