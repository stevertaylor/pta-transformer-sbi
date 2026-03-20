"""Transformer encoder over TOA tokens → context vector."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


class TimeEmbedding(nn.Module):
    """Learned continuous-time embedding from (t_norm, dt_prev) → d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t_norm: torch.Tensor, dt_prev: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t_norm, dt_prev : (B, L)
        Returns : (B, L, d_model)
        """
        x = torch.stack([t_norm, torch.log1p(dt_prev.clamp(min=0.0))], dim=-1)  # (B, L, 2)
        return self.mlp(x)


class TransformerEncoderModel(nn.Module):
    """Transformer encoder: tokens → CLS summary → context vector.

    Architecture:
    1. Token MLP: continuous features → d_model
    2. Learned time embedding added to token embedding
    3. Prepend learnable CLS token
    4. Standard transformer encoder (batch_first)
    5. CLS output projected to context_dim
    """

    def __init__(
        self,
        n_cont_features: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        context_dim: int = 64,
        n_backends: int = 4,
        backend_embed_dim: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # Token feature MLP
        in_dim = n_cont_features + backend_embed_dim
        self.backend_embed = nn.Embedding(n_backends, backend_embed_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Time embedding
        self.time_embed = TimeEmbedding(d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Context projection
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, context_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        backend_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features   : (B, L, n_cont_features)
        backend_id : (B, L) long
        mask       : (B, L) bool, True = valid

        Returns
        -------
        context : (B, context_dim)
        """
        B, L, _ = features.shape

        # Token embedding
        be = self.backend_embed(backend_id)              # (B, L, be_dim)
        tok_in = torch.cat([features, be], dim=-1)       # (B, L, in_dim)
        tok_emb = self.token_mlp(tok_in)                 # (B, L, d_model)

        # Time embedding (features[:, :, 0] = t_norm, [:, :, 1] = dt_prev)
        t_emb = self.time_embed(features[:, :, 0], features[:, :, 1])
        tok_emb = tok_emb + t_emb

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        seq = torch.cat([cls, tok_emb], dim=1)            # (B, 1+L, d_model)

        # Build causal-free attention mask: True positions are IGNORED
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)    # (B, 1+L)
        src_key_padding_mask = ~full_mask                 # True = pad → ignore

        out = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)
        cls_out = out[:, 0, :]                            # (B, d_model)

        return self.context_proj(cls_out)                 # (B, context_dim)
