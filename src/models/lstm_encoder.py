"""LSTM baseline encoder over TOA tokens → context vector."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoderModel(nn.Module):
    """LSTM encoder with the same interface as the transformer encoder.

    Architecture:
    1. Same token feature MLP and time embedding as the transformer
    2. Bidirectional LSTM
    3. Masked mean-pooling over valid tokens
    4. Project to context_dim
    """

    def __init__(
        self,
        n_cont_features: int = 6,
        d_model: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        context_dim: int = 64,
        n_backends: int = 4,
        backend_embed_dim: int = 8,
        factorized: bool = False,
        n_backends_max: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.factorized = factorized
        self.n_backends_max = n_backends_max

        # Token feature MLP (same as transformer)
        in_dim = n_cont_features + backend_embed_dim
        self.backend_embed = nn.Embedding(n_backends, backend_embed_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Time embedding (same architecture)
        self.time_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # Context projection (bidirectional → 2*hidden)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(2 * lstm_hidden),
            nn.Linear(2 * lstm_hidden, context_dim),
        )

        # Per-backend projection (factorized mode)
        if factorized:
            self.wn_context_proj = nn.Sequential(
                nn.LayerNorm(2 * lstm_hidden),
                nn.Linear(2 * lstm_hidden, context_dim),
            )

    def forward(
        self,
        features: torch.Tensor,
        backend_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Same signature as TransformerEncoderModel.forward."""
        B, L, _ = features.shape

        # Token embedding
        be = self.backend_embed(backend_id)
        tok_in = torch.cat([features, be], dim=-1)
        tok_emb = self.token_mlp(tok_in)

        # Time embedding
        t_input = torch.stack(
            [features[:, :, 0], torch.log1p(features[:, :, 1].clamp(min=0.0))], dim=-1
        )
        t_emb = self.time_embed(t_input)
        tok_emb = tok_emb + t_emb

        # Pack padded sequences for LSTM
        seq_lens = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            tok_emb, seq_lens, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=L
        )

        # Masked mean pool
        mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, 2*H)

        return self.context_proj(pooled)

    def forward_factorized(
        self,
        features: torch.Tensor,
        backend_id: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Return global context and per-backend contexts.

        Returns
        -------
        global_ctx  : (B, context_dim)
        backend_ctx : (B, n_backends_max, context_dim)
        """
        B, L, _ = features.shape

        # Encode
        be = self.backend_embed(backend_id)
        tok_in = torch.cat([features, be], dim=-1)
        tok_emb = self.token_mlp(tok_in)
        t_input = torch.stack(
            [features[:, :, 0], torch.log1p(features[:, :, 1].clamp(min=0.0))], dim=-1
        )
        t_emb = self.time_embed(t_input)
        tok_emb = tok_emb + t_emb

        seq_lens = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            tok_emb, seq_lens, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=L
        )

        # Global: masked mean pool
        mask_f = mask.unsqueeze(-1).float()
        global_pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        global_ctx = self.context_proj(global_pooled)

        # Per-backend: masked mean pool per backend
        backend_ctxs = []
        for b in range(self.n_backends_max):
            b_mask = mask & (backend_id == b)
            b_mask_f = b_mask.unsqueeze(-1).float()
            n_b = b_mask_f.sum(dim=1).clamp(min=1)
            b_pooled = (out * b_mask_f).sum(dim=1) / n_b
            backend_ctxs.append(self.wn_context_proj(b_pooled))

        backend_ctx = torch.stack(backend_ctxs, dim=1)
        return global_ctx, backend_ctx
