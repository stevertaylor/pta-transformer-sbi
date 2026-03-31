"""Transformer encoder over TOA tokens → context vector.

Supports two modes:
- Legacy (use_rope=False): Additive TimeEmbedding, CLS token, post-norm TransformerEncoder
- Modern (use_rope=True): RoPE, pre-norm blocks, attention pooling
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Shared: token feature MLP
# ---------------------------------------------------------------------------


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
        x = torch.stack([t_norm, torch.log1p(dt_prev.clamp(min=0.0))], dim=-1)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# RoPE components (for use_rope=True path)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Continuous-time rotary position embedding."""

    def __init__(self, d_k: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: torch.Tensor):
        """positions: (B, L) continuous time values → cos, sin each (B, L, d_k//2)."""
        freqs = positions.unsqueeze(-1) * self.inv_freq  # (B, L, d_k//2)
        return freqs.cos(), freqs.sin()


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to Q or K.

    x:   (B, nhead, L, d_k)
    cos: (B, L, d_k//2)
    sin: (B, L, d_k//2)
    """
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    cos = cos[:, None, :, :]  # (B, 1, L, d_half)
    sin = sin[:, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings on Q and K."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = self.d_k**-0.5

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.W_qkv(x).reshape(B, L, 3, self.nhead, self.d_k)
        q, k, v = qkv.unbind(2)  # each (B, L, nhead, d_k)
        q = q.transpose(1, 2)  # (B, nhead, L, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = _apply_rotary_emb(q, rope_cos, rope_sin)
        k = _apply_rotary_emb(k, rope_cos, rope_sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, nhead, L, L)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.W_out(out)


class PreNormBlock(nn.Module):
    """Pre-norm transformer block with RoPE self-attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_cos, rope_sin, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    """Learned scalar attention pooling over a masked sequence."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model), mask: (B, L) True=valid → (B, d_model)."""
        scores = self.score_proj(x).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = scores.softmax(dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (x * weights).sum(dim=1)


class CLSQueryPooling(nn.Module):
    """Learned [CLS] query token with multi-head cross-attention pooling.

    A single learned query vector attends over the encoder output sequence,
    allowing the model to learn *what* information to extract rather than
    relying on a fixed reduction (mean or scalar-attention).
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model), mask: (B, L) True=valid → (B, d_model)."""
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        key_padding_mask = ~mask  # True = ignore
        out, _ = self.cross_attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out.squeeze(1))  # (B, d_model)


class BackendQueryPooling(nn.Module):
    """Per-backend cross-attention pooling.

    Uses a shared learned query to attend over only the tokens belonging to
    each backend, producing one context vector per backend.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        backend_id: torch.Tensor,
        n_backends_max: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d_model)
        mask : (B, L) bool, True = valid
        backend_id : (B, L) int
        n_backends_max : int

        Returns
        -------
        (B, n_backends_max, d_model) per-backend context vectors
        """
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1)  # (B, 1, d_model)

        contexts = []
        for b in range(n_backends_max):
            b_mask = mask & (backend_id == b)  # (B, L)
            has_tokens = b_mask.any(dim=1)  # (B,)

            # For samples with no tokens from backend b, use full mask
            # as fallback to avoid NaN in softmax, then zero out.
            safe_mask = b_mask.clone()
            empty = ~has_tokens
            if empty.any():
                safe_mask[empty] = mask[empty]

            kpm = ~safe_mask
            out, _ = self.cross_attn(q, x, x, key_padding_mask=kpm)
            out = self.norm(out.squeeze(1))  # (B, d_model)
            out = out * has_tokens.float().unsqueeze(-1)
            contexts.append(out)

        return torch.stack(contexts, dim=1)  # (B, n_backends_max, d_model)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------


class TransformerEncoderModel(nn.Module):
    """Transformer encoder: tokens → summary → context vector.

    When use_rope=False (legacy):
        Additive TimeEmbedding + CLS token + post-norm nn.TransformerEncoder
    When use_rope=True (modern):
        Rotary position embeddings + pre-norm blocks + attention pooling
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
        use_rope: bool = False,
        rope_base: float = 10000.0,
        position_scale: float = 512.0,
        factorized: bool = False,
        n_backends_max: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_rope = use_rope
        self.position_scale = position_scale
        self.factorized = factorized
        self.n_backends_max = n_backends_max

        # Token feature MLP (shared)
        in_dim = n_cont_features + backend_embed_dim
        self.backend_embed = nn.Embedding(n_backends, backend_embed_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        if use_rope:
            d_k = d_model // nhead
            self.rope = RotaryEmbedding(d_k, base=rope_base)
            self.blocks = nn.ModuleList(
                [
                    PreNormBlock(d_model, nhead, dim_feedforward, dropout)
                    for _ in range(num_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(d_model)
            self.pool = CLSQueryPooling(d_model, nhead, dropout)
        else:
            self.time_embed = TimeEmbedding(d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

        # Context projection
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, context_dim),
        )

        # Per-backend pooling and projection (factorized mode)
        if factorized:
            self.backend_pool = BackendQueryPooling(d_model, nhead, dropout)
            self.wn_context_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, context_dim),
            )

    def _encode_trunk(
        self,
        features: torch.Tensor,
        backend_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder trunk, return (B, L, d_model) sequence output."""
        B, L, _ = features.shape

        be = self.backend_embed(backend_id)
        tok_in = torch.cat([features, be], dim=-1)
        tok_emb = self.token_mlp(tok_in)

        if self.use_rope:
            positions = features[:, :, 0] * self.position_scale
            rope_cos, rope_sin = self.rope(positions)
            pad_mask = ~mask
            x = tok_emb
            for block in self.blocks:
                x = block(x, rope_cos, rope_sin, key_padding_mask=pad_mask)
            return self.final_norm(x)
        else:
            t_emb = self.time_embed(features[:, :, 0], features[:, :, 1])
            tok_emb = tok_emb + t_emb
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, tok_emb], dim=1)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            out = self.transformer(seq, src_key_padding_mask=~full_mask)
            # Return just the token outputs (skip CLS position)
            return out[:, 1:, :]

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
        x = self._encode_trunk(features, backend_id, mask)

        if self.use_rope:
            pooled = self.pool(x, mask)
        else:
            # Legacy path: CLS output is at position 0 of trunk output
            # But _encode_trunk strips CLS for legacy path too,
            # so we need the CLS token from the transformer output.
            # Re-run through legacy path directly for backward compat.
            B, L, _ = features.shape
            be = self.backend_embed(backend_id)
            tok_in = torch.cat([features, be], dim=-1)
            tok_emb = self.token_mlp(tok_in)
            t_emb = self.time_embed(features[:, :, 0], features[:, :, 1])
            tok_emb = tok_emb + t_emb
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([cls, tok_emb], dim=1)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            out = self.transformer(seq, src_key_padding_mask=~full_mask)
            pooled = out[:, 0, :]

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
        x = self._encode_trunk(features, backend_id, mask)

        if self.use_rope:
            pooled = self.pool(x, mask)
        else:
            pooled = (x * mask.unsqueeze(-1).float()).sum(1) / mask.sum(
                1, keepdim=True
            ).float().clamp(min=1)

        global_ctx = self.context_proj(pooled)
        backend_raw = self.backend_pool(x, mask, backend_id, self.n_backends_max)
        backend_ctx = self.wn_context_proj(backend_raw)
        return global_ctx, backend_ctx
