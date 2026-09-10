"""Decoder-only Transformer in MLX (SmolLM2 / Llama family).

Layout of one block (pre-norm residual):

    x -> RMSNorm -> Attention -> +x  ->  RMSNorm -> SwiGLU MLP -> +x

Components, and why each is the modern choice:
  - RMSNorm      : like LayerNorm but no mean-subtraction and no bias - cheaper,
                   and empirically just as stable for pre-norm transformers.
  - RoPE         : rotates q/k by a position-dependent angle so attention scores
                   depend on *relative* position. No learned position table, and
                   it extrapolates to longer contexts better than learned/absolute.
  - GQA          : query heads share a smaller set of key/value heads. Shrinks the
                   KV cache (the thing that dominates memory at generation time)
                   with almost no quality loss.
  - SwiGLU       : gated MLP, `down(silu(gate(x)) * up(x))`. Beats plain GELU MLP
                   at equal parameter count.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from forge.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # normalize in float32 for stability, then cast back
        dt = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * self.weight).astype(dt)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None):
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # MLX SDPA handles GQA (n_heads > n_kv_heads) and the causal mask itself.
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask=mask, cache=cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def __call__(self, idx: mx.array, cache=None) -> mx.array:
        """idx: (B, T) int32 token ids -> logits (B, T, vocab_size)."""
        x = self.tok_emb(idx)

        mask = None
        if x.shape[1] > 1:
            mask = "causal"

        caches = cache if cache is not None else [None] * len(self.blocks)
        for block, c in zip(self.blocks, caches):
            x = block(x, mask=mask, cache=c)

        x = self.final_norm(x)
        if self.cfg.tie_embeddings:
            return self.tok_emb.as_linear(x)
        return self.lm_head(x)

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits = self(idx)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean"
        )

    @property
    def n_params(self) -> int:
        return self.cfg.n_params
