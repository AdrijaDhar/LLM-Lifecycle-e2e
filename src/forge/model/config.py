"""Model configuration and size presets.

The architecture is a modern decoder-only Transformer in the SmolLM2 / Llama
family: RMSNorm (pre-norm), Rotary Position Embeddings, SwiGLU MLP, Grouped-Query
Attention. The presets below match SmolLM2-135M and SmolLM2-360M layer-for-layer
so we have a sane reference to compare loss curves against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32769           # fw32k tokenizer: 32768 BPE + 1 special
    dim: int = 576                    # residual stream width (d_model)
    n_layers: int = 30
    n_heads: int = 9                  # query heads
    n_kv_heads: int = 3              # key/value heads (GQA: n_heads must be a multiple)
    hidden_dim: int = 1536           # SwiGLU inner width (~2.67x dim)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True       # share token-embedding and output projection

    def __post_init__(self) -> None:
        assert self.dim % self.n_heads == 0, "dim must divide evenly into n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def n_params(self) -> int:
        """Exact trainable parameter count for this config."""
        c = self
        emb = c.vocab_size * c.dim
        attn = c.dim * c.dim + 2 * (c.dim * c.n_kv_heads * c.head_dim) + c.dim * c.dim
        mlp = 3 * c.dim * c.hidden_dim
        norms = 2 * c.dim                       # two RMSNorm gains per block
        per_block = attn + mlp + norms
        total = emb + c.n_layers * per_block + c.dim  # + final norm
        if not c.tie_embeddings:
            total += c.vocab_size * c.dim
        return total


PRESETS: dict[str, ModelConfig] = {
    # ~135M params (SmolLM2-135M shape)
    "135m": ModelConfig(
        dim=576, n_layers=30, n_heads=9, n_kv_heads=3, hidden_dim=1536,
    ),
    # ~360M params (SmolLM2-360M shape)
    "360m": ModelConfig(
        dim=960, n_layers=32, n_heads=15, n_kv_heads=5, hidden_dim=2560,
    ),
    # tiny config for fast local smoke tests (~2M params)
    "tiny": ModelConfig(
        dim=128, n_layers=4, n_heads=4, n_kv_heads=2, hidden_dim=352,
        max_seq_len=256, vocab_size=32769,
    ),
}
