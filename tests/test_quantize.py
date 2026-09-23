"""Tests for quantized checkpoint save/load (weight-only INT4/INT8 via MLX).

Run: uv run pytest -q tests/test_quantize.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from forge.model.config import ModelConfig
from forge.model.transformer import Transformer
from forge.training import checkpoint

# dims divisible by group_size=64, as MLX's quantize requires
BASE = dict(dim=128, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=256, max_seq_len=32, vocab_size=320)


def test_quantize_shrinks_and_stays_finite() -> None:
    model = Transformer(ModelConfig(**BASE))
    mx.eval(model.parameters())
    before = sum(p.nbytes for _, p in tree_flatten(model.parameters()))

    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    after = sum(p.nbytes for _, p in tree_flatten(model.parameters()))

    assert after < before / 3  # 4-bit weights should be well under a third of fp32 size
    idx = mx.array([[1, 2, 3, 4]])
    out = model(idx)
    assert bool(mx.all(mx.isfinite(out)))


def test_quantized_checkpoint_roundtrip(tmp_path) -> None:
    m1 = Transformer(ModelConfig(**BASE))
    mx.eval(m1.parameters())
    nn.quantize(m1, group_size=64, bits=4)
    mx.eval(m1.parameters())
    checkpoint.save(tmp_path / "ckpt", m1, optim.AdamW(learning_rate=1e-3), step=5, extra={"quantized": True})

    m2 = Transformer(ModelConfig(**BASE))
    nn.quantize(m2, group_size=64, bits=4)  # must quantize BEFORE loading - matching structure
    st = checkpoint.load(tmp_path / "ckpt", m2)

    assert st["step"] == 5 and st["quantized"] is True
    idx = mx.array([[1, 2, 3, 4]])
    assert bool(mx.all(m1(idx) == m2(idx)))  # exact match: same packed integers loaded back
