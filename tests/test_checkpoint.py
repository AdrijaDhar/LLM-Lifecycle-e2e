"""Tests for checkpoint save/load, including the vocab-resize path used for SFT.

Run: uv run pytest -q tests/test_checkpoint.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from forge.model.config import ModelConfig
from forge.model.transformer import Transformer
from forge.training import checkpoint

BASE = dict(dim=32, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=64, max_seq_len=32)


def test_save_and_load_roundtrip(tmp_path) -> None:
    cfg = ModelConfig(vocab_size=100, **BASE)
    model = Transformer(cfg)
    opt = optim.AdamW(learning_rate=1e-3)
    mx.eval(model.parameters())

    checkpoint.save(tmp_path / "ckpt", model, opt, step=42, extra={"preset": "test"})

    model2 = Transformer(cfg)
    st = checkpoint.load(tmp_path / "ckpt", model2, opt)
    assert st["step"] == 42
    assert st["preset"] == "test"

    for (n1, p1), (n2, p2) in zip(
        sorted(tree_flatten(model.parameters())), sorted(tree_flatten(model2.parameters()))
    ):
        assert n1 == n2
        assert bool(mx.all(p1 == p2))


def test_load_resized_grows_embedding_and_keeps_old_rows(tmp_path) -> None:
    small_cfg = ModelConfig(vocab_size=100, **BASE)
    small = Transformer(small_cfg)
    mx.eval(small.parameters())
    checkpoint.save(tmp_path / "ckpt", small, optim.AdamW(learning_rate=1e-3), step=10)

    big_cfg = ModelConfig(vocab_size=103, **BASE)  # +3 tokens, like adding chat specials
    big = Transformer(big_cfg)
    mx.eval(big.parameters())
    fresh_new_rows = big.tok_emb.weight[100:103]

    st = checkpoint.load_resized(tmp_path / "ckpt", big)
    assert st["step"] == 10
    assert big.tok_emb.weight.shape == (103, small_cfg.dim)
    # old rows preserved exactly
    assert bool(mx.all(big.tok_emb.weight[:100] == small.tok_emb.weight))
    # new rows left at the fresh model's own random init (untouched by the smaller checkpoint)
    assert bool(mx.all(big.tok_emb.weight[100:103] == fresh_new_rows))


def test_load_resized_forward_pass_runs(tmp_path) -> None:
    small_cfg = ModelConfig(vocab_size=100, **BASE)
    small = Transformer(small_cfg)
    mx.eval(small.parameters())
    checkpoint.save(tmp_path / "ckpt", small, optim.AdamW(learning_rate=1e-3), step=1)

    big_cfg = ModelConfig(vocab_size=105, **BASE)
    big = Transformer(big_cfg)
    checkpoint.load_resized(tmp_path / "ckpt", big)

    idx = mx.array([[1, 2, 3, 101, 102]])  # includes a couple of new-token ids
    logits = big(idx)
    assert logits.shape == (1, 5, 105)
    assert bool(mx.all(mx.isfinite(logits)))
