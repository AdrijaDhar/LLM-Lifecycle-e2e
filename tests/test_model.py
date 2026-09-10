"""Shape / sanity tests for the Transformer.

Run: uv run pytest -q tests/test_model.py
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from forge.model.config import PRESETS, ModelConfig
from forge.model.transformer import Transformer


def _actual_param_count(model: nn.Module) -> int:
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def test_param_count_matches_formula() -> None:
    for name in ("tiny", "135m", "360m"):
        cfg = PRESETS[name]
        model = Transformer(cfg)
        assert _actual_param_count(model) == cfg.n_params, name


def test_preset_sizes_in_range() -> None:
    assert 125_000_000 < PRESETS["135m"].n_params < 145_000_000
    assert 340_000_000 < PRESETS["360m"].n_params < 380_000_000


def test_forward_shape() -> None:
    cfg = PRESETS["tiny"]
    model = Transformer(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, (2, 32))
    logits = model(idx)
    assert logits.shape == (2, 32, cfg.vocab_size)


def test_untrained_loss_near_ln_vocab() -> None:
    cfg = PRESETS["tiny"]
    model = Transformer(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, (4, 64))
    targets = mx.random.randint(0, cfg.vocab_size, (4, 64))
    loss = float(model.loss(idx, targets))
    # A well-initialised LM starts near uniform: -ln(1/V) = ln(V).
    assert abs(loss - math.log(cfg.vocab_size)) < 1.0, loss


def test_backward_produces_finite_grads() -> None:
    cfg = PRESETS["tiny"]
    model = Transformer(cfg)
    idx = mx.random.randint(0, cfg.vocab_size, (2, 32))
    targets = mx.random.randint(0, cfg.vocab_size, (2, 32))

    def loss_fn(m):
        return m.loss(idx, targets)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    flat = tree_flatten(grads)
    assert all(bool(mx.all(mx.isfinite(g))) for _, g in flat)
    assert float(loss) > 0


def test_gqa_config_validation() -> None:
    try:
        ModelConfig(dim=64, n_heads=8, n_kv_heads=3)
        raise AssertionError("should have rejected n_kv_heads not dividing n_heads")
    except AssertionError as e:
        assert "multiple" in str(e)
