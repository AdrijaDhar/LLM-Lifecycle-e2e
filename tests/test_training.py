"""Training-loop building blocks: schedule, loader, and an overfit-one-batch check.

Run: uv run pytest -q tests/test_training.py
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from forge.data.loader import TokenLoader
from forge.model.config import ModelConfig
from forge.model.transformer import Transformer
from forge.training.schedule import wsd_lr, wsd_schedule


# --- schedule ---------------------------------------------------------------

def test_wsd_warmup_is_linear() -> None:
    kw = dict(peak_lr=1.0, total_steps=1000, warmup_steps=100)
    assert wsd_lr(0, **kw) == 0.01
    assert abs(wsd_lr(49, **kw) - 0.50) < 1e-9
    assert abs(wsd_lr(99, **kw) - 1.00) < 1e-9


def test_wsd_stable_is_flat_at_peak() -> None:
    kw = dict(peak_lr=3e-3, total_steps=1000, warmup_steps=100, decay_frac=0.2)
    for s in (100, 300, 500, 799):
        assert wsd_lr(s, **kw) == 3e-3


def test_wsd_decay_monotone_to_final() -> None:
    kw = dict(peak_lr=1.0, total_steps=1000, warmup_steps=100, decay_frac=0.2, final_lr_frac=0.1)
    prev = 1.0
    for s in range(800, 1000):
        cur = wsd_lr(s, **kw)
        assert cur <= prev + 1e-12
        prev = cur
    assert abs(wsd_lr(1000, **kw) - 0.1) < 1e-9


def test_wsd_schedule_matches_phases() -> None:
    sched = wsd_schedule(peak_lr=1.0, total_steps=1000, warmup_steps=100, decay_frac=0.2)
    at = lambda s: float(sched(mx.array(s)))
    assert at(0) < 0.02                       # warmup start
    assert abs(at(100) - 1.0) < 1e-6          # peak reached
    assert abs(at(500) - 1.0) < 1e-6          # stable
    assert at(900) <= 1.0 + 1e-6              # decay begins
    assert at(999) < 0.05                     # nearly cooled down


# --- loader ----------------------------------------------------------------

def test_loader_shapes_and_shift(tmp_path) -> None:
    p = tmp_path / "train.bin"
    np.arange(5000, dtype=np.uint16).tofile(p)
    dl = TokenLoader(p, seq_len=64, batch_size=8, dtype="uint16", seed=0)
    x, y = dl.batch()
    assert x.shape == (8, 64) and y.shape == (8, 64)
    # y is x shifted by one position
    assert bool(mx.all(x[:, 1:] == y[:, :-1]))


# --- the whole machine: overfit a single batch ---------------------------

def test_overfit_one_batch() -> None:
    cfg = ModelConfig(
        vocab_size=128, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        hidden_dim=128, max_seq_len=64,
    )
    model = Transformer(cfg)
    model.train()
    opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.95], weight_decay=0.0)

    mx.random.seed(0)
    x = mx.random.randint(0, cfg.vocab_size, (4, 32))
    y = mx.random.randint(0, cfg.vocab_size, (4, 32))

    def loss_fn(m):
        return m.loss(x, y)

    lg = nn.value_and_grad(model, loss_fn)
    start = float(model.loss(x, y))
    for _ in range(60):
        loss, grads = lg(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    end = float(model.loss(x, y))

    assert start > math.log(cfg.vocab_size) - 1.0     # started near uniform
    assert end < start - 2.0                          # learned the batch hard
