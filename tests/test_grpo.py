"""Tests for the GRPO core: group-relative advantages and the policy loss.

Run: uv run pytest -q tests/test_grpo.py
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from forge.model.config import ModelConfig
from forge.model.transformer import Transformer
from forge.training.grpo import group_advantages, grpo_loss, token_logp


# --- group_advantages: the hand-reimplemented piece -------------------------

def test_group_advantages_hand_computed() -> None:
    # group 1: rewards [0, 1] -> mean .5, std .5 -> advantages [-1, 1] (up to eps)
    # group 2: rewards [3, 3] -> std 0 -> advantages ~0 (eps in the denominator avoids div-by-0;
    # eps=0 would be a genuine 0/0 here, which is why the function defaults to eps>0)
    rewards = np.array([0.0, 1.0, 3.0, 3.0])
    adv = group_advantages(rewards, group_size=2, eps=0.0)
    assert np.allclose(adv[:2], [-1.0, 1.0], atol=1e-3)
    adv_default_eps = group_advantages(rewards, group_size=2)
    assert np.allclose(adv_default_eps[2:], [0.0, 0.0], atol=1e-2)


def test_group_advantages_each_group_independent() -> None:
    # group 1 has a high mean, group 2 a low mean - advantages should only
    # reflect within-group rank, not cross-group reward level.
    rewards = np.array([10.0, 12.0, 0.0, 2.0])
    adv = group_advantages(rewards, group_size=2)
    assert adv[0] < adv[1]  # 10 below its group's mean (11)
    assert adv[2] < adv[3]  # 0 below its group's mean (1)
    assert abs(adv[0] - adv[2]) < 1e-3  # same relative position -> same advantage
    assert abs(adv[1] - adv[3]) < 1e-3


def test_group_advantages_rejects_bad_group_size() -> None:
    try:
        group_advantages(np.zeros(5), group_size=2)
        raise AssertionError("should have rejected non-divisible length")
    except ValueError:
        pass


def test_group_advantages_zero_variance_group_is_stable() -> None:
    # all completions in a group scored identically -> no learning signal,
    # but must not divide by zero / produce nan or inf.
    adv = group_advantages(np.array([5.0, 5.0, 5.0]), group_size=3)
    assert np.all(np.isfinite(adv))
    assert np.allclose(adv, 0.0, atol=1e-2)


# --- token_logp / grpo_loss ---------------------------------------------------

def _tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, n_kv_heads=1,
                       hidden_dim=64, max_seq_len=32)


def test_token_logp_shapes() -> None:
    cfg = _tiny_cfg()
    model = Transformer(cfg)
    ids = mx.random.randint(0, cfg.vocab_size, (3, 10))
    mask = mx.ones((3, 10))
    logp, m = token_logp(model, ids, mask)
    assert logp.shape == (3, 9) == m.shape


def test_grpo_kl_is_zero_when_policy_equals_ref() -> None:
    cfg = _tiny_cfg()
    policy = Transformer(cfg)
    ref = Transformer(cfg)
    ref.update(policy.parameters())

    ids = mx.random.randint(0, cfg.vocab_size, (4, 10))
    mask = mx.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]] * 4)
    advantages = mx.array([1.0, -1.0, 0.5, -0.5])

    loss, metrics = grpo_loss(policy, ref, ids, mask, advantages, beta_kl=0.04)
    assert abs(float(metrics["kl"])) < 1e-5  # exp(0) - 0 - 1 == 0 exactly


def test_grpo_training_increases_logp_of_positive_advantage_completions() -> None:
    cfg = _tiny_cfg()
    policy = Transformer(cfg)
    ref = Transformer(cfg)
    ref.update(policy.parameters())
    opt = optim.AdamW(learning_rate=5e-3)

    mx.random.seed(0)
    ids = mx.random.randint(0, cfg.vocab_size, (4, 12))
    mask = mx.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]] * 4)
    advantages = mx.array([1.0, 1.0, -1.0, -1.0])  # first two "good", last two "bad"

    def logp_sum(m):
        logp, mm = token_logp(m, ids, mask)
        return (logp * mm).sum(axis=1)

    start = logp_sum(policy)

    def loss_fn(m):
        loss, _ = grpo_loss(m, ref, ids, mask, advantages, beta_kl=0.01)
        return loss

    lg = nn.value_and_grad(policy, loss_fn)
    for _ in range(40):
        loss, grads = lg(policy)
        opt.update(policy, grads)
        mx.eval(policy.parameters(), opt.state)

    end = logp_sum(policy)
    # positive-advantage rows should gain log-prob relative to negative-advantage rows
    delta = end - start
    assert float(delta[0] + delta[1]) > float(delta[2] + delta[3])
