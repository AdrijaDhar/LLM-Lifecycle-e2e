"""GRPO (Group Relative Policy Optimization) - the core math, no rollouts here.

The idea PPO needs a learned value network for: "was this completion better
than expected?" GRPO answers it without one. For each prompt, sample a GROUP of
G completions, score each with a reward function, and normalize within the
group:

    advantage_i = (reward_i - mean(group rewards)) / (std(group rewards) + eps)

A completion that beat its group's average gets a positive advantage (push its
log-probability up); one that did worse gets pushed down. The group mean acts
as the free per-prompt baseline a value network would otherwise estimate.

Policy update (single rollout -> single gradient step, so the probability
ratio pi_theta/pi_theta_old is exactly 1 - see the note in grpo_loss):

    loss_per_token = -(advantage * log p_theta(token)) + beta * KL_per_token
    KL_per_token    = exp(logp_ref - logp_theta) - (logp_ref - logp_theta) - 1

That KL form (the "k3" estimator) is always >= 0 and cheap to compute per
token from log-probs alone (no extra forward pass beyond the reference model's
own logp) - it keeps the policy from drifting too far from the reference while
chasing reward.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from forge.model.transformer import Transformer


def group_advantages(rewards: np.ndarray, group_size: int, eps: float = 1e-4) -> np.ndarray:
    """rewards: shape (N,), N a multiple of group_size, where consecutive runs
    of `group_size` are the G completions of the SAME prompt. Returns advantages
    of the same shape, normalized independently within each group."""
    if len(rewards) % group_size != 0:
        raise ValueError(f"{len(rewards)} rewards not divisible by group_size {group_size}")
    groups = rewards.reshape(-1, group_size)
    mean = groups.mean(axis=1, keepdims=True)
    std = groups.std(axis=1, keepdims=True)
    adv = (groups - mean) / (std + eps)
    return adv.reshape(-1)


def token_logp(model: Transformer, ids: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
    """Per-token log p(target) for next-token prediction, plus the target-aligned
    mask. Shapes: ids/mask (B, L) in -> both outputs (B, L-1)."""
    logits = model(ids[:, :-1])
    targets = ids[:, 1:]
    neg_logp = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(ids.shape[0], -1)
    return -neg_logp, mask[:, 1:]


def grpo_loss(
    policy: Transformer, ref: Transformer,
    ids: mx.array, mask: mx.array, advantages: mx.array,
    beta_kl: float = 0.04,
) -> tuple[mx.array, dict[str, mx.array]]:
    """ids/mask: (B, L), one row per sampled completion (prompt+completion,
    right-padded, mask=1 on completion tokens only - same convention as DPO).
    advantages: (B,) one group-normalized scalar per completion.

    Single-rollout-per-update simplification: pi_theta_old (the policy that did
    the sampling) equals pi_theta at the moment this loss is first evaluated -
    no parameter update has happened yet - so the PPO-style probability ratio
    pi_theta/pi_theta_old is exactly 1 and the clipped surrogate degenerates to
    plain policy gradient: -(advantage * logp). We rely on this (one rollout,
    one gradient step) rather than implementing importance-ratio clipping for
    multiple updates per rollout.
    """
    policy_logp, m = token_logp(policy, ids, mask)
    ref_logp, _ = token_logp(ref, ids, mask)

    kl = mx.exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1  # k3 estimator, >= 0
    adv = advantages[:, None]
    per_token_loss = -(adv * policy_logp) + beta_kl * kl

    denom = mx.maximum(m.sum(), 1.0)
    loss = (per_token_loss * m).sum() / denom
    mean_kl = (kl * m).sum() / denom
    return loss, {"kl": mean_kl, "mean_advantage": mx.mean(advantages)}
