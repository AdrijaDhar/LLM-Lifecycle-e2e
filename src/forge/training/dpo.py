"""DPO (Direct Preference Optimization) loss.

DPO's trick: you can push a policy toward preferred responses WITHOUT training
a separate reward model or doing on-policy RL rollouts (like PPO/GRPO need).
Instead, compare the policy's preference for chosen-over-rejected against a
FROZEN reference model's (usually a copy of the model before this stage) same
preference:

    reward(response) ~= beta * (log pi_policy(response) - log pi_ref(response))

    L = -log sigmoid( reward(chosen) - reward(rejected) )

If the policy starts equal to the reference (step 0), reward(chosen) ==
reward(rejected) == 0, so the loss is exactly log(2) for every pair - a clean
sanity check (see tests). Training then increases the policy's *relative*
log-probability of chosen over rejected, without needing to know the "true"
reward - just which of two responses is better.

`beta` controls how far the policy is allowed to drift from the reference:
higher beta = stays closer to reference (more conservative); lower beta =
larger updates per preference (risks the "reward hacking" a real reward model
would show, informally).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from forge.model.transformer import Transformer


def sequence_logp(model: Transformer, ids: mx.array, mask: mx.array) -> mx.array:
    """Sum of log p(token) over positions where mask==1 (the completion span),
    for next-token prediction over `ids`. Returns one scalar per sequence."""
    logits = model(ids[:, :-1])
    targets = ids[:, 1:]
    neg_logp = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(ids.shape[0], -1)
    m = mask[:, 1:].astype(neg_logp.dtype)  # mask aligns with the TARGET at each position
    return -(neg_logp * m).sum(axis=1)


def dpo_loss(
    policy: Transformer, ref: Transformer,
    chosen_ids: mx.array, chosen_mask: mx.array,
    rejected_ids: mx.array, rejected_mask: mx.array,
    beta: float = 0.1,
) -> tuple[mx.array, dict[str, mx.array]]:
    logp_c = sequence_logp(policy, chosen_ids, chosen_mask)
    logp_r = sequence_logp(policy, rejected_ids, rejected_mask)
    ref_logp_c = sequence_logp(ref, chosen_ids, chosen_mask)
    ref_logp_r = sequence_logp(ref, rejected_ids, rejected_mask)

    chosen_reward = beta * (logp_c - ref_logp_c)
    rejected_reward = beta * (logp_r - ref_logp_r)
    logits = chosen_reward - rejected_reward

    # -log sigmoid(logits), computed stably as softplus(-logits)
    loss = mx.logaddexp(mx.zeros_like(logits), -logits)

    metrics = {
        "chosen_reward": mx.mean(chosen_reward),
        "rejected_reward": mx.mean(rejected_reward),
        "margin": mx.mean(chosen_reward - rejected_reward),
        "accuracy": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
    }
    return mx.mean(loss), metrics
