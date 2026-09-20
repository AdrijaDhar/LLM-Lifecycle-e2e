"""Warmup-Stable-Decay (WSD) learning-rate schedule.

Classic cosine decay ties the whole curve to a fixed total step count - you
can't extend a run or change the data mix partway without redoing the schedule.
WSD fixes that:

    lr
    |        ______________________________
    |       /                              \\
    |      /  (stable: constant peak lr)     \\  (decay)
    |     /                                   \\_____
    |    / (warmup)
    +---+----------------------------------+--------+---> step
        0                          decay_start   total

- **Warmup** (linear 0 -> peak): stops the first few steps from blowing up while
  Adam's moment estimates are still garbage.
- **Stable** (constant peak): the bulk of training. Because the lr is flat, a
  checkpoint from the middle of this phase is a perfectly good "base model" you
  can branch from - anneal it on any data mix you like.
- **Decay** (peak -> ~0): a short final cooldown. Most of the loss improvement of
  a cosine schedule turns out to come from this last drop; WSD just defers it.

We shape the decay as a half-cosine by default.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mlx.optimizers as optim


def wsd_lr(
    step: int,
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int = 2000,
    decay_frac: float = 0.2,
    final_lr_frac: float = 0.0,
    shape: str = "cosine",
) -> float:
    decay_steps = max(1, int(total_steps * decay_frac))
    decay_start = total_steps - decay_steps
    final_lr = peak_lr * final_lr_frac

    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step < decay_start:
        return peak_lr
    if step >= total_steps:
        return final_lr

    t = (step - decay_start) / decay_steps  # 0 -> 1 across the decay phase
    if shape == "linear":
        frac = 1.0 - t
    elif shape == "cosine":
        frac = 0.5 * (1.0 + math.cos(math.pi * t))
    elif shape == "sqrt":
        frac = 1.0 - math.sqrt(t)
    else:
        raise ValueError(f"unknown decay shape {shape!r}")
    return final_lr + (peak_lr - final_lr) * frac


def wsd_schedule(
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int = 2000,
    decay_frac: float = 0.2,
    final_lr_frac: float = 0.0,
) -> Callable:
    """WSD as a composed MLX schedule (callable of the optimizer's step count).

    Built from MLX primitives so it works inside `mx.compile`. Matches `wsd_lr`
    closely (cosine decay). Pass straight to `optim.AdamW(learning_rate=...)`.
    """
    warmup = optim.linear_schedule(peak_lr / max(warmup_steps, 1), peak_lr, warmup_steps)

    if decay_frac <= 0:
        # Warmup then constant peak forever. The run ends on --steps / --max-minutes;
        # anneal later with scripts/anneal.py. This is the WSD "decide length late" path.
        return optim.join_schedules(
            [warmup, optim.linear_schedule(peak_lr, peak_lr, 1 << 30)],
            [warmup_steps],
        )

    decay_steps = max(1, int(total_steps * decay_frac))
    stable_steps = max(1, total_steps - warmup_steps - decay_steps)
    final_lr = peak_lr * final_lr_frac
    return optim.join_schedules(
        [
            warmup,
            optim.linear_schedule(peak_lr, peak_lr, stable_steps),   # constant
            optim.cosine_decay(peak_lr, decay_steps, end=final_lr),
        ],
        [warmup_steps, warmup_steps + stable_steps],
    )
