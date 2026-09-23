"""Autoregressive generation with a KV cache.

At generation time the expensive thing is recomputing attention keys/values for
every past token on every new step. The KV cache stores them: we run the full
prompt once ("prefill"), then feed one token at a time ("decode"), each step
appending its k/v to the cache and attending over everything so far.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import numpy as np

from forge.model.transformer import Transformer


class KVCache:
    """Per-layer key/value store. `offset` is how many positions are cached
    (RoPE needs it to rotate new tokens to the right absolute position)."""

    def __init__(self) -> None:
        self.k: mx.array | None = None
        self.v: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)
        self.offset = self.k.shape[2]
        return self.k, self.v


def _sample(
    logits: mx.array, temperature: float, top_k: int | None,
    seen: set[int] | None = None, repetition_penalty: float = 1.0,
) -> mx.array:
    if repetition_penalty != 1.0 and seen:
        # Base models love collapsing into loops ("Paris. Paris. Paris...") once a
        # phrase repeats - it becomes the single most confident continuation.
        # Penalize already-generated tokens: divide positive logits, multiply
        # negative ones, so a "penalty" always pushes the logit down.
        arr = np.array(logits, copy=True)
        idx = np.fromiter(seen, dtype=np.int64)
        vals = arr[..., idx]
        arr[..., idx] = np.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        logits = mx.array(arr)

    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_k is not None:
        kth = mx.sort(logits, axis=-1)[..., -top_k][..., None]
        logits = mx.where(logits < kth, -mx.inf, logits)
    return mx.random.categorical(logits)


def generate(
    model: Transformer,
    prompt_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
    eot_id: int | None = None,
    repetition_penalty: float = 1.3,
) -> Iterator[int]:
    """Yield generated token ids one at a time."""
    cache = [KVCache() for _ in model.blocks]

    x = mx.array(prompt_ids, dtype=mx.int32)[None]
    logits = model(x, cache=cache)[:, -1]
    seen: set[int] = set(prompt_ids)
    # Never penalize the intended stop token, even if it already appears in the
    # prompt (e.g. a chat prompt's user turn closes with the same <|end|> the
    # assistant turn should end with) - otherwise repetition penalty actively
    # suppresses correct stopping.
    seen.discard(eot_id)

    for _ in range(max_new_tokens):
        next_id = _sample(logits, temperature, top_k, seen, repetition_penalty)
        mx.eval(next_id)
        tok = int(next_id.item())
        if eot_id is not None and tok == eot_id:
            return
        yield tok
        seen.add(tok)
        logits = model(next_id[None], cache=cache)[:, -1]
