"""Autoregressive generation with a KV cache.

At generation time the expensive thing is recomputing attention keys/values for
every past token on every new step. The KV cache stores them: we run the full
prompt once ("prefill"), then feed one token at a time ("decode"), each step
appending its k/v to the cache and attending over everything so far.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx

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


def _sample(logits: mx.array, temperature: float, top_k: int | None) -> mx.array:
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
) -> Iterator[int]:
    """Yield generated token ids one at a time."""
    cache = [KVCache() for _ in model.blocks]

    x = mx.array(prompt_ids, dtype=mx.int32)[None]
    logits = model(x, cache=cache)[:, -1]

    for _ in range(max_new_tokens):
        next_id = _sample(logits, temperature, top_k)
        mx.eval(next_id)
        tok = int(next_id.item())
        if eot_id is not None and tok == eot_id:
            return
        yield tok
        logits = model(next_id[None], cache=cache)[:, -1]
