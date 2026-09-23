"""Tests for sampling / generation, especially repetition-penalty edge cases.

Run: uv run pytest -q tests/test_generate.py
"""

from __future__ import annotations

from unittest.mock import patch

import mlx.core as mx

from forge.model.config import PRESETS
from forge.model.generate import _sample, generate
from forge.model.transformer import Transformer


def test_repetition_penalty_can_flip_the_argmax() -> None:
    # token 0 starts ahead of token 1; a strong penalty on 0 (already "seen")
    # should push 1's logit above it.
    logits = mx.array([[5.0, 4.0]])
    picked_no_penalty = int(_sample(logits, temperature=0.0, top_k=None, seen=set()).item())
    picked_penalized = int(_sample(logits, temperature=0.0, top_k=None, seen={0}, repetition_penalty=2.0).item())
    assert picked_no_penalty == 0
    assert picked_penalized == 1


def test_repetition_penalty_off_ignores_seen() -> None:
    logits = mx.array([[5.0, 4.0]])
    picked = int(_sample(logits, temperature=0.0, top_k=None, seen={0}, repetition_penalty=1.0).item())
    assert picked == 0


def test_eot_in_prompt_is_never_penalized() -> None:
    """Regression test: a chat prompt's user turn closes with the same <|end|>
    the assistant should stop with. That token must not be excluded from
    consideration by the repetition penalty just because it's already in the
    prompt - otherwise the model can never emit its own stop token."""
    cfg = PRESETS["tiny"]
    model = Transformer(cfg)
    eot_id = 5
    prompt_ids = [1, 2, eot_id, 3]  # eot_id deliberately present in the prompt

    with patch("forge.model.generate._sample", wraps=__import__(
        "forge.model.generate", fromlist=["_sample"])._sample) as spy:
        next(generate(model, prompt_ids, max_new_tokens=1, eot_id=eot_id, repetition_penalty=1.3), None)
        seen_arg = spy.call_args.args[3]
        assert eot_id not in seen_arg
