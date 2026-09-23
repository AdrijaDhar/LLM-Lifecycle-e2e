"""Tests for the lm-evaluation-harness adapter - pure logic, no network/task
downloads (those are smoke-tested separately against real harness tasks).

Run: uv run pytest -q tests/test_lm_eval_wrapper.py
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from lm_eval.api.instance import Instance

from forge.eval.lm_eval_wrapper import ForgeLM
from forge.model.config import ModelConfig
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer

TRAIN_TEXT = (
    "the cat sat on the mat. the dog ran to the cat. "
    "please answer the question about the weather today. "
) * 30


def _tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train_fast(TRAIN_TEXT, vocab_size=350)
    t.register_special("<|endoftext|>")
    return t


def _lm() -> ForgeLM:
    tok = _tok()
    cfg = ModelConfig(vocab_size=tok.vocab_size, dim=32, n_layers=2, n_heads=2, n_kv_heads=1,
                       hidden_dim=64, max_seq_len=64)
    model = Transformer(cfg)
    return ForgeLM(model, tok, max_length=64)


def _inst(request_type, args):
    return Instance(request_type=request_type, doc={}, arguments=args, idx=0)


def test_loglikelihood_matches_manual_cross_entropy() -> None:
    lm = _lm()
    context, continuation = "the cat sat", " on the mat"
    (lp, _), = lm.loglikelihood([_inst("loglikelihood", (context, continuation))])

    ctx_ids = lm._encode(context)
    cont_ids = lm._encode(continuation)
    ids = ctx_ids + cont_ids
    logits = lm.model(mx.array(ids[:-1])[None])[0]
    targets = mx.array(ids[len(ctx_ids):])
    manual = -float(nn.losses.cross_entropy(
        logits[len(ctx_ids) - 1:], targets, reduction="sum"
    ))
    assert abs(lp - manual) < 1e-3


def test_loglikelihood_empty_continuation_is_zero() -> None:
    lm = _lm()
    (lp, is_greedy), = lm.loglikelihood([_inst("loglikelihood", ("the cat sat", ""))])
    assert lp == 0.0 and is_greedy is True


def test_loglikelihood_handles_empty_context() -> None:
    lm = _lm()
    # must not crash - docstring for the harness interface requires this
    out = lm.loglikelihood([_inst("loglikelihood", ("", "the cat sat"))])
    assert len(out) == 1
    assert math.isfinite(out[0][0])


def test_loglikelihood_is_greedy_flag_is_consistent() -> None:
    lm = _lm()
    # whatever greedy decoding actually produces from this context must be flagged is_greedy=True
    from forge.model.generate import generate
    ctx = "the cat sat on the mat"
    ctx_ids = lm._encode(ctx)
    greedy_ids = list(generate(lm.model, ctx_ids, max_new_tokens=4, temperature=0.0, eot_id=None))
    greedy_text = lm.tok.decode(greedy_ids)
    (_, is_greedy), = lm.loglikelihood([_inst("loglikelihood", (ctx, greedy_text))])
    assert is_greedy is True


def test_loglikelihood_rolling_matches_sum_of_full_loglikelihood() -> None:
    lm = _lm()
    text = "the cat sat on the mat"
    (rolling,) = lm.loglikelihood_rolling([_inst("loglikelihood_rolling", (text,))])
    (full, _), = lm.loglikelihood([_inst("loglikelihood", ("", text))])
    assert abs(rolling - full) < 1e-3


def test_generate_until_stops_at_marker() -> None:
    lm = _lm()
    out, = lm.generate_until([_inst("generate_until", ("the cat sat", {"until": ["mat"], "max_gen_toks": 40}))])
    assert "mat" not in out  # truncated before the stop string


def test_generate_until_respects_max_gen_toks() -> None:
    # Re-encoding already-decoded text isn't a reliable token count (replacement
    # characters from a truncated multi-byte tail can re-tokenize to more
    # pieces than the model actually emitted) - so check the raw call instead.
    calls = []
    real_generate = __import__("forge.eval.lm_eval_wrapper", fromlist=["generate"]).generate

    def spy(*args, **kwargs):
        calls.append(kwargs.get("max_new_tokens"))
        return real_generate(*args, **kwargs)

    import forge.eval.lm_eval_wrapper as mod
    mod.generate = spy
    try:
        lm = _lm()
        lm.generate_until([_inst("generate_until", ("the cat", {"until": [], "max_gen_toks": 3}))])
    finally:
        mod.generate = real_generate

    assert calls == [3]
