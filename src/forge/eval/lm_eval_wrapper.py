"""Adapts our MLX Transformer + BPETokenizer to EleutherAI's lm-evaluation-harness
LM interface, so standard tasks (ARC, HellaSwag, MMLU, GSM8K, ...) run against
our own model instead of an HF `transformers` one - no torch dependency needed,
since we only use the harness's base package (task loading, few-shot
formatting, scoring/aggregation), not its built-in HF model backend.

The harness asks three things of an LM:
  loglikelihood(context, continuation)   - how likely is this continuation?
                                            (multiple-choice tasks: score each
                                            choice, pick the highest)
  loglikelihood_rolling(text)            - total log-likelihood of a whole
                                            text (perplexity-style tasks)
  generate_until(context, stop_strings)  - greedy-generate until a stop string
                                            (free-form tasks like GSM8K)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from lm_eval.api.model import LM

from forge.model.generate import generate
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer


class ForgeLM(LM):
    def __init__(self, model: Transformer, tok: BPETokenizer, max_length: int = 1024) -> None:
        super().__init__()
        self.model = model
        self.tok = tok
        self.max_length = max_length
        self.model.eval()

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def _score_continuation(self, ctx_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
        """Sum log p(cont_ids | ctx_ids), and whether greedy decoding would
        have produced cont_ids exactly (used for exact-match-style scoring)."""
        if not ctx_ids:
            ctx_ids = [self.tok.eot_id]  # anchor an empty context the same way BOS would
        ids = ctx_ids + cont_ids
        if len(ids) > self.max_length + 1:
            overflow = len(ids) - (self.max_length + 1)
            ctx_ids = ctx_ids[overflow:]  # truncate context from the left; keep the full continuation
            ids = ctx_ids + cont_ids

        x = mx.array(ids[:-1])[None]
        logits = self.model(x)[0].astype(mx.float32)
        logp = nn.log_softmax(logits, axis=-1)

        start = len(ctx_ids) - 1
        n = len(cont_ids)
        window = logp[start : start + n, :]
        idx = mx.array(cont_ids)
        chosen = window[mx.arange(n), idx]
        is_greedy = bool(mx.all(mx.argmax(window, axis=-1) == idx))
        mx.eval(chosen, is_greedy)
        return float(mx.sum(chosen)), is_greedy

    def loglikelihood(self, requests: list) -> list[tuple[float, bool]]:
        out = []
        for req in requests:
            context, continuation = req.args
            cont_ids = self._encode(continuation)
            if not cont_ids:
                out.append((0.0, True))
                continue
            out.append(self._score_continuation(self._encode(context), cont_ids))
        return out

    def loglikelihood_rolling(self, requests: list) -> list[float]:
        out = []
        for req in requests:
            (text,) = req.args
            ids = self._encode(text)
            total = 0.0
            ctx: list[int] = [self.tok.eot_id]
            pos = 0
            while pos < len(ids):
                chunk = ids[pos : pos + self.max_length]
                lp, _ = self._score_continuation(ctx, chunk)
                total += lp
                ctx = chunk[-1:]  # small running anchor for the next chunk
                pos += len(chunk)
            out.append(total)
        return out

    def generate_until(self, requests: list) -> list[str]:
        out = []
        for req in requests:
            context, gen_kwargs = req.args
            gen_kwargs = dict(gen_kwargs or {})
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_gen = int(gen_kwargs.get("max_gen_toks", 256))

            ctx_ids = self._encode(context)
            budget = max(self.max_length - max_gen, 1)
            if len(ctx_ids) > budget:
                ctx_ids = ctx_ids[-budget:]

            gen_ids: list[int] = []
            text = ""
            for tid in generate(self.model, ctx_ids, max_new_tokens=max_gen, temperature=0.0,
                                 eot_id=self.tok.eot_id):
                gen_ids.append(tid)
                text = self.tok.decode(gen_ids)
                hit = next((u for u in until if u and u in text), None)
                if hit is not None:
                    text = text.split(hit)[0]
                    break
            out.append(text)
        return out
