"""Encode (prompt, chosen, rejected) preference triples for DPO.

Each of `chosen`/`rejected` becomes ONE full sequence: the same chat-formatted
prompt, followed by that completion. Sequences are right-padded to a fixed
`max_len` with the tokenizer's <|endoftext|> id. Right-padding is safe for a
causal decoder with no extra masking work: attention only looks backward, so
padding at the end can never influence any real token's representation - we
just need to make sure padded positions never contribute to the loss, which
the returned mask (1 = completion token, 0 = prompt or pad) takes care of.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from forge.data.chat import ChatFormat
from forge.tokenizer.bpe import BPETokenizer


def _build_sequence(
    prompt_ids: list[int], completion: str, tok: BPETokenizer, fmt: ChatFormat,
    max_len: int, pad_id: int,
) -> tuple[list[int], list[int]] | None:
    comp_ids = tok.encode(completion) + [fmt.end_id]
    ids = prompt_ids + comp_ids
    mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
    if len(ids) > max_len:
        return None
    pad = max_len - len(ids)
    return ids + [pad_id] * pad, mask + [0] * pad


def encode_pair(
    tok: BPETokenizer, fmt: ChatFormat, prompt: str, chosen: str, rejected: str,
    max_len: int, pad_id: int | None = None,
) -> dict | None:
    """Returns None if either completion doesn't fit in max_len (caller should skip)."""
    pad_id = fmt.end_id if pad_id is None else pad_id  # any id is fine; masked out either way
    prompt_ids = [fmt.user_id] + tok.encode(prompt) + [fmt.end_id, fmt.assistant_id]

    c = _build_sequence(prompt_ids, chosen, tok, fmt, max_len, pad_id)
    r = _build_sequence(prompt_ids, rejected, tok, fmt, max_len, pad_id)
    if c is None or r is None:
        return None
    return {
        "chosen_ids": c[0], "chosen_mask": c[1],
        "rejected_ids": r[0], "rejected_mask": r[1],
    }


class PreferenceLoader:
    """Small enough (thousands of pairs, not billions of tokens) to just hold
    in memory and sample random batches - unlike TokenLoader/SFTLoader's
    memmapped streams, there's no giant file to avoid loading."""

    def __init__(self, npy_dir: str | Path, split: str, seed: int = 0) -> None:
        d = Path(npy_dir)
        self.chosen_ids = np.load(d / f"{split}_chosen_ids.npy")
        self.chosen_mask = np.load(d / f"{split}_chosen_mask.npy")
        self.rejected_ids = np.load(d / f"{split}_rejected_ids.npy")
        self.rejected_mask = np.load(d / f"{split}_rejected_mask.npy")
        self.n = len(self.chosen_ids)
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        idx = self.rng.integers(0, self.n, size=batch_size)
        return (
            mx.array(self.chosen_ids[idx].astype(np.int32)),
            mx.array(self.chosen_mask[idx].astype(np.float32)),
            mx.array(self.rejected_ids[idx].astype(np.int32)),
            mx.array(self.rejected_mask[idx].astype(np.float32)),
        )

    def iter_val(self, batch_size: int, max_batches: int):
        for b in range(min(max_batches, self.n // batch_size)):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            yield (
                mx.array(self.chosen_ids[sl].astype(np.int32)),
                mx.array(self.chosen_mask[sl].astype(np.float32)),
                mx.array(self.rejected_ids[sl].astype(np.int32)),
                mx.array(self.rejected_mask[sl].astype(np.float32)),
            )
