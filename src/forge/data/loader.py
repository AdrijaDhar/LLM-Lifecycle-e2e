"""Batch loader over packed uint16 token files.

The packed `.bin` is one long array of token ids. For next-token prediction a
training example is any window of `seq_len + 1` consecutive tokens: the first
`seq_len` are the input, the same window shifted by one is the target.

We sample windows at uniformly random offsets (with replacement). That's the
standard approach - across a run every token gets seen many times, and random
offsets decorrelate consecutive batches without the bookkeeping of a shuffled
epoch.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_meta(tokens_dir: str | Path) -> dict:
    return json.loads((Path(tokens_dir) / "meta.json").read_text())


class TokenLoader:
    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int,
        batch_size: int,
        dtype: str = "uint16",
        seed: int = 0,
    ) -> None:
        self.data = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        if len(self.data) < seq_len + 1:
            raise ValueError(f"{bin_path} has only {len(self.data)} tokens")

    @classmethod
    def from_dir(cls, tokens_dir: str | Path, split: str, seq_len: int, batch_size: int, seed: int = 0):
        meta = load_meta(tokens_dir)
        return cls(
            Path(tokens_dir) / f"{split}.bin",
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=meta["dtype"],
            seed=seed,
        )

    @property
    def n_tokens(self) -> int:
        return len(self.data)

    def batch(self) -> tuple[mx.array, mx.array]:
        hi = len(self.data) - self.seq_len - 1
        offs = self.rng.integers(0, hi, size=self.batch_size)
        # gather windows -> (batch, seq_len+1), then split into input / target
        idx = offs[:, None] + np.arange(self.seq_len + 1)[None, :]
        chunk = np.asarray(self.data[idx.reshape(-1)]).reshape(self.batch_size, self.seq_len + 1)
        x = mx.array(chunk[:, :-1].astype(np.int32))
        y = mx.array(chunk[:, 1:].astype(np.int32))
        return x, y

    def iter_val(self, max_batches: int):
        """Deterministic sequential sweep for validation loss."""
        step = self.batch_size * self.seq_len
        for b in range(max_batches):
            start = b * step
            if start + step + 1 > len(self.data):
                return
            block = np.asarray(self.data[start : start + step + 1])
            x = block[:-1].reshape(self.batch_size, self.seq_len).astype(np.int32)
            y = block[1:].reshape(self.batch_size, self.seq_len).astype(np.int32)
            yield mx.array(x), mx.array(y)


class SFTLoader(TokenLoader):
    """Like TokenLoader, but also carries a per-token loss mask (1 = train on
    this target token, i.e. it's assistant content; 0 = context, don't train).

    `mask_path` is a flat uint8 array the same length as the token array,
    produced alongside it by `scripts/prepare_sft_data.py`.
    """

    def __init__(self, bin_path, mask_path, seq_len, batch_size, dtype="uint16", seed=0):
        super().__init__(bin_path, seq_len, batch_size, dtype=dtype, seed=seed)
        self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
        if len(self.mask) != len(self.data):
            raise ValueError(f"token/mask length mismatch: {len(self.data)} vs {len(self.mask)}")

    @classmethod
    def from_dir(cls, tokens_dir, split, seq_len, batch_size, seed=0):
        meta = load_meta(tokens_dir)
        d = Path(tokens_dir)
        return cls(d / f"{split}.bin", d / f"{split}_mask.bin", seq_len, batch_size,
                   dtype=meta["dtype"], seed=seed)

    def batch(self) -> tuple[mx.array, mx.array, mx.array]:
        hi = len(self.data) - self.seq_len - 1
        offs = self.rng.integers(0, hi, size=self.batch_size)
        idx = offs[:, None] + np.arange(self.seq_len + 1)[None, :]
        flat = idx.reshape(-1)
        chunk = np.asarray(self.data[flat]).reshape(self.batch_size, self.seq_len + 1)
        mchunk = np.asarray(self.mask[flat]).reshape(self.batch_size, self.seq_len + 1)
        x = mx.array(chunk[:, :-1].astype(np.int32))
        y = mx.array(chunk[:, 1:].astype(np.int32))
        # mask aligned with the TARGET at each position (mask[t] says "was the
        # token being predicted, y[...,t], assistant content?")
        loss_mask = mx.array(mchunk[:, 1:].astype(np.float32))
        return x, y, loss_mask

    def iter_val(self, max_batches: int):
        step = self.batch_size * self.seq_len
        for b in range(max_batches):
            start = b * step
            if start + step + 1 > len(self.data):
                return
            block = np.asarray(self.data[start : start + step + 1])
            mblock = np.asarray(self.mask[start : start + step + 1])
            x = block[:-1].reshape(self.batch_size, self.seq_len).astype(np.int32)
            y = block[1:].reshape(self.batch_size, self.seq_len).astype(np.int32)
            m = mblock[1:].reshape(self.batch_size, self.seq_len).astype(np.float32)
            yield mx.array(x), mx.array(y), mx.array(m)
