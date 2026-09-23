"""Quantize a trained checkpoint to INT4 (or INT8) for faster, smaller local inference.

MLX quantizes per-layer, per-group: each weight matrix is split into groups of
`--group-size` values along its last dimension, and each group gets its own
scale (+ bias for affine mode) so `bits`-wide integers can represent it with
little precision loss - the classic weight-only post-training quantization
recipe (same idea as GGUF's Q4_0/Q4_1, GPTQ, etc., minus the per-group error
correction those add).

    uv run python scripts/quantize.py checkpoints/135m-grpo-0921-1902/last
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from forge.model.config import PRESETS
from forge.model.generate import generate
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer
from forge.tokenizer.stream import StreamDecoder
from forge.training import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = "The capital of France is"


def _size_mb(model) -> float:
    return sum(p.nbytes for _, p in tree_flatten(model.parameters())) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "fw32k-chat.bpe.json")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 6, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    st = json.loads((args.ckpt / "state.json").read_text())
    tok = BPETokenizer()
    tok.load(str(args.tokenizer))
    cfg = PRESETS[st["preset"]]
    cfg.vocab_size = tok.vocab_size

    print(f"loading fp32 checkpoint: {args.ckpt}")
    fp_model = Transformer(cfg)
    checkpoint.load(args.ckpt, fp_model)
    mx.eval(fp_model.parameters())
    fp_size = _size_mb(fp_model)

    fp_out = "".join(t for t in _sample(fp_model, tok, PROBE))
    print(f"fp32 sample: {PROBE!r} -> {fp_out!r}")

    print(f"quantizing: {args.bits}-bit, group_size={args.group_size} ...")
    nn.quantize(fp_model, group_size=args.group_size, bits=args.bits)
    mx.eval(fp_model.parameters())
    q_size = _size_mb(fp_model)

    q_out = "".join(t for t in _sample(fp_model, tok, PROBE))
    print(f"int{args.bits} sample: {PROBE!r} -> {q_out!r}")

    out_dir = args.out or (args.ckpt.parent / f"quantized-int{args.bits}")
    checkpoint.save(out_dir, fp_model, optim.AdamW(learning_rate=0.0), step=st["step"],
                    extra={**st, "quantized": True, "bits": args.bits, "group_size": args.group_size})

    print(f"\nsize: {fp_size:.0f} MB -> {q_size:.0f} MB ({fp_size / q_size:.1f}x smaller)")
    print(f"-> {out_dir}")


def _sample(model, tok, prompt, max_new_tokens=40):
    ids = tok.encode(prompt)
    dec = StreamDecoder(tok)
    model.eval()
    for tid in generate(model, ids, max_new_tokens=max_new_tokens, temperature=0.0, eot_id=tok.eot_id):
        yield dec.push(tid)
    yield dec.flush()


if __name__ == "__main__":
    main()
