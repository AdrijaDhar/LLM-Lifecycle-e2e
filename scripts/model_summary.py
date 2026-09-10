"""Print the parameter breakdown and a generation smoke test for a preset.

    uv run python scripts/model_summary.py 135m
    uv run python scripts/model_summary.py tiny --generate
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx.utils import tree_flatten

from forge.model.config import PRESETS
from forge.model.generate import generate
from forge.model.transformer import Transformer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--generate", action="store_true", help="run a short generation from random weights")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    model = Transformer(cfg)
    mx.eval(model.parameters())

    print(f"preset: {args.preset}")
    for f in ("dim", "n_layers", "n_heads", "n_kv_heads", "head_dim", "hidden_dim", "vocab_size", "max_seq_len"):
        print(f"  {f:12s} {getattr(cfg, f)}")
    print(f"  tie_embeddings {cfg.tie_embeddings}")

    # group params by component
    groups: dict[str, int] = {}
    for name, p in tree_flatten(model.parameters()):
        key = "embedding" if name.startswith("tok_emb") else (
            "final_norm" if name.startswith("final_norm") else (
                "lm_head" if name.startswith("lm_head") else name.split(".")[1] + "." + name.split(".")[3]
                if name.startswith("blocks") else name
            )
        )
        groups[key] = groups.get(key, 0) + p.size

    total = sum(groups.values())
    print(f"\n  total params: {total:,}  ({total / 1e6:.1f}M)")
    print(f"  formula says: {cfg.n_params:,}")
    emb = groups.get("embedding", 0)
    print(f"  embedding is {100 * emb / total:.1f}% of params "
          f"({'shared with output' if cfg.tie_embeddings else 'untied'})")

    # bytes at bf16 / fp32
    print(f"\n  weights: {total * 2 / 1e6:.0f} MB (bf16), {total * 4 / 1e6:.0f} MB (fp32)")

    if args.generate:
        prompt = [1, 2, 3, 4, 5]
        t0 = time.time()
        toks = list(generate(model, prompt, max_new_tokens=32, temperature=0.8))
        dt = time.time() - t0
        print(f"\n  generated {len(toks)} tokens from random weights in {dt:.2f}s "
              f"({len(toks) / dt:.1f} tok/s) -> {toks[:12]}...")


if __name__ == "__main__":
    main()
