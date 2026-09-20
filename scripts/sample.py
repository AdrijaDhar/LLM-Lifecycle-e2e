"""Load a trained checkpoint and generate text from it.

    uv run python scripts/sample.py checkpoints/135m-0911-0035/last
    uv run python scripts/sample.py checkpoints/135m-0911-0035/last \
        --prompt "The history of the Roman Empire" --temperature 0.7 --tokens 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from forge.model.config import PRESETS
from forge.model.generate import generate
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer
from forge.training import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPTS = [
    "The history of the Roman Empire",
    "Once upon a time, in a small village,",
    "Photosynthesis is the process by which",
    "The most important thing to remember about cooking pasta is",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="checkpoint dir, e.g. checkpoints/135m-.../last")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "fw32k.bpe.json")
    ap.add_argument("--prompt", action="append", default=None, help="repeatable; default = 4 sample prompts")
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.3,
                     help="1.0 = off. >1 discourages repeating tokens (fixes loops)")
    args = ap.parse_args()

    st = json.loads((args.ckpt / "state.json").read_text())
    cfg = PRESETS[st["preset"]]

    tok = BPETokenizer()
    tok.load(str(args.tokenizer))
    cfg.vocab_size = tok.vocab_size

    model = Transformer(cfg)
    checkpoint.load(args.ckpt, model)
    model.eval()
    mx.eval(model.parameters())

    print(f"loaded {st['preset']} ({cfg.n_params / 1e6:.1f}M params) "
          f"from step {st['step']} (val loss {st.get('val_loss', '?')})\n")

    for prompt in (args.prompt or DEFAULT_PROMPTS):
        ids = tok.encode(prompt)
        print(f"--- {prompt!r} ---")
        print(prompt, end="", flush=True)
        for tid in generate(model, ids, max_new_tokens=args.tokens,
                             temperature=args.temperature, top_k=args.top_k, eot_id=tok.eot_id,
                             repetition_penalty=args.repetition_penalty):
            print(tok.decode([tid]), end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
