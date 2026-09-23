"""Chat with an SFT'd checkpoint using the <|user|>/<|assistant|>/<|end|> template.

    uv run python scripts/sample_chat.py checkpoints/135m-sft-.../last
    uv run python scripts/sample_chat.py checkpoints/135m-sft-.../last \
        --prompt "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from forge.data.chat import ChatFormat
from forge.model.config import PRESETS
from forge.model.generate import generate
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer
from forge.tokenizer.stream import StreamDecoder
from forge.training import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Can you explain what photosynthesis is?",
    "Give me a simple recipe for cooking pasta.",
    "Tell me a short story about a village.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "fw32k-chat.bpe.json")
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    args = ap.parse_args()

    st = json.loads((args.ckpt / "state.json").read_text())
    cfg = PRESETS[st["preset"]]

    tok = BPETokenizer()
    tok.load(str(args.tokenizer))
    fmt = ChatFormat.register(tok)  # no-op if already in the saved tokenizer file
    cfg.vocab_size = tok.vocab_size

    model = Transformer(cfg)
    checkpoint.load(args.ckpt, model)
    model.eval()
    mx.eval(model.parameters())

    print(f"loaded {st['preset']} ({cfg.n_params / 1e6:.1f}M params) "
          f"from step {st['step']} (val loss {st.get('val_loss', '?')})\n")

    for prompt in (args.prompt or DEFAULT_PROMPTS):
        ids = [fmt.user_id] + tok.encode(prompt) + [fmt.end_id, fmt.assistant_id]
        print(f"> {prompt}")
        dec = StreamDecoder(tok)
        for tid in generate(model, ids, max_new_tokens=args.tokens, temperature=args.temperature,
                             top_k=args.top_k, eot_id=fmt.end_id, repetition_penalty=args.repetition_penalty):
            print(dec.push(tid), end="", flush=True)
        print(dec.flush(), end="")
        print("\n")


if __name__ == "__main__":
    main()
