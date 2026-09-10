"""Download a small slice of an open pretraining corpus to a local text file.

We stream from the Hugging Face Hub so we never download the full dataset -
we pull `--docs` documents and stop. Output is one document per line-block,
separated by a blank line, as plain UTF-8 text.

Examples
--------
    # ~5 MB of educational web text (good first tokenizer corpus)
    uv run python scripts/prepare_data.py fineweb-edu --docs 8000

    # tiny, extremely clean synthetic stories (fast sanity checks)
    uv run python scripts/prepare_data.py tinystories --docs 20000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# name -> (hf repo id, config name, text column)
SOURCES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "tinystories": ("roneneldan/TinyStories", None, "text"),
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("--docs", type=int, default=8000, help="number of documents to pull")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=None, help="output .txt path")
    args = ap.parse_args()

    repo_id, config, text_col = SOURCES[args.source]
    out_path = args.out or REPO_ROOT / "data" / "raw" / f"{args.source}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"streaming {repo_id} ({config or 'default'}) split={args.split}")
    ds = load_dataset(repo_id, config, split=args.split, streaming=True)

    n_chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(ds, total=args.docs, desc="docs")):
            if i >= args.docs:
                break
            text = row[text_col].strip()
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            n_chars += len(text)

    mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path.relative_to(REPO_ROOT)}  ({mb:.1f} MB, {n_chars:,} chars)")


if __name__ == "__main__":
    main()
    os._exit(0)  # HF datasets streaming stalls a clean exit; our file is closed
