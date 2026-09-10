"""Download an open pretraining corpus to local JSONL shards, up to a size budget.

Unlike `prepare_data.py` (a quick flat-text sample), this is the real ingestion:
  - streams from the Hugging Face Hub, stops at `--target-gb` of raw text
  - keeps one document per line as JSON, preserving useful metadata
    (quality score, url, id, token/char counts) for the filtering step
  - rotates to a new shard file every `--shard-mb` megabytes
  - resumable: re-running skips shards that already exist (use --overwrite to redo)

Examples
--------
    uv run python scripts/download_corpus.py fineweb-edu --target-gb 1.0
    uv run python scripts/download_corpus.py fineweb-edu --target-gb 5.0 --shard-mb 256
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# name -> (repo id, config, text column, list of metadata columns to keep if present)
SOURCES = {
    "fineweb-edu": (
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "text",
        ["id", "url", "score", "language_score", "token_count", "dump"],
    ),
    "tinystories": ("roneneldan/TinyStories", None, "text", []),
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("--target-gb", type=float, default=1.0, help="stop after this much raw text")
    ap.add_argument("--shard-mb", type=int, default=256, help="megabytes of text per shard file")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-chars", type=int, default=200, help="skip documents shorter than this")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo_id, config, text_col, meta_cols = SOURCES[args.source]
    out_dir = REPO_ROOT / "data" / "raw" / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for p in out_dir.glob("shard_*.jsonl"):
            p.unlink()

    existing = sorted(out_dir.glob("shard_*.jsonl"))
    bytes_done = sum(p.stat().st_size for p in existing)
    shard_idx = len(existing)
    target_bytes = int(args.target_gb * 1e9)
    shard_bytes_limit = args.shard_mb * 1_000_000

    if bytes_done >= target_bytes:
        print(f"already have {bytes_done / 1e9:.2f} GB in {len(existing)} shards; nothing to do")
        return

    print(f"streaming {repo_id} ({config or 'default'}); target {args.target_gb} GB, resuming at {bytes_done / 1e9:.2f} GB")
    ds = load_dataset(repo_id, config, split=args.split, streaming=True)

    # Skip forward over documents we already wrote in a previous run (approximate:
    # streaming datasets are deterministically ordered, so we re-read and skip).
    docs_to_skip = 0
    for p in existing:
        with p.open() as f:
            docs_to_skip += sum(1 for _ in f)

    n_written = 0
    n_skipped_short = 0
    shard_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
    shard_f = shard_path.open("a", encoding="utf-8")
    shard_size = shard_path.stat().st_size if shard_path.exists() else 0
    t0 = time.time()
    pbar = tqdm(total=target_bytes, initial=bytes_done, unit="B", unit_scale=True, desc="text")

    try:
        for i, row in enumerate(ds):
            if i < docs_to_skip:
                continue
            text = row[text_col].strip()
            if len(text) < args.min_chars:
                n_skipped_short += 1
                continue

            rec = {"text": text}
            for c in meta_cols:
                if c in row:
                    rec[c] = row[c]
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            enc = line.encode("utf-8")

            if shard_size + len(enc) > shard_bytes_limit:
                shard_f.close()
                shard_idx += 1
                shard_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
                shard_f = shard_path.open("w", encoding="utf-8")
                shard_size = 0

            shard_f.write(line)
            shard_size += len(enc)
            bytes_done += len(enc)
            n_written += 1
            pbar.update(len(enc))

            if bytes_done >= target_bytes:
                break
    finally:
        shard_f.close()
        pbar.close()

    dt = time.time() - t0
    print(
        f"wrote {n_written:,} docs ({bytes_done / 1e9:.2f} GB) across {shard_idx + 1} shards "
        f"in {dt / 60:.1f} min; skipped {n_skipped_short:,} short docs"
    )
    print(f"-> {out_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
    # HF `datasets` streaming leaves background threads that stall a clean exit.
    # Our files are flushed and closed above, so hard-exit instead of hanging.
    os._exit(0)
