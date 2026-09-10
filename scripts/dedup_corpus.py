"""Remove near-duplicate documents with MinHash + LSH.

Reads   data/clean/<source>/shard_*.jsonl
Writes  data/dedup/<source>/shard_*.jsonl
Prints  cluster stats + how many documents (and characters) were removed.

    uv run python scripts/dedup_corpus.py fineweb-edu
    uv run python scripts/dedup_corpus.py fineweb-edu --threshold 0.85 --num-perm 128 --bands 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from forge.data.dedup import LSHIndex, MinHasher, dedup, shingles

REPO_ROOT = Path(__file__).resolve().parents[1]


def iter_docs(shards):
    for shard in shards:
        with shard.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--shingle-k", type=int, default=5, help="words per shingle")
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.8, help="min estimated Jaccard to call a duplicate")
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = REPO_ROOT / "data" / "clean" / args.source
    out_dir = REPO_ROOT / "data" / "dedup" / args.source
    shards = sorted(in_dir.glob("shard_*.jsonl"))
    if not shards:
        raise SystemExit(f"no shards in {in_dir} - run filter_corpus.py first")

    lsh_preview = LSHIndex(num_perm=args.num_perm, bands=args.bands)
    print(f"num_perm={args.num_perm} bands={args.bands} rows={lsh_preview.rows} "
          f"-> LSH catches pairs above ~{lsh_preview.implied_threshold:.2f} Jaccard; "
          f"confirming at >= {args.threshold}")

    n_docs = sum(1 for s in shards for _ in s.open(encoding="utf-8"))
    mh = MinHasher(num_perm=args.num_perm, seed=1)

    # Pass 1: signatures + weights (weight = char length, so we keep the fullest copy).
    sigs = np.empty((n_docs, args.num_perm), dtype=np.uint64)
    weights = np.empty(n_docs, dtype=np.float64)
    t0 = time.time()
    for i, rec in enumerate(tqdm(iter_docs(shards), total=n_docs, desc="minhash")):
        text = rec["text"]
        sigs[i] = mh.signature(shingles(text, k=args.shingle_k))
        weights[i] = len(text)
    print(f"signatures in {time.time() - t0:.1f}s")

    # Pass 2: LSH + union-find -> indices to keep.
    t1 = time.time()
    keep_idx = dedup(sigs, weights=weights, bands=args.bands, threshold=args.threshold)
    keep_mask = np.zeros(n_docs, dtype=bool)
    keep_mask[keep_idx] = True
    n_removed = n_docs - len(keep_idx)
    print(f"clustering in {time.time() - t1:.1f}s")

    # Pass 3: write survivors.
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for p in out_dir.glob("shard_*.jsonl"):
            p.unlink()

    shard_idx = 0
    limit = args.shard_mb * 1_000_000
    shard_f = (out_dir / f"shard_{shard_idx:05d}.jsonl").open("w", encoding="utf-8")
    shard_bytes = 0
    kept_chars = removed_chars = 0

    for i, rec in enumerate(tqdm(iter_docs(shards), total=n_docs, desc="write")):
        if not keep_mask[i]:
            removed_chars += len(rec["text"])
            continue
        kept_chars += len(rec["text"])
        out = json.dumps(rec, ensure_ascii=False) + "\n"
        enc = out.encode("utf-8")
        if shard_bytes + len(enc) > limit:
            shard_f.close()
            shard_idx += 1
            shard_f = (out_dir / f"shard_{shard_idx:05d}.jsonl").open("w", encoding="utf-8")
            shard_bytes = 0
        shard_f.write(out)
        shard_bytes += len(enc)
    shard_f.close()

    total_chars = kept_chars + removed_chars
    print(f"\nremoved {n_removed:,} / {n_docs:,} docs "
          f"({100 * n_removed / n_docs:.1f}%), "
          f"{100 * removed_chars / max(total_chars, 1):.1f}% of characters")
    print(f"-> data/dedup/{args.source}/  ({shard_idx + 1} shards, {kept_chars / 1e6:.0f}M chars)")


if __name__ == "__main__":
    main()
