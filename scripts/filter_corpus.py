"""Apply heuristic quality filters to raw JSONL shards.

Reads   data/raw/<source>/shard_*.jsonl
Writes  data/clean/<source>/shard_*.jsonl   (kept documents, same schema)
Prints  keep rate + a histogram of *why* documents were dropped.

    uv run python scripts/filter_corpus.py fineweb-edu
    uv run python scripts/filter_corpus.py fineweb-edu --min-score 3.0 --sample 5000
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from forge.data.quality import QualityConfig, check_document

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--min-score", type=float, default=None, help="drop docs below this FineWeb-EDU score")
    ap.add_argument("--min-language-score", type=float, default=0.65)
    ap.add_argument("--sample", type=int, default=None, help="dry run: check N docs, write nothing, just report")
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = REPO_ROOT / "data" / "raw" / args.source
    out_dir = REPO_ROOT / "data" / "clean" / args.source
    shards = sorted(in_dir.glob("shard_*.jsonl"))
    if not shards:
        raise SystemExit(f"no shards in {in_dir} - run download_corpus.py first")

    cfg = QualityConfig(min_score=args.min_score, min_language_score=args.min_language_score)
    dry_run = args.sample is not None

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for p in out_dir.glob("shard_*.jsonl"):
                p.unlink()

    n_seen = n_kept = 0
    kept_chars = seen_chars = 0
    reasons: Counter[str] = Counter()
    combos: Counter[str] = Counter()

    shard_idx = 0
    shard_f = None
    shard_bytes = 0
    limit = args.shard_mb * 1_000_000
    t0 = time.time()

    def open_shard(i: int):
        return (out_dir / f"shard_{i:05d}.jsonl").open("w", encoding="utf-8")

    if not dry_run:
        shard_f = open_shard(shard_idx)

    total = args.sample if dry_run else sum(1 for s in shards for _ in s.open())
    pbar = tqdm(total=total, unit="doc", desc="filter")

    for shard in shards:
        with shard.open(encoding="utf-8") as f:
            for line in f:
                if dry_run and n_seen >= args.sample:
                    break
                pbar.update(1)
                rec = json.loads(line)
                text = rec["text"]
                n_seen += 1
                seen_chars += len(text)

                fails = check_document(text, rec, cfg)
                if fails:
                    reasons.update(fails)
                    combos["+".join(sorted(fails))] += 1
                    continue

                n_kept += 1
                kept_chars += len(text)
                if not dry_run:
                    out = json.dumps(rec, ensure_ascii=False) + "\n"
                    enc = out.encode("utf-8")
                    if shard_bytes + len(enc) > limit:
                        shard_f.close()
                        shard_idx += 1
                        shard_f = open_shard(shard_idx)
                        shard_bytes = 0
                    shard_f.write(out)
                    shard_bytes += len(enc)
        if dry_run and n_seen >= args.sample:
            break

    if shard_f is not None:
        shard_f.close()
    pbar.close()

    dt = time.time() - t0
    keep_pct = 100 * n_kept / max(n_seen, 1)
    char_pct = 100 * kept_chars / max(seen_chars, 1)

    print(f"\n{'DRY RUN - ' if dry_run else ''}checked {n_seen:,} docs in {dt:.1f}s")
    print(f"kept {n_kept:,} ({keep_pct:.1f}% of docs, {char_pct:.1f}% of characters)")
    print(f"\ndrop reasons (a doc can fail several):")
    for name, c in reasons.most_common():
        print(f"  {name:20s} {c:>9,}  ({100 * c / max(n_seen, 1):.1f}%)")
    print(f"\ntop drop combinations:")
    for combo, c in combos.most_common(8):
        print(f"  {c:>9,}  {combo}")
    if not dry_run:
        print(f"\n-> data/clean/{args.source}/  ({shard_idx + 1} shards)")


if __name__ == "__main__":
    main()
