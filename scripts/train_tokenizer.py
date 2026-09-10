"""Train the byte-level BPE tokenizer on a local text corpus.

    uv run python scripts/train_tokenizer.py --corpus data/raw/fineweb-edu.txt --vocab-size 8192

Outputs:
    data/tokenizer/<name>.bpe.json   merge table (a, b, new_id triples)
    data/tokenizer/<name>.meta.json  stats: vocab size, corpus, compression ratio
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from forge.tokenizer.bpe import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_corpus(path: Path, max_bytes: int) -> str:
    """Read a .txt file or concatenate the `text` fields of a dir of jsonl shards,
    stopping once we have `max_bytes` of UTF-8."""
    if path.is_file():
        raw = path.read_bytes()[:max_bytes]
        return raw.decode("utf-8", errors="ignore")

    shards = sorted(path.glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"{path} has no .jsonl shards")
    parts: list[str] = []
    total = 0
    for shard in shards:
        with shard.open(encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)["text"]
                parts.append(t)
                total += len(t.encode("utf-8")) + 1
                if total >= max_bytes:
                    return "\n".join(parts)
    return "\n".join(parts)


def _rel(p: Path) -> str:
    """Path relative to the repo root if possible, else absolute."""
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())

# A few strings we round-trip after training as a smoke test.
PROBES = [
    "The quick brown fox jumps over the lazy dog.",
    "resource \"aws_s3_bucket\" \"logs\" { bucket = \"my-log-bucket\" }",
    "def add(a, b):\n    return a + b  # 42\n",
    "Unicode: café, naïve, 日本語, \U0001f600",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help=".txt file OR a dir of *.jsonl shards")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--name", default=None, help="output basename (default: corpus stem)")
    ap.add_argument("--max-bytes", type=int, default=200_000_000, help="cap corpus text used for training")
    ap.add_argument("--special", action="append", default=[], help="extra special token (repeatable)")
    ap.add_argument("--naive", action="store_true", help="use the slow O(merges x corpus) trainer")
    args = ap.parse_args()

    args.corpus = args.corpus.resolve()
    name = args.name or (args.corpus.name if args.corpus.is_dir() else args.corpus.stem)
    out_dir = REPO_ROOT / "data" / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)

    text = _read_corpus(args.corpus, args.max_bytes)
    print(f"corpus: {args.corpus}  ({len(text):,} chars used, cap {args.max_bytes:,} bytes)")

    tok = BPETokenizer()
    trainer = tok.train if args.naive else tok.train_fast
    print(f"trainer: {trainer.__name__}")
    t0 = time.time()
    trainer(text, vocab_size=args.vocab_size, verbose=True)
    train_s = time.time() - t0
    for s in ["<|endoftext|>", *args.special]:
        tok.register_special(s)
    print(f"trained {tok.vocab_size} tokens ({len(tok.merges)} merges + "
          f"{len(tok.special_tokens)} special) in {train_s:.1f}s")

    # Compression ratio: chars per token over the training corpus.
    sample = text[:2_000_000]
    n_tokens = len(tok.encode(sample))
    ratio = len(sample) / n_tokens
    print(f"compression: {ratio:.2f} chars/token  (higher is better; ~4.8 is typical)")

    print("\nround-trip probes:")
    for p in PROBES:
        ok = tok.decode(tok.encode(p)) == p
        print(f"  [{'ok ' if ok else 'FAIL'}] {p[:60]!r} -> {len(tok.encode(p))} tok")

    bpe_path = out_dir / f"{name}.bpe.json"
    meta_path = out_dir / f"{name}.meta.json"
    tok.save(str(bpe_path))
    meta_path.write_text(
        json.dumps(
            {
                "name": name,
                "vocab_size": tok.vocab_size,
                "n_merges": len(tok.merges),
                "special_tokens": tok.special_tokens,
                "corpus": _rel(args.corpus),
                "corpus_chars": len(text),
                "train_seconds": round(train_s, 1),
                "chars_per_token": round(ratio, 3),
            },
            indent=2,
        )
    )
    print(f"\nsaved {bpe_path.relative_to(REPO_ROOT)}")
    print(f"saved {meta_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
