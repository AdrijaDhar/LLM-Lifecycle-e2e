"""Tokenize a cleaned+deduped corpus and pack it into flat binary token files.

Reads   data/dedup/<source>/shard_*.jsonl   (falls back to data/clean, then data/raw)
Uses    a trained tokenizer  (data/tokenizer/<name>.bpe.json)
Writes  data/tokens/<source>/train.bin      flat array of token ids
        data/tokens/<source>/val.bin
        data/tokens/<source>/meta.json

Each document is encoded and followed by the <|endoftext|> id, then all documents
are concatenated into one stream. The training dataloader will `np.memmap` these
files and slice fixed-length sequences out of them - no JSON, no tokenization in
the training hot loop.

    uv run python scripts/pack_corpus.py fineweb-edu --tokenizer data/tokenizer/fineweb-edu.bpe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from forge.tokenizer.bpe import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUSH_EVERY = 8_000_000  # tokens buffered before writing to disk


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def _find_input_dir(source: str) -> Path:
    for stage in ("dedup", "clean", "raw"):
        d = REPO_ROOT / "data" / stage / source
        if d.is_dir() and any(d.glob("shard_*.jsonl")):
            return d
    raise SystemExit(f"no shards for '{source}' under data/dedup|clean|raw/")


def _iter_texts(shards):
    for shard in shards:
        with shard.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


class BinWriter:
    def __init__(self, path: Path, dtype: np.dtype) -> None:
        self.path = path
        self.dtype = dtype
        self.buf: list[int] = []
        self.n = 0
        path.write_bytes(b"")  # truncate

    def add(self, ids: list[int]) -> None:
        self.buf.extend(ids)
        if len(self.buf) >= FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        arr = np.asarray(self.buf, dtype=self.dtype)
        with self.path.open("ab") as f:
            arr.tofile(f)
        self.n += len(self.buf)
        self.buf.clear()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.005, help="fraction of documents held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None, help="cap for a quick test pack")
    args = ap.parse_args()

    args.tokenizer = args.tokenizer.resolve()
    tok = BPETokenizer()
    tok.load(str(args.tokenizer))
    eot = tok.eot_id
    dtype = np.dtype(np.uint16) if tok.vocab_size <= 65536 else np.dtype(np.uint32)
    print(f"tokenizer: {tok.vocab_size} tokens, eot={eot}, packing as {dtype}")

    in_dir = _find_input_dir(args.source)
    shards = sorted(in_dir.glob("shard_*.jsonl"))
    n_docs_total = sum(1 for s in shards for _ in s.open(encoding="utf-8"))
    if args.max_docs:
        n_docs_total = min(n_docs_total, args.max_docs)
    print(f"input: {in_dir.relative_to(REPO_ROOT)}/  ({n_docs_total:,} docs)")

    out_dir = REPO_ROOT / "data" / "tokens" / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    train_w = BinWriter(out_dir / "train.bin", dtype)
    val_w = BinWriter(out_dir / "val.bin", dtype)

    rng = np.random.default_rng(args.seed)
    n_docs_val = n_chars = 0
    t0 = time.time()

    for i, text in enumerate(tqdm(_iter_texts(shards), total=n_docs_total, desc="encode")):
        if args.max_docs and i >= args.max_docs:
            break
        n_chars += len(text)
        ids = tok.encode(text)
        ids.append(eot)
        if rng.random() < args.val_frac:
            val_w.add(ids)
            n_docs_val += 1
        else:
            train_w.add(ids)

    train_w.flush()
    val_w.flush()
    dt = time.time() - t0

    n_docs = i + 1
    total_tokens = train_w.n + val_w.n
    meta = {
        "source": args.source,
        "tokenizer": _rel(args.tokenizer),
        "vocab_size": tok.vocab_size,
        "dtype": dtype.name,
        "eot_id": eot,
        "n_docs": n_docs,
        "n_docs_val": n_docs_val,
        "n_tokens_train": train_w.n,
        "n_tokens_val": val_w.n,
        "chars_per_token": round(n_chars / max(total_tokens, 1), 3),
        "seconds": round(dt, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nencoded {n_docs:,} docs in {dt / 60:.1f} min")
    print(f"train: {train_w.n:,} tokens   val: {val_w.n:,} tokens")
    print(f"compression: {meta['chars_per_token']} chars/token")
    print(f"train.bin = {train_w.path.stat().st_size / 1e6:.0f} MB")
    print(f"-> data/tokens/{args.source}/")


if __name__ == "__main__":
    main()
