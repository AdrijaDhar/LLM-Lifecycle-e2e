"""Build the SFT dataset: smol-smoltalk (general chat) + CodeAlpaca (code
instructions, upweighted) mixed together, chat-formatted, packed to
token+mask binaries the same way pack_corpus.py packs pretraining data.

Also grows the pretrained tokenizer with 3 chat special tokens
(<|user|>, <|assistant|>, <|end|>) and saves it as a new file - the base
tokenizer (fw32k.bpe.json) is left untouched so existing base-model scripts
keep working.

Writes:
    data/tokenizer/fw32k-chat.bpe.json
    data/sft/<out-name>/{train.bin, train_mask.bin, val.bin, val_mask.bin, meta.json}

    uv run python scripts/prepare_sft_data.py                       # smoltalk + code (default)
    uv run python scripts/prepare_sft_data.py --code-dataset ""     # smoltalk only, original recipe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from forge.data.chat import ChatFormat
from forge.tokenizer.bpe import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _smoltalk_messages(row: dict) -> list[dict] | None:
    return row["messages"] or None


def _codealpaca_messages(row: dict) -> list[dict] | None:
    prompt, completion = row.get("prompt"), row.get("completion")
    if not prompt or not completion:
        return None
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]


def _pack(rows, to_messages, tok, fmt, max_tokens_per_convo, desc):
    n_skipped_empty = n_skipped_long = n_ok = 0
    for row in tqdm(rows, desc=desc):
        msgs = to_messages(row)
        if not msgs:
            n_skipped_empty += 1
            continue
        ids, mask = fmt.encode_conversation(tok, msgs)
        if len(ids) > max_tokens_per_convo:
            n_skipped_long += 1
            continue
        n_ok += 1
        yield ids, mask
    print(f"  {desc}: kept {n_ok:,}, skipped {n_skipped_empty:,} empty / {n_skipped_long:,} over "
          f"{max_tokens_per_convo} tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "fw32k.bpe.json")
    ap.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    ap.add_argument("--code-dataset", default="HuggingFaceH4/CodeAlpaca_20K",
                     help="a second, code-focused dataset mixed in; pass \"\" to disable")
    ap.add_argument("--code-repeat", type=int, default=3,
                     help="repeat the (much smaller) code dataset this many times so it isn't diluted away")
    ap.add_argument("--out-name", default=None, help="output dir under data/sft/ (default: derived from --dataset)")
    ap.add_argument("--max-tokens-per-convo", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=None, help="cap examples per source (debugging/smoke tests)")
    args = ap.parse_args()

    tok = BPETokenizer()
    tok.load(str(args.base_tokenizer))
    eot_id = tok.eot_id
    fmt = ChatFormat.register(tok)
    chat_tok_path = args.base_tokenizer.with_name(args.base_tokenizer.stem.split(".")[0] + "-chat.bpe.json")
    tok.save(str(chat_tok_path))
    print(f"tokenizer: {tok.vocab_size} tokens (base {eot_id + 1} + 3 chat specials) -> {chat_tok_path.name}")
    print(f"special ids: user={fmt.user_id} assistant={fmt.assistant_id} end={fmt.end_id}")

    sources = []
    print(f"loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    print(f"{len(ds):,} conversations")
    sources.append((ds, _smoltalk_messages, 1, "smoltalk"))

    if args.code_dataset:
        print(f"loading {args.code_dataset} ...")
        code_ds = load_dataset(args.code_dataset, split="train")
        if args.max_examples:
            code_ds = code_ds.select(range(min(args.max_examples, len(code_ds))))
        print(f"{len(code_ds):,} code examples x{args.code_repeat}")
        sources.append((code_ds, _codealpaca_messages, args.code_repeat, "code"))

    out_dir = REPO_ROOT / "data" / "sft" / (args.out_name or ("smoltalk" if not args.code_dataset else "smoltalk-code"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    train_ids: list[int] = []
    train_mask: list[int] = []
    val_ids: list[int] = []
    val_mask: list[int] = []
    n_train = n_val = 0
    train_msg_tokens = 0
    per_source_counts: dict[str, int] = {}

    for ds, to_messages, repeat, name in sources:
        count = 0
        for _ in range(repeat):
            for ids, mask in _pack(ds, to_messages, tok, fmt, args.max_tokens_per_convo, name):
                ids = ids + [eot_id]
                mask = mask + [0]
                if rng.random() < args.val_frac:
                    val_ids.extend(ids)
                    val_mask.extend(mask)
                    n_val += 1
                else:
                    train_ids.extend(ids)
                    train_mask.extend(mask)
                    n_train += 1
                    train_msg_tokens += sum(mask)
                count += 1
        per_source_counts[name] = count

    # shuffle the packed conversation order isn't done at the bin level (loader
    # samples random windows anyway) - but source order matters for what ends
    # up adjacent, so this is fine as-is for a random-offset-window loader.

    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
    np.asarray(train_ids, dtype=dtype).tofile(out_dir / "train.bin")
    np.asarray(train_mask, dtype=np.uint8).tofile(out_dir / "train_mask.bin")
    np.asarray(val_ids, dtype=dtype).tofile(out_dir / "val.bin")
    np.asarray(val_mask, dtype=np.uint8).tofile(out_dir / "val_mask.bin")

    meta = {
        "datasets": {name: count for name, count in per_source_counts.items()},
        "tokenizer": str(chat_tok_path.relative_to(REPO_ROOT)),
        "vocab_size": tok.vocab_size,
        "dtype": np.dtype(dtype).name,
        "eot_id": eot_id,
        "chat_ids": {"user": fmt.user_id, "assistant": fmt.assistant_id, "end": fmt.end_id},
        "n_conversations_train": n_train,
        "n_conversations_val": n_val,
        "n_tokens_train": len(train_ids),
        "n_tokens_val": len(val_ids),
        "n_trainable_tokens_train": int(train_msg_tokens),
        "trainable_frac": round(train_msg_tokens / max(len(train_ids), 1), 3),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nmix: {per_source_counts}")
    print(f"train: {n_train:,} conversations, {len(train_ids):,} tokens "
          f"({train_msg_tokens:,} trainable = {meta['trainable_frac']:.1%})")
    print(f"val:   {n_val:,} conversations, {len(val_ids):,} tokens")
    print(f"-> {out_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
