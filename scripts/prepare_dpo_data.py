"""Build the DPO dataset: download UltraFeedback (binarized), chat-format each
(prompt, chosen, rejected) triple, pad to a fixed length, save as .npy arrays.

    uv run python scripts/prepare_dpo_data.py
    uv run python scripts/prepare_dpo_data.py --max-len 768 --max-examples 30000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from forge.data.chat import ChatFormat
from forge.data.preference import encode_pair
from forge.tokenizer.bpe import BPETokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rows(ds, max_examples):
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        yield row["prompt"], row["chosen"][-1]["content"], row["rejected"][-1]["content"]


def _pack(ds, tok, fmt, max_len, max_examples, desc):
    ids_c, mask_c, ids_r, mask_r = [], [], [], []
    n_skipped = 0
    for prompt, chosen, rejected in tqdm(_rows(ds, max_examples), total=min(len(ds), max_examples or len(ds)), desc=desc):
        out = encode_pair(tok, fmt, prompt, chosen, rejected, max_len=max_len)
        if out is None:
            n_skipped += 1
            continue
        ids_c.append(out["chosen_ids"]); mask_c.append(out["chosen_mask"])
        ids_r.append(out["rejected_ids"]); mask_r.append(out["rejected_mask"])
    print(f"  kept {len(ids_c):,}, skipped {n_skipped:,} (too long for max_len={max_len})")
    return np.array(ids_c, dtype=np.uint16), np.array(mask_c, dtype=np.uint8), \
           np.array(ids_r, dtype=np.uint16), np.array(mask_r, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "fw32k-chat.bpe.json",
                     help="use the SFT tokenizer (already has the chat special tokens)")
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--max-examples", type=int, default=30000, help="cap train examples; 0 = no cap")
    args = ap.parse_args()

    tok = BPETokenizer()
    tok.load(str(args.tokenizer))
    fmt = ChatFormat.register(tok)  # no-op: already present in this tokenizer file

    print(f"loading {args.dataset} ...")
    train_ds = load_dataset(args.dataset, split="train_prefs")
    val_ds = load_dataset(args.dataset, split="test_prefs")
    print(f"train_prefs: {len(train_ds):,}  test_prefs: {len(val_ds):,}")

    out_dir = REPO_ROOT / "data" / "dpo" / "ultrafeedback"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("packing train...")
    tc_ids, tc_mask, tr_ids, tr_mask = _pack(train_ds, tok, fmt, args.max_len, args.max_examples, "train")
    print("packing val...")
    vc_ids, vc_mask, vr_ids, vr_mask = _pack(val_ds, tok, fmt, args.max_len, args.max_examples // 10 or 1000, "val")

    np.save(out_dir / "train_chosen_ids.npy", tc_ids)
    np.save(out_dir / "train_chosen_mask.npy", tc_mask)
    np.save(out_dir / "train_rejected_ids.npy", tr_ids)
    np.save(out_dir / "train_rejected_mask.npy", tr_mask)
    np.save(out_dir / "val_chosen_ids.npy", vc_ids)
    np.save(out_dir / "val_chosen_mask.npy", vc_mask)
    np.save(out_dir / "val_rejected_ids.npy", vr_ids)
    np.save(out_dir / "val_rejected_mask.npy", vr_mask)

    meta = {
        "dataset": args.dataset,
        "tokenizer": str(args.tokenizer.relative_to(REPO_ROOT)),
        "max_len": args.max_len,
        "n_train": len(tc_ids),
        "n_val": len(vc_ids),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ntrain: {len(tc_ids):,} pairs   val: {len(vc_ids):,} pairs")
    print(f"-> {out_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
