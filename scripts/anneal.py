"""Phase 2 of WSD: a short cosine decay from a stable-phase checkpoint.

Run this AFTER an overnight `pretrain.py --decay-frac 0` run. It takes the stable
checkpoint (constant peak lr) and cools the learning rate to ~0 over a short
extra stretch of steps. Most of a schedule's final loss improvement comes from
this cooldown; WSD just lets you do it whenever you decide the run is "done".

    uv run python scripts/anneal.py checkpoints/135m-0911-2312/last \
        --steps 1500 --peak-lr 3e-3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from forge.data.loader import TokenLoader, load_meta
from forge.model.config import PRESETS
from forge.model.transformer import Transformer
from forge.training import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="stable-phase checkpoint dir")
    ap.add_argument("--tokens-dir", type=Path, default=REPO_ROOT / "data" / "tokens" / "fineweb-edu")
    ap.add_argument("--steps", type=int, default=1500, help="length of the decay")
    ap.add_argument("--peak-lr", type=float, default=3e-3, help="lr the stable run held")
    ap.add_argument("--final-lr", type=float, default=0.0)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    meta = load_meta(args.tokens_dir)
    st = json.loads((args.ckpt / "state.json").read_text())
    cfg = PRESETS[st["preset"]]
    cfg.vocab_size = meta["vocab_size"]
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    print(f"annealing {st['preset']} from step {st['step']} (val {st.get('val_loss', '?')}) "
          f"over {args.steps} steps, lr {args.peak_lr:g} -> {args.final_lr:g}")

    model = Transformer(cfg)
    model.train()
    schedule = optim.cosine_decay(args.peak_lr, args.steps, end=args.final_lr)
    optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=args.weight_decay)

    checkpoint.load(args.ckpt, model, optimizer)   # weights + Adam moments
    optimizer.state["step"] = mx.array(0)          # restart the schedule clock at 0

    train_loader = TokenLoader.from_dir(args.tokens_dir, "train", args.seq_len, args.batch_size, seed=99)
    val_loader = TokenLoader.from_dir(args.tokens_dir, "val", args.seq_len, args.batch_size, seed=0)
    max_norm = args.grad_clip or 1e9

    def loss_fn(m, x, y):
        return m.loss(x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, gnorm = optim.clip_grad_norm(grads, max_norm)
        optimizer.update(model, grads)
        return loss, gnorm

    out = args.ckpt.parent / "annealed"
    mx.eval(model.parameters())
    t0 = time.time()

    for step in range(args.steps):
        x, y = train_loader.batch()
        loss, gnorm = train_step(x, y)
        mx.eval(state, loss)
        if step % args.log_every == 0:
            lr = float(schedule(mx.array(step)))
            print(f"step {step:>5}/{args.steps} | loss {float(loss):6.3f} | lr {lr:.2e} | "
                  f"|g| {float(gnorm):5.2f} | {step * args.batch_size * args.seq_len / (time.time() - t0):,.0f} tok/s")

    model.eval()
    losses = [model.loss(x, y) for x, y in val_loader.iter_val(args.eval_batches)]
    vloss = float(mx.mean(mx.stack(losses)))
    print(f"\nannealed val loss {vloss:.3f}  (ppl {math.exp(vloss):.1f})")
    checkpoint.save(out, model, optimizer, st["step"] + args.steps,
                    extra={"preset": st["preset"], "val_loss": vloss, "annealed": True})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
