"""Supervised fine-tuning: take a pretrained base checkpoint and fine-tune it
on chat data, with loss computed only on assistant tokens.

    uv run python scripts/prepare_sft_data.py     # once, builds data/sft/smoltalk/
    uv run python scripts/sft.py checkpoints/135m-0912-1812/annealed \
        --steps 3000 --peak-lr 1e-4

Key differences from pretrain.py:
  - starts from trained weights (checkpoint.load_resized - the base checkpoint's
    vocab is smaller than the chat tokenizer's, so 3 new embedding rows are
    added, randomly initialised; everything else is the trained base model)
  - much lower learning rate (this is fine-tuning, not training from scratch)
  - loss is MASKED to assistant tokens only (model.loss_masked)
  - a normal single-shot cosine schedule (no stable-forever trick - SFT runs
    are short enough to just pick a step count and let it fully decay)
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

from forge.data.loader import SFTLoader, load_meta
from forge.model.config import PRESETS
from forge.model.transformer import Transformer
from forge.training import checkpoint
from forge.training.schedule import wsd_schedule

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def evaluate(model: Transformer, loader: SFTLoader, n_batches: int) -> float:
    model.eval()
    losses = [model.loss_masked(x, y, m) for x, y, m in loader.iter_val(n_batches)]
    model.train()
    return float(mx.mean(mx.stack(losses))) if losses else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_ckpt", type=Path, help="pretrained checkpoint dir, e.g. .../annealed")
    ap.add_argument("--sft-dir", type=Path, default=REPO_ROOT / "data" / "sft" / "smoltalk")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--decay-frac", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    meta = load_meta(args.sft_dir)
    base_state = json.loads((args.base_ckpt / "state.json").read_text())
    preset = base_state["preset"]
    cfg = PRESETS[preset]
    cfg.vocab_size = meta["vocab_size"]
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    run_name = Path(args.out) if args.out else (
        Path(args.resume).parent if args.resume else
        REPO_ROOT / "checkpoints" / f"{preset}-sft-{time.strftime('%m%d-%H%M')}"
    )
    run_name = run_name.resolve()
    run_name.mkdir(parents=True, exist_ok=True)
    log_f = (run_name / "log.jsonl").open("a")

    tokens_per_step = args.batch_size * args.seq_len
    print(f"run: {_rel(run_name)}")
    print(f"base: {_rel(args.base_ckpt)} (val loss {base_state.get('val_loss', '?')})")
    print(f"model: {preset} ({cfg.n_params / 1e6:.1f}M params)  chat vocab={cfg.vocab_size}")
    print(f"sft data: {meta['n_conversations_train']:,} conversations, "
          f"{meta['n_tokens_train']:,} tokens ({meta['trainable_frac']:.1%} trainable)")

    model = Transformer(cfg)
    model.train()
    schedule = wsd_schedule(peak_lr=args.peak_lr, total_steps=args.steps,
                             warmup_steps=args.warmup, decay_frac=args.decay_frac)
    optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    max_norm = args.grad_clip or 1e9

    start_step = 0
    if args.resume:
        st = checkpoint.load(args.resume, model, optimizer)
        start_step = st["step"]
        print(f"resumed from {_rel(args.resume)} at step {start_step}")
    else:
        checkpoint.load_resized(args.base_ckpt, model)
        print("loaded base weights (new chat-token embedding rows randomly initialised)")

    train_loader = SFTLoader.from_dir(args.sft_dir, "train", args.seq_len, args.batch_size, seed=start_step)
    val_loader = SFTLoader.from_dir(args.sft_dir, "val", args.seq_len, args.batch_size, seed=0)

    def loss_fn(m, x, y, mask):
        return m.loss_masked(x, y, mask)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    def _single(x, y, mask):
        loss, grads = loss_and_grad(model, x, y, mask)
        grads, gnorm = optim.clip_grad_norm(grads, max_norm)
        optimizer.update(model, grads)
        return loss, gnorm

    train_step = _single if args.no_compile else partial(mx.compile, inputs=state, outputs=state)(_single)

    mx.eval(model.parameters())
    t0 = time.time()
    tokens_seen = 0
    running = None
    step = start_step

    for step in range(start_step, args.steps):
        x, y, mask = train_loader.batch()
        loss, gnorm = train_step(x, y, mask)
        mx.eval(state, loss)

        lval = float(loss)
        running = lval if running is None else 0.9 * running + 0.1 * lval
        tokens_seen += tokens_per_step

        if not math.isfinite(lval):
            print(f"\n!! non-finite loss at step {step} - stopping.")
            checkpoint.save(run_name / "crash", model, optimizer, step, extra={"preset": preset})
            break

        if step % args.log_every == 0:
            tps = tokens_seen / (time.time() - t0)
            lr = float(schedule(mx.array(step)))
            print(f"step {step:>6} | loss {lval:6.3f} (ema {running:6.3f}) | "
                  f"lr {lr:.2e} | |g| {float(gnorm):5.2f} | {tps:>7,.0f} tok/s")
            log_f.write(json.dumps({"step": step, "loss": lval, "ema": running, "lr": lr,
                                    "grad_norm": float(gnorm)}) + "\n")
            log_f.flush()

        if step > start_step and step % args.eval_every == 0:
            vloss = evaluate(model, val_loader, args.eval_batches)
            print(f"  >>> step {step}: val loss {vloss:.3f}")
            log_f.write(json.dumps({"step": step, "val_loss": vloss}) + "\n")
            log_f.flush()
            checkpoint.save(run_name / "last", model, optimizer, step,
                            extra={"preset": preset, "val_loss": vloss})

    final_step = min(step + 1, args.steps)
    vloss = evaluate(model, val_loader, args.eval_batches)
    print(f"\ndone. step {final_step}, val loss {vloss:.3f}")
    checkpoint.save(run_name / "last", model, optimizer, final_step,
                    extra={"preset": preset, "val_loss": vloss})
    print(f"-> {_rel(run_name / 'last')}")


if __name__ == "__main__":
    main()
