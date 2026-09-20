"""Pretrain a Transformer on packed tokens with a WSD schedule.

    # smoke: 200 steps on the tiny model, watch loss fall from ~10.9
    uv run python scripts/pretrain.py --preset tiny --steps 200 --warmup 20 \
        --seq-len 512 --batch-size 16 --eval-every 100

    # overnight 135m: stable-only (constant lr), stop on time, anneal in the morning
    uv run python scripts/pretrain.py --preset 135m --decay-frac 0 \
        --steps 100000 --warmup 200 --seq-len 1024 --batch-size 32 \
        --peak-lr 3e-3 --eval-every 500 --max-minutes 420

    # normal fixed-length run (warmup -> stable -> cosine decay)
    uv run python scripts/pretrain.py --preset 135m --steps 8000 --warmup 200 \
        --seq-len 1024 --batch-size 32 --peak-lr 3e-3

    # resume
    uv run python scripts/pretrain.py --resume checkpoints/135m-<stamp>/last
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
from mlx.utils import tree_map

from forge.data.loader import TokenLoader, load_meta
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


def evaluate(model: Transformer, loader: TokenLoader, n_batches: int) -> float:
    model.eval()
    losses = [model.loss(x, y) for x, y in loader.iter_val(n_batches)]
    model.train()
    return float(mx.mean(mx.stack(losses))) if losses else float("nan")


# Schedule/batch knobs: precedence is CLI flag > resumed checkpoint > this default.
# (argparse default=None so we can tell "not passed" apart from "passed explicitly".)
_DEFAULTS = dict(
    peak_lr=3e-3, warmup=200, decay_frac=0.2, steps=8000, seq_len=1024,
    batch_size=32, weight_decay=0.1, grad_clip=1.0, min_lr_frac=0.0, grad_accum=1,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="135m")
    ap.add_argument("--tokens-dir", type=Path, default=REPO_ROOT / "data" / "tokens" / "fineweb-edu")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--peak-lr", type=float, default=None)
    ap.add_argument("--min-lr-frac", type=float, default=None)
    ap.add_argument("--decay-frac", type=float, default=None, help="0 = stable forever (anneal separately)")
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--no-compile", action="store_true", help="eager step: lower peak memory, no compile stall, ~1.5-2x slower")
    args = ap.parse_args()

    meta = load_meta(args.tokens_dir)
    preset = args.preset
    saved_targs = {}
    if args.resume:
        saved = json.loads((Path(args.resume) / "state.json").read_text())
        preset = saved.get("preset", preset)
        saved_targs = saved.get("train_args", {})
        print(f"resume: preset '{preset}' from checkpoint "
              f"(explicit CLI flags override its saved schedule/batch settings)")
    for k, default in _DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, saved_targs.get(k, default))
    cfg = PRESETS[preset]
    cfg.vocab_size = meta["vocab_size"]
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    if args.out:
        run_name = Path(args.out)
    elif args.resume:
        run_name = Path(args.resume).parent
    else:
        run_name = REPO_ROOT / "checkpoints" / f"{args.preset}-{time.strftime('%m%d-%H%M')}"
    run_name = run_name.resolve()
    run_name.mkdir(parents=True, exist_ok=True)
    log_f = (run_name / "log.jsonl").open("a")

    _targs = {k: getattr(args, k) for k in ("peak_lr", "warmup", "decay_frac", "steps", "seq_len",
              "batch_size", "weight_decay", "grad_clip", "min_lr_frac", "grad_accum")}
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    decay_start = args.steps - int(args.steps * args.decay_frac)
    print(f"run: {_rel(run_name)}")
    print(f"model: {preset} ({cfg.n_params / 1e6:.1f}M params)  vocab={cfg.vocab_size}")
    print(f"{tokens_per_step:,} tokens/step | "
          f"{'stable-only (--decay-frac 0)' if args.decay_frac <= 0 else f'warmup {args.warmup} / stable / decay from {decay_start}'}")

    model = Transformer(cfg)
    model.train()
    schedule = wsd_schedule(
        peak_lr=args.peak_lr, total_steps=args.steps, warmup_steps=args.warmup,
        decay_frac=args.decay_frac, final_lr_frac=args.min_lr_frac,
    )
    optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=args.weight_decay)
    max_norm = args.grad_clip or 1e9

    start_step = 0
    if args.resume:
        st = checkpoint.load(args.resume, model, optimizer)
        start_step = st["step"]
        print(f"resumed from {_rel(args.resume)} at step {start_step}")

    train_loader = TokenLoader.from_dir(args.tokens_dir, "train", args.seq_len, args.batch_size, seed=start_step)
    val_loader = TokenLoader.from_dir(args.tokens_dir, "val", args.seq_len, args.batch_size, seed=0)

    def loss_fn(m, x, y):
        return m.loss(x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    def _single(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, gnorm = optim.clip_grad_norm(grads, max_norm)
        optimizer.update(model, grads)
        return loss, gnorm

    train_step = _single if args.no_compile else partial(mx.compile, inputs=state, outputs=state)(_single)

    def accum_step(x0, y0):
        loss, grads = loss_and_grad(model, x0, y0)
        for _ in range(args.grad_accum - 1):
            xi, yi = train_loader.batch()
            li, gi = loss_and_grad(model, xi, yi)
            loss = loss + li
            grads = tree_map(lambda a, b: a + b, grads, gi)
        loss = loss / args.grad_accum
        grads = tree_map(lambda a: a / args.grad_accum, grads)
        grads, gnorm = optim.clip_grad_norm(grads, max_norm)
        optimizer.update(model, grads)
        return loss, gnorm

    compiled = args.grad_accum == 1
    if args.grad_accum > 1:
        path = f"eager x{args.grad_accum} accum"
    else:
        path = "eager single-batch" if args.no_compile else "compiled single-batch"
    print(f"grad path: {path}\n")

    mx.eval(model.parameters())
    t0 = time.time()
    tokens_seen = 0
    running = None
    step = start_step
    eta_shown = False

    for step in range(start_step, args.steps):
        x, y = train_loader.batch()
        loss, gnorm = train_step(x, y) if compiled else accum_step(x, y)
        mx.eval(state, loss)

        lval = float(loss)
        running = lval if running is None else 0.9 * running + 0.1 * lval
        tokens_seen += tokens_per_step

        if not math.isfinite(lval):
            print(f"\n!! non-finite loss ({lval}) at step {step} - stopping. "
                  f"resume from last/ with a lower --peak-lr.")
            checkpoint.save(run_name / "crash", model, optimizer, step, extra={"loss": lval})
            break

        if not eta_shown and step - start_step >= 40:
            tps = tokens_seen / (time.time() - t0)
            if args.max_minutes:
                reach = start_step + int(args.max_minutes * 60 * tps / tokens_per_step)
                tail = ("stable-only; anneal.py afterwards" if args.decay_frac <= 0
                        else f"decay starts {decay_start:,}"
                             + (" - won't anneal, use anneal.py" if reach < decay_start else ""))
                print(f"  ~{tps:,.0f} tok/s | in {args.max_minutes:g} min -> ~step {reach:,}  ({tail})\n")
            else:
                rem = (args.steps - step) * tokens_per_step / tps
                print(f"  ~{tps:,.0f} tok/s | {args.steps:,} steps ETA ~{rem / 3600:.1f} h\n")
            eta_shown = True

        if step % args.log_every == 0:
            tps = tokens_seen / (time.time() - t0)
            lr = float(schedule(mx.array(step)))
            print(f"step {step:>6} | loss {lval:6.3f} (ema {running:6.3f}) | "
                  f"lr {lr:.2e} | |g| {float(gnorm):5.2f} | {tps:>8,.0f} tok/s")
            log_f.write(json.dumps({"step": step, "loss": lval, "ema": running, "lr": lr,
                                    "grad_norm": float(gnorm), "tok_s": tps}) + "\n")
            log_f.flush()

        if step > start_step and step % args.eval_every == 0:
            vloss = evaluate(model, val_loader, args.eval_batches)
            print(f"  >>> step {step}: val loss {vloss:.3f}  (ppl {math.exp(vloss):.1f})")
            log_f.write(json.dumps({"step": step, "val_loss": vloss}) + "\n")
            log_f.flush()
            checkpoint.save(run_name / "last", model, optimizer, step,
                            extra={"preset": preset, "val_loss": vloss, "train_args": _targs})

        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print(f"\nhit --max-minutes at step {step}")
            break

    final_step = min(step + 1, args.steps)
    vloss = evaluate(model, val_loader, args.eval_batches)
    print(f"\ndone. step {final_step}, val loss {vloss:.3f}  (ppl {math.exp(vloss):.1f})")
    checkpoint.save(run_name / "last", model, optimizer, final_step,
                    extra={"preset": preset, "val_loss": vloss, "train_args": _targs})
    print(f"-> {_rel(run_name / 'last')}")


if __name__ == "__main__":
    main()
