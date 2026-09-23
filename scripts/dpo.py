"""DPO: fine-tune the SFT checkpoint to prefer better responses.

    uv run python scripts/prepare_dpo_data.py     # once
    uv run python scripts/dpo.py checkpoints/135m-sft-0920-1142/last --steps 1500

Runs eager (no mx.compile) - this stage trains two forward passes per step
(policy + frozen reference) on a much smaller dataset than pretraining/SFT, so
compile's startup cost and the added complexity of tracing two models aren't
worth it here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from forge.data.preference import PreferenceLoader
from forge.model.config import PRESETS
from forge.model.transformer import Transformer
from forge.training import checkpoint
from forge.training.dpo import dpo_loss

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def evaluate(policy, ref, loader: PreferenceLoader, batch_size: int, n_batches: int, beta: float) -> dict:
    policy.eval()
    losses, accs, margins = [], [], []
    for c_ids, c_mask, r_ids, r_mask in loader.iter_val(batch_size, n_batches):
        loss, m = dpo_loss(policy, ref, c_ids, c_mask, r_ids, r_mask, beta=beta)
        losses.append(float(loss)); accs.append(float(m["accuracy"])); margins.append(float(m["margin"]))
    policy.train()
    if not losses:
        return {"loss": float("nan"), "accuracy": float("nan"), "margin": float("nan")}
    return {"loss": sum(losses) / len(losses), "accuracy": sum(accs) / len(accs), "margin": sum(margins) / len(margins)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sft_ckpt", type=Path)
    ap.add_argument("--dpo-dir", type=Path, default=REPO_ROOT / "data" / "dpo" / "ultrafeedback")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--peak-lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta = json.loads((args.dpo_dir / "meta.json").read_text())
    base_state = json.loads((args.sft_ckpt / "state.json").read_text())
    preset = base_state["preset"]
    cfg = PRESETS[preset]
    cfg.vocab_size = json.loads((args.dpo_dir.parent.parent / "sft" / "smoltalk" / "meta.json").read_text())["vocab_size"] \
        if (args.dpo_dir.parent.parent / "sft" / "smoltalk" / "meta.json").exists() else cfg.vocab_size
    cfg.max_seq_len = max(cfg.max_seq_len, meta["max_len"])

    run_name = (Path(args.out) if args.out else
                REPO_ROOT / "checkpoints" / f"{preset}-dpo-{time.strftime('%m%d-%H%M')}").resolve()
    run_name.mkdir(parents=True, exist_ok=True)
    log_f = (run_name / "log.jsonl").open("a")

    print(f"run: {_rel(run_name)}")
    print(f"sft base: {_rel(args.sft_ckpt)} (val loss {base_state.get('val_loss', '?')})")
    print(f"dpo data: {meta['n_train']:,} train pairs, {meta['n_val']:,} val pairs, max_len={meta['max_len']}")

    policy = Transformer(cfg)
    checkpoint.load(args.sft_ckpt, policy)
    policy.train()

    ref = Transformer(cfg)
    checkpoint.load(args.sft_ckpt, ref)
    ref.eval()  # frozen: never passed to an optimizer, never appears as the value_and_grad target

    from forge.training.schedule import wsd_schedule
    schedule = wsd_schedule(peak_lr=args.peak_lr, total_steps=args.steps, warmup_steps=args.warmup, decay_frac=0.8)
    optimizer = optim.AdamW(learning_rate=schedule, betas=[0.9, 0.95], weight_decay=0.0)
    max_norm = args.grad_clip or 1e9

    train_loader = PreferenceLoader(args.dpo_dir, "train", seed=0)
    val_loader = PreferenceLoader(args.dpo_dir, "val", seed=1)

    def loss_fn(m, c_ids, c_mask, r_ids, r_mask):
        loss, metrics = dpo_loss(m, ref, c_ids, c_mask, r_ids, r_mask, beta=args.beta)
        return loss, metrics

    loss_and_grad = nn.value_and_grad(policy, loss_fn)

    t0 = time.time()
    running = None
    for step in range(args.steps):
        c_ids, c_mask, r_ids, r_mask = train_loader.batch(args.batch_size)
        (loss, metrics), grads = loss_and_grad(policy, c_ids, c_mask, r_ids, r_mask)
        grads, gnorm = optim.clip_grad_norm(grads, max_norm)
        optimizer.update(policy, grads)
        mx.eval(policy.parameters(), optimizer.state, loss)

        lval = float(loss)
        running = lval if running is None else 0.9 * running + 0.1 * lval

        if not math.isfinite(lval):
            print(f"\n!! non-finite loss at step {step} - stopping.")
            checkpoint.save(run_name / "crash", policy, optimizer, step, extra={"preset": preset})
            break

        if step % args.log_every == 0:
            lr = float(schedule(mx.array(step)))
            print(f"step {step:>5} | loss {lval:5.3f} (ema {running:5.3f}) | acc {float(metrics['accuracy']):.2f} | "
                  f"margin {float(metrics['margin']):+.3f} | lr {lr:.1e} | |g| {float(gnorm):4.2f} | "
                  f"{(step + 1) * args.batch_size / (time.time() - t0):.1f} pairs/s")
            log_f.write(json.dumps({"step": step, "loss": lval, "accuracy": float(metrics["accuracy"]),
                                    "margin": float(metrics["margin"]), "lr": lr}) + "\n")
            log_f.flush()

        if step > 0 and step % args.eval_every == 0:
            ev = evaluate(policy, ref, val_loader, args.batch_size, args.eval_batches, args.beta)
            print(f"  >>> step {step}: val loss {ev['loss']:.3f}  acc {ev['accuracy']:.2f}  margin {ev['margin']:+.3f}")
            log_f.write(json.dumps({"step": step, "val": ev}) + "\n")
            log_f.flush()
            checkpoint.save(run_name / "last", policy, optimizer, step, extra={"preset": preset, **ev})

    ev = evaluate(policy, ref, val_loader, args.batch_size, args.eval_batches, args.beta)
    print(f"\ndone. val loss {ev['loss']:.3f}  preference accuracy {ev['accuracy']:.2f}  margin {ev['margin']:+.3f}")
    checkpoint.save(run_name / "last", policy, optimizer, args.steps, extra={"preset": preset, **ev})
    print(f"-> {_rel(run_name / 'last')}")


if __name__ == "__main__":
    main()
