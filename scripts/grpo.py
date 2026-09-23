"""GRPO on GSM8K: on-policy rollouts, rule-based reward, group-relative advantage.

    uv run python scripts/grpo.py checkpoints/135m-dpo-0921-1805/last --steps 300

Each step: sample P prompts from GSM8K, generate G completions per prompt from
the CURRENT policy (temperature > 0, so they differ), score each completion
with the deterministic math_reward (format + correctness - no LLM judge),
normalize rewards within each prompt's group of G, then one gradient step on
grpo_loss. Runs eager (no mx.compile) - generation itself already dominates the
step time (autoregressive, one token at a time, done P*G times), and the
tensors change shape every step (completions vary in length), which compile
doesn't like anyway.

This is the slowest phase per unit of learning of the whole lifecycle - it's
also the one closest to "real" RLHF/RLVR. Expect small, noisy steps of
progress on GSM8K accuracy, not a dramatic jump, at 125M params / this rollout
budget.
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
import numpy as np
from datasets import load_dataset

from forge.data.chat import ChatFormat
from forge.data.math_reward import PROMPT_TEMPLATE, max_reward, reward
from forge.model.config import PRESETS
from forge.model.generate import generate
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer
from forge.training import checkpoint
from forge.training.grpo import group_advantages, grpo_loss

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def _pack(prompt_ids: list[int], completion_ids: list[int], pad_id: int, max_len: int) -> tuple[list[int], list[int]]:
    ids = (prompt_ids + completion_ids)[:max_len]
    mask = ([0] * len(prompt_ids) + [1] * len(completion_ids))[:max_len]
    pad = max_len - len(ids)
    return ids + [pad_id] * pad, mask + [0] * pad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy_ckpt", type=Path, help="starting checkpoint (DPO or SFT)")
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "fw32k-chat.bpe.json")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4, help="completions sampled per prompt (G)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=384, help="packed prompt+completion length")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--peak-lr", type=float, default=2e-6)
    ap.add_argument("--beta-kl", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-prompts", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_state = json.loads((args.policy_ckpt / "state.json").read_text())
    preset = base_state["preset"]

    tok = BPETokenizer()
    tok.load(str(args.tokenizer))
    fmt = ChatFormat.register(tok)
    cfg = PRESETS[preset]
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = max(cfg.max_seq_len, args.max_len)

    run_name = (Path(args.out) if args.out else
                REPO_ROOT / "checkpoints" / f"{preset}-grpo-{time.strftime('%m%d-%H%M')}").resolve()
    run_name.mkdir(parents=True, exist_ok=True)
    log_f = (run_name / "log.jsonl").open("a")

    print(f"run: {_rel(run_name)}")
    print(f"policy base: {_rel(args.policy_ckpt)}")
    print(f"rollout: {args.prompts_per_step} prompts x {args.group_size} completions "
          f"= {args.prompts_per_step * args.group_size} sequences/step, max {args.max_new_tokens} new tokens")

    policy = Transformer(cfg)
    checkpoint.load(args.policy_ckpt, policy)
    policy.train()

    ref = Transformer(cfg)
    checkpoint.load(args.policy_ckpt, ref)
    ref.eval()

    optimizer = optim.AdamW(learning_rate=args.peak_lr, betas=[0.9, 0.95], weight_decay=0.0)
    max_norm = args.grad_clip or 1e9

    def loss_fn(m, ids, mask, adv):
        loss, metrics = grpo_loss(m, ref, ids, mask, adv, beta_kl=args.beta_kl)
        return loss, metrics

    loss_and_grad = nn.value_and_grad(policy, loss_fn)

    print("loading gsm8k...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = np.random.default_rng(args.seed)

    def rollout(n_prompts: int, greedy_eval: bool = False):
        """Returns (ids, mask, rewards, correct_frac) for n_prompts x group_size completions."""
        idxs = rng.integers(0, len(ds), size=n_prompts)
        all_ids, all_mask, all_rewards = [], [], []
        n_correct = n_total = 0
        for i in idxs:
            row = ds[int(i)]
            question, gold = row["question"], row["answer"]
            prompt_text = PROMPT_TEMPLATE.format(question=question)
            prompt_ids = [fmt.user_id] + tok.encode(prompt_text) + [fmt.end_id, fmt.assistant_id]

            g = 1 if greedy_eval else args.group_size
            for _ in range(g):
                gen_ids = list(generate(
                    policy, prompt_ids, max_new_tokens=args.max_new_tokens,
                    temperature=0.0 if greedy_eval else args.temperature,
                    top_k=None if greedy_eval else args.top_k,
                    eot_id=fmt.end_id, repetition_penalty=args.repetition_penalty,
                ))
                text = tok.decode(gen_ids)
                r = reward(text, gold)
                all_rewards.append(r)
                n_total += 1
                n_correct += int(r >= max_reward())
                ids, mask = _pack(prompt_ids, gen_ids, pad_id=fmt.end_id, max_len=args.max_len)
                all_ids.append(ids); all_mask.append(mask)

        return (mx.array(all_ids), mx.array(all_mask, dtype=mx.float32),
                np.array(all_rewards, dtype=np.float32), n_correct / max(n_total, 1))

    t0 = time.time()
    running_reward = None
    for step in range(args.steps):
        ids, mask, rewards, correct_frac = rollout(args.prompts_per_step)
        advantages = mx.array(group_advantages(rewards, group_size=args.group_size))

        (loss, metrics), grads = loss_and_grad(policy, ids, mask, advantages)
        grads, gnorm = optim.clip_grad_norm(grads, max_norm)
        optimizer.update(policy, grads)
        mx.eval(policy.parameters(), optimizer.state, loss)

        mean_r = float(rewards.mean())
        running_reward = mean_r if running_reward is None else 0.9 * running_reward + 0.1 * mean_r
        elapsed = time.time() - t0
        completions_per_s = (step + 1) * args.prompts_per_step * args.group_size / elapsed

        if not math.isfinite(float(loss)):
            print(f"\n!! non-finite loss at step {step} - stopping.")
            checkpoint.save(run_name / "crash", policy, optimizer, step, extra={"preset": preset})
            break

        print(f"step {step:>4} | loss {float(loss):6.3f} | reward {mean_r:5.2f} (ema {running_reward:5.2f}, "
              f"max {max_reward():.0f}) | acc {correct_frac:.2f} | kl {float(metrics['kl']):.4f} | "
              f"|g| {float(gnorm):5.2f} | {completions_per_s:.2f} compl/s")
        log_f.write(json.dumps({"step": step, "loss": float(loss), "mean_reward": mean_r,
                                "accuracy": correct_frac, "kl": float(metrics["kl"])}) + "\n")
        log_f.flush()

        if step > 0 and step % args.eval_every == 0:
            _, _, eval_rewards, eval_acc = rollout(args.eval_prompts, greedy_eval=True)
            print(f"  >>> step {step}: greedy eval reward {eval_rewards.mean():.2f}  accuracy {eval_acc:.2f}")
            log_f.write(json.dumps({"step": step, "eval_reward": float(eval_rewards.mean()), "eval_accuracy": eval_acc}) + "\n")
            log_f.flush()
            checkpoint.save(run_name / "last", policy, optimizer, step,
                            extra={"preset": preset, "eval_accuracy": eval_acc})

    _, _, eval_rewards, eval_acc = rollout(args.eval_prompts, greedy_eval=True)
    print(f"\ndone. greedy eval reward {eval_rewards.mean():.2f}  accuracy {eval_acc:.2f}")
    checkpoint.save(run_name / "last", policy, optimizer, args.steps,
                    extra={"preset": preset, "eval_accuracy": eval_acc})
    print(f"-> {_rel(run_name / 'last')}")


if __name__ == "__main__":
    main()
