"""Run EleutherAI's lm-evaluation-harness against a trained checkpoint via our
own ForgeLM adapter - standardized, third-party-comparable numbers (the same
harness backs the Open LLM Leaderboard), not our own report card's numbers.

Runs the BASE pretrained model in plain-text few-shot mode (no chat template -
this matches how nanochat's own report card and most base-model leaderboards
score a model, and avoids needing chat-template plumbing in the harness).

    # quick check: 2 examples per task, fast, just proves the plumbing works
    uv run python scripts/run_eval_harness.py checkpoints/135m-0912-1812/annealed --limit 2

    # the real run (expect this to take a while - no torch/GPU batching here)
    uv run python scripts/run_eval_harness.py checkpoints/135m-0912-1812/annealed \
        --tasks arc_easy,arc_challenge,hellaswag,gsm8k --num-fewshot 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from lm_eval import evaluator
from lm_eval.utils import make_table

from forge.eval.lm_eval_wrapper import ForgeLM
from forge.model.config import PRESETS
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer
from forge.training import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]

# arc_easy/arc_challenge/hellaswag: loglikelihood multiple-choice, fast.
# gsm8k: generate_until, much slower (autoregressive, no batching here) - kept
# in the default list but expect it to dominate wall-clock time.
DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,gsm8k"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer" / "fw32k.bpe.json",
                     help="base (non-chat) tokenizer, matching the pretrained checkpoint")
    ap.add_argument("--tasks", default=DEFAULT_TASKS, help="comma-separated lm-eval task names")
    ap.add_argument("--num-fewshot", type=int, default=5)
    ap.add_argument("--limit", type=float, default=None, help="cap examples per task (int count, or float 0-1 fraction)")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    st = json.loads((args.ckpt / "state.json").read_text())
    tok = BPETokenizer()
    tok.load(str(args.tokenizer))
    cfg = PRESETS[st["preset"]]
    cfg.vocab_size = tok.vocab_size
    cfg.max_seq_len = max(cfg.max_seq_len, args.max_length)

    print(f"loading {args.ckpt} (preset {st['preset']}, val {st.get('val_loss', '?')}) ...")
    model = Transformer(cfg)
    checkpoint.load(args.ckpt, model)
    mx.eval(model.parameters())
    lm = ForgeLM(model, tok, max_length=args.max_length)

    tasks = args.tasks.split(",")
    print(f"tasks: {tasks}  num_fewshot={args.num_fewshot}  limit={args.limit}")

    results = evaluator.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=args.num_fewshot, limit=args.limit,
        bootstrap_iters=1000, random_seed=0, numpy_random_seed=0,
    )

    print("\n" + make_table(results))

    out_path = args.out or (args.ckpt / "eval_harness.json")
    slim = {k: v for k, v in results.items() if k in ("results", "configs", "n-shot")}
    out_path.write_text(json.dumps(slim, indent=2, default=str))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
