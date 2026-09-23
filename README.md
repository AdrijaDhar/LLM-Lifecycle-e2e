

A 125M-parameter language model, built and trained from raw bytes on one
Apple M3 Max, in [MLX](https://github.com/ml-explore/mlx) — no cloud compute
anywhere in this pipeline. Tokenizer → pretraining → SFT → DPO → GRPO,
every stage measured, every checkpoint real.



## Results

Standardized eval ([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
5-shot, full test sets — not our own eval code):

| Task | Forge 135M (1.6B tokens, one Mac) | nanochat d20 (11.2B tokens, 8×H100) |
|---|---|---|
| ARC-Easy | **0.4655** | 0.3876 |
| ARC-Challenge (norm) | 0.2415 | 0.2807 |
| HellaSwag (norm) | 0.2849 | — |
| GSM8K (base model, 5-shot) | 1.5% | — |

ARC-Easy beats a reference model trained on 7× more data across an 8×H100
node. ARC-Challenge and HellaSwag land where a 125M model at this data scale
honestly should — both outcomes are the real result, not just the flattering
one.

## How it was built

**Tokenizer.** Byte-level BPE, hand-trained two ways: a readable
`O(merges×corpus)` version, then an incremental heap-based one — proven
byte-identical on their merge tables before the fast one touched real data.
32,772 tokens, 4.573 chars/token.

**Data.** 8GB of FineWeb-EDU streamed, quality-filtered (structural heuristics
— repetition, bullet-density, symbol ratio), MinHash+LSH deduplicated, packed
to 1.615B training tokens. The MinHash estimator was biased 0.97 vs. a true
0.85 similarity on the first pass — a weak 32-bit hash family under-mixing
shingle hashes; fixed with splitmix64 bit-mixing, verified against a
from-scratch true-Jaccard test.

**Model.** SmolLM2-shaped decoder: RoPE, Grouped-Query Attention, SwiGLU,
RMSNorm, tied embeddings. 135M preset = 576-dim, 30 layers, 9 query / 3 KV
heads.

**Pretraining.** Two-phase WSD schedule (warmup → flat peak LR for as long as
the machine had → short cosine anneal whenever the run was called done) —
trained once as a pipeline proof (393M tokens), then rebuilt on 8× more real
data once the loop was trusted (1.615B tokens, val loss 3.343, ppl 28.3). The
real memory ceiling for `mx.compile`'d training turned out to be batch 32, not
64 — batch 96 OOM'd inside the compiled graph, batch 64 didn't crash but was
silently swapping to disk at 439 tok/s.

**SFT.** Chat template (`<|user|>`/`<|assistant|>`/`<|end|>`) grafted onto the
pretrained embedding table via `checkpoint.load_resized()`, loss masked to
assistant tokens only. `smol-smoltalk` + CodeAlpaca (3× upweighted, after the
smoltalk-only model gave weird answers to code questions) — val loss 1.702.
Two real bugs surfaced only by reading generated text closely: repetition
penalty was suppressing the model's own `<|end|>` stop token, and streaming
output mojibake from decoding one token at a time (byte-level tokens don't
align to UTF-8 boundaries).

**DPO.** Reward defined relative to a frozen reference copy of the SFT model
— no separate reward model needed. At step 0, policy == reference exactly, so
the loss opens at precisely `log(2)`; a clean closed-form sanity check that
also showed up correctly on the real run. UltraFeedback pairs, preference
accuracy 0.50 → 0.59.

**GRPO (RLVR).** On-policy: sample 4 completions per GSM8K problem, score
each with a deterministic `"#### <number>"` parser (no LLM judge), normalize
reward *within* the group of 4 — the group mean is the baseline, no value
network required. `group_advantages()` is hand-reimplemented and tested
against hand-computed examples. Greedy-eval accuracy 0.00 → 0.05.

**Quantization + serving.** INT4 via `mlx.nn.quantize` (500MB → 78MB), served
through a local chat UI — stdlib `http.server`, no new dependencies.

## Pipeline

```bash
scripts/download_corpus.py → filter_corpus.py → dedup_corpus.py → pack_corpus.py
scripts/train_tokenizer.py
scripts/pretrain.py            # + anneal.py for the cooldown phase
scripts/prepare_sft_data.py  → sft.py
scripts/prepare_dpo_data.py  → dpo.py
scripts/grpo.py
scripts/quantize.py
scripts/run_eval_harness.py
scripts/sample.py / sample_chat.py / chat_server.py    # talk to a checkpoint
```

## Quickstart

```bash
uv sync
uv run pytest -q                       # 103 tests
uv run python scripts/chat_server.py checkpoints/135m-dpo-0921-1805/last
# -> http://127.0.0.1:8800
```

`checkpoints/` and `data/` aren't in this repo (multi-GB binary artifacts,
gitignored) — regenerate them by running the pipeline above.

## Stack

MLX · Python 3.12 · [uv](https://github.com/astral-sh/uv) · Apple M3 Max, 128GB.

[![ci](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
— pytest on `macos-latest` (MLX needs real Metal hardware). Every PR also gets
a [graph-code](https://github.com/AdrijaDhar/graph-code) blast-radius comment
showing which tests actually exercise the changed code.
