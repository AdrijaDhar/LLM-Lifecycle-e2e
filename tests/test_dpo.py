"""Tests for preference-pair encoding and the DPO loss.

Run: uv run pytest -q tests/test_dpo.py
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np

from forge.data.chat import ChatFormat
from forge.data.preference import PreferenceLoader, encode_pair
from forge.model.config import ModelConfig
from forge.model.transformer import Transformer
from forge.tokenizer.bpe import BPETokenizer
from forge.training.dpo import dpo_loss, sequence_logp

TRAIN_TEXT = (
    "the cat sat on the mat. the dog ran to the cat. "
    "please answer the question about the weather today. "
) * 30


def _tok() -> tuple[BPETokenizer, ChatFormat]:
    t = BPETokenizer()
    t.train_fast(TRAIN_TEXT, vocab_size=350)
    t.register_special("<|endoftext|>")
    fmt = ChatFormat.register(t)
    return t, fmt


def test_encode_pair_shapes_and_padding() -> None:
    tok, fmt = _tok()
    out = encode_pair(tok, fmt, "what is the weather", "the weather is sunny", "the cat sat", max_len=40)
    assert out is not None
    assert len(out["chosen_ids"]) == 40 == len(out["chosen_mask"])
    assert len(out["rejected_ids"]) == 40 == len(out["rejected_mask"])
    # the two completions share the same prompt prefix (same mask up to where content starts)
    n_prompt = out["chosen_mask"].index(1)
    assert out["chosen_ids"][:n_prompt] == out["rejected_ids"][:n_prompt]


def test_encode_pair_rejects_too_long() -> None:
    tok, fmt = _tok()
    out = encode_pair(tok, fmt, "hi", "the cat sat on the mat " * 50, "ok", max_len=20)
    assert out is None


def test_encode_pair_mask_only_on_completion() -> None:
    tok, fmt = _tok()
    out = encode_pair(tok, fmt, "what is the weather", "sunny", "cat", max_len=30)
    for ids, mask in [(out["chosen_ids"], out["chosen_mask"]), (out["rejected_ids"], out["rejected_mask"])]:
        for i, tid in enumerate(ids):
            if tid == fmt.user_id:
                assert mask[i] == 0


def test_preference_loader_batch_shapes(tmp_path) -> None:
    n, L = 10, 16
    for name in ("chosen_ids", "rejected_ids"):
        np.save(tmp_path / f"train_{name}.npy", np.random.randint(0, 50, size=(n, L)))
    for name in ("chosen_mask", "rejected_mask"):
        np.save(tmp_path / f"train_{name}.npy", np.random.randint(0, 2, size=(n, L)))

    loader = PreferenceLoader(tmp_path, "train", seed=0)
    c_ids, c_mask, r_ids, r_mask = loader.batch(4)
    assert c_ids.shape == (4, L) and c_mask.shape == (4, L)
    assert r_ids.shape == (4, L) and r_mask.shape == (4, L)


# --- DPO loss ---------------------------------------------------------------

def _tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, n_kv_heads=1,
                       hidden_dim=64, max_seq_len=32)


def test_sequence_logp_matches_manual_sum() -> None:
    cfg = _tiny_cfg()
    model = Transformer(cfg)
    ids = mx.random.randint(0, cfg.vocab_size, (1, 8))
    mask = mx.array([[0, 0, 1, 1, 1, 0, 0, 0]])

    logp = sequence_logp(model, ids, mask)

    logits = model(ids[:, :-1])
    targets = ids[:, 1:]
    neg = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    m = mask[:, 1:].reshape(-1)
    manual = -(neg * m).sum()
    assert abs(float(logp[0]) - float(manual)) < 1e-4


def test_dpo_loss_at_initialisation_is_log2() -> None:
    # policy == ref exactly -> rewards are 0 for both -> loss = -log(sigmoid(0)) = log(2)
    cfg = _tiny_cfg()
    policy = Transformer(cfg)
    ref = Transformer(cfg)
    ref.update(policy.parameters())

    ids = mx.random.randint(0, cfg.vocab_size, (3, 10))
    mask = mx.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]] * 3)
    loss, metrics = dpo_loss(policy, ref, ids, mask, ids, mask, beta=0.1)
    assert abs(float(loss) - math.log(2)) < 1e-4
    assert abs(float(metrics["margin"])) < 1e-4


def test_dpo_training_increases_preference_for_chosen() -> None:
    cfg = _tiny_cfg()
    policy = Transformer(cfg)
    ref = Transformer(cfg)
    ref.update(policy.parameters())
    opt = optim.AdamW(learning_rate=5e-3)

    mx.random.seed(0)
    chosen = mx.random.randint(0, cfg.vocab_size, (4, 12))
    rejected = mx.random.randint(0, cfg.vocab_size, (4, 12))
    mask = mx.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]] * 4)

    def loss_fn(m):
        loss, _ = dpo_loss(m, ref, chosen, mask, rejected, mask, beta=0.1)
        return loss

    lg = nn.value_and_grad(policy, loss_fn)
    _, m0 = dpo_loss(policy, ref, chosen, mask, rejected, mask, beta=0.1)
    for _ in range(40):
        loss, grads = lg(policy)
        opt.update(policy, grads)
        mx.eval(policy.parameters(), opt.state)
    _, m1 = dpo_loss(policy, ref, chosen, mask, rejected, mask, beta=0.1)

    assert float(m1["margin"]) > float(m0["margin"])
    assert float(loss) < math.log(2)
