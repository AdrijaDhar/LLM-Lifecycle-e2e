"""Tests for chat formatting (SFT special tokens + loss masking).

Run: uv run pytest -q tests/test_chat.py
"""

from __future__ import annotations

from forge.data.chat import ChatFormat
from forge.tokenizer.bpe import BPETokenizer

TRAIN_TEXT = (
    "the cat sat on the mat. the cat sat on the hat. "
    "a dog ran to the cat. the cat ran to the dog. "
    "please answer the question about the weather today. "
) * 30


def _tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train_fast(TRAIN_TEXT, vocab_size=400)
    t.register_special("<|endoftext|>")
    return t


def test_register_ids_above_existing_specials() -> None:
    tok = _tok()
    base = tok.vocab_size  # 256 + merges + <|endoftext|>
    fmt = ChatFormat.register(tok)
    assert fmt.user_id == base
    assert fmt.assistant_id == base + 1
    assert fmt.end_id == base + 2
    assert tok.vocab_size == base + 3


def test_register_is_idempotent() -> None:
    tok = _tok()
    fmt1 = ChatFormat.register(tok)
    fmt2 = ChatFormat.register(tok)
    assert fmt1 == fmt2
    assert tok.vocab_size == fmt1.end_id + 1


def test_mask_shape_matches_ids() -> None:
    tok = _tok()
    fmt = ChatFormat.register(tok)
    turns = [
        {"role": "user", "content": "what is the weather today"},
        {"role": "assistant", "content": "the weather is sunny"},
    ]
    ids, mask = fmt.encode_conversation(tok, turns)
    assert len(ids) == len(mask)


def test_mask_is_zero_on_user_turn() -> None:
    tok = _tok()
    fmt = ChatFormat.register(tok)
    turns = [{"role": "user", "content": "the cat sat on the mat"}]
    ids, mask = fmt.encode_conversation(tok, turns)
    assert set(mask) == {0}
    assert ids[0] == fmt.user_id
    assert ids[-1] == fmt.end_id


def test_mask_is_one_on_assistant_content_and_end() -> None:
    tok = _tok()
    fmt = ChatFormat.register(tok)
    turns = [
        {"role": "user", "content": "the cat sat on the mat"},
        {"role": "assistant", "content": "the dog ran to the cat"},
    ]
    ids, mask = fmt.encode_conversation(tok, turns)
    a_start = ids.index(fmt.assistant_id)
    assert mask[a_start] == 0                 # the <|assistant|> tag itself: not trained on
    assert all(m == 1 for m in mask[a_start + 1 :])  # content + trailing <|end|>
    assert ids[-1] == fmt.end_id


def test_multi_turn_masks_only_assistant_spans() -> None:
    tok = _tok()
    fmt = ChatFormat.register(tok)
    turns = [
        {"role": "user", "content": "the cat sat on the mat"},
        {"role": "assistant", "content": "the dog ran to the cat"},
        {"role": "user", "content": "please answer the question"},
        {"role": "assistant", "content": "the weather is sunny"},
    ]
    ids, mask = fmt.encode_conversation(tok, turns)
    n_ones = sum(mask)
    n_assistant_turns = 2
    # each assistant span contributes >= 1 masked token (its <|end|>) plus content
    assert n_ones >= n_assistant_turns
    # nothing after the very last <|end|> and no mask=1 tokens land on <|user|> ids
    for i, tid in enumerate(ids):
        if tid == fmt.user_id:
            assert mask[i] == 0


def test_system_role_treated_like_user() -> None:
    tok = _tok()
    fmt = ChatFormat.register(tok)
    turns = [
        {"role": "system", "content": "please answer the question"},
        {"role": "assistant", "content": "the weather is sunny"},
    ]
    ids, mask = fmt.encode_conversation(tok, turns)
    assert ids[0] == fmt.user_id  # system reuses the user tag
    assert mask[0] == 0


def test_roundtrip_decodes_with_role_tags() -> None:
    tok = _tok()
    fmt = ChatFormat.register(tok)
    turns = [
        {"role": "user", "content": "what is the weather today"},
        {"role": "assistant", "content": "the weather is sunny"},
    ]
    ids, _ = fmt.encode_conversation(tok, turns)
    text = tok.decode(ids)
    assert "<|user|>" in text and "<|assistant|>" in text and "<|end|>" in text
