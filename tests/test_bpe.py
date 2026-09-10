"""Unit tests for the byte-level BPE tokenizer.

Run: uv run pytest -q
"""

from __future__ import annotations

import pytest

from forge.tokenizer.bpe import BPETokenizer

TRAIN_TEXT = (
    "the cat sat on the mat. the cat sat on the hat. "
    "a dog ran to the cat. the cat ran to the dog. "
    "resource blocks describe infrastructure; variables parameterize modules. "
    "training a tokenizer means learning frequent byte pairs from real text. "
) * 40

ROUNDTRIP_CASES = [
    "the cat sat on the mat",
    "",
    "a",
    "totally unseen words: xylophone quetzalcoatl",
    "emoji \U0001f600 and accents café naïve",
    "tabs\tand\nnewlines\n\n",
    "日本語のテキスト",
    'resource "aws_s3_bucket" "b" {}',
]


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train(TRAIN_TEXT, vocab_size=350)
    return t


def test_base_vocab_is_256_bytes() -> None:
    t = BPETokenizer()
    assert t.vocab_size == 256
    assert t.encode("abc") == [97, 98, 99]


def test_trains_to_requested_size(tok: BPETokenizer) -> None:
    # Training may stop early if the corpus runs out of mergeable pairs.
    assert 256 < tok.vocab_size <= 350
    assert len(tok.merges) == tok.vocab_size - 256


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_roundtrip(tok: BPETokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_arbitrary_bytes(tok: BPETokenizer) -> None:
    blob = bytes(range(256)).decode("latin-1")
    assert tok.decode(tok.encode(blob)) == blob


def test_merges_actually_compress(tok: BPETokenizer) -> None:
    ids = tok.encode("the cat sat on the mat. the cat sat on the mat.")
    raw = len("the cat sat on the mat. the cat sat on the mat.".encode("utf-8"))
    assert len(ids) < raw


def test_encode_is_deterministic(tok: BPETokenizer) -> None:
    s = "the cat ran to the dog and the dog ran to the cat"
    assert tok.encode(s) == tok.encode(s)


RICH_TEXT = (
    TRAIN_TEXT
    + "\n"
    + "".join(f"item {n}: value={n * n}, name=widget_{n % 7}. " for n in range(200))
    + "\nдвуязычный текст with 日本語 mixed in. " * 5
)


def test_fast_matches_naive() -> None:
    naive = BPETokenizer()
    naive.train(RICH_TEXT, vocab_size=500)
    fast = BPETokenizer()
    fast.train_fast(RICH_TEXT, vocab_size=500)
    assert fast.merges == naive.merges
    assert fast.vocab == naive.vocab
    probe = "item 42: value=1764, name=widget_0. 日本語"
    assert fast.encode(probe) == naive.encode(probe)


def test_save_load_roundtrip(tok: BPETokenizer, tmp_path) -> None:
    p = tmp_path / "t.bpe.json"
    tok.save(str(p))
    reloaded = BPETokenizer()
    reloaded.load(str(p))
    assert reloaded.merges == tok.merges
    s = "the cat sat on the mat"
    assert reloaded.encode(s) == tok.encode(s)


def _fresh_tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train_fast(TRAIN_TEXT, vocab_size=350)
    return t


def test_special_token_id_above_bpe_range() -> None:
    t = _fresh_tok()
    eot = t.eot_id
    assert eot == 256 + len(t.merges)
    assert t.vocab_size == 256 + len(t.merges) + 1


def test_special_default_is_literal_text() -> None:
    t = _fresh_tok()
    t.register_special("<|endoftext|>")
    ids = t.encode("a<|endoftext|>b")  # default: marker is just characters
    assert t.special_tokens["<|endoftext|>"] not in ids
    assert t.decode(ids) == "a<|endoftext|>b"


def test_special_recognised_when_allowed() -> None:
    t = _fresh_tok()
    eot = t.register_special("<|endoftext|>")
    ids = t.encode("the cat<|endoftext|>the dog", allowed_special="all")
    assert ids.count(eot) == 1
    assert ids == t.encode("the cat") + [eot] + t.encode("the dog")
    assert t.decode(ids) == "the cat<|endoftext|>the dog"


def test_special_survives_save_load(tmp_path) -> None:
    t = _fresh_tok()
    t.register_special("<|endoftext|>")
    t.register_special("<|user|>")
    p = tmp_path / "s.bpe.json"
    t.save(str(p))
    r = BPETokenizer()
    r.load(str(p))
    assert r.special_tokens == t.special_tokens
    txt = "hi<|user|>there<|endoftext|>"
    assert r.encode(txt, allowed_special="all") == t.encode(txt, allowed_special="all")
