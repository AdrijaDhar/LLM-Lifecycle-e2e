"""Tests for streaming-safe UTF-8 decoding.

Run: uv run pytest -q tests/test_stream.py
"""

from __future__ import annotations

from forge.tokenizer.bpe import BPETokenizer
from forge.tokenizer.stream import StreamDecoder

# ASCII-only training corpus: the tokenizer never learns to merge the two
# bytes of a non-ASCII character like 'é' (0xC3 0xA9), so encoding "café"
# yields them as two separate single-byte tokens - the exact split that
# breaks per-token decoding.
ASCII_TEXT = "the cat sat on the mat. the dog ran to the cat. " * 40


def _tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train_fast(ASCII_TEXT, vocab_size=300)
    return t


def test_naive_per_token_decode_can_mojibake() -> None:
    tok = _tok()
    ids = tok.encode("café")
    naive = "".join(tok.decode([i]) for i in ids)
    assert "�" in naive  # documents the bug this module fixes


def test_stream_decoder_reassembles_split_character() -> None:
    tok = _tok()
    ids = tok.encode("café résumé naïve")
    dec = StreamDecoder(tok)
    streamed = "".join(dec.push(i) for i in ids) + dec.flush()
    assert streamed == "café résumé naïve"
    assert streamed == tok.decode(ids)


def test_stream_decoder_matches_batch_decode_on_ascii() -> None:
    tok = _tok()
    ids = tok.encode("the cat sat on the mat")
    dec = StreamDecoder(tok)
    streamed = "".join(dec.push(i) for i in ids) + dec.flush()
    assert streamed == tok.decode(ids)


def test_stream_decoder_handles_special_tokens() -> None:
    tok = _tok()
    eot = tok.register_special("<|endoftext|>")
    dec = StreamDecoder(tok)
    out = dec.push(eot) + dec.flush()
    assert out == "<|endoftext|>"
