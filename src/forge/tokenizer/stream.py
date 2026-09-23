"""Streaming-safe decoding for token-at-a-time generation.

Our tokenizer is byte-level: a single token's bytes are not guaranteed to be a
complete UTF-8 character on their own (an accented/non-ASCII character can be
split across two adjacent tokens if the BPE merges never learned to fuse them).
Decoding `tok.decode([one_id])` in isolation can therefore emit the U+FFFD
replacement character (mojibake) even when the *full* sequence decodes fine.

`StreamDecoder` fixes this the same way any streaming UTF-8 consumer does:
buffer bytes and only emit text once a complete character boundary is reached.
"""

from __future__ import annotations

import codecs

from forge.tokenizer.bpe import BPETokenizer


class StreamDecoder:
    def __init__(self, tok: BPETokenizer) -> None:
        self.tok = tok
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        """Feed one token id; returns the text that's now safe to emit (often
        empty, if the byte(s) so far are mid-character)."""
        return self._decoder.decode(self.tok.id_to_bytes(token_id))

    def flush(self) -> str:
        """Call once generation ends, to drain any leftover buffered bytes."""
        return self._decoder.decode(b"", final=True)
