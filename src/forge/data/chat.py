"""Chat formatting: turn a list of {role, content} turns into (token_ids, loss_mask).

We add three new special tokens on top of the pretrained tokenizer's vocab:
    <|user|>        opens a user turn
    <|assistant|>    opens an assistant turn
    <|end|>          closes any turn

A conversation is flattened to one token stream:

    <|user|> {user text} <|end|> <|assistant|> {assistant text} <|end|> ...

The loss mask is 1 only on assistant CONTENT and its trailing <|end|> - the
model is trained to predict what an assistant says and when to stop, but not
to predict the user's turn (it doesn't get a say in that) or the role tags
themselves (those are fixed formatting, not a modelling choice worth spending
gradient on).
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.tokenizer.bpe import BPETokenizer

CHAT_SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]


@dataclass
class ChatFormat:
    user_id: int
    assistant_id: int
    end_id: int

    @classmethod
    def register(cls, tok: BPETokenizer) -> "ChatFormat":
        """Register the chat special tokens on `tok` (idempotent) and return their ids."""
        ids = [tok.register_special(t) for t in CHAT_SPECIAL_TOKENS]
        return cls(*ids)

    def encode_conversation(self, tok: BPETokenizer, turns: list[dict]) -> tuple[list[int], list[int]]:
        """turns: [{"role": "user"|"assistant", "content": str}, ...] -> (ids, loss_mask)."""
        ids: list[int] = []
        mask: list[int] = []
        for turn in turns:
            content_ids = tok.encode(turn["content"])
            if turn["role"] in ("user", "system"):
                # No separate <|system|> tag: a system prompt is context the model
                # reads, not something it's learning to produce - same as a user turn.
                ids += [self.user_id] + content_ids + [self.end_id]
                mask += [0] * (2 + len(content_ids))
            elif turn["role"] == "assistant":
                ids += [self.assistant_id]
                mask += [0]
                ids += content_ids + [self.end_id]
                mask += [1] * (len(content_ids) + 1)
            else:
                raise ValueError(f"unknown role {turn['role']!r}")
        return ids, mask
