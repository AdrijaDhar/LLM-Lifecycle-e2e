from __future__ import annotations

import heapq
import json
from collections import Counter

import regex as re

SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> Counter[tuple[int, ...]]:
    chunk_counts: Counter[tuple[int, ...]] = Counter()
    for chunk in SPLIT_PATTERN.findall(text):
        chunk_counts[tuple(chunk.encode("utf-8"))] += 1
    return chunk_counts


def count_pairs(chunk_counts: Counter[tuple[int, ...]]) -> Counter[tuple[int, int]]:
    pair_counts: Counter[tuple[int, int]] = Counter()
    for ids, freq in chunk_counts.items():
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += freq
    return pair_counts


def merge_chunk(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return tuple(merged)


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Special tokens (e.g. "<|endoftext|>") get IDs *above* the BPE range and
        # are never split or merged. Register them after train()/load().
        self.special_tokens: dict[str, int] = {}
        self.special_ids: dict[int, str] = {}
        # Per-chunk encode cache. Pretokens (words) repeat heavily across a corpus,
        # so caching their token sequences turns O(corpus) BPE work into O(unique
        # words). Invalidated whenever merges change (train/load).
        self._chunk_cache: dict[str, list[int]] = {}

    def register_special(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        new_id = 256 + len(self.merges) + len(self.special_tokens)
        self.special_tokens[token] = new_id
        self.special_ids[new_id] = token
        return new_id

    @property
    def eot_id(self) -> int:
        """ID of "<|endoftext|>"; registers it on first access."""
        return self.register_special("<|endoftext|>")

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256
        self._chunk_cache.clear()
        chunk_counts = pretokenize(text)
        num_merges = vocab_size - 256

        for i in range(num_merges):
            pair_counts = count_pairs(chunk_counts)
            if not pair_counts:
                break
            pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            new_id = 256 + i

            new_chunk_counts: Counter[tuple[int, ...]] = Counter()
            for ids, freq in chunk_counts.items():
                new_ids = merge_chunk(ids, pair, new_id)
                new_chunk_counts[new_ids] += freq
            chunk_counts = new_chunk_counts

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and (i < 10 or i % 50 == 0):
                token_repr = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {i + 1}/{num_merges}: {pair} -> {new_id} ({token_repr!r}, count={pair_counts[pair]})")

    def train_fast(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Same result as `train`, but with incremental pair counting.

        Instead of re-scanning the whole corpus every merge (O(merges x corpus)),
        we keep a running `pair_counts` dict, an index `pair_pos` from each pair to
        the words containing it, and a max-heap of pairs by count. A merge only
        touches the words that actually contained the merged pair, and only those
        words' pair counts are updated. Produces byte-for-byte identical `merges`
        to `train` for the same input (same tie-break: highest count, then lowest
        pair). See `tests/test_bpe.py::test_fast_matches_naive`.
        """
        assert vocab_size >= 256
        self._chunk_cache.clear()
        chunk_counts = pretokenize(text)
        words: list[list[int]] = [list(k) for k in chunk_counts]
        freqs: list[int] = list(chunk_counts.values())

        pair_counts: dict[tuple[int, int], int] = {}
        pair_pos: dict[tuple[int, int], set[int]] = {}
        for wi, w in enumerate(words):
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + f
                pair_pos.setdefault(p, set()).add(wi)

        # (-count, pair): tuple order gives highest count first, then lowest pair,
        # matching `train`'s max(key=(count, -a, -b)). Entries go stale as counts
        # change, so we re-validate against pair_counts on pop (lazy deletion).
        heap: list[tuple[int, tuple[int, int]]] = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        num_merges = vocab_size - 256
        for i in range(num_merges):
            pair: tuple[int, int] | None = None
            while heap:
                neg_c, cand = heapq.heappop(heap)
                if pair_counts.get(cand, 0) == -neg_c and -neg_c > 0:
                    pair = cand
                    break
            if pair is None:
                break

            new_id = 256 + i
            a, b = pair
            count = -neg_c

            for wi in list(pair_pos.get(pair, ())):
                w = words[wi]
                f = freqs[wi]
                old_pairs = set(zip(w, w[1:]))

                merged: list[int] = []
                j = 0
                while j < len(w):
                    if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(w[j])
                        j += 1
                words[wi] = merged
                new_pairs = set(zip(merged, merged[1:]))

                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                for p in zip(merged, merged[1:]):
                    pair_counts[p] = pair_counts.get(p, 0) + f
                for p in old_pairs - new_pairs:
                    s = pair_pos.get(p)
                    if s is not None:
                        s.discard(wi)
                for p in new_pairs:
                    pair_pos.setdefault(p, set()).add(wi)
                for p in old_pairs | new_pairs:
                    c = pair_counts.get(p, 0)
                    if c > 0:
                        heapq.heappush(heap, (-c, p))

            pair_pos.pop(pair, None)
            pair_counts.pop(pair, None)

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

            if verbose and (i < 10 or i % 500 == 0):
                token_repr = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {i + 1}/{num_merges}: {pair} -> {new_id} ({token_repr!r}, count={count})")

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached
        chunk_ids = tuple(chunk.encode("utf-8"))
        while len(chunk_ids) >= 2:
            pairs = set(zip(chunk_ids, chunk_ids[1:]))
            candidates = pairs & self.merges.keys()
            if not candidates:
                break
            pair = min(candidates, key=lambda p: self.merges[p])
            chunk_ids = merge_chunk(chunk_ids, pair, self.merges[pair])
        result = list(chunk_ids)
        self._chunk_cache[chunk] = result
        return result

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str, allowed_special: set[str] | str = "none") -> list[int]:
        """Encode text to token IDs.

        allowed_special: "none" (default) treats special-token strings as literal
        text; "all" recognises every registered special; or pass a set of the
        specific special strings to recognise. Recognised specials become their
        reserved ID; text between them is BPE-encoded normally.
        """
        if allowed_special == "all":
            specials = set(self.special_tokens)
        elif allowed_special == "none":
            specials = set()
        else:
            specials = set(allowed_special)

        if not specials:
            return self._encode_ordinary(text)

        pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
        ids: list[int] = []
        for part in re.split(pattern, text):
            if part in specials:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def id_to_bytes(self, i: int) -> bytes:
        """Raw bytes for one token id, before UTF-8 decoding. A single id's bytes
        are not necessarily valid UTF-8 on their own - a multi-byte character can
        be split across adjacent tokens (see StreamDecoder)."""
        if i in self.special_ids:
            return self.special_ids[i].encode("utf-8")
        return self.vocab[i]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_bytes(i) for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def save(self, path: str) -> None:
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        payload = {
            "merges": [[a, b, v] for (a, b), v in ordered],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        # Back-compat: old format was a bare list of [a, b, v] triples.
        merges = data if isinstance(data, list) else data["merges"]
        specials = {} if isinstance(data, list) else data.get("special_tokens", {})

        self._chunk_cache.clear()
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for a, b, v in merges:
            self.merges[(a, b)] = v
            self.vocab[v] = self.vocab[a] + self.vocab[b]
        self.special_tokens = {tok: int(i) for tok, i in specials.items()}
        self.special_ids = {i: tok for tok, i in self.special_tokens.items()}
