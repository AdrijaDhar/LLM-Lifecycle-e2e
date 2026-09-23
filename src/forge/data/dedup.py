"""Near-duplicate detection with MinHash + LSH.

Why: exact-dup removal misses documents that are 95% the same (same article with
a different header/footer, mirrored pages, templated spam). Training on those
wastes compute and teaches memorization. We want to drop documents whose
*Jaccard similarity* on word n-grams exceeds a threshold (~0.7-0.8).

Computing Jaccard for every pair is O(N^2) - impossible at 200k docs. The trick:

1. SHINGLES. Represent a document as the *set* of its overlapping word n-grams
   ("shingles"). Jaccard(A, B) = |shingles(A) & shingles(B)| / |union|.

2. MINHASH. For K random hash functions, a document's signature is the K minimum
   hash values over its shingles. Key property: P(minhash_i(A) == minhash_i(B))
   == Jaccard(A, B) exactly. So the fraction of matching signature slots is an
   unbiased estimate of Jaccard - K numbers instead of thousands of shingles.

3. LSH (Locality-Sensitive Hashing). Split the K-slot signature into B bands of
   R rows (K = B*R). Two docs are "candidates" if they match exactly on any full
   band. The probability of that rises sharply around a threshold set by B and R
   (roughly t ~= (1/B)^(1/R)) - so similar docs collide, dissimilar ones almost
   never do, and we only compare candidates.

`tests/test_dedup.py` checks the MinHash estimate against true Jaccard and the
end-to-end removal.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np

_U64_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)
WORD_RE = re.compile(r"\w+")

# splitmix64 finalizer constants
_SM1 = np.uint64(0x9E3779B97F4A7C15)
_SM2 = np.uint64(0xBF58476D1CE4E5B9)
_SM3 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)


def _splitmix64(z: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 bit-mixer. uint64 arithmetic wraps mod 2^64 (intended)."""
    z = z + _SM1
    z = (z ^ (z >> _S30)) * _SM2
    z = (z ^ (z >> _S27)) * _SM3
    return z ^ (z >> _S31)


def shingles(text: str, k: int = 5) -> set[str]:
    """Set of overlapping k-word n-grams, lowercased."""
    words = WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class MinHasher:
    """Turns a shingle set into a length-`num_perm` MinHash signature."""

    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        # One random 64-bit offset per permutation. hash_i(s) = splitmix64(crc32(s)
        # + offset_i): crc32 is a cheap base hash, splitmix64 scrambles it into a
        # high-quality, near-independent value for each permutation.
        self.offsets = rng.integers(0, 1 << 64, size=num_perm, dtype=np.uint64)

    def signature(self, sh: set[str]) -> np.ndarray:
        if not sh:
            return np.full(self.num_perm, _U64_MAX, dtype=np.uint64)
        base = np.fromiter(
            (zlib.crc32(s.encode("utf-8")) for s in sh),
            dtype=np.uint64,
            count=len(sh),
        )
        # (num_perm, n_shingles) -> min hash value over shingles, per permutation
        mixed = _splitmix64(base[None, :] + self.offsets[:, None])
        return mixed.min(axis=1)


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))


class LSHIndex:
    """Banded LSH over MinHash signatures. `add` then `iter_buckets`."""

    def __init__(self, num_perm: int = 128, bands: int = 16) -> None:
        if num_perm % bands:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self._buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]

    @property
    def implied_threshold(self) -> float:
        """Rough similarity where detection probability crosses ~0.5."""
        return (1.0 / self.bands) ** (1.0 / self.rows)

    def add(self, doc_id: int, sig: np.ndarray) -> None:
        for bi in range(self.bands):
            key = sig[bi * self.rows : (bi + 1) * self.rows].tobytes()
            self._buckets[bi][key].append(doc_id)

    def iter_buckets(self):
        """Yield each group of >=2 doc ids that collided in some band."""
        for band in self._buckets:
            for ids in band.values():
                if len(ids) > 1:
                    yield ids


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return out


def dedup(
    signatures: np.ndarray,
    weights: np.ndarray | None = None,
    bands: int = 16,
    threshold: float = 0.8,
) -> np.ndarray:
    """Given an (N, num_perm) signature matrix, return the indices to KEEP.

    Candidates that collide in LSH and whose estimated Jaccard >= threshold are
    unioned into a cluster; from each cluster we keep the single highest-weight
    document (default weight = 1, i.e. keep the lowest index).
    """
    n, num_perm = signatures.shape
    weights = np.ones(n) if weights is None else weights

    lsh = LSHIndex(num_perm=num_perm, bands=bands)
    for i in range(n):
        lsh.add(i, signatures[i])

    uf = UnionFind(n)
    for ids in lsh.iter_buckets():
        anchor = ids[0]
        for other in ids[1:]:
            if estimated_jaccard(signatures[anchor], signatures[other]) >= threshold:
                uf.union(anchor, other)

    keep = np.ones(n, dtype=bool)
    for members in uf.clusters().values():
        if len(members) == 1:
            continue
        winner = max(members, key=lambda i: (weights[i], -i))
        for m in members:
            if m != winner:
                keep[m] = False
    return np.flatnonzero(keep)
