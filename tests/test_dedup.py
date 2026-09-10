"""Tests for MinHash + LSH dedup.

Run: uv run pytest -q tests/test_dedup.py
"""

from __future__ import annotations

import numpy as np

from forge.data.dedup import (
    LSHIndex,
    MinHasher,
    dedup,
    estimated_jaccard,
    shingles,
)


def true_jaccard(a: str, b: str, k: int = 5) -> float:
    sa, sb = shingles(a, k), shingles(b, k)
    return len(sa & sb) / len(sa | sb)


LOREM = (
    "the endosymbiotic theory explains the origin of mitochondria and chloroplasts "
    "as free living bacteria that were engulfed by an ancestral eukaryotic cell "
    "and gradually became permanent organelles retaining a small circular genome "
).split()


def _doc(words: list[str]) -> str:
    return " ".join(words)


def test_shingles_count() -> None:
    assert shingles("a b c d e f", k=5) == {"a b c d e", "b c d e f"}
    assert shingles("short", k=5) == {"short"}


def test_minhash_estimates_true_jaccard() -> None:
    mh = MinHasher(num_perm=256, seed=0)
    a = _doc(LOREM)
    b = _doc(LOREM[:35] + ["extra", "trailing", "words", "here", "now"])  # ~85% overlap
    est = estimated_jaccard(mh.signature(shingles(a)), mh.signature(shingles(b)))
    true = true_jaccard(a, b)
    assert abs(est - true) < 0.08, (est, true)


def test_identical_docs_identical_signature() -> None:
    mh = MinHasher(num_perm=64, seed=3)
    s = mh.signature(shingles(_doc(LOREM)))
    assert np.array_equal(s, mh.signature(shingles(_doc(LOREM))))
    assert estimated_jaccard(s, s) == 1.0


def test_disjoint_docs_low_estimate() -> None:
    mh = MinHasher(num_perm=128, seed=1)
    a = mh.signature(shingles("alpha beta gamma delta epsilon zeta eta theta iota kappa"))
    b = mh.signature(shingles("one two three four five six seven eight nine ten eleven"))
    assert estimated_jaccard(a, b) < 0.05


def test_lsh_buckets_similar_not_dissimilar() -> None:
    mh = MinHasher(num_perm=128, seed=2)
    lsh = LSHIndex(num_perm=128, bands=16)
    near1 = _doc(LOREM)
    near2 = _doc(LOREM[:38] + ["and", "later", "evolved"])
    far = _doc("completely different content about quantum field theory and gauge symmetry breaking")
    for i, d in enumerate([near1, near2, far]):
        lsh.add(i, mh.signature(shingles(d)))
    bucketed_pairs = {tuple(sorted((ids[0], j))) for ids in lsh.iter_buckets() for j in ids[1:]}
    assert (0, 1) in bucketed_pairs
    assert (0, 2) not in bucketed_pairs and (1, 2) not in bucketed_pairs


def test_dedup_end_to_end() -> None:
    mh = MinHasher(num_perm=128, seed=7)
    docs = [
        _doc(LOREM),                                   # 0
        _doc(LOREM[:39] + ["!!!"]),                     # 1  near-dup of 0
        _doc(LOREM),                                    # 2  exact dup of 0
        "a totally unrelated paragraph about volcanic soil chemistry and basalt weathering rates",  # 3
    ]
    sigs = np.vstack([mh.signature(shingles(d)) for d in docs])
    keep = dedup(sigs, bands=16, threshold=0.7)
    assert 3 in keep
    assert len([i for i in keep if i in (0, 1, 2)]) == 1  # only one of the dup cluster survives


def test_dedup_keeps_highest_weight() -> None:
    mh = MinHasher(num_perm=128, seed=8)
    docs = [_doc(LOREM), _doc(LOREM), _doc(LOREM)]
    sigs = np.vstack([mh.signature(shingles(d)) for d in docs])
    keep = dedup(sigs, weights=np.array([1.0, 9.0, 1.0]), bands=16, threshold=0.9)
    assert list(keep) == [1]
