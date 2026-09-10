"""Tests for the heuristic quality filters.

Run: uv run pytest -q tests/test_quality.py
"""

from __future__ import annotations

from forge.data.quality import (
    QualityConfig,
    check_document,
    line_fracs,
    symbol_word_ratio,
    word_stats,
)

GOOD = (
    "The mitochondrion is the organelle responsible for producing most of the "
    "chemical energy that cells need. It does this through respiration, converting "
    "nutrients into adenosine triphosphate. Mitochondria have their own small "
    "genome, a fact that supports the endosymbiotic theory of their origin. "
    "Researchers continue to study how mitochondrial dysfunction contributes to "
    "aging and to a range of metabolic diseases in humans and other animals."
)


def test_good_doc_passes() -> None:
    assert check_document(GOOD) == []


def test_too_short() -> None:
    assert "too_short" in check_document("Just a few words here.")


def test_word_stats_basic() -> None:
    n, mean, alpha = word_stats("cat dog bird")
    assert n == 3
    assert 3.0 <= mean <= 4.0
    assert alpha == 1.0


def test_symbol_ratio_flags_hash_spam() -> None:
    spam = ("#buy #cheap #now #deal #sale " * 30)
    assert symbol_word_ratio(spam) > 0.10
    assert "symbol_ratio" in check_document(spam)


def test_dup_lines_flagged() -> None:
    doc = ("Click here to subscribe to our newsletter today.\n" * 40)
    _, _, dup = line_fracs(doc)
    assert dup > 0.3
    assert "dup_lines" in check_document(doc)


def test_mostly_bullets_flagged() -> None:
    doc = "\n".join(f"- item number {i} in a very long navigation list" for i in range(60))
    assert "mostly_bullets" in check_document(doc)


def test_low_alpha_words_flagged() -> None:
    doc = " ".join(str(i) for i in range(200))  # all numbers
    assert "low_alpha_words" in check_document(doc)


def test_score_threshold_uses_meta() -> None:
    cfg = QualityConfig(min_score=3.0)
    assert "low_score" in check_document(GOOD, {"score": 2.1}, cfg)
    assert "low_score" not in check_document(GOOD, {"score": 3.5}, cfg)


def test_language_score_threshold() -> None:
    cfg = QualityConfig(min_language_score=0.65)
    assert "low_language_score" in check_document(GOOD, {"language_score": 0.4}, cfg)
    assert check_document(GOOD, {"language_score": 0.95}, cfg) == []


def test_missing_meta_is_lenient() -> None:
    # No score/language_score in meta -> those rules don't fire.
    cfg = QualityConfig(min_score=3.0, min_language_score=0.65)
    assert check_document(GOOD, {}, cfg) == []
