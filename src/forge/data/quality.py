"""Heuristic quality filters for pretraining documents.

These are the classic "Gopher / RefinedWeb / FineWeb" rules: cheap, per-document
checks that catch machine-generated spam, navigation dumps, word-salad, and
broken markup that a quality *classifier* score alone misses. Each function is
pure and independently testable.

`check_document` returns a list of the rule names a document FAILED. An empty
list means keep it. The driver (`scripts/filter_corpus.py`) tallies these names
so you can see *why* documents are dropped, not just how many.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD_RE = re.compile(r"\S+")
# "bad" symbols whose ratio to words signals junk (hash = markdown headers gone
# wrong / SEO stuffing, ellipsis = truncated listicles)
BAD_SYMBOLS = ("#", "...", "…")
BULLET_PREFIXES = ("•", "-", "*", "‣", "◦", "⁃", "∙")


@dataclass
class QualityConfig:
    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_len: float = 3.0
    max_mean_word_len: float = 10.0
    max_symbol_word_ratio: float = 0.10
    min_alpha_word_frac: float = 0.80          # fraction of words containing >=1 a-z
    max_bullet_line_frac: float = 0.90         # whole doc is a bullet list
    max_ellipsis_line_frac: float = 0.30       # lines ending in "..."
    max_dup_line_frac: float = 0.30            # identical lines / total lines
    max_dup_para_frac: float = 0.30            # identical paragraphs
    min_score: float | None = None             # FineWeb-EDU classifier score (0-5)
    min_language_score: float | None = 0.65    # FineWeb-EDU langid confidence


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def word_stats(text: str) -> tuple[int, float, float]:
    """(n_words, mean_word_len, fraction_of_words_with_a_letter)"""
    words = WORD_RE.findall(text)
    if not words:
        return 0, 0.0, 0.0
    total_len = sum(len(w) for w in words)
    n_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    return len(words), total_len / len(words), n_alpha / len(words)


def symbol_word_ratio(text: str) -> float:
    words = WORD_RE.findall(text)
    if not words:
        return 1.0
    n_sym = sum(text.count(s) for s in BAD_SYMBOLS)
    return n_sym / len(words)


def _dup_frac(items: list[str]) -> float:
    """Fraction of entries that are repeats: (total - distinct) / total."""
    if len(items) <= 1:
        return 0.0
    return (len(items) - len(set(items))) / len(items)


def line_fracs(text: str) -> tuple[float, float, float]:
    """(bullet_line_frac, ellipsis_line_frac, dup_line_frac)"""
    lines = _lines(text)
    if not lines:
        return 1.0, 1.0, 1.0
    n_bullet = sum(1 for ln in lines if ln.startswith(BULLET_PREFIXES))
    n_ellipsis = sum(1 for ln in lines if ln.endswith(("...", "…")))
    return n_bullet / len(lines), n_ellipsis / len(lines), _dup_frac(lines)


def check_document(text: str, meta: dict | None = None, cfg: QualityConfig | None = None) -> list[str]:
    cfg = cfg or QualityConfig()
    meta = meta or {}
    fails: list[str] = []

    n_words, mean_len, alpha_frac = word_stats(text)
    if n_words < cfg.min_words:
        fails.append("too_short")
    if n_words > cfg.max_words:
        fails.append("too_long")
    if n_words and not (cfg.min_mean_word_len <= mean_len <= cfg.max_mean_word_len):
        fails.append("mean_word_len")
    if n_words and alpha_frac < cfg.min_alpha_word_frac:
        fails.append("low_alpha_words")
    if symbol_word_ratio(text) > cfg.max_symbol_word_ratio:
        fails.append("symbol_ratio")

    bullet_f, ellipsis_f, dup_line_f = line_fracs(text)
    if bullet_f > cfg.max_bullet_line_frac:
        fails.append("mostly_bullets")
    if ellipsis_f > cfg.max_ellipsis_line_frac:
        fails.append("ellipsis_lines")
    if dup_line_f > cfg.max_dup_line_frac:
        fails.append("dup_lines")
    if _dup_frac(_paragraphs(text)) > cfg.max_dup_para_frac:
        fails.append("dup_paragraphs")

    if cfg.min_score is not None and meta.get("score") is not None and meta["score"] < cfg.min_score:
        fails.append("low_score")
    if (
        cfg.min_language_score is not None
        and meta.get("language_score") is not None
        and meta["language_score"] < cfg.min_language_score
    ):
        fails.append("low_language_score")

    return fails
