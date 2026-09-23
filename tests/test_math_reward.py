"""Tests for the GSM8K rule-based reward.

Run: uv run pytest -q tests/test_math_reward.py
"""

from __future__ import annotations

from forge.data.math_reward import extract_answer, max_reward, reward


def test_extract_answer_basic() -> None:
    assert extract_answer("some reasoning\n#### 42") == "42"


def test_extract_answer_strips_commas() -> None:
    assert extract_answer("#### 1,234") == "1234"


def test_extract_answer_negative_and_decimal() -> None:
    assert extract_answer("#### -3.5") == "-3.5"


def test_extract_answer_uses_last_occurrence() -> None:
    # a rambling completion might mention #### mid-reasoning by mistake; the
    # LAST one is the model's actual final answer.
    assert extract_answer("#### 1\nwait let me redo this\n#### 2") == "2"


def test_extract_answer_missing_returns_none() -> None:
    assert extract_answer("I don't know the answer") is None


def test_reward_no_format_is_zero() -> None:
    assert reward("just rambling, no marker", "#### 5") == 0.0


def test_reward_format_only() -> None:
    r = reward("#### 7", "#### 5")
    assert r == 1.0  # format credit, wrong number


def test_reward_format_and_correct() -> None:
    r = reward("blah blah\n#### 5", "#### 5")
    assert r == max_reward() == 4.0


def test_reward_tolerates_float_noise() -> None:
    assert reward("#### 5.0000001", "#### 5") == max_reward()


def test_reward_gold_as_plain_number() -> None:
    # gold_answer without a #### marker - falls back to parsing it directly
    assert reward("#### 12", "12") == max_reward()
