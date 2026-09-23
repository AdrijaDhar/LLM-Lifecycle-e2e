"""Deterministic, rule-based rewards for GSM8K-style math problems.

No LLM judge, no learned reward model - just parse the completion and check it
against ground truth. This determinism is the whole point of RLVR
(reinforcement learning from VERIFIABLE rewards): the signal can't be gamed by
sounding plausible, only by actually being right.

GSM8K's own solutions end with "#### <number>" - we ask the model (via the
prompt) to do the same, and reward that exact, easy-to-parse convention.
"""

from __future__ import annotations

import re

ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

PROMPT_TEMPLATE = (
    "Solve this math problem step by step. On the final line, write your "
    "answer in the exact format: #### <number>\n\n{question}"
)


def extract_answer(text: str) -> str | None:
    """Pull the number after the LAST '####' in the text, GSM8K-style.
    Returns a normalised string (commas stripped) or None if absent."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def reward(completion: str, gold_answer: str, format_weight: float = 1.0, correct_weight: float = 3.0) -> float:
    """format_weight if the '#### N' pattern is present at all; correct_weight
    ADDITIONALLY if the number matches gold (within float tolerance)."""
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    r = format_weight

    gold = _to_float(extract_answer(gold_answer) or gold_answer)
    pred_f = _to_float(pred)
    if gold is not None and pred_f is not None and abs(pred_f - gold) < 1e-4:
        r += correct_weight
    return r


def max_reward(format_weight: float = 1.0, correct_weight: float = 3.0) -> float:
    return format_weight + correct_weight
