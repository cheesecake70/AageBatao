"""Message filters and code-mixing statistics (PRD section 5, rule 2)."""

import itertools

MIN_TOKENS = 3
MAX_TOKENS = 60
MIN_CMI = 10.0


def n_switch(lid: list[str]) -> int:
    """Count hi/en switches, ignoring `other` tokens between them."""
    langs = [label for label in lid if label in ("hi", "en")]
    return sum(a != b for a, b in itertools.pairwise(langs))


def cmi(lid: list[str]) -> float:
    """Code-Mixing Index: 100 * minority language tokens / language tokens (0 = monolingual)."""
    hi, en = lid.count("hi"), lid.count("en")
    total = hi + en
    return 100.0 * min(hi, en) / total if total else 0.0


def length_ok(tokens: list[str]) -> bool:
    return MIN_TOKENS <= len(tokens) <= MAX_TOKENS


def is_code_mixed(lid: list[str]) -> bool:
    """Keep messages with both languages, or CMI at or above the threshold."""
    return ("hi" in lid and "en" in lid) or cmi(lid) >= MIN_CMI
