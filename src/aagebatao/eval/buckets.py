"""Assign every prediction target to an evaluation bucket (PRD section 9).

Buckets, from the LID tag of the TARGET token and its PREDECESSOR:
  hi            target hi, previous hi
  en            target en, previous en
  switch_hi_en  previous hi, target en
  switch_en_hi  previous en, target hi
  other         target is `other` (number, emoji, url, punctuation, named entity)

TODO 9 -> 10 live here. Tests: tests/test_buckets.py (hand-labelled examples = P1 checkbox 4)
"""

BUCKETS = ("hi", "en", "switch_hi_en", "switch_en_hi", "other")
ALL_COLUMNS = ("overall",) + BUCKETS


def assign(lid: list[str]) -> list[str]:
    valid = {"hi", "en", "other"}
    out: list[str] = []
    prev: str | None = None  # None = <bos>
    for t, cur in enumerate(lid):
        if cur not in valid:
            raise ValueError(
                f"unknown LID label {cur!r} at position {t}; expected one of {sorted(valid)}"
            )
        if cur == "other":
            out.append("other")
        elif prev is None or prev == "other" or prev == cur:
            out.append(cur)  # start of sentence, after punctuation, or same language
        elif prev == "hi":  # and cur == "en"
            out.append("switch_hi_en")
        else:  # prev == "en", cur == "hi"
            out.append("switch_en_hi")
        prev = cur
    return out


def target_buckets(lid: list[str]) -> list[str]:
    return assign(lid) + ["other"]  # the <eos> target
