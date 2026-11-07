"""Hand-labelled bucket examples: this file IS the 'validated against hand-labelled
examples' part of P1 checkbox 4. Add more real sentences from test_a as you look at them."""

import pytest

from aagebatao.eval.buckets import ALL_COLUMNS, BUCKETS, assign, target_buckets

# "Main kal office ja raha hoon ."  hand-tagged
CASE_1 = (
    ["hi", "hi", "en", "hi", "hi", "hi", "other"],
    ["hi", "hi", "switch_hi_en", "switch_en_hi", "hi", "hi", "other"],
)
# "Hello . Kaise ho ?" -> first token after <bos> is own language; after punctuation own language
CASE_2 = (
    ["en", "other", "hi", "en", "other"],
    ["en", "other", "hi", "switch_hi_en", "other"],
)
# "movie bahut good thi"
CASE_3 = (
    ["en", "hi", "en", "hi"],
    ["en", "switch_en_hi", "switch_hi_en", "switch_en_hi"],
)
# pure english
CASE_4 = (["en", "en", "en"], ["en", "en", "en"])
# starts with number: "<num> log aaye"
CASE_5 = (["other", "hi", "hi"], ["other", "hi", "hi"])


@pytest.mark.parametrize("lid,expected", [CASE_1, CASE_2, CASE_3, CASE_4, CASE_5])
def test_assign_hand_labelled(lid, expected):
    assert assign(lid) == expected


def test_assign_rejects_unknown_label():
    with pytest.raises(ValueError):
        assign(["hindi", "en"])


def test_target_buckets_appends_eos_as_other():
    lid = ["hi", "en"]
    assert target_buckets(lid) == ["hi", "switch_hi_en", "other"]
    assert len(target_buckets(lid)) == len(lid) + 1


def test_bucket_names_match_prd_section_14():
    assert BUCKETS == ("hi", "en", "switch_hi_en", "switch_en_hi", "other")
    assert ALL_COLUMNS[0] == "overall"
    assert all(b in ALL_COLUMNS for b in set(CASE_1[1]) | set(CASE_3[1]))
