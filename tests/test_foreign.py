from aagebatao.data.foreign import coverage, is_foreign

VOCAB = {"kya", "hua", "yaar", "good"}


def test_coverage_ignores_punctuation_and_specials():
    assert coverage(["kya", "hua", "?", "<user>"], VOCAB) == 1.0


def test_foreign_sentence_flagged():
    assert is_foreign(["je", "ne", "sais", "pas", "yaar"], VOCAB)
    assert not is_foreign(["kya", "hua", "yaar"], VOCAB)
