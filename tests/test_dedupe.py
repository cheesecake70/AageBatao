from aagebatao.data.dedupe import HeldOutIndex, jaccard, shingles


def test_jaccard_identical():
    s = shingles(["kya", "hua", "yaar"])
    assert jaccard(s, s) == 1.0


def test_index_finds_near_duplicate_case_insensitive():
    base = [
        "aaj",
        "movie",
        "dekhne",
        "chalte",
        "hain",
        "kya",
        "bhai",
        "bahut",
        "mast",
        "hai",
        "yaar",
        "sach",
        "mein",
    ]
    index = HeldOutIndex([base])
    assert index.is_near_duplicate([t.upper() for t in base])
    assert not index.is_near_duplicate(["mujhe", "kal", "office", "jana", "hai", "subah", "jaldi"])
