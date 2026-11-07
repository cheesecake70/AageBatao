import json

import pytest

from aagebatao.data.tokenize import BOS, EOS, PAD, SPECIALS, UNK, Vocab, tokenize


def test_punctuation_and_specials():
    assert tokenize("kya hua, <user>? <num> don't") == [
        "kya",
        "hua",
        ",",
        "<user>",
        "?",
        "<num>",
        "don't",
    ]


def test_emoji_is_single_token():
    assert tokenize("mast 😂😂") == ["mast", "😂", "😂"]


# ---- Vocab (TODO 1-3) -------------------------------------------------------------------

SENTENCES = [
    ["kya", "hai", "yeh"],
    ["kya", "kar", "raha", "hai"],
    ["kya", "yaar"],
]


def test_specials_have_fixed_ids():
    v = Vocab.build(SENTENCES)
    assert v.itos[:4] == list(SPECIALS) == [PAD, UNK, BOS, EOS]
    assert (v.pad_id, v.unk_id, v.bos_id, v.eos_id) == (0, 1, 2, 3)


def test_build_orders_by_frequency_then_alphabet():
    v = Vocab.build(SENTENCES)
    # kya x3, hai x2, then the singletons alphabetically
    assert v.itos[4:] == ["kya", "hai", "kar", "raha", "yaar", "yeh"]
    assert len(v) == 10


def test_build_caps_size_including_specials():
    v = Vocab.build(SENTENCES, max_size=6)
    assert len(v) == 6
    assert v.itos[4:] == ["kya", "hai"]


def test_encode_maps_unknown_to_unk_and_round_trips():
    v = Vocab.build(SENTENCES, max_size=6)
    ids = v.encode(["kya", "bolta", "hai"])
    assert ids == [v.stoi["kya"], v.unk_id, v.stoi["hai"]]
    assert v.decode(ids) == ["kya", UNK, "hai"]


def test_save_load_round_trip(tmp_path):
    v = Vocab.build(SENTENCES)
    path = tmp_path / "vocab.json"
    v.save(path)
    assert json.loads(path.read_text(encoding="utf-8")) == v.itos
    assert Vocab.load(path).itos == v.itos


def test_constructor_rejects_missing_specials():
    with pytest.raises(ValueError):
        Vocab(["a", "b", "c", "d"])


def test_build_is_deterministic():
    assert Vocab.build(SENTENCES).itos == Vocab.build(list(reversed(SENTENCES))).itos
