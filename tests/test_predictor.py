import pytest
import torch

from aagebatao.data.tokenize import SPECIALS, Vocab
from aagebatao.models.lstm import WordLSTM
from aagebatao.serve.predictor import Prediction, Predictor

WORDS = ["kya", "kar", "raha", "hai", "ho", "hoon", "office", "<num>", ".", "Kal"]


@pytest.fixture
def predictor():
    torch.manual_seed(0)
    vocab = Vocab(list(SPECIALS) + WORDS)
    model = WordLSTM(len(vocab), 16, 16, 1, 0.0)
    return Predictor(vocab, model)


def test_split_input_detects_partial_word():
    assert Predictor.split_input("kya kar") == (["kya"], "kar")
    assert Predictor.split_input("kya kar ") == (["kya", "kar"], "")
    assert Predictor.split_input("") == ([], "")
    assert Predictor.split_input("kya kar?") == (
        ["kya", "kar", "?"],
        "",
    )  # punctuation is never partial


def test_split_input_normalises_like_training():
    # numbers and urls become the P0 placeholders before tokenising
    ctx, partial = Predictor.split_input("main 5 baje http://x.co aaya ")
    assert ctx == ["main", "<num>", "baje", "<url>", "aaya"] and partial == ""


def test_predict_returns_k_real_words(predictor):
    pred = predictor.predict("kya kar ", k=3)
    assert isinstance(pred, Prediction)
    assert len(pred.words) == 3 and len(pred.probs) == 3
    assert pred.partial == ""
    assert not any(w.startswith("<") for w in pred.words)  # no <unk>/<num>/<bos>...
    assert all(0 < p <= 1 for p in pred.probs)
    assert pred.probs == sorted(pred.probs, reverse=True)
    assert pred.latency_ms >= 0


def test_predict_filters_by_typed_prefix(predictor):
    pred = predictor.predict("kya kar raha h", k=3)
    assert pred.partial == "h"
    assert pred.words and all(w.lower().startswith("h") for w in pred.words)
    assert set(pred.words) <= {"hai", "ho", "hoon"}


def test_predict_prefix_is_case_insensitive_and_skips_exact(predictor):
    pred = predictor.predict("k", k=5)
    assert all(w.lower().startswith("k") for w in pred.words)
    pred2 = predictor.predict("kya", k=5)
    assert "kya" not in pred2.words  # fully typed already


def test_predict_empty_text_gives_first_word(predictor):
    pred = predictor.predict("", k=2)
    assert len(pred.words) == 2 and pred.partial == ""


def test_load_from_checkpoint(tmp_path):
    import yaml

    vocab = Vocab(list(SPECIALS) + WORDS)
    vocab.save(tmp_path / "vocab.json")
    with open("configs/model/lstm_xs.yaml") as f:
        cfg = yaml.safe_load(f)
    from aagebatao.models.lstm import from_config

    model = from_config(cfg, len(vocab))
    torch.save({"model_state": model.state_dict(), "cfg": {}}, tmp_path / "m_best.pt")
    p = Predictor.load(tmp_path / "m_best.pt", "xs", tmp_path / "vocab.json")
    assert len(p.predict("kya ", k=2).words) == 2


@pytest.fixture
def swear_predictor():
    torch.manual_seed(0)
    vocab = Vocab(list(SPECIALS) + ["kya", "chutiya", "Chutiya", "chal", "chutti", "cheez"])
    model = WordLSTM(len(vocab), 16, 16, 1, 0.0)
    return Predictor(vocab, model, blocklist=frozenset({"chutiya", "notinvocab"}))


def test_blocklist_words_never_suggested(swear_predictor):
    words = swear_predictor.predict("kya ", k=10).words
    assert "chutiya" not in words and "Chutiya" not in words  # any casing


def test_blocklist_applies_to_prefix_completions(swear_predictor):
    words = swear_predictor.predict("kya ch", k=10).words
    assert not any(w.lower() == "chutiya" for w in words)
    assert {"chal", "chutti", "cheez"} <= set(words)  # innocent near-misses still come through


def test_shipped_blocklist_loads():
    from aagebatao.serve.predictor import load_blocklist

    bad = load_blocklist()
    assert "chutiya" in bad and "madarchod" in bad
    assert not {"chota", "chutti", "chodo", "gandhi"} & bad  # innocent words stay allowed
