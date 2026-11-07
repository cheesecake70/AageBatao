import shutil

import numpy as np
import pandas as pd
import pytest

from aagebatao.data.tokenize import Vocab
from aagebatao.models.ngram import KENLM_UNK, LOG10_TO_LN, write_train_text

kenlm = pytest.importorskip("kenlm")


@pytest.fixture
def vocab():
    return Vocab(["<pad>", "<unk>", "<bos>", "<eos>", "kya", "hai", "yeh"])


def test_write_train_text_substitutes_unk(tmp_path, vocab):
    df = pd.DataFrame({"tokens": [["kya", "hai"], ["yeh", "bolta", "hai"]]})
    path = write_train_text(df, vocab, tmp_path / "train.txt")
    assert path.read_text(encoding="utf-8") == f"kya hai\nyeh {KENLM_UNK} hai\n"
    assert "<unk>" not in path.read_text(encoding="utf-8")  # lmplz would refuse it


def test_log10_constant():
    assert LOG10_TO_LN == pytest.approx(np.log(10))


@pytest.mark.skipif(shutil.which("lmplz") is None, reason="KenLM binaries not built")
def test_ngram_end_to_end(tmp_path, vocab):
    from aagebatao.models.ngram import NgramModel, train_ngram

    rng = np.random.default_rng(0)
    toks = ["kya", "hai", "yeh", "zzz"]
    df = pd.DataFrame({"tokens": [list(rng.choice(toks, size=6)) for _ in range(200)]})
    text = write_train_text(df, vocab, tmp_path / "t.txt")
    binary = train_ngram(3, text, tmp_path / "lm3.bin")
    m = NgramModel(binary, vocab)
    ids = [2, 4, 5, 1, 3]
    lp = m.log_probs(ids)
    assert lp.shape == (4, len(vocab))
    assert np.exp(lp).sum(axis=1) == pytest.approx(np.ones(4), abs=1e-6)
    assert np.isfinite(lp).all()
