"""M0: Kneser-Ney smoothed n-gram via KenLM (PRD section 7).

The bar every neural model must clear. Three jobs:
  1. dump train as one sentence per line with <unk> substituted (shared vocab with the LSTM)
  2. shell out to `lmplz` and `build_binary`
  3. wrap the binary so it speaks the `log_probs(ids)` interface from eval/evaluate.py

Install: `uv add kenlm` gives the Python module. The `lmplz` / `build_binary` binaries come from
building https://github.com/kpu/kenlm (cmake) or `brew install kenlm`; put them on PATH or pass
their paths explicitly.

TODO 17 -> 19 live here. Tests: tests/test_ngram.py (KenLM tests skip if not installed)
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from aagebatao.data.tokenize import Vocab

LOG10_TO_LN = float(np.log(10.0))  # kenlm scores are log10; our interface is natural log
# lmplz refuses the literal words <s>, </s> and <unk> in its training text, so the n-gram
# side spells our <unk> as this placeholder. NgramModel maps unk_id -> KENLM_UNK when scoring,
# so the two sides stay consistent and the trained unk probability is actually used.
KENLM_UNK = "<unkw>"


def write_train_text(df: pd.DataFrame, vocab: Vocab, path: str | Path) -> Path:
    # TODO 17: write one sentence per line: tokens space-joined, every token NOT in vocab.stoi
    #   replaced by KENLM_UNK (lmplz rejects a literal "<unk>"). Do NOT add <bos>/<eos>: lmplz adds <s> </s>
    #   itself. Return Path(path). (Why substitute? So the n-gram and the LSTM see the same
    #   vocabulary and their perplexities are comparable.)
    with Path(path).open("w", encoding="utf-8") as f:
        for line in df["tokens"]:
            list = []

            for word in line:
                if word in vocab.stoi:
                    list.append(word)
                else:
                    list.append(KENLM_UNK)

            f.write(" ".join(list) + "\n")
    return Path(path)


def train_ngram(
    order: int,
    text_path: str | Path,
    out_path: str | Path,
    lmplz: str = "lmplz",
    build_binary: str = "build_binary",
) -> Path:
    for tool in (lmplz, build_binary):
        if shutil.which(tool) is None:
            raise FileNotFoundError(
                f"{tool!r} not found on PATH; build KenLM (cmake) and copy bin/lmplz and "
                "bin/build_binary somewhere on PATH, or pass their paths explicitly"
            )
    text_path, out_path = Path(text_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arpa_path = Path(str(out_path) + ".arpa")
    # lmplz reads the corpus on stdin and writes the ARPA text model on stdout
    with text_path.open("rb") as fin, arpa_path.open("wb") as fout:
        subprocess.run(
            [lmplz, "-o", str(order), "--discount_fallback"],
            stdin=fin,
            stdout=fout,
            check=True,
        )
    # build_binary compacts the ARPA text into KenLM's fast binary format
    subprocess.run([build_binary, str(arpa_path), str(out_path)], check=True)
    return out_path


class NgramModel:
    """KenLM binary + our vocab, exposing log_probs(ids) -> [T, V]."""

    def __init__(self, binary_path: str | Path, vocab: Vocab):
        import kenlm  # local import so the rest of the package works without kenlm

        self.lm = kenlm.Model(str(binary_path))
        self.vocab = vocab
        # KenLM uses "<s>", "</s>", "<unk>" where we use <bos>, <eos>, <unk>:
        self._kenlm_word = {
            vocab.bos_id: "<s>",
            vocab.eos_id: "</s>",
            vocab.unk_id: KENLM_UNK,  # the trained placeholder, not kenlm's built-in <unk>
            vocab.pad_id: "<unk>",  # <pad> never appears in text; give it kenlm's OOV prob
        }

        self._words = [self._word(i) for i in range(len(vocab))]

    def _word(self, i: int) -> str:
        return self._kenlm_word.get(i, self.vocab.itos[i])

    def log_probs(self, ids: list[int]) -> np.ndarray:
        import kenlm

        T = len(ids) - 1
        V = len(self.vocab)
        out = np.empty((T, V), dtype=np.float64)

        state = kenlm.State()
        self.lm.BeginSentenceWrite(state)  # context so far = <s>, standing in for ids[0] (<bos>)
        scratch = kenlm.State()  # receives the context after a *candidate*; we never read it
        score = self.lm.BaseScore
        words = self._words

        for t in range(T):
            # 1. score every vocabulary word as the next token given the current context
            row = out[t]
            for v in range(V):
                row[v] = score(state, words[v], scratch)
            # 2. advance the real context by the true next token ids[t + 1]
            new_state = kenlm.State()
            score(state, words[ids[t + 1]], new_state)
            state = new_state

        out *= LOG10_TO_LN  # kenlm is log10; the eval interface is natural log
        # renormalise each row so it is a proper distribution over OUR vocabulary
        row_max = out.max(axis=1, keepdims=True)
        log_norm = row_max + np.log(np.exp(out - row_max).sum(axis=1, keepdims=True))
        out -= log_norm
        return out
