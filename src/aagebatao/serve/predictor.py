"""Next-word prediction for the demo (plan Stage A4; a thin slice of PRD section 11).

The input text goes through the SAME normalize -> tokenize functions the training data went
through (scripts/build_dataset.py), so the model never sees text in a form it was not trained
on. No hidden-state cache and no char-LSTM fallback yet: both are PRD P6 polish.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from aagebatao.data.normalize import normalize
from aagebatao.data.tokenize import SPECIALS, Vocab, tokenize
from aagebatao.models.lstm import WordLSTM, from_config

CANDIDATE_POOL = 200  # how many top words to consider before prefix-filtering a partial word


@dataclass
class Prediction:
    words: list[str]
    probs: list[float]
    partial: str  # the partially typed word the suggestions complete ("" if none)
    latency_ms: float


class Predictor:
    def __init__(self, vocab: Vocab, model: WordLSTM, device: str = "cpu"):
        self.vocab = vocab
        self.model = model.to(device).eval()
        self.device = device
        # never suggest control tokens: <pad> <unk> <bos> <eos> <num> <url> <user>
        self._blocked = torch.tensor(
            [i for i, w in enumerate(vocab.itos) if w in SPECIALS or w.startswith("<")],
            dtype=torch.long,
        )

    @classmethod
    def load(
        cls, checkpoint: str | Path, size: str, vocab_path: str | Path, device: str = "cpu"
    ) -> Predictor:
        vocab = Vocab.load(vocab_path)
        with open(f"configs/model/lstm_{size}.yaml") as f:
            model = from_config(yaml.safe_load(f), len(vocab))
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
        return cls(vocab, model, device)

    @staticmethod
    def split_input(text: str) -> tuple[list[str], str]:
        """-> (context tokens, partial word). A word is partial if the text does not end in a
        space and its last token is alphanumeric (punctuation is never 'partial')."""
        tokens = tokenize(normalize(text))
        if tokens and text and not text[-1].isspace() and tokens[-1][0].isalnum():
            return tokens[:-1], tokens[-1]
        return tokens, ""

    @torch.no_grad()
    def predict(self, text: str, k: int = 3) -> Prediction:
        t0 = time.perf_counter()
        context, partial = self.split_input(text)
        ids = [self.vocab.bos_id] + self.vocab.encode(context)
        x = torch.tensor(ids, dtype=torch.long, device=self.device)[None]  # [1, T]
        logits, _ = self.model(x)
        probs = torch.softmax(logits[0, -1].float(), dim=-1)  # distribution over the NEXT word
        probs[self._blocked] = 0.0

        pool = min(CANDIDATE_POOL if partial else k, len(self.vocab))
        top_p, top_i = torch.topk(probs, pool)
        words, ps = [], []
        low = partial.lower()
        for p_, i in zip(top_p.tolist(), top_i.tolist()):
            w = self.vocab.itos[i]
            if partial and not w.lower().startswith(low):
                continue
            if partial and w.lower() == low:
                continue  # already fully typed; not a useful completion
            words.append(w)
            ps.append(round(p_, 5))
            if len(words) == k:
                break
        return Prediction(words, ps, partial, round((time.perf_counter() - t0) * 1000, 2))
