"""Run any model over a list of sentences and produce the bucketed results dict.

The ONE interface every model implements (n-gram, NumPy LSTM, torch LSTM, uniform baseline):

    model.log_probs(ids: list[int]) -> np.ndarray of shape [len(ids) - 1, V]

row t is the model's log-probability distribution over the token at position t+1 given
ids[: t + 1]. Natural log. `ids` includes <bos> and <eos>.

TODO 15 lives here. Tests: tests/test_report.py (uniform model must give perplexity == V)
"""

from typing import Protocol

import numpy as np

from aagebatao.eval.buckets import target_buckets
from aagebatao.eval.metrics import aggregate, score_sentence


class LanguageModel(Protocol):
    def log_probs(self, ids: list[int]) -> np.ndarray: ...


class UniformModel:
    """Every word gets probability 1/V. Its perplexity must equal V exactly; this is how we
    prove the scoring code is right before any real model exists."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def log_probs(self, ids: list[int]) -> np.ndarray:
        return np.full((len(ids) - 1, self.vocab_size), -np.log(self.vocab_size))


def evaluate_model(
    model: LanguageModel, sentences: list[tuple[list[int], list[str]]], unk_id: int
) -> dict[str, dict[str, float]]:
    all_nll, all_rank, all_unk, all_buckets = [], [], [], []
    for ids, lid in sentences:
        targets = ids[1:]
        buckets = target_buckets(lid)
        if len(buckets) != len(targets):
            raise ValueError(
                f"lid has {len(lid)} tags but ids has {len(ids)} tokens (expected len(lid) == len(ids) - 2)"
            )
        lp = model.log_probs(ids)
        if lp.shape[0] != len(targets):
            raise ValueError(f"model returned {lp.shape[0]} rows for {len(targets)} targets")
        nll, rank, is_unk = score_sentence(lp, targets, unk_id)
        all_nll.append(nll)
        all_rank.append(rank)
        all_unk.append(is_unk)
        all_buckets.extend(buckets)
    if not all_nll:
        raise ValueError("no sentences to evaluate")
    return aggregate(
        np.concatenate(all_nll), np.concatenate(all_rank), np.concatenate(all_unk), all_buckets
    )
