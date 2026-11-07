"""Per-bucket perplexity, top-k accuracy and <unk> rate (PRD section 9).

Model-agnostic: everything here works from a [T, V] matrix of log-probabilities, so the
n-gram, the NumPy LSTM and the torch LSTM are all scored by the same code.

TODO 11 -> 12 live here. Tests: tests/test_metrics.py
"""

import numpy as np

from aagebatao.eval.buckets import ALL_COLUMNS

METRICS = ("perplexity", "top1", "top3", "top5", "unk_rate", "share", "n_tokens")
# perplexity: exp(mean nll); top1/top3/top5/unk_rate/share: PERCENT (0-100); n_tokens: int


def score_sentence(
    log_probs: np.ndarray, targets: list[int], unk_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_probs = np.asarray(log_probs, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int64)
    if log_probs.ndim != 2 or log_probs.shape[0] != targets_arr.shape[0]:
        raise ValueError(
            f"log_probs {log_probs.shape} does not match {targets_arr.shape[0]} targets"
        )
    rows = np.arange(len(targets_arr))
    target_lp = log_probs[rows, targets_arr]
    nll = -target_lp
    rank = (log_probs > target_lp[:, None]).sum(axis=1)
    is_unk = targets_arr == unk_id
    return nll, rank, is_unk


def aggregate(
    nll: np.ndarray, rank: np.ndarray, is_unk: np.ndarray, buckets: list[str]
) -> dict[str, dict[str, float]]:
    nll = np.asarray(nll, dtype=np.float64)
    rank = np.asarray(rank)
    is_unk = np.asarray(is_unk, dtype=bool)
    buckets_arr = np.asarray(buckets)
    if not (len(nll) == len(rank) == len(is_unk) == len(buckets_arr)):
        raise ValueError("nll, rank, is_unk and buckets must have the same length")
    total = len(nll)
    results: dict[str, dict[str, float]] = {}
    for col in ALL_COLUMNS:
        mask = np.ones(total, dtype=bool) if col == "overall" else buckets_arr == col
        n = int(mask.sum())
        if n == 0:
            results[col] = {
                "perplexity": float("nan"),
                "top1": float("nan"),
                "top3": float("nan"),
                "top5": float("nan"),
                "unk_rate": float("nan"),
                "share": 0.0,
                "n_tokens": 0,
            }
            continue
        r = rank[mask]
        results[col] = {
            "perplexity": float(np.exp(nll[mask].mean())),
            "top1": float(100 * (r < 1).mean()),
            "top3": float(100 * (r < 3).mean()),
            "top5": float(100 * (r < 5).mean()),
            "unk_rate": float(100 * is_unk[mask].mean()),
            "share": float(100 * n / total),
            "n_tokens": n,
        }
    return results
