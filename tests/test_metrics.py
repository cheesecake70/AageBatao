import math

import numpy as np
import pytest

from aagebatao.eval.buckets import ALL_COLUMNS
from aagebatao.eval.metrics import METRICS, aggregate, score_sentence

UNK = 1


def _lp(rows):
    """rows of probabilities -> log probs."""
    return np.log(np.array(rows, dtype=float))


def test_score_sentence_nll_rank_unk():
    lp = _lp([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.5, 0.5, 0.0 + 1e-12]])
    nll, rank, is_unk = score_sentence(lp, [0, 1, 1], unk_id=UNK)
    assert nll == pytest.approx([-math.log(0.7), -math.log(0.1), -math.log(0.5)])
    # rank = number of entries strictly greater than the target's
    assert rank.tolist() == [0, 1, 0]  # tie in row 3 counts as top-1
    assert is_unk.tolist() == [False, True, True]


def test_score_sentence_shape_mismatch():
    with pytest.raises(ValueError):
        score_sentence(_lp([[0.5, 0.5]]), [0, 1], unk_id=UNK)


def test_aggregate_per_bucket_values():
    nll = np.array([math.log(2), math.log(4), math.log(8), math.log(16)])
    rank = np.array([0, 2, 5, 1])
    is_unk = np.array([False, True, False, False])
    buckets = ["hi", "hi", "en", "other"]
    r = aggregate(nll, rank, is_unk, buckets)

    assert set(r) == set(ALL_COLUMNS)
    assert set(r["overall"]) == set(METRICS)

    # overall: mean nll = mean(log 2, log 4, log 8, log 16) = log(2^(10/4))
    assert r["overall"]["perplexity"] == pytest.approx(2**2.5)
    assert r["overall"]["top1"] == pytest.approx(25.0)
    assert r["overall"]["top3"] == pytest.approx(75.0)
    assert r["overall"]["top5"] == pytest.approx(75.0)
    assert r["overall"]["unk_rate"] == pytest.approx(25.0)
    assert r["overall"]["share"] == pytest.approx(100.0)
    assert r["overall"]["n_tokens"] == 4

    assert r["hi"]["perplexity"] == pytest.approx(math.sqrt(8))
    assert r["hi"]["top1"] == pytest.approx(50.0)
    assert r["hi"]["top3"] == pytest.approx(100.0)
    assert r["hi"]["unk_rate"] == pytest.approx(50.0)
    assert r["hi"]["share"] == pytest.approx(50.0)
    assert r["hi"]["n_tokens"] == 2

    assert r["en"]["perplexity"] == pytest.approx(8.0)
    assert r["en"]["top5"] == pytest.approx(0.0)


def test_aggregate_empty_bucket_is_nan_with_zero_share():
    r = aggregate(np.array([1.0]), np.array([0]), np.array([False]), ["hi"])
    assert math.isnan(r["switch_hi_en"]["perplexity"])
    assert math.isnan(r["switch_hi_en"]["top1"])
    assert r["switch_hi_en"]["share"] == 0
    assert r["switch_hi_en"]["n_tokens"] == 0
