import pytest

from aagebatao.train.naming import run_name

CFG = {"exp": "e2", "family": "lstm_w", "size": "m", "data": "300k", "mix": "mix50_g3", "seed": 42}


def test_run_name_matches_prd_example():
    assert run_name(CFG) == "e2_lstm_w_m_300k_mix50_g3_s42"


def test_run_name_rejects_unknown_seed():
    with pytest.raises(ValueError):
        run_name({**CFG, "seed": 1})
