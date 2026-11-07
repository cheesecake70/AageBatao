import numpy as np
import pytest
import torch

from aagebatao.models.lstm import WordLSTM, from_config

V = 50


def test_forward_shapes_and_state():
    m = WordLSTM(V, emb=16, hidden=16, layers=2, dropout=0.1)
    x = torch.randint(0, V, (3, 7))
    logits, (h, c) = m(x)
    assert logits.shape == (3, 7, V)
    assert h.shape == (2, 3, 16) and c.shape == (2, 3, 16)
    logits2, _ = m(x, (h, c))
    assert logits2.shape == (3, 7, V)


def test_tied_weights_share_storage():
    m = WordLSTM(V, 16, 16, 1, 0.0, tie_weights=True)
    assert m.out.weight is m.embed.weight
    with pytest.raises(ValueError):
        WordLSTM(V, 16, 32, 1, 0.0, tie_weights=True)
    WordLSTM(V, 16, 32, 1, 0.0, tie_weights=False)  # fine


def test_log_probs_interface():
    m = WordLSTM(V, 16, 16, 1, 0.0)
    ids = [2, 5, 9, 4, 3]
    lp = m.log_probs(ids)
    assert isinstance(lp, np.ndarray) and lp.dtype == np.float64
    assert lp.shape == (len(ids) - 1, V)
    assert np.exp(lp).sum(axis=1) == pytest.approx(np.ones(len(ids) - 1), abs=1e-5)


def test_count_params():
    m = WordLSTM(V, 16, 16, 1, 0.0, tie_weights=True)
    total, non_emb = m.count_params()
    assert total == sum(p.numel() for p in m.parameters())
    assert non_emb == total - V * 16
    m2 = WordLSTM(V, 16, 16, 1, 0.0, tie_weights=False)
    total2, non_emb2 = m2.count_params()
    assert total2 == total + V * 16
    assert non_emb2 == non_emb


def test_from_config_matches_yaml_keys():
    import yaml

    with open("configs/model/lstm_xs.yaml") as f:
        cfg = yaml.safe_load(f)
    m = from_config(cfg, V)
    assert m.lstm.hidden_size == 64 and m.lstm.num_layers == 1
