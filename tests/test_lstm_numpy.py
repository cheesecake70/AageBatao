import numpy as np
import pytest

from aagebatao.models.check_grad import ATOL, compare_gradients
from aagebatao.models.lstm_numpy import PARAM_NAMES, NumpyLSTM, sigmoid, softmax, train_small

V, E, H, T = 23, 6, 10, 7


@pytest.fixture
def model():
    return NumpyLSTM(V, E, H, seed=0)


@pytest.fixture
def window():
    rng = np.random.default_rng(1)
    seq = rng.integers(0, V, size=T + 1)
    return seq[:-1], seq[1:]


def test_param_shapes(model):
    p = model.params
    assert set(p) == set(PARAM_NAMES)
    assert p["emb"].shape == (V, E)
    assert p["W"].shape == (E + H, 4 * H)
    assert p["b"].shape == (4 * H,)
    assert p["Wout"].shape == (H, V) and p["bout"].shape == (V,)
    assert all(a.dtype == np.float64 for a in p.values())


def test_forward_shapes_and_loss(model, window):
    x, y = window
    logits, _cache = model.forward(x)
    assert logits.shape == (T, V)
    loss = model.loss(logits, y)
    assert np.isfinite(loss)
    # fresh small weights -> near-uniform output -> loss near ln V
    assert abs(loss - np.log(V)) < 1.0


def test_helpers():
    assert sigmoid(np.array([0.0])) == pytest.approx([0.5])
    assert softmax(np.array([[1.0, 1.0]])).sum() == pytest.approx(1.0)


def test_backward_shapes(model, window):
    x, y = window
    _, cache = model.forward(x)
    grads = model.backward(cache, y)
    for k in PARAM_NAMES:
        assert grads[k].shape == model.params[k].shape, k


def test_gradients_match_torch(model, window):
    """P1 checkbox 3, first half: every parameter gradient within 1e-6 of torch.nn.LSTM."""
    x, y = window
    diffs = compare_gradients(model, x, y)
    for k in (*PARAM_NAMES, "loss"):
        assert diffs[k] < ATOL, f"{k}: {diffs[k]:.2e}"


def test_gradients_match_torch_with_repeated_ids(model):
    # embedding scatter must ACCUMULATE when the same id appears twice
    x = np.array([3, 3, 7, 3, 7])
    y = np.array([3, 7, 3, 7, 1])
    diffs = compare_gradients(model, x, y)
    assert diffs["emb"] < ATOL


def test_sgd_step_clips_and_moves(model, window):
    x, y = window
    _, cache = model.forward(x)
    grads = model.backward(cache, y)
    before = {k: v.copy() for k, v in model.params.items()}
    model.sgd_step(grads, lr=0.1, clip=1e-3)
    moved = np.sqrt(sum(((model.params[k] - before[k]) ** 2).sum() for k in PARAM_NAMES))
    assert 0 < moved <= 0.1 * 1e-3 + 1e-12  # global norm clipped to 1e-3, times lr


def test_loss_descends_on_tiny_data():
    rng = np.random.default_rng(0)
    sents = [list(rng.integers(4, 12, size=20)) for _ in range(30)]
    losses = train_small(
        sents,
        vocab_size=12,
        hidden=16,
        emb=16,
        window=8,
        lr=0.5,
        steps=300,
        seed=0,
        log_every=10_000,
    )
    assert len(losses) == 300
    assert np.mean(losses[-50:]) < np.mean(losses[:50])
