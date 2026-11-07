import numpy as np
import pandas as pd
import pytest
import torch

from aagebatao.data.dataset import WindowDataset, make_loader
from aagebatao.models.lstm import WordLSTM
from aagebatao.train.trainer import Trainer, make_scheduler, seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(13)
    a = (torch.rand(3), np.random.rand(3))
    seed_everything(13)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_scheduler_warmup_then_cosine():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = make_scheduler(opt, warmup_steps=10, total_steps=110)
    lrs = []
    for _ in range(110):
        lrs.append(sch.get_last_lr()[0])
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[5] == pytest.approx(0.5)
    assert lrs[10] == pytest.approx(1.0)
    assert lrs[60] == pytest.approx(0.5, abs=0.02)  # halfway through cosine
    assert lrs[-1] < 0.01
    assert max(lrs) <= 1.0 + 1e-9


def test_scheduler_zero_warmup():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = make_scheduler(opt, warmup_steps=0, total_steps=10)
    assert sch.get_last_lr()[0] == pytest.approx(1.0)


def _tiny_setup(tmp_path):
    seed_everything(42)
    # a highly predictable stream: repeating 0..9
    stream = [list(range(10)) * 40]
    ds = WindowDataset(stream, window=8)
    train = make_loader(ds, batch_size=4, shuffle=True)
    val = make_loader(ds, batch_size=4, shuffle=False)
    model = WordLSTM(10, 16, 16, 1, 0.0)
    cfg = {"lr": 1e-2, "warmup_steps": 5, "max_epochs": 4, "patience": 3}
    return Trainer(
        model, train, val, "e0_lstm_w_xs_test_real_s42", cfg, artifacts_dir=tmp_path, device="cpu"
    )


def test_fit_learns_saves_and_logs(tmp_path):
    tr = _tiny_setup(tmp_path)
    ppl0 = tr.evaluate_val()
    assert ppl0 == pytest.approx(10.0, rel=0.5)  # ~uniform over 10 symbols at init
    hist = tr.fit()
    assert 1 <= len(hist) <= 4
    assert set(hist[0]) >= {"epoch", "step", "train_loss", "val_ppl", "lr"}
    assert hist[-1]["val_ppl"] < ppl0
    ckpt = tmp_path / "e0_lstm_w_xs_test_real_s42_best.pt"
    assert ckpt.exists()
    state = torch.load(ckpt)
    assert "model_state" in state
    log = pd.read_csv(tmp_path / "e0_lstm_w_xs_test_real_s42_log.csv")
    assert len(log) == len(hist)


def test_early_stopping_triggers(tmp_path):
    tr = _tiny_setup(tmp_path)
    tr.cfg["max_epochs"] = 50
    tr.cfg["patience"] = 1
    # make val never improve: monkeypatch evaluate_val to return a constant
    tr.evaluate_val = lambda: 7.0  # type: ignore[method-assign]
    hist = tr.fit()
    assert len(hist) == 2  # epoch 1 sets best, epoch 2 has no improvement -> stop


def test_detach_cuts_graph():
    h = torch.zeros(1, requires_grad=True) * 2
    (h2,) = Trainer._detach((h,))
    assert not h2.requires_grad
    assert Trainer._detach(None) is None
