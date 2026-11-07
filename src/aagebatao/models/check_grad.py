"""Compare NumpyLSTM gradients against torch.nn.LSTM on the same tiny batch (PRD section 7, M4).

Weight mapping (torch gate order is i, f, g, o, the same as ours):
    torch lstm.weight_ih_l0  [4H, E]  = W[:E].T
    torch lstm.weight_hh_l0  [4H, H]  = W[E:].T
    torch lstm.bias_ih_l0    [4H]     = b          (set bias_hh_l0 to zeros; torch adds both)
    torch embed.weight       [V, E]   = emb
    torch out.weight         [V, H]   = Wout.T ;  out.bias = bout

TODO 26 -> 27 live here. Tests: tests/test_lstm_numpy.py::test_gradients_match_torch
"""

import numpy as np
import torch
from torch import nn

from aagebatao.models.lstm_numpy import PARAM_NAMES, NumpyLSTM

ATOL = 1e-6


def torch_reference(model: NumpyLSTM) -> tuple[nn.Embedding, nn.LSTM, nn.Linear]:
    V, E, H = model.V, model.E, model.H
    p = model.params
    embed = nn.Embedding(V, E).double()
    lstm = nn.LSTM(E, H, num_layers=1, batch_first=True).double()
    out = nn.Linear(H, V).double()
    with torch.no_grad():
        embed.weight.copy_(torch.from_numpy(p["emb"]))
        lstm.weight_ih_l0.copy_(torch.from_numpy(p["W"][:E].T))  # [4H, E]
        lstm.weight_hh_l0.copy_(torch.from_numpy(p["W"][E:].T))  # [4H, H]
        lstm.bias_ih_l0.copy_(torch.from_numpy(p["b"]))
        lstm.bias_hh_l0.zero_()  # torch adds both biases; ours lives entirely in bias_ih
        out.weight.copy_(torch.from_numpy(p["Wout"].T))  # [V, H]
        out.bias.copy_(torch.from_numpy(p["bout"]))
    return embed, lstm, out


def torch_grads(
    modules: tuple[nn.Embedding, nn.LSTM, nn.Linear], ids: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray]:
    """Run the torch reference on the same window and return grads in OUR parameter layout."""
    embed, lstm, out = modules
    for m in modules:
        m.zero_grad()
    x = torch.as_tensor(ids, dtype=torch.long)[None]  # [1, T]
    y = torch.as_tensor(targets, dtype=torch.long)
    h, _ = lstm(embed(x))
    logits = out(h)[0]  # [T, V]
    loss = nn.functional.cross_entropy(logits, y)  # mean over T, same as NumpyLSTM.loss
    loss.backward()
    g = lambda t: t.grad.detach().numpy()
    return {
        "emb": g(embed.weight),
        "W": np.concatenate([g(lstm.weight_ih_l0).T, g(lstm.weight_hh_l0).T], axis=0),
        # torch gives bias_ih and bias_hh the SAME gradient (z = ... + b_ih + b_hh), so take one
        "b": g(lstm.bias_ih_l0),
        "Wout": g(out.weight).T,
        "bout": g(out.bias),
    }


def compare_gradients(model: NumpyLSTM, ids: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    ids = np.asarray(ids)
    targets = np.asarray(targets)
    # ours
    logits, cache = model.forward(ids)
    np_loss = model.loss(logits, targets)
    np_grads = model.backward(cache, targets)
    # torch, same weights, same window
    modules = torch_reference(model)
    embed, lstm, out = modules
    with torch.no_grad():
        x = torch.as_tensor(ids, dtype=torch.long)[None]
        h, _ = lstm(embed(x))
        torch_loss = nn.functional.cross_entropy(
            out(h)[0], torch.as_tensor(targets, dtype=torch.long)
        ).item()
    th_grads = torch_grads(modules, ids, targets)
    diffs = {name: float(np.abs(np_grads[name] - th_grads[name]).max()) for name in PARAM_NAMES}
    diffs["loss"] = abs(np_loss - torch_loss)
    return diffs


def main() -> None:
    rng = np.random.default_rng(0)
    model = NumpyLSTM(vocab_size=50, emb=8, hidden=16, seed=0)
    seq = rng.integers(0, 50, size=9)
    diffs = compare_gradients(model, seq[:-1], seq[1:])
    for name, d in diffs.items():
        print(f"{name:5s} max|Δ| = {d:.2e}  {'OK' if d < ATOL else 'FAIL'}")


if __name__ == "__main__":
    main()
