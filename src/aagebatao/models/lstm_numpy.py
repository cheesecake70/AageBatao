"""M4: a single-layer word LSTM in NumPy, forward + hand-derived BPTT (PRD section 7, 15).

Purpose: understanding and a verification claim. check_grad.py compares every gradient here
against torch.nn.LSTM at 1e-6. Everything is float64.

Parameter layout (chosen to match torch so the gradient check is a plain copy):
    emb   [V, E]        word embeddings
    W     [E + H, 4H]   ONE stacked matrix for all gates; rows 0:E act on x_t, rows E: on h_{t-1}
    b     [4H]
    Wout  [H, V], bout [V]
Gate order inside the 4H axis is torch's: i, f, g, o  (input, forget, cell-candidate, output).

Cell equations, t = 0..T-1, h_{-1} = c_{-1} = 0:
    z   = [x_t ; h_{t-1}] @ W + b           # [4H]
    i   = sigmoid(z[0:H])      f = sigmoid(z[H:2H])
    g   = tanh(z[2H:3H])       o = sigmoid(z[3H:4H])
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)
    logits_t = h_t @ Wout + bout

TODO 20 -> 25 live here. Tests: tests/test_lstm_numpy.py
"""

import numpy as np

PARAM_NAMES = ("emb", "W", "b", "Wout", "bout")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class NumpyLSTM:
    def __init__(self, vocab_size: int, emb: int, hidden: int, seed: int = 0):
        self.V, self.E, self.H = vocab_size, emb, hidden
        rng = np.random.default_rng(seed)

        k = 1.0 / np.sqrt(hidden)  # torch's default init scale for nn.LSTM
        self.params: dict[str, np.ndarray] = {
            # word id -> E-vector; the model's input layer
            "emb": rng.uniform(-k, k, size=(vocab_size, emb)),
            # ONE stacked matrix for all four gates: rows 0:E act on x_t, rows E: on h_{t-1};
            # columns are [i | f | g | o], H each (torch's order, so check_grad is a plain copy)
            "W": rng.uniform(-k, k, size=(emb + hidden, 4 * hidden)),
            "b": np.zeros(4 * hidden),
            # hidden state -> one logit per vocabulary word
            "Wout": rng.uniform(-k, k, size=(hidden, vocab_size)),
            "bout": np.zeros(vocab_size),
        }
        # forget-gate bias starts at 1 so the cell keeps its memory early in training
        self.params["b"][hidden : 2 * hidden] = 1.0

    def forward(self, ids: np.ndarray) -> tuple[np.ndarray, dict]:

        ids = np.asarray(ids, dtype=np.int64)  # accept a list or array; ids must index `emb`
        T, E, H = len(ids), self.E, self.H  # T timesteps, E embedding size, H hidden size
        emb, W, b, Wout, bout = (self.params[k] for k in PARAM_NAMES)  # local names for speed

        # One row per timestep. Backward reads row t of each; pre-allocating keeps them arrays.
        xs = np.empty((T, E))  # x_t: the embedding vector fed in at step t
        h_prev = np.empty((T, H))  # h_{t-1}: hidden state BEFORE step t (row 0 is zeros)
        c_prev = np.empty((T, H))  # c_{t-1}: cell state BEFORE step t (row 0 is zeros)
        i_g = np.empty((T, H))  # input gate after sigmoid: how much new candidate to write
        f_g = np.empty((T, H))  # forget gate after sigmoid: how much old memory to keep
        g_g = np.empty((T, H))  # candidate memory after tanh: WHAT to write
        o_g = np.empty((T, H))  # output gate after sigmoid: how much memory to expose
        cs = np.empty((T, H))  # c_t: cell state AFTER step t
        tanh_c = np.empty((T, H))  # tanh(c_t): saved because backward needs it twice
        hs = np.empty((T, H))  # h_t: hidden state AFTER step t; feeds the output layer

        h = np.zeros(H)  # h_{-1}: the LSTM starts every window with no memory
        c = np.zeros(H)  # c_{-1}
        for t in range(T):
            x = emb[ids[t]]  # embedding lookup: integer word id -> E-vector (row of emb)
            h_prev[t], c_prev[t] = h, c  # remember the state we are about to overwrite

            z = np.concatenate([x, h]) @ W + b  # [E+H] @ [E+H, 4H] -> all four gate pre-activations
            i, f, g, o = np.split(z, 4)  # slice the 4H vector into 4 chunks of H, torch order
            i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)  # squash the three gates into (0, 1)
            g = np.tanh(g)  # squash the candidate into (-1, 1)

            c = f * c + i * g  # new memory = keep fraction f of old + write fraction i of candidate
            tc = np.tanh(c)  # bounded view of the memory
            h = o * tc  # output gate chooses how much of that view leaves the cell

            xs[t], i_g[t], f_g[t], g_g[t], o_g[t] = x, i, f, g, o  # record this step's values
            cs[t], tanh_c[t], hs[t] = c, tc, h  # ...and the states it produced

        logits = hs @ Wout + bout  # [T, H] @ [H, V] + [V] -> one score per vocab word per step
        cache = {  # everything backward needs; keys match the equations in the docstring
            "ids": ids,  # which emb rows were used, so demb can be scattered back (np.add.at)
            "x": xs,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "i": i_g,
            "f": f_g,
            "g": g_g,
            "o": o_g,
            "c": cs,
            "tanh_c": tanh_c,
            "h": hs,
            "logits": logits,  # saved so backward can take softmax without recomputing
        }
        return logits, cache

    def loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Mean softmax cross-entropy over the T positions, computed as log-softmax directly.

        Never log(softmax(x)): softmax can underflow to exactly 0 and log(0) = -inf. The
        log-sum-exp form below is exact and stable, which the torch comparison relies on.
        """
        targets = np.asarray(targets, dtype=np.int64)
        T = len(targets)
        m = logits.max(axis=1, keepdims=True)  # row max, subtracted for stability
        log_probs = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))  # [T, V]
        return float(-log_probs[np.arange(T), targets].mean())  # pick each row's target column

    def backward(self, cache: dict, targets: np.ndarray) -> dict[str, np.ndarray]:
        """BPTT: the hand-derived gradient of `loss` w.r.t. every parameter.

        Written in the reverse order of forward. Each `d<name>` is dL/d<name>.
        """
        targets = np.asarray(targets, dtype=np.int64)
        T, E, H = len(targets), self.E, self.H
        W, Wout = self.params["W"], self.params["Wout"]
        ids, xs, h_prev, c_prev = cache["ids"], cache["x"], cache["h_prev"], cache["c_prev"]
        i_g, f_g, g_g, o_g = cache["i"], cache["f"], cache["g"], cache["o"]
        tanh_c, hs, logits = cache["tanh_c"], cache["h"], cache["logits"]

        # ---- output layer: loss -> logits -> (Wout, bout, h) ------------------------------
        dlogits = softmax(logits)  # [T, V]
        dlogits[np.arange(T), targets] -= 1.0  # d(cross-entropy)/d(logits) = softmax - onehot
        dlogits /= T  # the loss is a MEAN over T positions
        dWout = hs.T @ dlogits  # [H, T] @ [T, V]
        dbout = dlogits.sum(axis=0)  # [V]
        dh_from_out = dlogits @ Wout.T  # [T, H]: how each h_t moved the loss directly

        # ---- recurrent part: walk time backwards, carrying dh and dc to the previous step ----
        dW = np.zeros_like(W)
        db = np.zeros(4 * H)
        demb = np.zeros_like(self.params["emb"])
        dh_next = np.zeros(H)  # gradient arriving at h_t from step t+1 (none after the last step)
        dc_next = np.zeros(H)  # gradient arriving at c_t from step t+1
        for t in reversed(range(T)):
            i, f, g, o, tc = i_g[t], f_g[t], g_g[t], o_g[t], tanh_c[t]

            dh = dh_from_out[t] + dh_next  # total gradient on h_t: direct + from the future
            # h_t = o * tanh(c_t)
            do = dh * tc
            dc = dh * o * (1.0 - tc**2) + dc_next  # through tanh, plus what the future sent to c_t
            # c_t = f * c_{t-1} + i * g
            df = dc * c_prev[t]
            di = dc * g
            dg = dc * i
            # back through the activations to the pre-activations z
            dz_i = di * i * (1.0 - i)  # sigmoid'
            dz_f = df * f * (1.0 - f)
            dz_g = dg * (1.0 - g**2)  # tanh'
            dz_o = do * o * (1.0 - o)
            dz = np.concatenate([dz_i, dz_f, dz_g, dz_o])  # [4H], same order as W's columns
            # z = [x_t ; h_{t-1}] @ W + b
            xh = np.concatenate([xs[t], h_prev[t]])  # [E+H]
            dW += np.outer(xh, dz)  # [E+H, 4H]
            db += dz
            dxh = dz @ W.T  # [E+H]: gradient on the concatenated input
            np.add.at(
                demb, ids[t], dxh[:E]
            )  # scatter into the looked-up row; accumulates on repeats
            dh_next = dxh[E:]  # flows to h_{t-1}
            dc_next = dc * f  # c_{t-1} reaches c_t only through the forget gate: the additive path

        return {"emb": demb, "W": dW, "b": db, "Wout": dWout, "bout": dbout}

    def sgd_step(self, grads: dict[str, np.ndarray], lr: float, clip: float = 1.0) -> None:
        # global norm over ALL parameters, as torch's clip_grad_norm_ does
        norm = float(np.sqrt(sum(float((g**2).sum()) for g in grads.values())))
        scale = clip / norm if norm > clip else 1.0  # shrink every gradient by the same factor
        for name in PARAM_NAMES:
            self.params[name] -= lr * scale * grads[name]


def train_small(
    encoded: list[list[int]],
    vocab_size: int,
    hidden: int = 64,
    emb: int = 64,
    window: int = 32,
    lr: float = 0.5,
    steps: int = 2000,
    seed: int = 42,
    log_every: int = 100,
) -> list[float]:
    rng = np.random.default_rng(seed)
    # same windowing as data/dataset.WindowDataset: a flat stream cut into fixed windows,
    # x = window[:-1], y = window[1:]
    stream = np.fromiter((tok for sent in encoded for tok in sent), dtype=np.int64)
    n_windows = (len(stream) - 1) // window
    if n_windows == 0:
        raise ValueError("not enough tokens for a single window")
    starts = np.arange(n_windows) * window
    rng.shuffle(starts)

    model = NumpyLSTM(vocab_size, emb, hidden, seed=seed)
    losses: list[float] = []
    for step in range(steps):
        s0 = starts[step % n_windows]  # cycle through the shuffled windows
        x = stream[s0 : s0 + window]
        y = stream[s0 + 1 : s0 + window + 1]
        logits, cache = model.forward(x)
        loss = model.loss(logits, y)
        grads = model.backward(cache, y)
        model.sgd_step(grads, lr)
        losses.append(loss)
        if log_every and (step + 1) % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            print(f"step {step + 1:6d}  loss {recent:.4f}  (uniform = {np.log(vocab_size):.4f})")
    return losses
