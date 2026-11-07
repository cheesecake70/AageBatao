"""M4 demonstration: the NumPy LSTM's loss descends on real sentences (P1 checkbox 3).

    uv run python scripts/train_numpy.py --version smoke --sentences 2000 --steps 500   # quick
    uv run python scripts/train_numpy.py --version v0.1 --sentences 50000               # PRD run

Writes reports/results/p1_numpy_loss.png and .csv, and prints the gradient check first.
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aagebatao.data.dataset import encode_sentences, load_split
from aagebatao.data.tokenize import Vocab
from aagebatao.models.check_grad import ATOL, NumpyLSTM, compare_gradients
from aagebatao.models.lstm_numpy import train_small


def gradient_check() -> bool:
    rng = np.random.default_rng(0)
    model = NumpyLSTM(vocab_size=50, emb=8, hidden=16, seed=0)
    seq = rng.integers(0, 50, size=9)
    diffs = compare_gradients(model, seq[:-1], seq[1:])
    ok = all(d < ATOL for d in diffs.values())
    print("gradient check vs torch.nn.LSTM (tolerance 1e-6):")
    for name, d in diffs.items():
        print(f"  {name:5s} max|Δ| = {d:.2e}  {'OK' if d < ATOL else 'FAIL'}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="smoke")
    p.add_argument("--sentences", type=int, default=50_000)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--vocab-size", type=int, default=5000, help="small vocab keeps the [H,V] matmul cheap"
    )
    p.add_argument("--out", default="reports/results/p1_numpy_loss.png")
    args = p.parse_args()

    grad_ok = gradient_check()

    df = load_split(args.version, "train").head(args.sentences)
    # M4 is a verification claim, not a leaderboard entry: it gets its own small vocab
    vocab = Vocab.build(df["tokens"], max_size=args.vocab_size)
    encoded = encode_sentences(df, vocab)
    print(f"\n{len(df)} sentences, vocab {len(vocab)}, H={args.hidden}, {args.steps} steps")
    t0 = time.time()
    losses = train_small(
        encoded,
        len(vocab),
        hidden=args.hidden,
        emb=args.hidden,
        window=args.window,
        lr=args.lr,
        steps=args.steps,
        seed=args.seed,
        log_every=max(1, args.steps // 20),
    )
    print(f"trained in {time.time() - t0:.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    losses_arr = np.asarray(losses)
    k = min(50, len(losses_arr))
    running = np.convolve(losses_arr, np.ones(k) / k, mode="valid")
    uniform = np.log(len(vocab))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses_arr, alpha=0.25, lw=0.8, label="per-window loss")
    ax.plot(np.arange(k - 1, len(losses_arr)), running, lw=2, label=f"running mean ({k})")
    ax.axhline(uniform, ls="--", c="gray", label=f"uniform = ln V = {uniform:.2f}")
    ax.set_xlabel("SGD step (one window each)")
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_title(
        f"NumPy LSTM, H={args.hidden}, {len(df)} sentences, grad check {'OK' if grad_ok else 'FAIL'}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        w.writerows((i + 1, round(float(loss), 6)) for i, loss in enumerate(losses_arr))

    first, last = losses_arr[:k].mean(), losses_arr[-k:].mean()
    print(f"loss: first {k} steps {first:.3f} -> last {k} steps {last:.3f} (uniform {uniform:.3f})")
    print(
        f"{'DESCENDS' if last < first else 'DID NOT DESCEND'}; wrote {out} and {out.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
