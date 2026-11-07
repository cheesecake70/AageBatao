"""Latency of the demo predictor (PRD section 11 target: p50 <= 15 ms, p95 <= 30 ms on CPU).

    uv run python scripts/bench_latency.py --checkpoint artifacts/<run>_best.pt --size m --version v0.1
    uv run python scripts/bench_latency.py --url http://localhost:8000      # against a live server

In-process by default (measures the model only); --url adds HTTP overhead.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aagebatao.data.dataset import load_split
from aagebatao.serve.predictor import Predictor


def sample_prefixes(version: str, n: int, seed: int = 0) -> list[str]:
    """Random prefixes of real test_a sentences, 1..len tokens, so the bench sees real lengths."""
    rng = np.random.default_rng(seed)
    df = load_split(version, "test_a")
    texts = []
    for _ in range(n):
        tokens = list(df["tokens"].iloc[int(rng.integers(len(df)))])
        cut = int(rng.integers(1, max(2, len(tokens) + 1)))
        texts.append(" ".join(tokens[:cut]) + (" " if rng.random() < 0.5 else ""))
    return texts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--size", default="m")
    p.add_argument("--version", default="v0.1")
    p.add_argument("--url", help="bench a running server instead of in-process")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default=None, help="json file for the numbers")
    args = p.parse_args()

    prefixes = sample_prefixes(args.version, args.n)
    times = []
    if args.url:
        import httpx

        with httpx.Client(base_url=args.url, timeout=10) as client:
            client.post("/predict", json={"text": prefixes[0], "k": 3})  # warm up
            for text in prefixes:
                t0 = time.perf_counter()
                client.post("/predict", json={"text": text, "k": 3}).raise_for_status()
                times.append((time.perf_counter() - t0) * 1000)
        mode = f"http ({args.url})"
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required without --url")
        pred = Predictor.load(
            args.checkpoint, args.size, Path("data/hingmix") / args.version / "vocab.json"
        )
        pred.predict(prefixes[0])  # warm up
        for text in prefixes:
            t0 = time.perf_counter()
            pred.predict(text, 3)
            times.append((time.perf_counter() - t0) * 1000)
        mode = f"in-process ({args.size}, cpu)"

    arr = np.asarray(times)
    stats = {
        "mode": mode,
        "n": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(arr.max()), 2),
        "target_p95_ms": 30,
        "meets_target": bool(np.percentile(arr, 95) <= 30),
    }
    print(json.dumps(stats, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
