"""Train KenLM 3/4/5-grams (PRD M0) and write the bucketed tables (P1 checkbox 1).

    uv run python scripts/train_ngram.py --version smoke
    uv run python scripts/train_ngram.py --version v0.1 --orders 3 4 5

Writes artifacts/ngram<order>_<version>.bin and reports/results/e0_ngram<order>_<version>_real_<split>.csv.
"""

import argparse
import sys
import time
from pathlib import Path

from aagebatao.data.dataset import eval_sentences, load_split
from aagebatao.eval.evaluate import evaluate_model
from aagebatao.eval.report import write_report
from aagebatao.models.ngram import NgramModel, train_ngram, write_train_text

sys.path.insert(0, str(Path(__file__).parent))
from train import load_or_build_vocab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="smoke")
    p.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--splits", nargs="+", default=["test_a", "test_b"])
    p.add_argument("--lmplz", default="lmplz")
    p.add_argument("--build-binary", default="build_binary")
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args()

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    train_df = load_split(args.version, "train")
    vocab = load_or_build_vocab(args.version, train_df)
    text = write_train_text(train_df, vocab, artifacts / f"ngram_{args.version}_train.txt")
    print(f"wrote {len(train_df)} training lines -> {text}")

    test_sets = {s: eval_sentences(load_split(args.version, s), vocab) for s in args.splits}
    headline: dict[tuple[int, str], float] = {}
    for order in args.orders:
        t0 = time.time()
        binary = train_ngram(
            order,
            text,
            artifacts / f"ngram{order}_{args.version}.bin",
            args.lmplz,
            args.build_binary,
        )
        print(f"trained {order}-gram in {time.time() - t0:.0f}s -> {binary}")
        model = NgramModel(binary, vocab)
        for split, sentences in test_sets.items():
            t0 = time.time()
            results = evaluate_model(model, sentences, vocab.unk_id)
            write_report(results, f"e0_ngram{order}_{args.version}_real", split)
            headline[(order, split)] = results["overall"]["perplexity"]
            print(f"  scored {split} in {time.time() - t0:.0f}s")

    print("\n==== n-gram perplexity (overall) ====")
    for (order, split), ppl in sorted(headline.items()):
        print(f"  {order}-gram  {split:7s}  {ppl:10.2f}")
    top = max(args.orders)
    if (top, "test_a") in headline:
        bar = headline[(top, "test_a")]
        print(
            f"\nLSTM `m` must reach test_a ppl <= {0.85 * bar:.2f} to beat the {top}-gram by 15 %."
        )


if __name__ == "__main__":
    main()
