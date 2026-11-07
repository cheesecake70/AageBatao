"""Bucketed evaluation of any model on a test split (PRD section 9).

    uv run python scripts/evaluate.py --model uniform --version smoke --split test_a
    uv run python scripts/evaluate.py --model lstm --checkpoint artifacts/<run>_best.pt \
        --size m --version v0.1 --split test_a

TODO 16 lives here.
"""

import argparse
from pathlib import Path

import torch
import yaml

from aagebatao.data.dataset import eval_sentences, load_split
from aagebatao.data.tokenize import Vocab
from aagebatao.eval.evaluate import UniformModel, evaluate_model
from aagebatao.eval.report import write_report
from aagebatao.models.lstm import from_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["uniform", "lstm", "ngram"], required=True)
    p.add_argument("--version", default="smoke")
    p.add_argument("--split", default="test_a", choices=["val", "test_a", "test_b"])
    p.add_argument("--checkpoint", default=None, help="lstm: artifacts/<run>_best.pt")
    p.add_argument("--size", default="m", help="lstm: which configs/model/lstm_<size>.yaml")
    p.add_argument("--binary", default=None, help="ngram: kenlm .bin path")
    p.add_argument("--name", default=None, help="results csv prefix; defaults from checkpoint")
    args = p.parse_args()
    vocab_path = Path("data/hingmix") / args.version / "vocab.json"
    if not vocab_path.exists():
        train_df = load_split(args.version, "train")
        vocab = Vocab.build(train_df["tokens"])
        vocab.save(vocab_path)
        print(f"built vocab ({len(vocab)} types) -> {vocab_path}")
    else:
        vocab = Vocab.load(vocab_path)

    if args.model == "uniform":
        model = UniformModel(len(vocab))
        default_name = f"e0_uniform_{args.version}"
    elif args.model == "lstm":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --model lstm")
        with open(f"configs/model/lstm_{args.size}.yaml") as f:
            cfg = yaml.safe_load(f)
        model = from_config(cfg, len(vocab))
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
        model.eval()
        default_name = Path(args.checkpoint).stem.replace("_best", "")
    else:
        if not args.binary:
            raise SystemExit("--binary is required for --model ngram")
        from aagebatao.models.ngram import NgramModel

        model = NgramModel(args.binary, vocab)
        default_name = f"e0_{Path(args.binary).stem}_{args.version}"

    sentences = eval_sentences(load_split(args.version, args.split), vocab)
    results = evaluate_model(model, sentences, vocab.unk_id)
    path = write_report(results, args.name or default_name, args.split)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
