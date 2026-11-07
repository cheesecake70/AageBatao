"""Train the word LSTM (PRD M1). P1 runs:

    uv run python scripts/train.py --version smoke --size xs --max-epochs 2    # pipeline check
    uv run python scripts/train.py --version v0.1 --size m                     # the P1 run

Writes artifacts/<run_name>_best.pt (early-stopped checkpoint) and artifacts/<run_name>_log.csv.
Builds data/hingmix/<version>/vocab.json on first use; every other script reloads it.
"""

import argparse
import time
from pathlib import Path

import torch
import yaml

from aagebatao.data.dataset import WindowDataset, encode_sentences, load_split, make_loader
from aagebatao.data.tokenize import Vocab
from aagebatao.models.lstm import from_config
from aagebatao.train.naming import run_name
from aagebatao.train.trainer import Trainer, seed_everything

HINGMIX = Path("data/hingmix")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--version", default="smoke", help="data/hingmix/<version>")
    p.add_argument("--size", default="m", choices=["xs", "s", "m", "l", "xl"])
    p.add_argument("--exp", default="e0")
    p.add_argument("--data", default=None, help="data-size token in the run name; default <rows>k")
    p.add_argument("--mix", default="real")
    p.add_argument("--seed", type=int, default=42, choices=[13, 42, 2024])
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="override the config's batch size")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: best available)")
    p.add_argument("--artifacts", default="artifacts")
    return p.parse_args()


def load_or_build_vocab(version: str, train_df) -> Vocab:
    """Shared by every script: one vocab.json per dataset version."""
    path = HINGMIX / version / "vocab.json"
    if path.exists():
        return Vocab.load(path)
    vocab = Vocab.build(train_df["tokens"])
    vocab.save(path)
    print(f"built vocab: {len(vocab)} types -> {path}")
    return vocab


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    with open(f"configs/model/lstm_{args.size}.yaml") as f:
        model_cfg = yaml.safe_load(f)
    batch_size = args.batch_size or model_cfg["batch_size"]

    t0 = time.time()
    train_df = load_split(args.version, "train")
    val_df = load_split(args.version, "val")
    vocab = load_or_build_vocab(args.version, train_df)
    train_ds = WindowDataset(encode_sentences(train_df, vocab), window=model_cfg["window"])
    val_ds = WindowDataset(encode_sentences(val_df, vocab), window=model_cfg["window"])
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)
    print(
        f"data: {len(train_df)} train sentences -> {len(train_ds)} windows, "
        f"{len(val_df)} val sentences -> {len(val_ds)} windows ({time.time() - t0:.0f}s)"
    )

    model = from_config(model_cfg, len(vocab))
    total, non_emb = model.count_params()
    print(
        f"model lstm_w {args.size}: {total / 1e6:.2f}M params, {non_emb / 1e6:.2f}M non-embedding"
    )

    data_tag = args.data or f"{round(len(train_df) / 1000)}k"
    name = run_name(
        {
            "exp": args.exp,
            "family": "lstm_w",
            "size": args.size,
            "data": data_tag,
            "mix": args.mix,
            "seed": args.seed,
        }
    )
    # float() guards against YAML reading "2e-3" as a string
    trainer_cfg = {"lr": float(model_cfg["lr"]), "weight_decay": float(model_cfg["weight_decay"])}
    if args.max_epochs:
        trainer_cfg["max_epochs"] = args.max_epochs
    if args.patience:
        trainer_cfg["patience"] = args.patience
    device = pick_device(args.device)
    print(f"run {name} on {device}; batch {batch_size}, {len(train_loader)} steps/epoch")

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        name,
        cfg=trainer_cfg,
        artifacts_dir=Path(args.artifacts),
        device=device,
    )
    history = trainer.fit()
    best = min(h["val_ppl"] for h in history)
    print(f"done: best val ppl {best:.2f} -> {Path(args.artifacts) / (name + '_best.pt')}")
    print(
        f"next: uv run python scripts/evaluate.py --model lstm --checkpoint {Path(args.artifacts) / (name + '_best.pt')} --size {args.size} --version {args.version} --split test_a"
    )


if __name__ == "__main__":
    main()
