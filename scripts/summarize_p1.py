"""Collect every P1 result csv into one table and three figures (plan Stage A3).

    uv run python scripts/summarize_p1.py --version v0.1
    uv run python scripts/summarize_p1.py --version smoke          # works on whatever exists

Reads  reports/results/e0_*_<version>*_test_{a,b}.csv  and  artifacts/*_log.csv
Writes reports/results/p1_summary_<version>.csv, p1_summary_<version>.md and
       reports/figures/figA_scaling_<version>.png, figB_buckets_<version>.png, figC_curves_<version>.png
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aagebatao.eval.buckets import ALL_COLUMNS
from aagebatao.models.lstm import from_config

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
ARTIFACTS = Path("artifacts")
SIZES = ["xs", "s", "m", "l", "xl"]
# result csv names: e0_ngram5_v0.1_real_test_a.csv  |  e0_lstm_w_m_475k_real_s42_test_a.csv
NGRAM_RE = re.compile(r"^e0_ngram(\d)_(?P<version>.+?)_real_(?P<split>test_[ab])\.csv$")
LSTM_RE = re.compile(
    r"^e0_lstm_w_(?P<size>xs|s|m|l|xl)_(?P<data>\w+?)_real_s(?P<seed>\d+)_(?P<split>test_[ab])\.csv$"
)


def model_label(name: str) -> tuple[str, str] | None:
    """csv filename -> (model label, split) or None if it is not a P1 result."""
    if m := NGRAM_RE.match(name):
        return f"ngram{m.group(1)}", m.group("split")
    if m := LSTM_RE.match(name):
        return f"lstm_{m.group('size')}", m.group("split")
    return None


def collect(version: str, results_dir: Path = RESULTS, data_tag: str | None = None) -> pd.DataFrame:
    """Long table: one row per (model, split, bucket) with every metric.

    n-gram csvs carry the dataset version in their name; LSTM csvs carry the PRD run name, whose
    data token is the training-set size (e.g. 475k), so they are filtered by `data_tag` instead
    (None = take every LSTM result present).
    """
    rows = []
    for path in sorted(results_dir.glob("e0_*_test_*.csv")):
        lab = model_label(path.name)
        if lab is None:
            continue
        if path.name.startswith("e0_ngram") and f"_{version}_" not in path.name:
            continue
        if path.name.startswith("e0_lstm") and data_tag and f"_{data_tag}_" not in path.name:
            continue
        model, split = lab
        df = pd.read_csv(path)
        df.insert(0, "split", split)
        df.insert(0, "model", model)
        rows.append(df)
    if not rows:
        raise SystemExit(f"no P1 result csvs for version {version!r} in {results_dir}")
    return pd.concat(rows, ignore_index=True)


def default_data_tag(version: str) -> str | None:
    """Same rule as scripts/train.py: <train rows / 1000>k, read from MANIFEST.json."""
    manifest = Path("data/hingmix") / version / "MANIFEST.json"
    if not manifest.exists():
        return None
    import json

    rows = json.loads(manifest.read_text())["rows"]["train"]["total"]
    return f"{round(rows / 1000)}k"


def model_order(models: list[str]) -> list[str]:
    def key(m: str) -> tuple[int, int]:
        if m.startswith("ngram"):
            return (0, int(m[-1]))
        return (1, SIZES.index(m.split("_")[1]))

    return sorted(set(models), key=key)


def wide_summary(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, split); columns <bucket>_<metric>."""
    keep = ["perplexity", "top1", "top3", "unk_rate"]
    out = []
    for (model, split), g in long.groupby(["model", "split"], sort=False):
        row = {"model": model, "split": split}
        for _, r in g.iterrows():
            for k in keep:
                row[f"{r['bucket']}_{k}"] = r[k]
        out.append(row)
    wide = pd.DataFrame(out)
    order = model_order(list(wide["model"]))
    wide["model"] = pd.Categorical(wide["model"], order, ordered=True)
    return wide.sort_values(["split", "model"]).reset_index(drop=True)


def non_embedding_params(size: str, vocab_size: int) -> int:
    with open(f"configs/model/lstm_{size}.yaml") as f:
        return from_config(yaml.safe_load(f), vocab_size).count_params()[1]


def fig_a_scaling(wide: pd.DataFrame, vocab_size: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for split, marker in (("test_a", "o"), ("test_b", "s")):
        color = None
        d = wide[(wide.split == split) & wide.model.astype(str).str.startswith("lstm")]
        if len(d):
            xs = [non_embedding_params(m.split("_")[1], vocab_size) for m in d.model.astype(str)]
            (line,) = ax.plot(xs, d.overall_perplexity, marker=marker, label=f"LSTM, {split}")
            color = line.get_color()
            for x, y, m in zip(xs, d.overall_perplexity, d.model.astype(str)):
                ax.annotate(
                    m.split("_")[1], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8
                )
        ng = wide[(wide.split == split) & (wide.model.astype(str) == "ngram5")]
        if len(ng):
            ax.axhline(
                float(ng.overall_perplexity.iloc[0]),
                ls="--",
                lw=1,
                color=color,
                label=f"5-gram, {split}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("non-embedding parameters")
    ax.set_ylabel("perplexity (overall)")
    ax.set_title("Fig A. Scaling preview: perplexity vs recurrent-core size (1 seed)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_b_buckets(long: pd.DataFrame, split: str, out: Path) -> None:
    d = long[long.split == split]
    models = model_order(list(d.model))
    buckets = [b for b in ALL_COLUMNS if b != "overall"]
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(buckets))
    for i, m in enumerate(models):
        vals = [float(d[(d.model == m) & (d.bucket == b)].top3.iloc[0]) for b in buckets]
        ax.bar(x + (i - len(models) / 2 + 0.5) * width, vals, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel("top-3 accuracy (%)")
    ax.set_title(f"Fig B. Who wins where: top-3 per bucket on {split}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_c_curves(version: str, out: Path, artifacts: Path = ARTIFACTS) -> bool:
    logs = sorted(artifacts.glob("e0_lstm_w_*_log.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    drew = False
    for log in logs:
        m = re.match(r"e0_lstm_w_(xs|s|m|l|xl)_(\w+?)_real_s\d+_log\.csv", log.name)
        if not m:
            continue
        df = pd.read_csv(log)
        ax.plot(df.epoch, df.val_ppl, marker="o", ms=3, label=f"{m.group(1)} ({m.group(2)})")
        drew = True
    if not drew:
        plt.close(fig)
        return False
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val perplexity")
    ax.set_title("Fig C. Training curves (early-stopped on val)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def acceptance(wide: pd.DataFrame) -> str:
    """PRD P1: LSTM m beats the 5-gram on test_a by >= 15 % relative perplexity."""
    a = wide[wide.split == "test_a"].set_index(wide[wide.split == "test_a"].model.astype(str))
    if "lstm_m" not in a.index or "ngram5" not in a.index:
        return "acceptance: need both lstm_m and ngram5 on test_a (not both present yet)"
    lstm, ng = a.loc["lstm_m", "overall_perplexity"], a.loc["ngram5", "overall_perplexity"]
    gain = 100 * (ng - lstm) / ng
    verdict = "PASS" if gain >= 15 else "FAIL"
    return f"acceptance: lstm_m ppl {lstm:.2f} vs 5-gram {ng:.2f} on test_a -> {gain:+.1f} % relative ({verdict}, need >= 15 %)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v0.1")
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--data", default=None, help="LSTM data tag to include (default: from MANIFEST)")
    args = p.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    data_tag = args.data or default_data_tag(args.version)
    long = collect(args.version, data_tag=data_tag)
    wide = wide_summary(long)
    csv_path = RESULTS / f"p1_summary_{args.version}.csv"
    wide.to_csv(csv_path, index=False)

    show = [
        "model",
        "split",
        "overall_perplexity",
        "hi_perplexity",
        "en_perplexity",
        "switch_hi_en_perplexity",
        "switch_en_hi_perplexity",
        "overall_top3",
        "switch_hi_en_top3",
        "switch_en_hi_top3",
        "overall_unk_rate",
    ]
    md = wide[show].to_markdown(index=False, floatfmt=".2f")
    md_path = RESULTS / f"p1_summary_{args.version}.md"
    verdict = acceptance(wide)
    md_path.write_text(f"# P1 summary ({args.version})\n\n{md}\n\n{verdict}\n")
    print(md)
    print()
    print(verdict)

    fig_a_scaling(wide, args.vocab_size, FIGURES / f"figA_scaling_{args.version}.png")
    for split in sorted(long.split.unique()):
        fig_b_buckets(long, split, FIGURES / f"figB_buckets_{split}_{args.version}.png")
    has_c = fig_c_curves(args.version, FIGURES / f"figC_curves_{args.version}.png")
    print(
        f"\nwrote {csv_path}, {md_path} and figures in {FIGURES}/"
        + ("" if has_c else " (no training logs for fig C yet)")
    )


if __name__ == "__main__":
    main()
