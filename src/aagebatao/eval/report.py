"""Fill the PRD section 9 results template -> markdown + csv.

TODO 13 -> 14 live here. Tests: tests/test_report.py
"""

import math
from pathlib import Path

import pandas as pd

from aagebatao.eval.buckets import ALL_COLUMNS
from aagebatao.eval.metrics import METRICS

RESULTS_DIR = Path("reports/results")
ROW_LABELS = (
    ("perplexity", "Perplexity"),
    ("top1", "Top-1 %"),
    ("top3", "Top-3 %"),
    ("top5", "Top-5 %"),
    ("unk_rate", "`<unk>` %"),
    ("share", "Share of tokens %"),
)
HEADER = ("Metric", "Overall", "hi", "en", "switch hi→en", "switch en→hi", "other")


def to_markdown(results: dict[str, dict[str, float]]) -> str:
    def fmt(v: float) -> str:
        return "–" if isinstance(v, float) and math.isnan(v) else f"{v:.2f}"

    lines = ["| " + " | ".join(HEADER) + " |", "|" + " --- |" * len(HEADER)]
    for key, label in ROW_LABELS:
        cells = [fmt(results[col][key]) for col in ALL_COLUMNS]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines) + "\n"


def to_dataframe(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = [{"bucket": col, **{m: results[col][m] for m in METRICS}} for col in ALL_COLUMNS]
    return pd.DataFrame(rows, columns=["bucket", *METRICS])


def write_report(
    results: dict[str, dict[str, float]], name: str, test_set: str, results_dir: Path = RESULTS_DIR
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}_{test_set}.csv"
    to_dataframe(results).to_csv(path, index=False)
    print(f"\n{name} on {test_set}\n")
    print(to_markdown(results))
    return path
