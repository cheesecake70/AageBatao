"""Per-source readers yielding raw message text."""

import csv
import random
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from aagebatao.data.download import HINGCORPUS_FILE, RAW

PHINC_TEST_FRACTION = 0.1


def hingcorpus(limit: int) -> Iterator[str]:
    """Stream the first lines of the shuffled training file without extracting it."""
    member = "R11_final_data/concatenated_train_final_shuffled.txt"
    proc = subprocess.Popen(["tar", "-xzOf", str(HINGCORPUS_FILE), member], stdout=subprocess.PIPE)
    assert proc.stdout
    for n, line in enumerate(proc.stdout):
        if n >= limit:
            break
        yield line.decode("utf-8", "replace")
    proc.kill()
    proc.wait()


def cmu_test() -> list[str]:
    df = pd.read_parquet(Path(RAW / "cmu_dog/hinglish/test.parquet"))
    return [t["hi_en"] for t in df["translation"]]


def phinc_test(seed: int) -> list[str]:
    """Hold out a seeded 10% of PHINC; the rest is reserved for generator G3."""
    path = RAW / "phinc/hinglish_english_parallel_corpus.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r["Sentence"] for r in csv.DictReader(f)]
    random.Random(seed).shuffle(rows)
    return rows[: int(len(rows) * PHINC_TEST_FRACTION)]
