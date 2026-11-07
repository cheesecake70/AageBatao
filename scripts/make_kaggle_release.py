"""Build release/kaggle/: the HingMix v0.1 splits as one file per source, ready to upload.

The three sources have licences that cannot be merged into one file (CC BY-NC-SA 4.0,
CC BY-SA 3.0, CC BY 4.0), and the v0.1 splits already line up with sources, so each split is
written as its own file, named after its source. Columns that are empty in a real-only release
(synthetic-data bookkeeping) are dropped.

    uv run python scripts/make_kaggle_release.py
"""

import json
import shutil
from pathlib import Path

import pandas as pd

SRC = Path("data/hingmix/v0.1")
OUT = Path("release/kaggle")
TEMPLATE = Path("release/kaggle_template")
FILES = {
    "train": "hingcorpus_train.parquet",
    "val": "hingcorpus_val.parquet",
    "test_a": "cmu_dog_test.parquet",
    "test_b": "phinc_test.parquet",
}
EXPECTED_SOURCE = {"train": "hingcorpus", "val": "hingcorpus", "test_a": "cmu", "test_b": "phinc"}
DROP = ["is_synthetic", "generator_version", "parent_id"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, name in FILES.items():
        df = pd.read_parquet(SRC / f"{split}.parquet")
        prefixes = set(df["id"].str.split("_").str[0])
        if prefixes != {EXPECTED_SOURCE[split]}:
            raise SystemExit(f"{split} mixes sources {prefixes}; licences would mix too")
        if df["is_synthetic"].any():
            raise SystemExit(f"{split} has synthetic rows; this script is for real-only releases")
        df = df.drop(columns=DROP)
        df.to_parquet(OUT / name, index=False)
        counts[name] = len(df)
    shutil.copy(SRC / "vocab.json", OUT / "vocab.json")
    for f in TEMPLATE.iterdir():
        shutil.copy(f, OUT / f.name)
    print(json.dumps(counts, indent=1))
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()
