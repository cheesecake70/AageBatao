"""MANIFEST.json writer."""

import json
from pathlib import Path

import pandas as pd


def write_manifest(out: Path, frames: dict[str, pd.DataFrame], extra: dict) -> None:
    manifest = {
        **extra,
        "rows": {
            name: {"total": len(df), "by_source": df["source"].value_counts().to_dict()}
            for name, df in frames.items()
        },
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
