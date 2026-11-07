"""Stratified train/val split (PRD section 5, rule 5)."""

import numpy as np
import pandas as pd

VAL_FRACTION = 0.05


def split_train_val(df: pd.DataFrame, seed: int) -> pd.Series:
    """Return a train/val label per row, stratified by CMI decile."""
    rng = np.random.default_rng(seed)
    deciles = pd.qcut(df["cmi"], 10, labels=False, duplicates="drop")
    labels = pd.Series("train", index=df.index)
    for idx in df.groupby(deciles).groups.values():
        chosen = rng.choice(idx, size=round(len(idx) * VAL_FRACTION), replace=False)
        labels.loc[chosen] = "val"
    return labels
