"""Parquet rows -> integer sequences -> fixed-length training windows (PRD section 7, 13).

Two consumers:
  * training wants a flat stream cut into 64-token windows (WindowDataset)
  * evaluation wants whole sentences with their LID tags aligned (eval_sentences)

TODO 4 -> 8 live here. Tests: tests/test_dataset.py
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from aagebatao.data.tokenize import Vocab

HINGMIX_ROOT = Path("data/hingmix")
WINDOW = 64  # truncated-BPTT window length for the word LSTM (PRD section 7)


def load_split(version: str, split: str, root: Path = HINGMIX_ROOT) -> pd.DataFrame:
    path = Path(root) / version / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `make dataset-smoke` or `make dataset` first"
        )
    return pd.read_parquet(path)


def encode_sentences(df: pd.DataFrame, vocab: Vocab) -> list[list[int]]:
    # tokens were produced in P0; never re-tokenise `text` here
    return [[vocab.bos_id] + vocab.encode(list(toks)) + [vocab.eos_id] for toks in df["tokens"]]


class WindowDataset(Dataset):
    """Concatenate all encoded sentences into one stream and cut it into windows.

    Item i is (x, y) with x = stream[i*W : i*W + W] and y = stream[i*W + 1 : i*W + W + 1].
    The one-position shift IS next-token prediction with teacher forcing.
    """

    def __init__(self, encoded: list[list[int]], window: int = WINDOW):
        flat = [tok for sent in encoded for tok in sent]
        self.data = torch.tensor(flat, dtype=torch.long)
        self.window = window

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.window)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = i * self.window
        x = self.data[start : start + self.window]
        y = self.data[start + 1 : start + self.window + 1]
        return x, y


def make_loader(ds: Dataset, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def eval_sentences(df: pd.DataFrame, vocab: Vocab) -> list[tuple[list[int], list[str]]]:
    ids = encode_sentences(df, vocab)
    return [(seq, list(lid)) for seq, lid in zip(ids, df["lid"], strict=True)]
