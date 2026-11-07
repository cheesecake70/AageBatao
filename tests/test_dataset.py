import pandas as pd
import pytest
import torch

from aagebatao.data.dataset import (
    WindowDataset,
    encode_sentences,
    eval_sentences,
    load_split,
    make_loader,
)
from aagebatao.data.tokenize import Vocab


@pytest.fixture
def vocab():
    return Vocab(["<pad>", "<unk>", "<bos>", "<eos>", "kya", "hai", "yeh"])


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "tokens": [["kya", "hai"], ["yeh", "bolta", "hai"]],
            "lid": [["hi", "hi"], ["hi", "hi", "hi"]],
        }
    )


def test_encode_sentences_wraps_with_bos_eos_and_unk(df, vocab):
    enc = encode_sentences(df, vocab)
    assert enc == [[2, 4, 5, 3], [2, 6, 1, 5, 3]]


def test_window_dataset_shapes_and_shift():
    enc = [list(range(10)), list(range(10, 20))]  # stream of 20 ints
    ds = WindowDataset(enc, window=4)
    assert len(ds) == (20 - 1) // 4 == 4
    x, y = ds[0]
    assert x.dtype == torch.long and y.dtype == torch.long
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]  # y is x shifted by one: next-token prediction
    x, y = ds[3]
    assert x.tolist() == [12, 13, 14, 15] and y.tolist() == [13, 14, 15, 16]


def test_window_dataset_exact_fit_needs_one_extra_token():
    ds = WindowDataset([list(range(8))], window=4)
    assert len(ds) == 1  # 8 tokens: one full window + its target; not two


def test_make_loader_fixed_batch_shape():
    ds = WindowDataset([list(range(100))], window=8)
    loader = make_loader(ds, batch_size=4, shuffle=False)
    batches = list(loader)
    assert all(x.shape == (4, 8) and y.shape == (4, 8) for x, y in batches)
    assert len(batches) == len(ds) // 4  # drop_last


def test_eval_sentences_keeps_lid_aligned(df, vocab):
    out = eval_sentences(df, vocab)
    assert len(out) == 2
    for (ids, lid), row in zip(out, df.itertuples()):
        assert ids[0] == vocab.bos_id and ids[-1] == vocab.eos_id
        assert len(ids) == len(lid) + 2
        assert lid == list(row.lid)


def test_load_split_reads_parquet(tmp_path, df):
    (tmp_path / "v9").mkdir()
    df.to_parquet(tmp_path / "v9" / "val.parquet")
    out = load_split("v9", "val", root=tmp_path)
    assert list(out["tokens"].iloc[1]) == ["yeh", "bolta", "hai"]


def test_load_split_missing_file_is_helpful(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split("nope", "train", root=tmp_path)
