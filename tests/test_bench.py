import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_latency as bl


def test_sample_prefixes_are_real_prefixes(tmp_path, monkeypatch):
    df = pd.DataFrame({"tokens": [["kya", "kar", "raha", "hai"], ["theek", "hai"]]})
    (tmp_path / "v9").mkdir()
    df.to_parquet(tmp_path / "v9" / "test_a.parquet")
    monkeypatch.setattr(
        bl, "load_split", lambda v, s: pd.read_parquet(tmp_path / v / f"{s}.parquet")
    )
    out = bl.sample_prefixes("v9", 20, seed=1)
    assert len(out) == 20
    for text in out:
        words = text.split()
        assert 1 <= len(words) <= 4
        assert any(" ".join(t[: len(words)]) == " ".join(words) for t in df["tokens"])
