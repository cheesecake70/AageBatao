"""The ruler test: a uniform model must score perplexity == V through the whole chain."""

import pandas as pd
import pytest

from aagebatao.eval.buckets import ALL_COLUMNS
from aagebatao.eval.evaluate import UniformModel, evaluate_model
from aagebatao.eval.report import HEADER, to_dataframe, to_markdown, write_report

V = 37
SENTENCES = [
    ([2, 5, 6, 7, 3], ["hi", "en", "hi"]),
    ([2, 9, 1, 3], ["en", "other"]),
]


@pytest.fixture
def results():
    return evaluate_model(UniformModel(V), SENTENCES, unk_id=1)


def test_uniform_perplexity_equals_vocab_size(results):
    for col in ALL_COLUMNS:
        if results[col]["n_tokens"]:
            assert results[col]["perplexity"] == pytest.approx(V)
    assert results["overall"]["n_tokens"] == 7  # 3+1 and 2+1 targets (eos included)


def test_uniform_topk_is_all_ties(results):
    # every word has the same prob -> nothing is strictly greater -> rank 0 everywhere
    assert results["overall"]["top1"] == pytest.approx(100.0)


def test_unk_rate_and_shares(results):
    assert results["overall"]["unk_rate"] == pytest.approx(100 * 1 / 7)
    assert results["other"]["n_tokens"] == 3  # two <eos> + one 'other' token
    shares = sum(results[c]["share"] for c in ALL_COLUMNS if c != "overall")
    assert shares == pytest.approx(100.0)


def test_markdown_table_shape(results):
    md = to_markdown(results)
    lines = md.strip().splitlines()
    assert lines[0].strip("| ").split(" | ") == list(HEADER)
    assert set(lines[1].replace("|", "").split()) == {"---"}
    assert len(lines) == 2 + 6
    assert lines[2].startswith("| Perplexity") and "37.00" in lines[2]
    assert "–" not in md  # these two sentences fill every bucket


def test_markdown_empty_bucket_renders_dash():
    only_hindi = evaluate_model(UniformModel(V), [([2, 5, 6, 3], ["hi", "hi"])], unk_id=1)
    md = to_markdown(only_hindi)
    assert "–" in md  # switch and en buckets are empty -> dash, not "nan"
    assert "nan" not in md


def test_dataframe_and_csv(results, tmp_path):
    df = to_dataframe(results)
    assert list(df["bucket"]) == list(ALL_COLUMNS)
    assert "perplexity" in df.columns and len(df) == 6
    path = write_report(results, "e0_uniform_test", "test_a", results_dir=tmp_path)
    assert path == tmp_path / "e0_uniform_test_test_a.csv"
    back = pd.read_csv(path)
    assert back.loc[back.bucket == "overall", "perplexity"].item() == pytest.approx(V)


def test_evaluate_model_rejects_misaligned_lid():
    with pytest.raises(ValueError):
        evaluate_model(UniformModel(V), [([2, 5, 3], ["hi", "hi", "hi"])], unk_id=1)
