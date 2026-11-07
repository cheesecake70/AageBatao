import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_p1 as sp

from aagebatao.eval.buckets import ALL_COLUMNS
from aagebatao.eval.evaluate import UniformModel, evaluate_model
from aagebatao.eval.report import write_report

SENTS = [([2, 5, 6, 7, 3], ["hi", "en", "hi"]), ([2, 9, 1, 3], ["en", "other"])]


def _fake_results(results_dir: Path, name: str, v: int):
    for split in ("test_a", "test_b"):
        write_report(
            evaluate_model(UniformModel(v), SENTS, 1), name, split, results_dir=results_dir
        )


def test_model_label_parses_both_naming_schemes():
    assert sp.model_label("e0_ngram5_v0.1_real_test_a.csv") == ("ngram5", "test_a")
    assert sp.model_label("e0_lstm_w_m_475k_real_s42_test_b.csv") == ("lstm_m", "test_b")
    assert sp.model_label("e0_uniform_smoke_test_a.csv") is None
    assert sp.model_label("p0_lid_handcheck.csv") is None


def test_collect_wide_and_acceptance(tmp_path, capsys):
    _fake_results(tmp_path, "e0_ngram5_v9_real", 100)  # ppl 100
    _fake_results(tmp_path, "e0_lstm_w_m_475k_real_s42", 80)  # ppl 80 -> 20 % better
    _fake_results(tmp_path, "e0_lstm_w_xs_475k_real_s42", 95)
    long = sp.collect("v9", results_dir=tmp_path)
    assert set(long.model) == {"ngram5", "lstm_m", "lstm_xs"}
    assert set(long.bucket) == set(ALL_COLUMNS)
    wide = sp.wide_summary(long)
    assert list(wide[wide.split == "test_a"].model.astype(str)) == ["ngram5", "lstm_xs", "lstm_m"]
    assert wide.loc[
        (wide.model == "lstm_m") & (wide.split == "test_a"), "overall_perplexity"
    ].item() == pytest.approx(80)
    verdict = sp.acceptance(wide)
    assert "PASS" in verdict and "+20.0 %" in verdict


def test_acceptance_fail_and_missing(tmp_path):
    _fake_results(tmp_path, "e0_ngram5_v9_real", 100)
    _fake_results(tmp_path, "e0_lstm_w_m_475k_real_s42", 90)  # only 10 %
    wide = sp.wide_summary(sp.collect("v9", results_dir=tmp_path))
    assert "FAIL" in sp.acceptance(wide)
    only = sp.wide_summary(sp.collect("v9", results_dir=tmp_path))
    only = only[only.model.astype(str) != "lstm_m"]
    assert "not both present" in sp.acceptance(only)


def test_collect_raises_when_nothing_matches(tmp_path):
    with pytest.raises(SystemExit):
        sp.collect("nothing", results_dir=tmp_path)


def test_figures_render(tmp_path):
    _fake_results(tmp_path, "e0_ngram5_v9_real", 100)
    _fake_results(tmp_path, "e0_lstm_w_xs_475k_real_s42", 95)
    _fake_results(tmp_path, "e0_lstm_w_m_475k_real_s42", 80)
    long = sp.collect("v9", results_dir=tmp_path)
    wide = sp.wide_summary(long)
    sp.fig_a_scaling(wide, 30000, tmp_path / "a.png")
    sp.fig_b_buckets(long, "test_a", tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 1000 and (tmp_path / "b.png").stat().st_size > 1000
    # fig C with a fake training log
    pd.DataFrame({"epoch": [1, 2], "val_ppl": [500.0, 300.0]}).to_csv(
        tmp_path / "e0_lstm_w_xs_475k_real_s42_log.csv", index=False
    )
    assert sp.fig_c_curves("v9", tmp_path / "c.png", artifacts=tmp_path) is True
    assert sp.fig_c_curves("v9", tmp_path / "c2.png", artifacts=tmp_path / "empty") is False
