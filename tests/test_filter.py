from aagebatao.data.filter import cmi, is_code_mixed, n_switch


def test_n_switch_skips_other():
    assert n_switch(["hi", "other", "en", "en", "hi"]) == 2


def test_cmi():
    assert cmi(["hi", "hi", "hi", "en"]) == 25.0
    assert cmi(["other"]) == 0.0


def test_code_mixed():
    assert is_code_mixed(["hi", "en"])
    assert not is_code_mixed(["hi", "hi", "other"])
