from aagebatao.data.normalize import normalize


def test_replaces_entities():
    assert normalize("RT @abc: dekho https://x.co/a 100 baar") == "dekho <url> <num> baar"


def test_preserves_case_and_spelling():
    assert normalize("  Yaar   BAHOT   acha ") == "Yaar BAHOT acha"


def test_nfc_and_emoji():
    assert normalize("café 😂") == "café 😂"


def test_digits_inside_words_stay():
    assert normalize("h1n1 aur 2g") == "h1n1 aur 2g"
