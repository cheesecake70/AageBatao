"""Text normalisation (PRD section 5, rule 1)."""

import re
import unicodedata

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_USER = re.compile(r"(?<!\w)@\w+")
_RT = re.compile(r"^\s*RT\b:?\s*(?:<user>\s*:?\s*)?")
_NUM = re.compile(r"(?<![\w<])\d+(?:[.,]\d+)*(?![\w>])")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalise one message. Casing and spelling are deliberately preserved."""
    text = unicodedata.normalize("NFC", text)
    text = _URL.sub("<url>", text)
    text = _USER.sub("<user>", text)
    text = _RT.sub("", text)
    text = _NUM.sub("<num>", text)
    return _SPACE.sub(" ", text).strip()
