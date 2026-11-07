"""Word tokeniser and the Vocab class shared by every model (PRD section 7).

P1 build order: TODO 1 -> 2 -> 3 live here. Tests: tests/test_tokenize.py
"""

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

_TOKEN = re.compile(r"<(?:url|user|num)>|[^\W_]+(?:['’][^\W_]+)*|[^\w\s]|_", re.UNICODE)

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)  # ids 0, 1, 2, 3 in every vocabulary, always
DEFAULT_VOCAB_SIZE = 30_000


def tokenize(text: str) -> list[str]:
    """Split normalised text into tokens."""
    return _TOKEN.findall(text)


class Vocab:
    """Maps tokens <-> integer ids. `itos` is the list of strings; the id is the list index."""

    def __init__(self, itos: list[str]):
        if tuple(itos[:4]) != SPECIALS:
            raise ValueError(f"itos must start with {SPECIALS}")
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    pad_id, unk_id, bos_id, eos_id = 0, 1, 2, 3

    @classmethod
    def build(cls, token_lists: Iterable[list[str]], max_size: int = DEFAULT_VOCAB_SIZE) -> "Vocab":
        # TODO 1: build the vocabulary from the TRAIN split only.
        #   - Count every token across all token_lists (collections.Counter).
        #   - Keep the (max_size - 4) most frequent tokens; the 4 specials take the first slots.
        #   - Break count ties alphabetically so two builds on the same data give the same vocab.
        #   - Return cls(list(SPECIALS) + kept_tokens).
        counts: Counter[str] = Counter()
        for tokens in token_lists:
            counts.update(tokens)
        # most frequent first; ties broken alphabetically so the vocab is reproducible
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [tok for tok, _ in ordered[: max_size - len(SPECIALS)]]
        return cls(list(SPECIALS) + kept)

    def encode(self, tokens: list[str]) -> list[int]:
        # TODO 2a: map each token to its id; anything not in stoi becomes unk_id.
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        # TODO 2b: inverse of encode (ids -> strings via itos).
        return [self.itos[i] for i in ids]

    def save(self, path: str | Path) -> None:
        # TODO 3a: write self.itos as a JSON list to `path` (UTF-8, ensure_ascii=False).
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        # TODO 3b: read the JSON list and return cls(itos).
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(itos)
