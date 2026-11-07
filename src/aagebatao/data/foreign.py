"""Drop non-Hinglish sentences by coverage of a vocabulary learned from HingLID."""

from collections import Counter
from pathlib import Path

GOLD = Path("data/raw/hinglid/train.txt")
MIN_FREQ = 2
MIN_COVERAGE = 0.5
MIN_MARKERS = 2
# Common Spanish, French, Turkish, Indonesian and Tagalog words that do not collide with
# Roman Hindi or English. Words like `se`, `de`, `la`, `ben` and `sama` are left out on purpose.
MARKERS = frozenset(
    [
        "que",
        "los",
        "las",
        "del",
        "para",
        "por",
        "pero",
        "una",
        "muy",
        "como",
        "esta",
        "nos",
        "porque",
        "hola",
        "gracias",
        "algo",
        "todo",
        "les",
        "des",
        "une",
        "pour",
        "dans",
        "avec",
        "qui",
        "je",
        "moi",
        "vous",
        "nous",
        "sont",
        "cette",
        "mais",
        "c'est",
        "j'ai",
        "il",
        "elle",
        "bir",
        "ile",
        "icin",
        "gibi",
        "daha",
        "degil",
        "yok",
        "yang",
        "ini",
        "itu",
        "aja",
        "gak",
        "juga",
        "dengan",
        "untuk",
        "tidak",
        "kabar",
        "ako",
        "ikaw",
        "ang",
        "mga",
        "naman",
        "lang",
        "kayo",
    ]
)


def load_vocab(path: Path = GOLD) -> set[str]:
    counts: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[line.rpartition("\t")[0].lower()] += 1
    return {w for w, n in counts.items() if n >= MIN_FREQ}


def coverage(tokens: list[str], vocab: set[str]) -> float:
    words = [t.lower() for t in tokens if any(c.isalpha() for c in t) and not t.startswith("<")]
    return sum(w in vocab for w in words) / len(words) if words else 1.0


def is_foreign(tokens: list[str], vocab: set[str]) -> bool:
    markers = sum(t.lower() in MARKERS for t in tokens)
    return markers >= MIN_MARKERS or coverage(tokens, vocab) < MIN_COVERAGE
