"""Near-duplicate detection between train and test (PRD section 5, rule 3)."""

from datasketch import MinHash, MinHashLSH

THRESHOLD = 0.8
NUM_PERM = 128


def shingles(tokens: list[str], n: int = 3) -> set[str]:
    lowered = [t.lower() for t in tokens]
    if len(lowered) < n:
        return {" ".join(lowered)}
    return {" ".join(lowered[i : i + n]) for i in range(len(lowered) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _minhash(items: set[str]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    m.update_batch([s.encode("utf-8") for s in items])
    return m


class HeldOutIndex:
    """LSH at a loose threshold for recall; candidates confirmed by exact Jaccard."""

    def __init__(self, test_tokens: list[list[str]]) -> None:
        self.lsh = MinHashLSH(threshold=0.6, num_perm=NUM_PERM)
        self.sets = [shingles(t) for t in test_tokens]
        for i, s in enumerate(self.sets):
            self.lsh.insert(str(i), _minhash(s))

    def is_near_duplicate(self, tokens: list[str]) -> bool:
        s = shingles(tokens)
        return any(
            jaccard(s, self.sets[int(key)]) >= THRESHOLD for key in self.lsh.query(_minhash(s))
        )
