"""Score HingBERT-LID against HingLID gold and report throughput."""

import json
import sys
import time
from collections import Counter
from pathlib import Path

from aagebatao.data.lid import LidTagger

GOLD = Path("data/raw/hinglid/test.txt")


def read_conll(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    words, labels, w, l = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if w:
                words.append(w)
                labels.append(l)
            w, l = [], []
            continue
        token, _, label = line.rpartition("\t")
        w.append(token)
        l.append(label.strip())
    if w:
        words.append(w)
        labels.append(l)
    return words, labels


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    words, gold = read_conll(GOLD)
    words, gold = words[:limit], gold[:limit]
    tagger = LidTagger()
    print("device", tagger.device, "labels", tagger.id2label)
    start = time.time()
    pred = tagger.tag(words)
    secs = time.time() - start
    map_gold = {"HI": "hi", "EN": "en"}
    confusion: Counter = Counter()
    for gs, ps in zip(gold, pred):
        for g, p in zip(gs, ps):
            confusion[(map_gold.get(g, "other"), p)] += 1
    total = sum(confusion.values())
    correct = sum(n for (g, p), n in confusion.items() if g == p)
    result = {
        "sentences": len(words),
        "tokens": total,
        "token_accuracy": round(correct / total, 4),
        "sentences_per_sec": round(len(words) / secs, 1),
        "gold_labels": Counter(g for gs in gold for g in gs).most_common(),
        "confusion": {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())},
    }
    print(json.dumps(result, indent=1))
    Path("reports/results").mkdir(parents=True, exist_ok=True)
    Path("reports/results/p0_lid_validation.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
