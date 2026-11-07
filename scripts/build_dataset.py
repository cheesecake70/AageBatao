"""Build HingMix real-only versions. Tagged chunks are cached, so a run can resume."""

import argparse
import hashlib
from itertools import islice
from pathlib import Path

import pandas as pd

from aagebatao.data import download, ingest
from aagebatao.data.dedupe import HeldOutIndex
from aagebatao.data.filter import cmi, is_code_mixed, length_ok, n_switch
from aagebatao.data.foreign import is_foreign, load_vocab
from aagebatao.data.lid import LidTagger
from aagebatao.data.manifest import write_manifest
from aagebatao.data.normalize import normalize
from aagebatao.data.split import split_train_val
from aagebatao.data.tokenize import tokenize

SEED = 42
CHUNK = 50_000


def records(texts, source: str, genre: str, split: str, tagger: LidTagger) -> pd.DataFrame:
    rows, seen = [], set()
    for raw in texts:
        text = normalize(raw)
        tokens = tokenize(text)
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if length_ok(tokens) and key not in seen:
            seen.add(key)
            rows.append((f"{source}_{key[:12]}", text, tokens))
    labels = tagger.tag([r[2] for r in rows])
    out = []
    for (id_, text, tokens), lid in zip(rows, labels):
        if is_code_mixed(lid):
            out.append(
                {
                    "id": id_,
                    "text": text,
                    "tokens": tokens,
                    "lid": lid,
                    "n_switch": n_switch(lid),
                    "cmi": cmi(lid),
                    "source": source,
                    "is_synthetic": False,
                    "generator_version": "",
                    "parent_id": "",
                    "split": split,
                    "genre": genre,
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=550_000)
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--stage", choices=["tag", "final", "all"], default="all")
    args = parser.parse_args()

    download.fetch()
    download.verify()
    cache = Path(f"data/interim/{args.version}_{args.limit}")
    cache.mkdir(parents=True, exist_ok=True)
    tagger = LidTagger()

    tests = {}
    for name, texts, genre in (
        ("test_a", ingest.cmu_test(), "chat"),
        ("test_b", ingest.phinc_test(SEED), "translation_pair"),
    ):
        path = cache / f"{name}.parquet"
        source = "cmu_dog" if name == "test_a" else "phinc"
        if not path.exists():
            records(texts, source, genre, name, tagger).to_parquet(path)
        tests[name] = pd.read_parquet(path)

    stream = ingest.hingcorpus(args.limit)
    n = 0
    while chunk := list(islice(stream, CHUNK)):
        path = cache / f"train_{n:05d}.parquet"
        if not path.exists():
            records(chunk, "hingcorpus", "tweet", "train", tagger).to_parquet(path)
            print("tagged chunk", n, flush=True)
        n += 1
    if args.stage == "tag":
        return

    pool = pd.concat(pd.read_parquet(p) for p in sorted(cache.glob("train_*.parquet")))
    pool = pool.drop_duplicates("id").reset_index(drop=True)
    vocab = load_vocab()
    foreign = [is_foreign(list(t), vocab) for t in pool["tokens"]]
    foreign_dropped = sum(foreign)
    pool[pd.Series(foreign, index=pool.index)].sample(
        n=min(200, foreign_dropped), random_state=SEED
    )[["text"]].to_csv(Path("reports/results") / f"foreign_removed_{args.version}.csv", index=False)
    pool = pool[[not f for f in foreign]].reset_index(drop=True)
    before = len(pool)
    test_tokens = [list(t) for df in tests.values() for t in df["tokens"]]
    index = HeldOutIndex(test_tokens)
    keep = [not index.is_near_duplicate(list(t)) for t in pool["tokens"]]
    pool = pool[keep].reset_index(drop=True)
    pool["split"] = split_train_val(pool, SEED)

    frames = {
        "train": pool[pool["split"] == "train"],
        "val": pool[pool["split"] == "val"],
        **tests,
    }
    out = Path(f"data/hingmix/{args.version}")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    write_manifest(
        out,
        frames,
        {
            "version": args.version,
            "seed": SEED,
            "hingcorpus_lines_read": args.limit,
            "foreign_dropped_from_train_val": foreign_dropped,
            "near_duplicates_dropped_from_train": before - len(pool),
        },
    )
    print("built", {k: len(v) for k, v in frames.items()})


if __name__ == "__main__":
    main()
