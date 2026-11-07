"""Fetch sources that are not stored in Git, and verify every raw file by checksum."""

import hashlib
import json
from pathlib import Path

import requests

RAW = Path("data/raw")
CHECKSUMS = RAW / "checksums.json"
CMU = "https://huggingface.co/api/datasets/festvox/cmu_hinglish_dog/parquet/default/{}/0.parquet"
HINGCORPUS = (
    "https://drive.usercontent.google.com/download"
    "?id=1s_6eHO9zDhxQ-xVN1TyNguszV1meZkl9&export=download&confirm=t"
)
HINGCORPUS_FILE = RAW / "hingcorpus/L3Cube-HingCorpus_roman.tar.gz"
SOURCES = {
    **{
        RAW / f"cmu_dog/hinglish/{s}.parquet": CMU.format(s)
        for s in ("train", "validation", "test")
    },
    HINGCORPUS_FILE: HINGCORPUS,
}
LOCAL = [
    RAW / "phinc/hinglish_english_parallel_corpus.csv",
    *[RAW / f"hinglid/{s}.txt" for s in ("train", "validation", "test")],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch() -> None:
    for path, url in SOURCES.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)


def verify(write: bool = False) -> None:
    actual = {str(p): sha256(p) for p in [*SOURCES, *LOCAL]}
    if write:
        CHECKSUMS.write_text(json.dumps(actual, indent=1) + "\n")
        return
    if actual != json.loads(CHECKSUMS.read_text()):
        raise SystemExit("raw data checksum mismatch")
