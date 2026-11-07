"""Token-level language tagging with HingBERT-LID."""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_ID = "l3cube-pune/hing-bert-lid"
_LABELS = {"HI": "hi", "EN": "en"}


def is_other(token: str) -> bool:
    """The model has no `other` class, so non-words are assigned it by rule."""
    return token in ("<url>", "<user>", "<num>") or not any(c.isalpha() for c in token)


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class LidTagger:
    """Tag pre-tokenised messages; each word takes its first subtoken's prediction."""

    def __init__(self, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        self.device = device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def tag(self, messages: list[list[str]]) -> list[list[str]]:
        order = sorted(range(len(messages)), key=lambda i: len(messages[i]))
        out: list[list[str]] = [[] for _ in messages]
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer(
                [messages[i] for i in idx],
                is_split_into_words=True,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            word_ids = [enc.word_ids(b) for b in range(len(idx))]
            preds = self.model(**enc.to(self.device)).logits.argmax(-1).cpu().tolist()
            for b, i in enumerate(idx):
                labels = ["other"] * len(messages[i])
                seen = set()
                for pos, word in enumerate(word_ids[b]):
                    if word is not None and word not in seen:
                        seen.add(word)
                        raw = self.id2label[preds[b][pos]]
                        token = messages[i][word]
                        labels[word] = "other" if is_other(token) else _LABELS[raw]
                out[i] = labels
        return out
