import pytest
import torch
from fastapi.testclient import TestClient

from aagebatao.data.tokenize import SPECIALS, Vocab
from aagebatao.models.lstm import WordLSTM
from aagebatao.serve.app import create_app
from aagebatao.serve.predictor import Predictor

WORDS = ["kya", "kar", "raha", "hai", "ho", "hoon", "office"]


@pytest.fixture
def client():
    torch.manual_seed(0)
    vocab = Vocab(list(SPECIALS) + WORDS)
    predictor = Predictor(vocab, WordLSTM(len(vocab), 16, 16, 1, 0.0))
    with TestClient(create_app(predictor)) as c:
        yield c


def test_index_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Aage Batao" in r.text and "/predict" in r.text


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["vocab_size"] == 4 + len(WORDS) and body["params"] > 0


def test_predict_endpoint(client):
    r = client.post("/predict", json={"text": "kya kar ", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"words", "probs", "partial", "latency_ms"}
    assert len(body["words"]) == 3 and body["partial"] == ""


def test_predict_partial_and_validation(client):
    r = client.post("/predict", json={"text": "kya h", "k": 2})
    assert r.status_code == 200 and r.json()["partial"] == "h"
    assert client.post("/predict", json={"text": "x", "k": 0}).status_code == 422
    assert client.post("/predict", json={"text": "x", "k": 99}).status_code == 422


def test_app_without_predictor_needs_env(monkeypatch):
    monkeypatch.delenv("AAGEBATAO_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError), TestClient(create_app()):
        pass
