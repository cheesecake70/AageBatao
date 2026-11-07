"""FastAPI service for the demo (plan Stage A4).

    make demo
    # or
    AAGEBATAO_CHECKPOINT=artifacts/<run>_best.pt AAGEBATAO_SIZE=m AAGEBATAO_VERSION=v0.1 \
        uv run uvicorn aagebatao.serve.app:app --port 8000

POST /predict {"text": "...", "k": 3} -> {"words", "probs", "partial", "latency_ms"}
GET  /        the type-ahead page
GET  /health  model + vocab info
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from aagebatao.serve.predictor import Predictor

STATIC = Path(__file__).parent / "static"


class PredictRequest(BaseModel):
    text: str = Field("", max_length=2000)
    k: int = Field(3, ge=1, le=10)


class PredictResponse(BaseModel):
    words: list[str]
    probs: list[float]
    partial: str
    latency_ms: float


def predictor_from_env() -> Predictor:
    ckpt = os.environ.get("AAGEBATAO_CHECKPOINT")
    size = os.environ.get("AAGEBATAO_SIZE", "m")
    version = os.environ.get("AAGEBATAO_VERSION", "v0.1")
    if not ckpt:
        raise RuntimeError(
            "set AAGEBATAO_CHECKPOINT=artifacts/<run>_best.pt (and AAGEBATAO_SIZE, "
            "AAGEBATAO_VERSION) or use `make demo CKPT=... SIZE=... VERSION=...`"
        )
    return Predictor.load(ckpt, size, Path("data/hingmix") / version / "vocab.json")


def create_app(predictor: Predictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or predictor_from_env()
        yield

    app = FastAPI(title="Aage Batao", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict:
        p: Predictor = app.state.predictor
        total, non_emb = p.model.count_params()
        return {"vocab_size": len(p.vocab), "params": total, "non_embedding_params": non_emb}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        try:
            pred = app.state.predictor.predict(req.text, req.k)
        except Exception as e:  # keep the UI alive on odd input
            raise HTTPException(status_code=400, detail=str(e)) from e
        return PredictResponse(**pred.__dict__)

    return app


app = create_app()
