# Aage Batao demo service: FastAPI + the word-LSTM `m` checkpoint, CPU inference.
#
#   docker build -t aagebatao .
#   docker run -p 8080:8080 aagebatao
#   # or straight to Cloud Run, which builds this file remotely:
#   make deploy
#
# Only the serving path is installed: torch (CPU wheel, ~200 MB instead of ~2.5 GB with CUDA),
# fastapi, uvicorn, numpy, pyyaml. Training-only deps (kenlm, transformers, pandas, ...) are not.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Versions match uv.lock so the container runs the same code paths the tests ran.
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.15.0 \
        numpy==2.4.6 \
        fastapi==0.162.0 \
        uvicorn==0.54.0 \
        pyyaml==6.0.3

# Source, the size configs the predictor reads, the vocab and the checkpoint.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs/model ./configs/model
COPY data/hingmix/v0.1/vocab.json ./data/hingmix/v0.1/vocab.json
COPY artifacts/e0_lstm_w_m_475k_real_s42_best.pt ./artifacts/e0_lstm_w_m_475k_real_s42_best.pt

# Install the package itself without pulling the training dependency list.
RUN pip install --no-deps .

ENV AAGEBATAO_CHECKPOINT=artifacts/e0_lstm_w_m_475k_real_s42_best.pt \
    AAGEBATAO_SIZE=m \
    AAGEBATAO_VERSION=v0.1 \
    PORT=8080

EXPOSE 8080

# Cloud Run injects PORT; one worker is right for a 35 MB model that answers in ~3 ms.
CMD ["sh", "-c", "uvicorn aagebatao.serve.app:app --host 0.0.0.0 --port ${PORT}"]
