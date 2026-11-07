.PHONY: setup test lint

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts
	uv run black --check src tests scripts

.PHONY: dataset dataset-smoke

dataset:
	uv run python scripts/build_dataset.py

dataset-smoke:
	uv run python scripts/build_dataset.py --limit 20000 --version smoke

# ---- P1: baselines and LSTM internals -------------------------------------------------
.PHONY: vocab ngram train train-smoke eval eval-uniform grad-check numpy-lstm
VERSION ?= v0.1
SIZE ?= m

ngram:          ## KenLM 3/4/5-gram tables on test_a / test_b
	uv run python scripts/train_ngram.py --version $(VERSION)

train-smoke:    ## quick end-to-end check on the smoke dataset
	uv run python scripts/train.py --version smoke --size xs --max-epochs 2

train:          ## the P1 run: size m on real v0.1
	uv run python scripts/train.py --version $(VERSION) --size $(SIZE) --exp e0 --data 475k

eval:           ## bucketed table for a checkpoint: make eval CKPT=artifacts/<run>_best.pt
	uv run python scripts/evaluate.py --model lstm --checkpoint $(CKPT) --size $(SIZE) --version $(VERSION) --split test_a
	uv run python scripts/evaluate.py --model lstm --checkpoint $(CKPT) --size $(SIZE) --version $(VERSION) --split test_b

eval-uniform:   ## proves the ruler: perplexity must equal vocab size
	uv run python scripts/evaluate.py --model uniform --version $(VERSION) --split test_a

grad-check:     ## M4 NumPy LSTM vs torch, 1e-6
	uv run python -m aagebatao.models.check_grad

numpy-lstm:     ## M4 loss curve
	uv run python scripts/train_numpy.py --version $(VERSION)

# ---- P1: summary and demo -------------------------------------------------------------
.PHONY: summary demo bench
CKPT ?= artifacts/e0_lstm_w_m_475k_real_s42_best.pt
PORT ?= 8000
DATA ?= 475k

summary:        ## one table + figs A/B/C from everything in reports/results and artifacts
	uv run python scripts/summarize_p1.py --version $(VERSION) --data $(DATA)

demo:           ## make demo CKPT=artifacts/<run>_best.pt SIZE=m VERSION=v0.1
	AAGEBATAO_CHECKPOINT=$(CKPT) AAGEBATAO_SIZE=$(SIZE) AAGEBATAO_VERSION=$(VERSION) \
	uv run uvicorn aagebatao.serve.app:app --host 127.0.0.1 --port $(PORT)

bench:          ## p50 / p95 latency of the predictor, in-process
	uv run python scripts/bench_latency.py --checkpoint $(CKPT) --size $(SIZE) --version $(VERSION) --out reports/results/p1_latency_$(VERSION).json

test-report:    ## run the suite + lint and write reports/test_report.md
	uv run python scripts/test_report.py

# ---- Deploy: Docker image / Google Cloud Run ----------------------------------------------
.PHONY: docker-build docker-run deploy
IMAGE ?= aagebatao
SERVICE ?= aagebatao
REGION ?= asia-south1

docker-build:   ## build the serving image locally (needs Docker)
	docker build -t $(IMAGE) .

docker-run:     ## run the image on http://127.0.0.1:8080
	docker run --rm -p 8080:8080 $(IMAGE)

deploy:         ## build remotely and deploy to Cloud Run: make deploy REGION=asia-south1
	gcloud run deploy $(SERVICE) --source . --region $(REGION) --allow-unauthenticated \
	  --cpu 1 --memory 1Gi --min-instances 0 --max-instances 3 --port 8080
