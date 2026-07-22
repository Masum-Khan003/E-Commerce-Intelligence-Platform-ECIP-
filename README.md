# E-CIP — E-Commerce Intelligence Platform

## Project Objective

E-CIP is a solo-built, production-grade ML platform that demonstrates end-to-end machine
learning engineering — not just model notebooks, but the full lifecycle: data validation,
training, calibration, explainability, serving, observability, and a live operations
dashboard. It follows an 18-week phased build against a 47-gap "blueprint fix" hardening
pass (documented decisions, not shortcuts).

## Scope

Three intelligent modules behind a single FastAPI backend and a Next.js operations
dashboard:

1. **Product Intelligence** — EfficientNet-B3 image classifier for product category
   prediction, with Grad-CAM explainability and Mahalanobis-distance OOD detection.
2. **Sentiment Intelligence** — DistilBERT fine-tuned for review sentiment + zero-shot NLI
   aspect-based sentiment analysis (ABSA).
3. **Retention Prediction** — XGBoost + LightGBM ensemble on RFM/behavioral features,
   calibrated and SHAP-explained, predicting 90-day customer churn.

Supporting infrastructure: PostgreSQL prediction logging, Redis-cached bcrypt auth, Celery
task queues, Prometheus/Alertmanager observability, feature drift detection (PSI/KS), and a
Next.js dashboard with a BFF proxy so the API key never reaches the browser.

> **Current status: Product and Sentiment need one remaining step — GPU training on
> Colab/Kaggle.** Everything else (data pipelines, training/eval code, API serving paths,
> dashboard, tests, infra) is built and verified. Retention is fully trained on real data.
> See [`HANDOFF.md`](HANDOFF.md) for the exact next steps and full build history.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Next.js Dashboard (dashboard/)                                          │
│  BFF proxy — API key stays server-side, never reaches the browser        │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 │  X-API-Key (bcrypt-verified, Redis-cached)
┌────────────────────────────────▼──────────────────────────────────────────┐
│  FastAPI (api/)                                                          │
│  ┌───────────────┐  ┌──────────────────┐  ┌────────────────────────┐     │
│  │ Module 1       │  │ Module 2          │  │ Module 3                │  │
│  │ Product        │  │ Sentiment         │  │ Retention               │  │
│  │ EfficientNet-B3│  │ DistilBERT + ABSA │  │ XGBoost + LightGBM      │  │
│  │ (pending GPU   │  │ (pending GPU      │  │ ensemble — trained,     │  │
│  │  training)     │  │  training)        │  │ calibrated, SHAP-explained│ │
│  └───────────────┘  └──────────────────┘  └────────────────────────┘     │
│  Startup warm-up loader · Celery (gpu/cpu/maintenance queues) · Prometheus │
└───────┬───────────────────────────────────────┬───────────────────────────┘
        │                                        │
┌───────▼─────────┐  ┌──────────────┐  ┌─────────▼─────────┐
│ PostgreSQL       │  │ Redis        │  │ MLflow              │
│ prediction_logs, │  │ auth cache,  │  │ experiment tracking, │
│ review_queue,    │  │ Celery broker│  │ model artifacts       │
│ drift_events     │  └──────────────┘  └────────────────────┘
└──────────────────┘
```

Data flow: `data/pipelines/` (image/text/tabular) → Great Expectations validation gates →
`data/feature_store/` (Parquet) → model training (`models/*/train.py`) → MLflow-tracked
artifacts → served by `api/main.py`'s warm-up loader.

## Module status

| Module | Status | Real metrics |
|---|---|---|
| **Retention** (XGBoost + LightGBM) | Trained on real data | CV ROC-AUC 0.912, calibrated ECE 0.014 — see [model card](models/retention/model_card.md) |
| **Product** (EfficientNet-B3) | Structure complete, **GPU training pending** | — see [model card](models/product/model_card.md) |
| **Sentiment** (DistilBERT + ABSA) | Structure complete, **GPU training pending** | — see [model card](models/sentiment/model_card.md) |

Product and Sentiment require GPU training on Colab/Kaggle (see `HANDOFF.md` §7) — out of
scope for a local CPU session. Retention is CPU-only and fully trained end-to-end against the
real [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset.

## Local setup

Requires Python 3.12, Docker, and Node 18+.

```bash
# 1. Install Python dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,train]"

# 2. Start PostgreSQL + Redis + MLflow
docker compose -f docker-compose.dev.yml up -d

# 3. Download the retention dataset (UCI Online Retail II, ~45MB, no auth needed)
mkdir -p data/raw/online_retail2
wget -c "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip" -P data/raw/online_retail2/
cd data/raw/online_retail2 && unzip -o "online+retail+ii.zip" && cd -

# 4. Run the full retention pipeline (churn labels -> features -> tune -> train -> calibrate)
python models/retention/churn_label_engineer.py
python data/scripts/synthesize_demo_sentiment.py   # see model card — synthetic, documented
python data/pipelines/tabular_pipeline.py
python mlops/optuna_search.py --n-trials 50 --data data/feature_store/customer_features/rfm_behavioral_v2.parquet
python models/retention/train.py --data data/feature_store/customer_features/rfm_behavioral_v2.parquet
python models/retention/calibrate.py --data data/feature_store/customer_features/rfm_behavioral_v2.parquet

# 5. Create a dev API key
python data/scripts/create_api_key.py --name dev-local

# 6. Start the API
uvicorn api.main:app --reload
# -> http://localhost:8000/docs (Swagger), http://localhost:8000/health/ready

# 7. Start the dashboard (separate terminal)
cd dashboard
npm install
cp .env.local.example .env.local   # fill in ECIP_API_KEY from step 5
npm run dev
# -> http://localhost:3000
```

## Training Product and Sentiment (GPU required)

Not runnable locally — both need a GPU. Full step-by-step instructions (dataset prep,
Colab bootstrap cells, training commands, evaluation gates, bringing weights back into the
repo) are in [`HANDOFF.md` §7](HANDOFF.md). Once the two weight files exist at
`models/product/weights/efficientnet_b3_best.pt` and
`models/sentiment/weights/distilbert_sentiment_best.pt`, the API's warm-up loader and the
dashboard pick them up automatically — no code changes needed.

## Tests

```bash
ruff check . && mypy . --ignore-missing-imports
pytest tests/unit/ tests/integration/ tests/model_tests/ -v --timeout=120
```

Integration and model tests need the Docker stack (step 2) and a trained retention ensemble
(step 4) — they skip gracefully (not fail) if either is missing.

## Deployment

See [`docs/deployment_guide.md`](docs/deployment_guide.md) for the production
`docker-compose.yml` (8-service stack with memory limits) and cloud deployment steps
(Render/Vercel/Upstash free tiers).

## Project structure

- `data/` — pipelines, Great Expectations validation, feature store
- `models/` — training/calibration/explainability scripts + model cards per module
- `mlops/` — Optuna search, drift detection, Celery Beat schedule, promotion gate
- `api/` — FastAPI app, routers, auth middleware, Celery tasks
- `observability/` — Prometheus metrics, Alertmanager rules
- `dashboard/` — Next.js operations dashboard
- `tests/` — unit, integration, cross-module, and k6 load tests
- `docs/decisions/` — architecture decision records (ADRs)

## Documentation

- [`HANDOFF.md`](HANDOFF.md) — full build history, phase-by-phase status, next steps
- [`docs/decisions/ADR-001-churn-label.md`](docs/decisions/ADR-001-churn-label.md) — churn
  label definition rationale (90-day horizon, UK-only scope)
- Model cards: [`models/retention/model_card.md`](models/retention/model_card.md),
  [`models/product/model_card.md`](models/product/model_card.md),
  [`models/sentiment/model_card.md`](models/sentiment/model_card.md)
