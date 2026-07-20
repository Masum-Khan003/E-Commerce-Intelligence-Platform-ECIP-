# E-CIP — E-Commerce Intelligence Platform

A production-grade, end-to-end ML system with three intelligent modules behind a single
FastAPI backend and a Next.js operations dashboard: automated product image classification,
transformer-based review sentiment + aspect analysis, and explainable customer retention
(churn) prediction.

Built solo against a 47-gap-hardened production blueprint. See [`HANDOFF.md`](HANDOFF.md) for
the full build history and current status of every module.

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
| **Product** (EfficientNet-B3) | Structure complete, GPU training pending | — see [model card](models/product/model_card.md) |
| **Sentiment** (DistilBERT + ABSA) | Structure complete, GPU training pending | — see [model card](models/sentiment/model_card.md) |

Product and Sentiment require GPU training on Colab/Kaggle (see `HANDOFF.md` §14) — out of
scope for a local session. Retention is CPU-only and fully trained end-to-end against the
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
