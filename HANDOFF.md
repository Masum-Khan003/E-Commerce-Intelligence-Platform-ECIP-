# E-CIP v3.0 — Project Handoff Document
**Date:** 2026-07-21
**Purpose:** Pick up tomorrow with GPU training for Module 1 (Product) and Module 2
(Sentiment) on Colab/Kaggle — the one remaining piece of the whole platform.

---

## 1. Project Overview

E-Commerce Intelligence Platform (E-CIP) v3.0 is a production-grade, end-to-end ML system
built as a senior ML engineering portfolio project: three ML modules behind one FastAPI
backend and a Next.js dashboard.

**Three intelligent modules:**
- Module 1 — Product Intelligence Engine (EfficientNet-B3 image classifier)
- Module 2 — Sentiment Intelligence Engine (DistilBERT fine-tuned + zero-shot ABSA)
- Module 3 — Retention Prediction Engine (XGBoost + LightGBM ensemble)

**Target metrics:**
- EfficientNet-B3 Top-1 accuracy ≥ 92%, Macro F1 ≥ 0.90, inference p95 < 120ms
- DistilBERT Macro F1 ≥ 0.88, Negative Recall ≥ 0.85, inference p95 < 50ms
- ABSA (zero-shot NLI) F1 ≥ 0.72 on SemEval-2014 laptop domain
- XGBoost/LightGBM ROC-AUC ≥ 0.87, ECE < 0.05, inference p95 < 12ms

**Blueprint documents** (read-only, in project workspace): `ecip_v3_blueprint.html` (full
architecture spec), `ecip_production_plan.html` (week-by-week execution plan).

---

## 2. Repository Location

`/home/mak/_project/E-CIP/ecip/` — git repo, `main` branch, clean, 45+ commits.

**Virtual environment:** `/home/mak/_project/E-CIP/ecip/.venv` — activate with
`source .venv/bin/activate`.

---

## 3. Current State — What's Actually Done

**Everything except GPU training is complete and verified for real** (not just written —
actually run against real data/services and confirmed working):

| Module | Status | Real metrics |
|---|---|---|
| **Retention** (XGBoost + LightGBM) | ✅ Trained on real UCI Online Retail II data | CV ROC-AUC **0.912**, calibrated ECE **0.014** |
| **Product** (EfficientNet-B3) | ⏳ Code complete, **GPU training pending** | — |
| **Sentiment** (DistilBERT + ABSA) | ⏳ Code complete, **GPU training pending** | — |

**Infrastructure, all built and verified end-to-end:**
- FastAPI backend (`api/`) — startup warm-up loader, bcrypt auth (Redis-cached), Celery
  (gpu/cpu/maintenance queues), Prometheus metrics, all `/v1/*` endpoints for all 3 modules
- PostgreSQL schema (5 tables), real prediction logging, review queue, drift events
- `mlops/` — Optuna search, feature drift detection (PSI/KS), Celery Beat schedule,
  promotion gate, retraining GitHub Actions workflow
- Next.js dashboard (`dashboard/`) — 6 pages (Overview, Product/Sentiment/Retention
  Analytics, Review Queue, Drift Monitor), BFF proxy so the API key never reaches the
  browser, ROI estimator
- `docker-compose.yml` — full 8-service production stack, built and run successfully
- `tests/` — 24 unit + 7 integration + 4 cross-module tests, all passing; k6 load tests for
  all 3 modules

**Why Product/Sentiment show empty on the dashboard right now:** `api/main.py`'s warm-up
loader checks for real weight files (`models/product/weights/efficientnet_b3_best.pt`,
`models/sentiment/weights/distilbert_sentiment_best.pt`) and honestly reports
`"missing — requires GPU training on Colab/Kaggle"` when they don't exist, rather than
faking a loaded state. The dashboard's Product/Sentiment Analytics pages read that status
directly and show an explanatory empty state. **No code changes are needed once training is
done** — drop the weight files in place, restart the API, and both the API and dashboard
light up automatically.

---

## 4. Tech Stack (as actually installed and verified)

| Layer | Tools |
|---|---|
| Language | Python 3.12.3 |
| Dev stack | Docker (PostgreSQL 16, Redis 7, MLflow 2.12.1) |
| ML frameworks | PyTorch 2.13, HuggingFace Transformers 5.14, XGBoost 3.3, LightGBM 4.7, SHAP 0.52 |
| API | FastAPI 0.139 + Celery 5.6 + Redis |
| Frontend | Next.js 14.2 (App Router, TypeScript, Tailwind) |
| Observability | Prometheus + Alertmanager + Grafana |
| Linting/typing | ruff, mypy (`explicit_package_bases = true` — see §7) |

**Docker dev stack start:**
```bash
docker compose -f docker-compose.dev.yml up -d
```
Services: PostgreSQL `localhost:5432` (db=ecip, user=ecip, pass=ecip_dev), Redis
`localhost:6379`, MLflow UI `http://localhost:5000`.

**Full production stack** (8 services — verified building and running clean):
```bash
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up -d
```

**Run the API + dashboard locally (two terminals):**
```bash
# Terminal 1
source .venv/bin/activate
uvicorn api.main:app --reload

# Terminal 2
cd dashboard && npm run dev
```
`dashboard/.env.local` already has a working API key and URL for local dev.

---

## 5. Code Quality Rules (NON-NEGOTIABLE — unchanged, still enforced)

```bash
ruff check --fix .
ruff check .
mypy . --ignore-missing-imports
```

**Known gotchas already hit and fixed in this codebase** (don't reintroduce):
- Never use `X` or `l` as variable names (N803/N806/E741); `import torch.nn.functional as F`
  is banned, use `as functional` (N812); class names must be CapWords (N801).
- `mypy` needs `explicit_package_bases = true` + `mypy_path = "."` in `pyproject.toml` —
  the repo has no `__init__.py` files anywhere, so without this, two same-named modules in
  different packages (e.g. `api/schemas/explain.py` vs `api/routers/explain.py`) collide
  under mypy's default module naming.
- `shap` must stay pinned `>=0.52` — older versions can't parse this XGBoost version's
  `base_score` format (`ValueError: could not convert string to float: '[5E-1]'`).
- `great-expectations` is pinned `>=1.0` — `shap>=0.52` requires `numpy>=2`, which conflicts
  with `great-expectations<1.0`'s `numpy<2.0` pin. Verified safe: `data/validation/setup_ge.py`
  doesn't actually import the `great_expectations` library, it's pure config scaffolding.
- Any code that needs Postgres/Redis reads `POSTGRES_DSN` / `REDIS_URL` /
  `REDIS_URL_AUTH` env vars (falling back to `localhost` for local dev) — never hardcode
  `localhost`, it breaks the moment that code runs inside a container where `localhost`
  means the container itself, not sibling service containers.

---

## 6. Commit Convention

```
feat(p{phase}-w{week}): description
fix: description
docs: description
chore: description
```

---

## 7. IMMEDIATE NEXT TASK — Train Module 1 + Module 2 on Colab/Kaggle

This is the one thing left. Both modules' code, data pipelines, evaluation scripts, and API
serving paths are already built and tested — they just need real trained weights.

### 7.0 Prerequisites checklist

- [ ] **Kaggle API credentials** — Module 1's primary dataset (Products-10K) downloads via
  the Kaggle CLI, which needs `~/.kaggle/kaggle.json` (API token from kaggle.com/settings).
  Without this, use the FEIDEGGER backup dataset instead (see `data/scripts/download.py`,
  `git_clone` method, no auth needed).
- [ ] **Google account** with Colab access (free tier works, Colab Pro speeds things up).
- [ ] **This repo pushed to GitHub** (or otherwise accessible from Colab) so the Colab
  notebook can `git clone` it.

### 7.1 Module 1 — Product Intelligence (EfficientNet-B3)

**Local, before Colab:**
```bash
source .venv/bin/activate
python data/scripts/download.py --module 1        # Products-10K via Kaggle (or FEIDEGGER backup)
python data/pipelines/image_pipeline.py            # validates images, builds train/val/test splits
python models/product/baseline_resnet18.py         # quick baseline, runs fine on CPU (~45 min)
```

**On Colab (GPU required for the real EfficientNet-B3 run):**
```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/YOUR_USERNAME/ecip.git
%cd ecip
%pip install -q -e ".[train]"

import mlflow, os
os.makedirs("/content/drive/MyDrive/ecip_mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:///content/drive/MyDrive/ecip_mlruns")

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — switch runtime to GPU'}")
```
Then either `dvc pull` / re-download the dataset into the Colab environment, or upload
`data/processed/images/` from the local run above (zip it, upload to Drive, unzip in Colab).

```python
!python models/product/train.py --data-dir data/processed/images
# 2-phase training: 10 epochs frozen backbone, 20 epochs fine-tune. ~2-4 hrs on a T4.
# Checkpoints saved every epoch that improves — safe to resume if the Colab session drops.
```

**Verify before pulling the weights back:**
```bash
python models/product/evaluate.py --model-path models/product/weights/efficientnet_b3_best.pt
python models/product/gradcam.py   # spot-check ~20 heatmaps per category — they should
                                     # highlight the actual product, not the background
```

**Bring the trained weights home:**
```bash
# Copy efficientnet_b3_best.pt from Colab's Drive-synced output back to:
models/product/weights/efficientnet_b3_best.pt
# And the OOD reference (Mahalanobis distances), if generated:
data/feature_store/product_features/mahalanobis_reference_v1.json
```

### 7.2 Module 2 — Sentiment Intelligence (DistilBERT + ABSA)

**Local, before Colab:**
```bash
python data/scripts/download.py --module 2        # Amazon Reviews 2023 — Electronics + Fashion
python data/pipelines/text_pipeline.py             # cleans reviews, saves tokenizer artifact
python models/sentiment/setfit_baseline.py --data data/processed/reviews/train_reviews.csv
```

**On Colab** (same bootstrap cell as above), then:
```python
!python models/sentiment/finetune.py --data data/processed/reviews
# DistilBERT + Focal Loss, head+tail 512-token truncation. ~1-2 hrs on a T4.
```

**Verify:**
```bash
python models/sentiment/evaluate.py --model-path models/sentiment/weights/distilbert_sentiment_best.pt
# Checks domain-shift F1 on 4 out-of-domain categories — flag any category with F1 < 0.78.
python models/sentiment/absa_pipeline.py   # zero-shot ABSA, no training needed —
                                             # runs on CPU, evaluate against SemEval-2014
```

**Bring home:**
```bash
models/sentiment/weights/distilbert_sentiment_best.pt
data/feature_store/artifacts/tokenizer_v1/   # the WHOLE directory — Fix #6: never
                                               # re-initialize the tokenizer from Hub at
                                               # inference, it must be this exact saved copy
```

### 7.3 After training — bring it all together

```bash
# Restart the API — it'll auto-detect the new weight files on next warm-up
uvicorn api.main:app --reload

# Verify both modules report "loaded"
curl http://localhost:8000/health/ready

# Update the model cards with real numbers (currently say "TBD after GPU training")
#   models/product/model_card.md
#   models/sentiment/model_card.md

# The dashboard's Product/Sentiment Analytics pages will show real data automatically
# once real predictions get logged — no code changes needed (see §3 above).
```

---

## 8. Key Design Decisions (unchanged, still binding)

**Churn label:** 90-day horizon, UK-only, snapshot_date=2010-11-30. See
`docs/decisions/ADR-001-churn-label.md`.

**SMOTE placement:** INSIDE the CV loop, training fold only — verified correct in
`models/retention/train.py`.

**Tokenizer:** Saved to `data/feature_store/artifacts/tokenizer_v1/` at training time.
NEVER call `AutoTokenizer.from_pretrained(hub_name)` at inference — `api/main.py` already
loads from this path.

**DVC vs MLflow boundary:** DVC owns raw data/processed features/reference distributions.
MLflow owns model weights/tokenizer/scaler/encoder artifacts.

**Retention's sentiment fusion is SYNTHETIC** (documented in
`models/retention/model_card.md` Known Limitations) — UCI Online Retail II has no review
text, so there's no real linkage to Module 2. This does not change once Module 2 is trained;
they're genuinely different datasets with no shared customer IDs.

**Grad-CAM TTL:** 1 hour, swept by Celery Beat every 30 minutes.
`storage/gradcam/{request_id}.png` (gitignored — Docker volume mount).

---

## 9. Known, Documented Gaps (not hidden, see model cards / README for detail)

- Retention's `/v1/retention/score` doesn't yet meet its 12ms p95 SLO under 20 concurrent
  VUs on a single dev machine (real fixes applied — `asyncio.to_thread` offload, background
  task logging — real improvement measured: ~1.7s → ~710ms p95 with 4 worker processes;
  still short of 12ms). See `tests/load/retention_slo.js` and the model card's "Performance
  Under Load" section.
- No Celery Prometheus exporter — `observability/alertmanager/rules.yml`'s
  `CeleryQueueBacklog` rule is unwired (would need a `celery-exporter` sidecar).
- Deployment (Render/Vercel/Upstash) is prep-only — configs and docs exist
  (`docs/deployment_guide.md`), no account-bound deployment has been attempted.

---

## 10. Quick Reference — Every Verified Command

```bash
# Environment
source .venv/bin/activate

# Dev services
docker compose -f docker-compose.dev.yml up -d

# Quality gate
ruff check --fix . && ruff check . && mypy . --ignore-missing-imports

# Tests
pytest tests/ -v --timeout=120

# API
uvicorn api.main:app --reload

# Dashboard
cd dashboard && npm run dev

# Create a dev API key
python data/scripts/create_api_key.py --name dev-local

# Retention pipeline (already run once — real trained artifacts exist)
python models/retention/churn_label_engineer.py
python data/scripts/synthesize_demo_sentiment.py
python data/pipelines/tabular_pipeline.py
python mlops/optuna_search.py --n-trials 50 --data data/feature_store/customer_features/rfm_behavioral_v2.parquet
python models/retention/train.py --data data/feature_store/customer_features/rfm_behavioral_v2.parquet
python models/retention/calibrate.py --data data/feature_store/customer_features/rfm_behavioral_v2.parquet

# Feature drift check
python mlops/drift_detector.py --module retention --write-db

# Production stack
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up -d

# Load tests (native k6 binary — Docker's bridge networking doesn't route to host
# services in some sandboxed environments; if that's not an issue for you, the
# grafana/k6 Docker image works too)
k6 run --env API_URL=http://localhost:8000 --env API_KEY=<key> tests/load/retention_slo.js
```

---

*Handoff document rewritten 2026-07-21 after completing Weeks 11–18 (retention modeling,
API hardening, observability, dashboard, load testing, deployment prep). Pick up from:
§7 — Module 1 + Module 2 GPU training on Colab/Kaggle.*
