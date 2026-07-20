# E-CIP v3.0 — Project Handoff Document
**Date:** 2026-07-21  
**Handoff From:** Claude (claude.ai conversation)  
**Handoff To:** Claude Code  
**Purpose:** Continue building E-CIP v3.0 from Phase 4, Week 11

---

## 1. Project Overview

E-Commerce Intelligence Platform (E-CIP) v3.0 is a production-grade,
end-to-end ML system built as a senior ML engineering portfolio project.

**Three intelligent modules:**
- Module 1 — Product Intelligence Engine (EfficientNet-B3 image classifier)
- Module 2 — Sentiment Intelligence Engine (DistilBERT fine-tuned + zero-shot ABSA)
- Module 3 — Retention Prediction Engine (XGBoost + LightGBM ensemble)

**Constraints:** 100% free tools, solo developer, ~18-week timeline.

**Target metrics:**
- EfficientNet-B3 Top-1 accuracy ≥ 92%
- DistilBERT Macro F1 ≥ 0.88
- XGBoost/LightGBM ROC-AUC ≥ 0.87, ECE < 0.05

**Blueprint documents** (read-only, in project workspace):
- `ecip_v3_blueprint.html` — full architecture specification
- `ecip_production_plan.html` — week-by-week execution plan

---

## 2. Repository Location

/home/mak/_project/E-CIP/ecip/

**Git status:** clean, 25+ commits on `main` branch.

**Virtual environment:** `/home/mak/_project/E-CIP/ecip/.venv`

**Activate:** `source .venv/bin/activate`

---

## 3. Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12.3 |
| Dev stack | Docker (PostgreSQL 16, Redis 7, MLflow 2.12.1) |
| Data versioning | DVC (local remote at /tmp/ecip-dvc-storage) |
| Data validation | Great Expectations 0.18.19 |
| ML frameworks | PyTorch 2.2, HuggingFace Transformers 4.39, XGBoost 2.x, LightGBM 4.x |
| HPO | Optuna 3.x (SQLite persistence) |
| Experiment tracking | MLflow (local SQLite) |
| API | FastAPI + Celery + Redis |
| Linting/typing | ruff 0.15.x, mypy 2.1.0 |
| GPU training | Google Colab / Kaggle (free tiers) |

**Docker dev stack start:**
```bash
docker compose -f docker-compose.dev.yml up -d
```

**Services:**
- PostgreSQL: localhost:5432 (db=ecip, user=ecip, pass=ecip_dev)
- Redis: localhost:6379
- MLflow UI: http://localhost:5000

---

## 4. Code Quality Rules (NON-NEGOTIABLE)

Every code block must pass before committing:

```bash
ruff check --fix .
ruff check .
mypy <file>.py --ignore-missing-imports
```

**Known ruff rules to pre-empt:**
- Never use `X` or `l` as variable names (N803, N806, E741)
- Never use `import torch.nn.functional as F` — use `as functional` (N812)
- Class names must be CapWords — `_NullContext` not `_null_context` (N801)
- No unused imports (F401)
- Always add trailing newline to files (W292)

**Known mypy rules to pre-empt:**
- Type all `dict` constants explicitly: `dict[str, Any]`
- Cast `Any` returns explicitly before returning from typed functions
- Use `list[float]` not bare `list` for typed collections
- HPARAMS dict values accessed as `int(HPARAMS["key"])` when int expected

**One block at a time strategy:**
Write → run → ruff → mypy → commit. Never proceed to next block
until current block is clean.

---

## 5. Commit Convention

feat(p{phase}-w{week}): description
fix: description
docs: description
chore: description

Examples from history:
- feat(p4-w11): XGBoost+LightGBM train — SMOTE inside CV Fix #10
- feat(p4-w11): Optuna search — SQLite persistence Fix #19 Fix #21

---

## 6. Directory Structure (Current State)

ecip/
├── data/
│ ├── pipelines/
│ │ ├── image_pipeline.py ✓ Complete
│ │ ├── text_pipeline.py ✓ Complete
│ │ └── tabular_pipeline.py ✓ Complete
│ ├── scripts/
│ │ ├── download.py ✓ Complete
│ │ └── verify_access.py ✓ Complete
│ ├── validation/
│ │ └── setup_ge.py ✓ Complete (8-gate GE framework)
│ ├── feature_store/
│ │ ├── artifacts/
│ │ │ └── tokenizer_v1/ ✓ Stub (populated after download)
│ │ └── product_features/
│ │ └── transform_spec.json ✓ Complete
│ └── reference_distributions/
│ └── customer_features_ref_v1.json ✓ Stub
├── models/
│ ├── product/
│ │ ├── baseline_resnet18.py ✓ Complete
│ │ ├── train.py ✓ Complete
│ │ ├── evaluate.py ✓ Complete
│ │ ├── gradcam.py ✓ Complete
│ │ └── model_card.md ✓ Complete (metrics TBD after training)
│ ├── sentiment/
│ │ ├── setfit_baseline.py ✓ Complete
│ │ ├── finetune.py ✓ Complete
│ │ ├── absa_pipeline.py ✓ Complete
│ │ ├── evaluate.py ✓ Complete
│ │ └── model_card.md ✓ Complete (metrics TBD after training)
│ └── retention/
│ └── churn_label_engineer.py ✓ Complete
├── mlops/
│ └── optuna_search.py ⚠ INCOMPLETE — mypy error on line 316
├── api/
│ ├── routers/
│ │ ├── products.py ✓ Complete
│ │ └── sentiment.py ✓ Complete
│ └── storage/
│ └── image_store.py ✓ Complete
├── db/
│ └── schema.sql ✓ Complete (5 tables)
├── docs/
│ └── decisions/
│ └── ADR-001-churn-label.md ✓ Complete
├── .github/
│ └── workflows/
│ └── ci.yml ✓ Complete (skeleton)
├── docker-compose.dev.yml ✓ Complete
├── dvc.yaml ✓ Complete (skeleton)
└── pyproject.toml ✓ Complete

---

## 7. What Was Completed (Phases 0–3)

### Phase 0 — Foundation (Weeks 1–2) ✓
- Repository structure, `.gitignore`, `pyproject.toml`
- Docker dev stack (PostgreSQL + Redis + MLflow)
- PostgreSQL schema (5 tables with GIN indexes)
- DVC initialised with local remote
- Dataset access verification — 7/7 datasets confirmed accessible
- Great Expectations 8-gate validation framework (G1–G8)
- Churn label engineer — 90-day horizon, UK scope (Critical Fix #2, #5)
- Dataset download script (all methods: kaggle, wget, git_clone)
- ADR-001: churn label definition documented
- CI skeleton with pip caching

### Phase 1 — Data Pipelines (Weeks 3–4) ✓
- `image_pipeline.py` — validate_image() Fix #3, SHA256 dedup, stratified split
- `text_pipeline.py` — head+tail truncation Fix #12, tokenizer artifact Fix #6
- `tabular_pipeline.py` — RFM features, NaN-safe gap CV Fix #15,
  time_decay_lambda as parameter Fix #21, UK scope Fix #24
- Reference distributions saved for drift detection baseline

### Phase 2 — Product Intelligence Engine (Weeks 5–7) ✓
- `baseline_resnet18.py` — ResNet-18 baseline Fix #16
- `train.py` — EfficientNet-B3 two-phase training, get_device() Fix #18
- `evaluate.py` — per-class F1, confusion matrix, ECE, OOD Fix #20
- `gradcam.py` — Grad-CAM heatmap, TTL storage Fix #9 Fix #30
- `image_store.py` — upload validation, storage backend
- `api/routers/products.py` — /v1/products/classify endpoint Fix #38
- `models/product/model_card.md`

### Phase 3 — Sentiment Intelligence Engine (Weeks 8–10) ✓
- `setfit_baseline.py` — 100-shot SetFit Fix #17
- `finetune.py` — DistilBERT + Focal Loss + head+tail Fix #6 Fix #12
- `absa_pipeline.py` — zero-shot NLI ABSA, 6 aspects Fix #3
- `evaluate.py` — subgroup analysis, domain-shift Fix #32
- `api/routers/sentiment.py` — /v1/sentiment/analyze endpoint
- `models/sentiment/model_card.md`

---

## 8. Immediate Next Task — Fix mlops/optuna_search.py

**Problem:** One remaining mypy error on line 316.

**Diagnosis command:**
```bash
sed -n '310,325p' mlops/optuna_search.py
```

**Root cause:** A `return {}` inside a try/except block returns
`dict[Any, Any]` instead of `dict[str, Any]`. Fix pattern:

```python
# Wrong:
return {}

# Correct:
empty: dict[str, Any] = {}
return empty
```

**After fixing, verify:**
```bash
ruff check --fix .
ruff check .
mypy mlops/optuna_search.py --ignore-missing-imports
```

**Then commit:**
```bash
git add .
git commit -m "fix(p4-w11): resolve mypy no-any-return in optuna_search.py"
```

---

## 9. Remaining Build Plan

### Phase 4 — Retention Engine (Weeks 11–13)

#### Week 11 — Remaining after Optuna fix

**Block: `models/retention/train.py`**

Implements:
- `train_with_smote_cv()` — XGBoost with SMOTE INSIDE CV loop (Critical Fix #10)
- `train_lgbm_cv()` — LightGBM equivalent
- `build_ensemble_prediction()` — weighted average of XGBoost + LightGBM
- Two-phase training: Optuna best params → full retrain on all folds
- MLflow experiment logging

Key constraints from blueprint:
```python
# SMOTE MUST be applied INSIDE the CV loop on training fold only
# Applying before split = data leakage (Critical Fix #10)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    # train on X_train_res — NEVER on full X before split
```

Also implement sentiment feature merge:
- Load `data/feature_store/text_features/review_sentiment_v1.parquet`
- Load `data/feature_store/text_features/aspect_sentiment_v1.parquet`
- Merge onto customer feature table with G8 causal integrity:
  `WHERE review_date < snapshot_date`
- Update `data/feature_store/customer_features/rfm_behavioral_v2.parquet`

#### Week 12 — Calibration + SHAP

**Block: `models/retention/calibrate.py`**
- Platt Scaling for n_calibration < 1000
- Isotonic Regression for n >= 1000
- Reliability diagram saved to MLflow
- Calibration set held out from training AND SMOTE

**Block: `models/retention/shap_explain.py`**
- TreeSHAP via `shap.TreeExplainer`
- `ShapExplanationResponse` Pydantic schema
- Top-10 features with direction and magnitude
- SHAP sum consistency check: |sum(shap) + expected - prediction| < 0.05

Subgroup evaluation:
- ROC-AUC by tenure quartile (0–90d, 91–180d, 181–365d, 365d+)
- Precision/Recall by frequency band (1, 2–5, 6–20, 20+)
- Single-purchase vs multi-purchase customer performance

#### Week 13 — Retention API + Cross-Module Tests

**Block: `api/routers/retention.py`**

Full API contract from blueprint Section 05:
```json
{
  "request_id": "req_c7e19230",
  "customer_id": "cust_10452",
  "churn_probability": 0.73,
  "risk_band": "HIGH",
  "recommended_action": "RETENTION_OFFER",
  "top_risk_factors": [...],
  "churn_label_definition": "no_purchase_90d",
  "is_single_purchase_customer": false,
  "model_version": "retention_ensemble_v1.2.0",
  "calibration_method": "platt_scaling",
  "decision_threshold": 0.55,
  "inference_ms": 12
}
```

**Block: `api/schemas/explain.py`** — ShapExplanationResponse schema

**Block: `tests/model_tests/test_cross_module.py`**

Four critical tests (blueprint Section 24):
```python
def test_negative_sentiment_increases_churn_risk()
    # GIVEN: two identical customers, different avg_sentiment_score
    # THEN: negative sentiment customer has STRICTLY higher churn_probability
    # This is the most important invariant in the system

def test_sentiment_score_range()
    # sentiment_score must always be in [-1, 1]

def test_shap_sum_consistency()
    # |sum(shap_values) + expected_value - prediction| < 0.05

def test_causal_integrity_no_future_sentiment()
    # Zero rows with review_date > snapshot_date in feature table
```

**Block: `models/retention/model_card.md`**

Subgroup specifications (blueprint Section 18):
- ROC-AUC by tenure quartile
- Precision/Recall by frequency band
- Single-purchase vs multi-purchase performance
- Calibration reliability diagram
- SHAP top-10 features with directionality

---

### Phase 5 — MLOps & API Hardening (Weeks 14–15)

#### Week 14

**Block: `api/main.py`** — FastAPI startup warm-up loader (Fix #7)
```python
# Lifespan context — loads all models at startup
# Readiness probe returns 503 until ALL models loaded
# Loading order: XGBoost → EfficientNet → DistilBERT (lightest to heaviest)
MODEL_URIS = {
    "efficientnet": "models:/product_classifier/Production",
    "distilbert":   "models:/sentiment_model/Production",
    "tokenizer":    "models:/sentiment_tokenizer/Production",
    "xgb_ensemble": "models:/retention_ensemble/Production",
    "calibrator":   "models:/retention_calibrator/Production",
    "scaler":       "models:/feature_scaler/Production",
    "mahal_ref":    "models:/ood_reference/Production",
}
```

**Block: `api/middleware/auth.py`** — bcrypt API key auth (Fix #14)
- Keys bcrypt-hashed in PostgreSQL api_keys table
- Redis 5-min cache to avoid bcrypt on every request

**Block: `api/workers/celery_tasks.py`** — per-queue Celery config (Fix #8)
```python
# GPU queue: concurrency=1, pool=solo
# CPU queue: concurrency=4, pool=prefork
# Maintenance queue: cleanup tasks
task_routes = {
    "workers.batch_classify_images":   {"queue": "gpu_queue"},
    "workers.batch_score_sentiment":   {"queue": "gpu_queue"},
    "workers.batch_score_retention":   {"queue": "cpu_queue"},
    "workers.cleanup_expired_gradcam": {"queue": "maintenance_queue"},
}
result_expires = 3600  # Fix #33
```

**Block: `mlops/drift_detector.py`** — feature drift detection (Fix #11)
- PSI + KS tests on INPUT feature distributions (not output)
- Compare against reference snapshots from tabular_pipeline.py
- Results written to drift_events PostgreSQL table
- Prometheus counter: ecip_feature_drift_total

#### Week 15

**Block: `observability/prometheus/metrics.py`** — all 6 metrics
```python
ecip_inference_latency_seconds  # histogram by module
ecip_prediction_confidence      # histogram by module
ecip_ood_flags_total            # counter
ecip_review_queue_depth         # gauge
ecip_feature_drift_total        # counter by feature
ecip_model_warmup_seconds       # histogram
```

**Block: `mlops/beat_schedule.py`** — Celery Beat cron (Fix #26)
```python
CELERYBEAT_SCHEDULE = {
    "daily-drift-check":    crontab(hour=2, minute=0),
    "gradcam-cleanup":      crontab(minute="*/30"),
    "weekly-perf-snapshot": crontab(day_of_week=1, hour=3),
}
```

**Block: `.github/workflows/model_retrain.yml`** — fully specified (Fix #44)
Steps: data pull → GE validate → Optuna 50 trials → train → promotion gate → promote or open Issue

**Block: `.github/workflows/ci.yml`** — expand with model tests
Add: `pytest tests/model_tests/ -v --timeout=120`

**Block: `docker-compose.yml`** — full production stack
All 8 services with mem_limit per blueprint Section 23:
- api: mem_limit 2g
- celery_gpu: mem_limit 2g
- celery_cpu: mem_limit 1g
- celery_beat: mem_limit 256m
- postgres: mem_limit 512m
- redis: mem_limit 256m

---

### Phase 6 — Dashboard & Portfolio (Weeks 16–18)

#### Week 16

**Block: `api/bff/`** — BFF proxy route handlers (Fix #23)
Next.js route handlers forward dashboard API calls server-side.
API key lives in server env — NEVER in browser.

**Block: `dashboard/`** — Next.js 14 app
Pages required (each with loading/success/error states — Fix #41):
- Overview: KPI cards, prediction volume, model version strip
- Product Analytics: category distribution, confidence histogram, Grad-CAM viewer
- Sentiment Analytics: sentiment trend, aspect radar, review volume by rating
- Retention Analytics: risk band distribution, SHAP bar chart, ROI estimator
- Review Queue: pending list, module filter, resolve/dismiss actions
- Drift Monitor: per-feature drift gauges, event timeline

**ROI Estimator formula (blueprint Section 16 — Fix #40):**
n_high_risk = customers with churn_prob > threshold
n_targeted = n_high_risk × reach_pct
n_saved = n_targeted × conversion_rate (default 25%)
revenue_saved = n_saved × avg_ltv (default £480)
campaign_cost = n_targeted × cost_per_offer (default £15)
net_roi = revenue_saved - campaign_cost

Client-side calculation only — no API call on input change.
3×3 sensitivity grid (low/mid/high conversion × low/mid/high LTV).

#### Week 17
- k6 load tests for all 3 modules (SLO verification)
- `tests/load/retention_slo.js` — p95 < 12ms @ 20 VUs
- `tests/load/product_slo.js` — p95 < 120ms @ 10 VUs
- `tests/load/sentiment_slo.js` — p95 < 50ms @ 10 VUs

#### Week 18
- Finalise all three model cards with actual metrics
- `README.md` with architecture diagram, 1-command setup, demo link
- Deploy API to Render free tier
- Deploy dashboard to Vercel free tier
- Pre-populate demo PostgreSQL with synthetic prediction logs
- Record 5-min Loom screencast demo
- GitHub Release v1.0.0

---

## 10. Blueprint Gap Fixes Still Pending

The following fixes from the 47-gap matrix are not yet implemented:

| Fix # | Description | Target File | Phase |
|---|---|---|---|
| #10 | SMOTE inside CV loop | models/retention/train.py | P4 W11 |
| #19 | Optuna SQLite persistence (mypy fix pending) | mlops/optuna_search.py | P4 W11 |
| #21 | time_decay_lambda in Optuna search | mlops/optuna_search.py | P4 W11 |
| #7  | FastAPI warm-up loader | api/main.py | P5 W14 |
| #8  | Celery per-queue concurrency | api/workers/celery_tasks.py | P5 W14 |
| #14 | bcrypt API key auth | api/middleware/auth.py | P5 W14 |
| #11 | Drift on feature distributions | mlops/drift_detector.py | P5 W14 |
| #26 | Celery Beat schedule | mlops/beat_schedule.py | P5 W15 |
| #44 | model_retrain.yml fully specified | .github/workflows/ | P5 W15 |
| #33 | Celery result_expires=3600 | api/workers/celery_tasks.py | P5 W14 |
| #23 | BFF proxy — API key server-side | api/bff/ | P6 W16 |
| #41 | Dashboard error/loading states | dashboard/ | P6 W16 |
| #40 | ROI estimator fully specified | dashboard/ | P6 W16 |
| #35 | k6 load tests | tests/load/ | P6 W17 |
| #45 | Cross-module integration tests | tests/model_tests/ | P4 W13 |

---

## 11. Key Design Decisions

**Churn label:** 90-day horizon, UK-only, snapshot_date=2010-11-30.
Documented in `docs/decisions/ADR-001-churn-label.md`.

**SMOTE placement:** INSIDE CV loop on training fold only.
Applying before split = data leakage (Critical Fix #10).

**Tokenizer:** Saved to MLflow artifact at training time.
NEVER call `AutoTokenizer.from_pretrained(hub_name)` at inference.

**DVC vs MLflow boundary:**
- DVC owns: raw data, processed features, reference distributions
- MLflow owns: model weights, tokenizer, scaler, encoder artifacts

**Drift detection:** PSI + KS on INPUT feature distributions.
Reference snapshots saved by tabular_pipeline.py at training time.

**OOD detection:** Mahalanobis distance on EfficientNet penultimate layer.
Threshold = 99th percentile of training set distances.

**Grad-CAM TTL:** 1 hour. Cleanup via Celery Beat every 30 minutes.
Storage at `storage/gradcam/{request_id}.png` (Docker volume mount).

**UK-only scope:** UCI Online Retail II scoped to UK customers.
Avoids mixed-currency monetary_value corruption.

---

## 12. Important File Paths

```bash
# Activate environment
source /home/mak/_project/E-CIP/ecip/.venv/bin/activate

# Start Docker services
cd /home/mak/_project/E-CIP/ecip
docker compose -f docker-compose.dev.yml up -d

# Run verification scripts
python data/scripts/verify_access.py      # 7/7 datasets
python data/validation/setup_ge.py        # GE framework

# Check code quality
ruff check --fix . && ruff check . && mypy . --ignore-missing-imports

# Run tests
pytest tests/ -v

# Git log
git log --oneline | head -30
```

---

## 13. Dataset Download Commands (when ready)

```bash
# Module 1 — Products
kaggle datasets download -d hirune924/products10k -p data/raw/products10k --unzip

# Module 2 — Amazon Reviews (Electronics + Fashion)
wget -c "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz" -P data/raw/amazon_reviews/
wget -c "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz" -P data/raw/amazon_reviews/

# Module 3 — UCI Online Retail II
wget -c "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip" -P data/raw/online_retail2/
cd data/raw/online_retail2 && unzip -o "online+retail+ii.zip"
```

---

## 14. GPU Training Strategy

EfficientNet-B3 and DistilBERT require GPU (Colab/Kaggle).
Everything else (tabular, API, tests) runs on local CPU.

**Colab setup cell:**
```python
from google.colab import drive
drive.mount("/content/drive")
import subprocess
subprocess.run(["git", "clone", "https://github.com/YOUR_USER/ecip.git"])
%cd ecip
%pip install -q -e ".[train]"
import mlflow, os
os.makedirs("/content/drive/MyDrive/ecip_mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:///content/drive/MyDrive/ecip_mlruns")
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
```

---

*Handoff document generated 2026-07-21. Pick up from: fixing mlops/optuna_search.py mypy error, then models/retention/train.py.*