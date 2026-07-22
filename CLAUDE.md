# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

E-CIP v3.0 (E-Commerce Intelligence Platform) is a production-grade, end-to-end ML
system built solo as a portfolio project, following an 18-week phased plan. It has
three ML modules behind a FastAPI backend:

- **Module 1 — Product Intelligence** (`models/product/`, `api/routers/products.py`):
  EfficientNet-B3 image classifier, target Top-1 accuracy ≥ 92%.
- **Module 2 — Sentiment Intelligence** (`models/sentiment/`, `api/routers/sentiment.py`):
  DistilBERT fine-tuned + zero-shot NLI ABSA (aspect-based sentiment), target Macro F1 ≥ 0.88.
- **Module 3 — Retention Prediction** (`models/retention/`, `api/routers/retention.py`):
  XGBoost + LightGBM ensemble on RFM/behavioral features, target ROC-AUC ≥ 0.87, ECE < 0.05.

Read `HANDOFF.md` first — it is the authoritative, living project state document:
current phase/week, what's built vs. pending, the full 47-gap "blueprint fix" matrix,
and the exact next task. Update it as phases complete; do not let it go stale.

## Commands

```bash
# Activate environment (project uses a single .venv, not per-tool envs)
source .venv/bin/activate

# Start dev infra: PostgreSQL 16, Redis 7, MLflow 2.12.1 (UI at localhost:5000)
docker compose -f docker-compose.dev.yml up -d

# Code quality gate — required clean before every commit, run in this order
ruff check --fix .
ruff check .
mypy . --ignore-missing-imports        # or target a single file: mypy path/to/file.py --ignore-missing-imports

# Tests
pytest tests/ -v
pytest tests/unit/ -v --timeout=30
pytest tests/integration/ -v --timeout=60
pytest tests/model_tests/ -v --timeout=120
pytest path/to/test_file.py::test_name -v   # single test

# Data pipeline DAG (DVC)
dvc repro <stage_name>                 # download_datasets | image_pipeline | text_pipeline | tabular_pipeline
dvc repro                              # full DAG

# Data validation
python data/validation/setup_ge.py     # 8-gate Great Expectations framework
python data/scripts/verify_access.py   # dataset access check
```

Install extras as needed for the area you're touching: `pip install -e ".[train]"` (torch,
transformers, xgboost, optuna, mlflow, dvc, great-expectations, ...), `pip install -e ".[api]"`
(fastapi, celery, redis, bcrypt, prometheus-client, ...), or `pip install -e ".[dev]"` (pytest,
ruff, mypy).

## Non-negotiable code quality rules

Every code block must pass `ruff check --fix . && ruff check . && mypy <file> --ignore-missing-imports`
before it is committed. Work one block at a time: write → run → ruff → mypy → commit; don't
move to the next block until the current one is clean.

Ruff/mypy gotchas that have bitten this codebase before (pre-empt these, don't wait for CI):
- Never use `X` or `l` as variable names (N803/N806/E741).
- `import torch.nn.functional as F` is banned — use `as functional` (N812).
- Class names must be CapWords, including private ones — `_NullContext` not `_null_context` (N801).
- Type dict constants explicitly (`dict[str, Any]`), use `list[float]` not bare `list`, and
  cast/annotate before returning `Any` from a typed function (`warn_return_any = true` in mypy config)
  — e.g. a bare `return {}` inside try/except infers `dict[Any, Any]`; assign to an annotated
  variable first.
- Trailing newline required on every file (W292); no unused imports (F401).

`pyproject.toml` config: ruff targets py312, line-length 100, `E501` ignored; mypy has
`warn_return_any` and `warn_unused_configs` on.

## Commit convention

```
feat(p{phase}-w{week}): description
fix: description
docs: description
chore: description
```

Phase/week numbers track the plan in `HANDOFF.md` (e.g. `feat(p4-w11): XGBoost+LightGBM train
— SMOTE inside CV loop Fix #10`). "Fix #N" references the blueprint's 47-gap fix matrix (§10 of
`HANDOFF.md`) when a commit closes one of those specific items.

## Architecture

**Data flow / storage boundary** — this split is load-bearing, don't blur it:
- **DVC** owns raw data, processed features, and reference distributions
  (`data/raw/`, `data/processed/`, `data/reference_distributions/`).
- **MLflow** owns model weights, tokenizer, scaler, and encoder artifacts. The sentiment
  tokenizer in particular is saved to MLflow at training time — inference code must load it
  from there, **never** call `AutoTokenizer.from_pretrained(hub_name)` at inference time
  (that would silently drift from the version the model was trained/calibrated against).

**Data pipelines** (`data/pipelines/`) feed the DVC DAG (`dvc.yaml`): `image_pipeline.py`,
`text_pipeline.py`, `tabular_pipeline.py` each validate/dedup/split raw data into
`data/processed/` and populate `data/feature_store/`. Great Expectations gates (G1–G8, see
`data/validation/setup_ge.py`) enforce data quality including causal integrity — G8 requires
`review_date < snapshot_date` when sentiment features are merged onto the customer feature
table, preventing future information leaking into churn labels.

**Cross-module dependency**: Module 3 (retention) consumes Module 2 (sentiment) output.
`data/feature_store/text_features/{review_sentiment_v1,aspect_sentiment_v1}.parquet` get
merged onto `data/feature_store/customer_features/rfm_behavioral_v2.parquet`. The system's
most important invariant, enforced in `tests/model_tests/test_cross_module.py`, is that two
otherwise-identical customers with different sentiment scores must produce strictly different
(sentiment-correlated) churn probabilities — sentiment must actually move the retention model,
not just ride along as an unused feature.

**Churn label definition** (Module 3): 90-day no-purchase horizon, UK-only customers (avoids
mixed-currency `monetary_value` corruption in UCI Online Retail II), snapshot_date=2010-11-30.
Rationale in `docs/decisions/ADR-001-churn-label.md` — don't redefine this without an ADR update.

**SMOTE placement**: class imbalance resampling must happen *inside* the CV loop, on the
training fold only, never before the train/val split — doing it beforehand is data leakage
(blueprint Critical Fix #10) and will silently inflate validation metrics.

**OOD / drift detection**:
- Out-of-distribution: Mahalanobis distance on the EfficientNet penultimate layer; threshold =
  99th percentile of training-set distances.
- Feature drift (`mlops/drift_detector.py`): PSI + KS tests on *input* feature distributions
  (not model outputs), compared against reference snapshots written by `tabular_pipeline.py`
  at training time. Results land in the `drift_events` Postgres table.

**Model serving lifecycle**: `api/main.py` loads all models at startup via a lifespan
context; the readiness probe stays 503 until every model in `MODEL_URIS` is loaded, in order
lightest→heaviest (XGBoost → EfficientNet → DistilBERT). Low-confidence or OOD-flagged
predictions get routed to the `review_queue` Postgres table rather than returned as-is.

**Postgres schema** (`db/schema.sql`, auto-applied by the `postgres` container on first boot):
`api_keys` (bcrypt-hashed, never plaintext — Fix #14), `prediction_logs` (JSONB prediction
payload, GIN-indexed, partial indexes on `risk_band`/`confidence` per module),
`review_queue`, `drift_events`. Dashboard queries are expected to hit the module+created_at
and JSONB GIN indexes rather than scanning.

**Celery queues** (`api/workers/celery_tasks.py`, planned): GPU queue (concurrency=1,
`pool=solo`) for image/text batch inference, CPU queue (concurrency=4, `pool=prefork`) for
retention scoring, maintenance queue for cleanup. `result_expires=3600`. Grad-CAM images are
written to `storage/gradcam/{request_id}.png` with a 1-hour TTL, swept by a Celery Beat task
every 30 minutes — don't assume they persist past that window.

**Auth**: API keys are bcrypt-hashed in Postgres; a 5-minute Redis cache avoids re-hashing
bcrypt on every request (`api/middleware/auth.py`, planned).

**Dashboard** (`dashboard/`, planned): Next.js 14. The BFF layer (`api/bff/`) proxies calls
server-side specifically so the backend API key never reaches the browser — don't add a path
that calls the E-CIP API directly from client-side dashboard code. The ROI estimator on the
Retention Analytics page is a client-side-only calculation (no API call on input change); its
formula and default constants (25% conversion, £480 avg LTV, £15 cost per offer) live in
`HANDOFF.md` §9 Week 16 if you need to reimplement it.

## GPU training

EfficientNet-B3 and DistilBERT training require GPU and are run on Colab/Kaggle free tiers,
not locally — everything else (tabular models, API, tests) runs on local CPU. When adding
training code for those two models, keep it Colab-runnable (see the Colab setup cell in
`HANDOFF.md` §14) rather than assuming local GPU availability.
