# Deployment Guide

Three deployment targets. This document is prep only — configs and exact steps, not an
executed deployment. Actual account-bound deployment (Render/Vercel/Upstash sign-up,
GitHub Release, screencast) is a manual follow-up.

## 1. Local — full production stack

```bash
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up -d
```

8 services, mem-limited per the resource budget in the model cards (~4GB peak — host needs
6GB+ RAM):

| Service | mem_limit | Role |
|---|---|---|
| `api` | 2g | FastAPI — warm-up loader, all `/v1/*` routers |
| `celery_gpu` | 2g | concurrency=1, pool=solo — image/sentiment batch tasks |
| `celery_cpu` | 1g | concurrency=4, pool=prefork — retention batch tasks |
| `celery_beat` | 256m | scheduled drift check, Grad-CAM cleanup, perf snapshot |
| `postgres` | 512m | `prediction_logs`, `review_queue`, `drift_events`, `api_keys` |
| `redis` | 256m | Celery broker, auth cache |
| `prometheus` | 256m | scrapes `api:8000/metrics`, loads `observability/alertmanager/rules.yml` |
| `grafana` | 256m | dashboards (`admin`/`admin` by default — **change this in any real deployment**) |

`docker-compose.dev.yml` (postgres + redis + mlflow only) remains the lighter option for
local development without the API/Celery/observability layer.

Health check: `curl http://localhost:8000/health/ready` — 503 until warm-up completes, 200
after with per-model load status.

## 2. Cloud demo

**API — Render.com free tier**
1. Push this repo to GitHub.
2. New Web Service on Render, connect the repo, select "Docker" runtime (uses the root
   `Dockerfile`).
3. Environment variables: `REDIS_URL` (from Upstash, step below).
4. Free tier cold-starts after inactivity (~30s) — the dashboard's Overview page should show
   a "warming up" state while `/health/ready` returns 503 (see `dashboard/lib/api.ts`'s error
   handling); this is expected behavior on a free-tier deploy, not a bug.
5. Add a managed PostgreSQL instance (Render's free tier includes one, 90-day retention) and
   run `db/schema.sql` against it once provisioned.

**Redis — Upstash free tier**
1. Create a free Redis database at upstash.com (10K commands/day free tier).
2. Use its connection string as `REDIS_URL` in Render's environment variables.

**Dashboard — Vercel free tier**
1. Import `dashboard/` as a separate Vercel project (set the root directory to `dashboard`).
2. Environment variables: `ECIP_API_URL` (the Render API URL), `ECIP_API_KEY` (from
   `python data/scripts/create_api_key.py` run against the deployed Postgres).
3. Both env vars are server-side only (no `NEXT_PUBLIC_` prefix) — Vercel keeps them out of
   the client bundle automatically; the BFF proxy (`dashboard/app/api/bff/[...path]/route.ts`)
   is the only code that reads them.

**Pre-populating the demo**
For a cloud demo to look populated rather than empty, run a handful of real
`/v1/retention/score` calls against the deployed API before sharing a link — real predictions
write to `prediction_logs`, which the dashboard's Overview/Retention Analytics pages read
directly. There is no synthetic-data seeding script for this — the predictions should be
real API calls, consistent with this project's "verify for real" approach throughout.

## 3. GitHub Pages (docs only)

Not yet built. Would host: Great Expectations HTML data docs
(`data/validation/great_expectations/uncommitted/data_docs/local_site/`), model cards
rendered from Markdown, an OpenAPI-generated static reference (FastAPI already serves this
live at `/docs` and `/redoc` — a static export would use `fastapi-code-generator` or similar),
and an ADR index (`docs/decisions/`).

## Known gaps (documented, not hidden)

- **Module 1/2 have no trained weights.** `models/product/weights/` and
  `models/sentiment/weights/` are empty — GPU training on Colab/Kaggle is required first (see
  `HANDOFF.md` §14). The API and dashboard both handle this state honestly (503/empty-state,
  not fabricated data) rather than requiring it to deploy.
- **Retention's `/v1/retention/score` SLO (p95 < 12ms @ 20 VUs) is not met on a single
  worker process** — measured ~1.7s p95, improved to ~710ms with `uvicorn --workers 4` (see
  `tests/load/retention_slo.js` for the full measurement history and what would close the
  remaining gap). Scale `api`'s replica count / worker count accordingly in any real
  deployment carrying this kind of concurrent load.
- **No Celery Prometheus exporter** — `observability/alertmanager/rules.yml`'s
  `CeleryQueueBacklog` rule needs a `celery-exporter` sidecar this compose file doesn't
  include yet.
