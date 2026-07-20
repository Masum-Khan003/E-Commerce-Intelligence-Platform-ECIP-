# E-CIP Dashboard

Next.js 14 (App Router) operations dashboard for the E-Commerce Intelligence Platform.

## Local setup

```bash
npm install
cp .env.local.example .env.local   # fill in ECIP_API_URL / ECIP_API_KEY
npm run dev
```

Requires the FastAPI backend running (`uvicorn api.main:app`) and a valid API key
(`python data/scripts/create_api_key.py --name dashboard-dev`).

## Architecture

All API calls go through the BFF proxy at `app/api/bff/[...path]/route.ts` — a
server-side Next.js route handler that attaches `X-API-Key` from the server-only
`ECIP_API_KEY` env var. Client components never see the key; they call
`/api/bff/<path>` via the `useBffData` hook in `lib/api.ts`, never the E-CIP API
directly.

## Pages

- `/` — Overview: KPI cards, prediction volume by module, model version strip
- `/products` — Product Analytics (honest empty state until Module 1 is GPU-trained)
- `/sentiment` — Sentiment Analytics (honest empty state until Module 2 is GPU-trained)
- `/retention`, `/review-queue`, `/drift` — see Stage 7

Every data-driven page implements loading / success / error states explicitly
(see `components/StatusStates.tsx`) rather than a bare spinner or blank screen.
