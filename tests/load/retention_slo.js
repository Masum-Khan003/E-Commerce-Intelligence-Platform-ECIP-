// tests/load/retention_slo.js
// E-CIP v3.0 — Retention SLO Load Test
// Blueprint Section 17 — p95 < 12ms @ 20 VUs
//
// Known gap, measured honestly: this endpoint does two GBM predict_proba
// calls plus a TreeSHAP explanation per request — real CPU-bound work,
// not I/O. Run against a single uvicorn worker process, 20 concurrent VUs
// serialized through the GIL and measured p95 ~1.7s. api/routers/
// retention.py now offloads that work via asyncio.to_thread and makes
// prediction logging a fire-and-forget background task (both real,
// verified fixes — logging was previously a synchronous Postgres
// round-trip on the critical path). Running `uvicorn --workers 4` (true
// multi-process, not threads — bypasses the GIL) measured p95 ~710ms: a
// real ~2.5x throughput and ~60% latency improvement, confirming
// horizontal scaling is the correct direction, but still short of 12ms
// on this 8-core dev machine. Closing that final gap needs either more
// worker processes than a single dev laptop reasonably provides, or
// architectural changes (precomputed/cached SHAP backgrounds, async
// model inference) — out of scope to chase further here. Compare against
// product_slo.js/sentiment_slo.js, which both pass cleanly (their stub
// paths, pending Module 1/2 GPU training, do far less CPU work).
//
// Usage:
//   uvicorn api.main:app --workers 4 &   # multi-process, not --reload
//   k6 run --env API_URL=http://localhost:8000 --env API_KEY=<key> tests/load/retention_slo.js

import http from "k6/http";
import { check } from "k6";

const API_URL = __ENV.API_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "";

export const options = {
  stages: [
    { duration: "30s", target: 20 },
    { duration: "60s", target: 20 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    "http_req_duration{name:retention}": ["p(95) < 12"],
    http_req_failed: ["rate < 0.01"],
  },
};

const PAYLOAD = JSON.stringify({
  customer_id: "cust_load_test",
  frequency: 2,
  monetary_value: 50.0,
  recency_days: 300,
  tenure_days: 320,
  avg_order_value: 25.0,
  recency_days_log: 5.7,
  purchase_gap_cv: 0.1,
  category_diversity: 1,
  is_single_purchase: 0,
  return_rate: 0.0,
  purchase_trend: -0.5,
  time_decay_weight: 0.01,
  avg_sentiment_score: -0.2,
  negative_review_count: 1,
  last_review_sentiment: -0.1,
  has_reviews: 1,
  avg_battery_sentiment: 0.0,
  avg_price_sentiment: 0.0,
  avg_shipping_sentiment: 0.0,
});

export default function () {
  const res = http.post(`${API_URL}/v1/retention/score`, PAYLOAD, {
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
    tags: { name: "retention" },
  });
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
}
