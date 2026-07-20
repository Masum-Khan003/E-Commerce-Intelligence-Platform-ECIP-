// tests/load/sentiment_slo.js
// E-CIP v3.0 — Sentiment SLO Load Test
// Blueprint Section 17 — p95 < 50ms @ 10 VUs
//
// NOTE: Module 2 (DistilBERT) has no trained model in this build (GPU
// training pending — see HANDOFF.md), so this test exercises
// api/routers/sentiment.py's stub response path, not real inference.
// The endpoint/auth/latency-measurement plumbing this test verifies is
// identical either way; the SLO itself can only be confirmed for real
// once Module 2 is trained.
//
// Usage:
//   docker run --rm -i --network host grafana/k6 run \
//     --env API_URL=http://localhost:8000 --env API_KEY=<key> - < tests/load/sentiment_slo.js

import http from "k6/http";
import { check } from "k6";

const API_URL = __ENV.API_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "";

export const options = {
  stages: [
    { duration: "30s", target: 10 },
    { duration: "60s", target: 10 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    "http_req_duration{name:sentiment}": ["p(95) < 50"],
    http_req_failed: ["rate < 0.01"],
  },
};

const PAYLOAD = JSON.stringify({
  text: "The battery life is disappointing but shipping was fast.",
  include_aspects: true,
});

export default function () {
  const res = http.post(`${API_URL}/v1/sentiment/analyze`, PAYLOAD, {
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
    tags: { name: "sentiment" },
  });
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
}
