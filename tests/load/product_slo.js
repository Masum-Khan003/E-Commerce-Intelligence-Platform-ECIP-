// tests/load/product_slo.js
// E-CIP v3.0 — Product Classification SLO Load Test
// Blueprint Section 17 — p95 < 120ms @ 10 VUs
//
// NOTE: Module 1 (EfficientNet-B3) has no trained model in this build
// (GPU training pending — see HANDOFF.md), so this test exercises
// api/routers/products.py's real validation path plus its stub response
// (image_store.py's validate_upload runs regardless of whether a model
// is loaded) — not real inference. fixtures/test_image.png is a real,
// validly-sized (>5000 bytes) noise PNG so the request clears validation
// rather than being rejected at 422 before reaching the classify logic.
//
// Usage:
//   docker run --rm -i --network host grafana/k6 run \
//     --env API_URL=http://localhost:8000 --env API_KEY=<key> - < tests/load/product_slo.js

import http from "k6/http";
import { check } from "k6";

const API_URL = __ENV.API_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "";

const testImage = open("./fixtures/test_image.png", "b");

export const options = {
  stages: [
    { duration: "30s", target: 10 },
    { duration: "60s", target: 10 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    "http_req_duration{name:product}": ["p(95) < 120"],
    http_req_failed: ["rate < 0.01"],
  },
};

export default function () {
  const formData = {
    file: http.file(testImage, "test.png", "image/png"),
  };
  const res = http.post(`${API_URL}/v1/products/classify`, formData, {
    headers: { "X-API-Key": API_KEY },
    tags: { name: "product" },
  });
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
}
