# observability/prometheus/metrics.py
# E-CIP v3.0 — Prometheus Metrics
# Blueprint Section 14
#
# Six metrics exposed at /metrics (mounted in api/main.py):
#   ecip_inference_latency_seconds  — histogram, labeled by module
#   ecip_prediction_confidence      — histogram, labeled by module
#   ecip_ood_flags_total            — counter
#   ecip_review_queue_depth         — gauge
#   ecip_feature_drift_total        — counter, labeled by feature
#   ecip_model_warmup_seconds       — histogram

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INFERENCE_LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0
)

inference_latency_seconds = Histogram(
    "ecip_inference_latency_seconds",
    "Inference latency in seconds, by module",
    labelnames=["module"],
    buckets=INFERENCE_LATENCY_BUCKETS,
)

prediction_confidence = Histogram(
    "ecip_prediction_confidence",
    "Model prediction confidence, by module",
    labelnames=["module"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 1.0),
)

ood_flags_total = Counter(
    "ecip_ood_flags_total",
    "Count of predictions flagged as out-of-distribution",
)

review_queue_depth = Gauge(
    "ecip_review_queue_depth",
    "Current count of pending items in the human review queue",
)

feature_drift_total = Counter(
    "ecip_feature_drift_total",
    "Count of drift-detected events, by feature",
    labelnames=["feature"],
)

model_warmup_seconds = Histogram(
    "ecip_model_warmup_seconds",
    "Time taken for the FastAPI startup warm-up sequence to complete",
    buckets=(1, 2, 5, 10, 15, 20, 30, 60, 90, 120),
)


def record_inference(module: str, latency_seconds: float, confidence: float | None = None) -> None:
    """Convenience helper — records latency and (if available) confidence for one prediction."""
    inference_latency_seconds.labels(module=module).observe(latency_seconds)
    if confidence is not None:
        prediction_confidence.labels(module=module).observe(confidence)


def record_drift_events(results: dict[str, dict[str, object]]) -> None:
    """Increments ecip_feature_drift_total for every feature flagged as drifted."""
    for feature, metrics in results.items():
        if metrics.get("drift_detected"):
            feature_drift_total.labels(feature=feature).inc()


def set_review_queue_depth(depth: int) -> None:
    review_queue_depth.set(depth)
