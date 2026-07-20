# api/workers/celery_tasks.py
# E-CIP v3.0 — Celery Task Definitions
# Blueprint Section 12/23 — Fix #8, Fix #33
#
# Three queues, matched to each module's resource profile:
#   gpu_queue         — concurrency=1, pool=solo   (image + sentiment batch —
#                       GPU models must not run concurrently in one process)
#   cpu_queue         — concurrency=4, pool=prefork (retention batch — cheap,
#                       parallelisable, no GPU contention)
#   maintenance_queue — concurrency=2               (Grad-CAM TTL cleanup)
#
# Fix #33: result_expires=3600 — task results are cleared from Redis after
# 1 hour so completed batch results don't accumulate indefinitely.
#
# Start workers:
#   celery -A api.workers.celery_tasks worker -Q gpu_queue --concurrency=1 --pool=solo
#   celery -A api.workers.celery_tasks worker -Q cpu_queue --concurrency=4 --pool=prefork
#   celery -A api.workers.celery_tasks worker -Q maintenance_queue --concurrency=2

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery("ecip", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_routes={
        "api.workers.celery_tasks.batch_classify_images": {"queue": "gpu_queue"},
        "api.workers.celery_tasks.batch_score_sentiment": {"queue": "gpu_queue"},
        "api.workers.celery_tasks.batch_score_retention": {"queue": "cpu_queue"},
        "api.workers.celery_tasks.cleanup_expired_gradcam": {"queue": "maintenance_queue"},
    },
    result_expires=3600,  # Fix #33
    task_acks_late=True,
    worker_max_tasks_per_child=50,
)


# ─── GPU queue tasks ────────────────────────────────────────────────────────────
# Module 1/2 models require GPU training not available in this environment
# (see HANDOFF.md) — these tasks report that honestly per item rather than
# silently returning fabricated predictions.

@app.task(name="api.workers.celery_tasks.batch_classify_images")
def batch_classify_images(file_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch product image classification. Blueprint Section 12."""
    weights_path = Path("models/product/weights/efficientnet_b3_best.pt")
    if not weights_path.exists():
        return [
            {"status": "unavailable", "reason": "EfficientNet-B3 not yet trained (GPU required)"}
            for _ in file_data
        ]
    # Real inference wiring reuses api.routers.products.preprocess_image /
    # run_inference once Module 1 is trained — not duplicated here.
    return [{"status": "pending_implementation"} for _ in file_data]


@app.task(name="api.workers.celery_tasks.batch_score_sentiment")
def batch_score_sentiment(texts: list[str]) -> list[dict[str, Any]]:
    """Batch sentiment analysis. Blueprint Section 12."""
    weights_path = Path("models/sentiment/weights/distilbert_sentiment_best.pt")
    if not weights_path.exists():
        return [
            {"status": "unavailable", "reason": "DistilBERT not yet trained (GPU required)"}
            for _ in texts
        ]
    return [{"status": "pending_implementation"} for _ in texts]


# ─── CPU queue task ─────────────────────────────────────────────────────────────
# Retention is fully trained and real — this task is genuinely functional.

@app.task(name="api.workers.celery_tasks.batch_score_retention")
def batch_score_retention(customer_feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Batch churn scoring. Reuses the exact scaling + ensemble + calibration
    logic api/routers/retention.py uses for single-request scoring.
    """
    from api.routers.retention import (
        _load_local_retention_artifacts,
        _risk_band,
        build_scaled_feature_row,
    )
    from models.retention.train import build_ensemble_prediction

    registry = _load_local_retention_artifacts()
    xgb_model = registry.get("xgb_final")
    lgbm_model = registry.get("lgbm_final")
    calibrator = registry.get("calibrator")
    feature_columns = registry.get("feature_columns")
    scaler = registry.get("scaler")
    calib_metrics = registry.get("calibration_metrics", {})

    if not (xgb_model and lgbm_model and calibrator and feature_columns):
        return [
            {"status": "unavailable", "reason": "Retention models not trained yet"}
            for _ in customer_feature_rows
        ]

    method = calib_metrics.get("calibration_method", "isotonic_regression")
    results: list[dict[str, Any]] = []

    for features in customer_feature_rows:
        row = build_scaled_feature_row(features, feature_columns, scaler)
        xgb_proba = float(xgb_model.predict_proba(row)[:, 1][0])
        lgbm_proba = float(lgbm_model.predict_proba(row)[:, 1][0])
        ensemble_proba = build_ensemble_prediction(xgb_proba, lgbm_proba)

        if method == "platt_scaling":
            import numpy as np

            calibrated = float(calibrator.predict_proba(np.array([[ensemble_proba]]))[0, 1])
        else:
            import numpy as np

            calibrated = float(calibrator.predict(np.array([ensemble_proba]))[0])

        results.append({
            "status": "scored",
            "churn_probability": round(calibrated, 4),
            "risk_band": _risk_band(calibrated),
        })

    return results


# ─── Maintenance queue task ─────────────────────────────────────────────────────

@app.task(name="api.workers.celery_tasks.cleanup_expired_gradcam")
def cleanup_expired_gradcam() -> dict[str, Any]:
    """
    Blueprint Section 11/15 — sweeps storage/gradcam/ for files older than
    the TTL image_store.py enforces. Run every 30 minutes via Celery Beat
    (mlops/beat_schedule.py).
    """
    from api.storage.image_store import GRADCAM_TTL_HOURS

    gradcam_dir = Path("storage/gradcam")
    if not gradcam_dir.exists():
        return {"deleted": 0, "checked": 0}

    ttl_seconds = GRADCAM_TTL_HOURS * 3600
    now = time.time()
    deleted = 0
    checked = 0

    for path in gradcam_dir.iterdir():
        if not path.is_file():
            continue
        checked += 1
        if now - path.stat().st_mtime > ttl_seconds:
            path.unlink()
            deleted += 1

    return {"deleted": deleted, "checked": checked}
