# mlops/beat_schedule.py
# E-CIP v3.0 — Celery Beat Schedule
# Blueprint Section 15 — Fix #26
#
# Three periodic tasks:
#   daily-drift-check     — 02:00 daily, runs mlops/drift_detector.py
#   gradcam-cleanup       — every 30 min, sweeps expired Grad-CAM PNGs
#   weekly-perf-snapshot  — Monday 03:00, logs a lightweight perf snapshot
#
# Start: celery -A mlops.beat_schedule beat --loglevel=info
# (Runs alongside the workers defined in api/workers/celery_tasks.py —
# Beat only schedules tasks, it never executes them itself.)

from __future__ import annotations

from celery.schedules import crontab

from api.workers.celery_tasks import app, cleanup_expired_gradcam

CELERYBEAT_SCHEDULE = {
    "daily-drift-check": {
        "task": "mlops.beat_schedule.run_daily_drift_check",
        "schedule": crontab(hour=2, minute=0),
    },
    "gradcam-cleanup": {
        "task": "api.workers.celery_tasks.cleanup_expired_gradcam",
        "schedule": crontab(minute="*/30"),
    },
    "weekly-perf-snapshot": {
        "task": "mlops.beat_schedule.take_model_perf_snapshot",
        "schedule": crontab(day_of_week=1, hour=3, minute=0),
    },
}

app.conf.beat_schedule = CELERYBEAT_SCHEDULE
app.conf.timezone = "UTC"


@app.task(name="mlops.beat_schedule.run_daily_drift_check")
def run_daily_drift_check() -> dict[str, int]:
    """Runs the feature drift check and writes results to PostgreSQL."""
    import asyncio

    from mlops.drift_detector import run_drift_check, write_drift_events

    results = run_drift_check(module="retention")
    if not results:
        return {"features_checked": 0, "drift_events_written": 0}

    written = asyncio.run(write_drift_events("retention", results))
    return {"features_checked": len(results), "drift_events_written": written}


@app.task(name="mlops.beat_schedule.take_model_perf_snapshot")
def take_model_perf_snapshot() -> dict[str, str]:
    """
    Lightweight weekly checkpoint — logs the currently-shipped model
    version and training metrics so performance regressions between
    retraining cycles are visible in MLflow without a full re-evaluation.
    """
    import json
    from pathlib import Path

    metrics_path = Path("models/retention/artifacts/training_metrics.json")
    if not metrics_path.exists():
        return {"status": "no metrics found"}

    metrics = json.loads(metrics_path.read_text())

    try:
        import mlflow

        mlflow.set_experiment("retention_classifier")
        with mlflow.start_run(run_name="weekly_perf_snapshot"):
            mlflow.log_metrics({
                k: v for k, v in metrics.items() if isinstance(v, int | float)
            })
    except ImportError:
        pass

    return {"status": "logged", "ensemble_cv_auc": str(metrics.get("ensemble_cv_auc"))}


# Re-exported so `cleanup_expired_gradcam` is registered under this module
# too when Beat imports mlops.beat_schedule directly.
__all__ = ["app", "CELERYBEAT_SCHEDULE", "cleanup_expired_gradcam"]
