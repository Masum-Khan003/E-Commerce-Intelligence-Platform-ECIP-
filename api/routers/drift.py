# api/routers/drift.py
# E-CIP v3.0 — Drift Events API
# Blueprint Section 15/16 — Drift Monitor dashboard page
#
# GET /v1/drift-events — per-feature drift gauges + event timeline, backed
# by real rows mlops/drift_detector.py writes to the drift_events table.

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

POSTGRES_DSN = "postgresql://ecip:ecip_dev@localhost:5432/ecip"

router = APIRouter(prefix="/v1/drift-events", tags=["Drift Monitor"])


class DriftEvent(BaseModel):
    module: str
    feature_name: str | None
    metric: str
    metric_value: float
    threshold: float
    alert_triggered: bool
    reference_version: str | None
    created_at: str


class FeatureDriftSummary(BaseModel):
    feature_name: str
    latest_psi: float
    alert_triggered: bool


class DriftEventsResponse(BaseModel):
    events: list[DriftEvent]
    by_feature: list[FeatureDriftSummary]


@router.get("", response_model=DriftEventsResponse)
async def list_drift_events(
    module: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
) -> DriftEventsResponse:
    try:
        import asyncpg

        conn = await asyncpg.connect(POSTGRES_DSN)
        try:
            params: list[object] = []
            where = ""
            if module:
                params.append(module)
                where = "WHERE module = $1"

            rows = await conn.fetch(
                f"""
                SELECT module, feature_name, metric, metric_value, threshold,
                       alert_triggered, reference_version, created_at
                FROM drift_events
                {where}
                ORDER BY created_at DESC
                LIMIT ${len(params) + 1}
                """,
                *params,
                limit,
            )
            latest_per_feature = await conn.fetch(
                f"""
                SELECT DISTINCT ON (feature_name)
                    feature_name, metric_value, alert_triggered
                FROM drift_events
                {where}
                ORDER BY feature_name, created_at DESC
                """,
                *params,
            )
        finally:
            await conn.close()
    except Exception:
        rows = []
        latest_per_feature = []

    events = [
        DriftEvent(
            module=row["module"],
            feature_name=row["feature_name"],
            metric=row["metric"],
            metric_value=float(row["metric_value"]),
            threshold=float(row["threshold"]),
            alert_triggered=bool(row["alert_triggered"]),
            reference_version=row["reference_version"],
            created_at=row["created_at"].isoformat(),
        )
        for row in rows
    ]
    by_feature = [
        FeatureDriftSummary(
            feature_name=row["feature_name"],
            latest_psi=float(row["metric_value"]),
            alert_triggered=bool(row["alert_triggered"]),
        )
        for row in latest_per_feature
    ]

    return DriftEventsResponse(events=events, by_feature=by_feature)
