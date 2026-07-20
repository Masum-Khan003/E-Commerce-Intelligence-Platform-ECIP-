# api/routers/analytics.py
# E-CIP v3.0 — Dashboard Analytics Endpoints
#
# Aggregates real rows from prediction_logs for the dashboard's Overview
# page (blueprint Section 16). Only retention currently writes real rows
# (api/routers/retention.py:_log_prediction) — Product/Sentiment have no
# trained models yet, so their aggregates are honestly empty rather than
# fabricated.

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

POSTGRES_DSN = "postgresql://ecip:ecip_dev@localhost:5432/ecip"

router = APIRouter(prefix="/v1/analytics", tags=["Analytics"])


class ModuleVolume(BaseModel):
    module: str
    count: int
    avg_latency_ms: float


class OverviewResponse(BaseModel):
    total_predictions: int
    by_module: list[ModuleVolume]
    model_versions: dict[str, str]


@router.get("/overview", response_model=OverviewResponse)
async def get_overview() -> OverviewResponse:
    model_versions = {
        "retention": "retention_ensemble_v1.0.0",
        "product": "pending GPU training",
        "sentiment": "pending GPU training",
    }

    try:
        import asyncpg

        conn = await asyncpg.connect(POSTGRES_DSN)
        try:
            rows = await conn.fetch(
                """
                SELECT module, COUNT(*) AS count, AVG(latency_ms) AS avg_latency_ms
                FROM prediction_logs
                GROUP BY module
                ORDER BY count DESC
                """
            )
        finally:
            await conn.close()
    except Exception:
        rows = []

    by_module = [
        ModuleVolume(
            module=row["module"],
            count=row["count"],
            avg_latency_ms=float(row["avg_latency_ms"] or 0.0),
        )
        for row in rows
    ]
    total = sum(m.count for m in by_module)

    return OverviewResponse(
        total_predictions=total,
        by_module=by_module,
        model_versions=model_versions,
    )


class RiskBandCount(BaseModel):
    risk_band: str
    count: int


class RetentionAnalyticsResponse(BaseModel):
    risk_band_distribution: list[RiskBandCount]
    total_scored: int


@router.get("/retention", response_model=RetentionAnalyticsResponse)
async def get_retention_analytics() -> RetentionAnalyticsResponse:
    """
    Risk band distribution, read via the JSONB path db/schema.sql already
    indexes for exactly this query (idx_pred_risk_band).
    """
    try:
        import asyncpg

        conn = await asyncpg.connect(POSTGRES_DSN)
        try:
            rows = await conn.fetch(
                """
                SELECT prediction->>'risk_band' AS risk_band, COUNT(*) AS count
                FROM prediction_logs
                WHERE module = 'retention'
                GROUP BY prediction->>'risk_band'
                ORDER BY count DESC
                """
            )
        finally:
            await conn.close()
    except Exception:
        rows = []

    distribution = [
        RiskBandCount(risk_band=row["risk_band"], count=row["count"]) for row in rows
    ]
    return RetentionAnalyticsResponse(
        risk_band_distribution=distribution,
        total_scored=sum(r.count for r in distribution),
    )
