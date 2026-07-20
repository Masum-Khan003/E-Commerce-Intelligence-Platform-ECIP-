# tests/integration/test_api.py
# E-CIP v3.0 — API integration tests
# Blueprint Section 24 / production plan testing-strategy table:
#   POST /v1/products/classify -> 200, auth rejects invalid key -> 401,
#   readiness 503 before models loaded / 200 after.
#
# Uses a real API key created via data/scripts/create_api_key.py against
# the actual docker-compose Postgres/Redis containers — these tests are
# skipped if that stack isn't reachable rather than mocked, since the
# whole point is verifying the real bcrypt/Postgres/Redis auth path.

from __future__ import annotations

import asyncio

import pytest


def _postgres_reachable() -> bool:
    try:
        import asyncpg

        async def _check() -> bool:
            conn = await asyncpg.connect(
                "postgresql://ecip:ecip_dev@localhost:5432/ecip", timeout=2
            )
            await conn.close()
            return True

        return asyncio.run(_check())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable at localhost:5432 — run "
           "`docker compose -f docker-compose.dev.yml up -d` first.",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def api_key() -> str:
    from data.scripts.create_api_key import create_api_key

    return create_api_key("integration-test")


RETENTION_PAYLOAD = {
    "customer_id": "cust_integration_test",
    "frequency": 2,
    "monetary_value": 50.0,
    "recency_days": 300,
    "tenure_days": 320,
    "avg_order_value": 25.0,
    "recency_days_log": 5.7,
    "purchase_gap_cv": 0.1,
    "category_diversity": 1,
    "is_single_purchase": 0,
    "return_rate": 0.0,
    "purchase_trend": -0.5,
    "time_decay_weight": 0.01,
    "avg_sentiment_score": -0.8,
    "negative_review_count": 3,
    "last_review_sentiment": -0.9,
    "has_reviews": 1,
    "avg_battery_sentiment": -0.5,
    "avg_price_sentiment": -0.2,
    "avg_shipping_sentiment": -0.6,
}


class TestHealthEndpoints:
    def test_liveness_always_200_no_auth(self, client) -> None:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_readiness_200_after_full_startup(self, client) -> None:
        """
        TestClient's context manager runs the lifespan startup sequence to
        completion before any request can be made, so by the time we can
        call this endpoint, warm-up has already finished — this confirms
        the post-warm-up 200 path, not the pre-warm-up 503 path (covered
        separately in TestReadinessBeforeWarmup below).
        """
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert "xgb_ensemble" in body["models"]


class TestReadinessBeforeWarmup:
    def test_returns_503_before_ready_flag_is_set(self) -> None:
        """Directly exercises api.main's readiness gate before warm-up completes."""
        from fastapi import HTTPException

        import api.main as main_module

        original_ready = main_module._ready
        main_module._ready = False
        try:
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(main_module.readiness())
            assert exc_info.value.status_code == 503
        finally:
            main_module._ready = original_ready


class TestRetentionAuth:
    def test_missing_api_key_returns_401(self, client) -> None:
        response = client.post("/v1/retention/score", json=RETENTION_PAYLOAD)
        assert response.status_code == 401

    def test_invalid_api_key_returns_401(self, client) -> None:
        response = client.post(
            "/v1/retention/score",
            json=RETENTION_PAYLOAD,
            headers={"X-API-Key": "not-a-real-key"},
        )
        assert response.status_code == 401

    def test_valid_api_key_returns_200_with_real_prediction(self, client, api_key) -> None:
        response = client.post(
            "/v1/retention/score",
            json=RETENTION_PAYLOAD,
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
        assert body["churn_label_definition"] == "no_purchase_90d"
        # The negative-sentiment payload above should score as elevated risk.
        assert body["risk_band"] == "HIGH"

    def test_explain_endpoint_round_trips_from_score(self, client, api_key) -> None:
        score_response = client.post(
            "/v1/retention/score",
            json=RETENTION_PAYLOAD,
            headers={"X-API-Key": api_key},
        )
        request_id = score_response.json()["request_id"]

        explain_response = client.get(
            f"/v1/explain/shap/{request_id}",
            headers={"X-API-Key": api_key},
        )
        assert explain_response.status_code == 200
        body = explain_response.json()
        assert len(body["top_features"]) > 0
        assert abs(body["shap_sum_check"]) < 0.05
