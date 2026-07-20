# tests/model_tests/test_cross_module.py
# E-CIP v3.0 — Cross-Module Integration Tests
# Blueprint Section 24 — Fix #45
#
# Cross-module behavioural tests that verify system-level correctness, not
# just unit-level correctness. test_negative_sentiment_increases_churn_risk
# is the single most important invariant in this system: sentiment (Module 2)
# must actually move the retention model's (Module 3) prediction, in the
# correct direction — not just sit in the feature table unused.
#
# Run: pytest tests/model_tests/ -v --timeout=120

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

WEIGHTS_DIR = Path("models/retention/weights")
ARTIFACTS_DIR = Path("models/retention/artifacts")
TEXT_FEATURES_DIR = Path("data/feature_store/text_features")
SCALER_PATH = Path("data/feature_store/artifacts/scaler_v1.joblib")
SNAPSHOT_DATE_STR = "2010-11-30"

pytestmark = pytest.mark.skipif(
    not (WEIGHTS_DIR / "xgb_final.joblib").exists(),
    reason="Trained retention models not found — run models/retention/train.py "
           "and models/retention/calibrate.py first.",
)


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def xgb_model() -> Any:
    import joblib

    return joblib.load(WEIGHTS_DIR / "xgb_final.joblib")


@pytest.fixture(scope="module")
def lgbm_model() -> Any:
    import joblib

    return joblib.load(WEIGHTS_DIR / "lgbm_final.joblib")


@pytest.fixture(scope="module")
def calibrator() -> Any:
    import joblib

    return joblib.load(ARTIFACTS_DIR / "calibrator.joblib")


@pytest.fixture(scope="module")
def calibration_method() -> str:
    metrics = json.loads((ARTIFACTS_DIR / "calibration_metrics.json").read_text())
    method: str = metrics["calibration_method"]
    return method


@pytest.fixture(scope="module")
def scaler() -> Any:
    import joblib

    return joblib.load(SCALER_PATH)


@pytest.fixture(scope="module")
def feature_columns() -> list[str]:
    cols: list[str] = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text())
    return cols


def _calibrated_probability(
    features: dict[str, float],
    feature_columns: list[str],
    scaler: Any,
    xgb_model: Any,
    lgbm_model: Any,
    calibrator: Any,
    calibration_method: str,
) -> float:
    import numpy as np

    from api.routers.retention import build_scaled_feature_row
    from models.retention.train import build_ensemble_prediction

    row = build_scaled_feature_row(features, feature_columns, scaler)
    xgb_proba = float(xgb_model.predict_proba(row)[:, 1][0])
    lgbm_proba = float(lgbm_model.predict_proba(row)[:, 1][0])
    ensemble_proba = build_ensemble_prediction(xgb_proba, lgbm_proba)

    if calibration_method == "platt_scaling":
        return float(calibrator.predict_proba(np.array([[ensemble_proba]]))[0, 1])
    return float(calibrator.predict(np.array([ensemble_proba]))[0])


BASE_CUSTOMER_FEATURES: dict[str, float] = {
    "frequency": 6.0,
    "monetary_value": 320.0,
    "recency_days": 45.0,
    "tenure_days": 420.0,
    "avg_order_value": 53.0,
    "recency_days_log": 3.83,
    "purchase_gap_cv": 0.4,
    "category_diversity": 3.0,
    "is_single_purchase": 0.0,
    "return_rate": 0.05,
    "purchase_trend": 0.1,
    "time_decay_weight": 0.5,
    "negative_review_count": 0.0,
    "has_reviews": 1.0,
}


class TestCrossModuleCausality:
    """Blueprint Section 24 — Fix #45."""

    def test_negative_sentiment_increases_churn_risk(
        self, xgb_model, lgbm_model, calibrator, calibration_method, scaler, feature_columns
    ) -> None:
        """
        GIVEN two otherwise-identical customers differing only in sentiment
        (avg_sentiment_score, last_review_sentiment, and the three aspect
        scores) THEN the negative-sentiment customer's churn_probability
        must be STRICTLY greater. This is the most important business
        invariant in the system — if sentiment doesn't move the retention
        model in this direction, the cross-module fusion is decorative.
        """
        positive_customer = {
            **BASE_CUSTOMER_FEATURES,
            "avg_sentiment_score": 0.8,
            "last_review_sentiment": 0.9,
            "avg_battery_sentiment": 0.7,
            "avg_shipping_sentiment": 0.7,
            "avg_price_sentiment": 0.7,
        }
        negative_customer = {
            **BASE_CUSTOMER_FEATURES,
            "avg_sentiment_score": -0.8,
            "last_review_sentiment": -0.9,
            "avg_battery_sentiment": -0.7,
            "avg_shipping_sentiment": -0.7,
            "avg_price_sentiment": -0.7,
        }

        prob_positive = _calibrated_probability(
            positive_customer, feature_columns, scaler,
            xgb_model, lgbm_model, calibrator, calibration_method,
        )
        prob_negative = _calibrated_probability(
            negative_customer, feature_columns, scaler,
            xgb_model, lgbm_model, calibrator, calibration_method,
        )

        assert prob_negative > prob_positive, (
            f"FAILED: negative-sentiment customer churn_probability "
            f"({prob_negative:.4f}) must exceed positive-sentiment customer "
            f"churn_probability ({prob_positive:.4f})."
        )

    def test_sentiment_score_range(self) -> None:
        """
        sentiment_score must always be in [-1, 1] for safe downstream use.
        Module 2 (DistilBERT) requires GPU training not available in this
        environment — this validates the actual sentiment signal this
        project runs against: the synthetic demo generator's output
        (data/scripts/synthesize_demo_sentiment.py), which every retention
        feature in production would need to satisfy the same contract.
        """
        import pandas as pd

        review_path = TEXT_FEATURES_DIR / "review_sentiment_v1.parquet"
        aspect_path = TEXT_FEATURES_DIR / "aspect_sentiment_v1.parquet"
        if not review_path.exists() or not aspect_path.exists():
            pytest.skip("Synthetic sentiment data not generated — run "
                        "data/scripts/synthesize_demo_sentiment.py first.")

        review_df = pd.read_parquet(review_path)
        aspect_df = pd.read_parquet(aspect_path)

        assert review_df["sentiment_score"].between(-1.0, 1.0).all(), (
            "sentiment_score out of [-1, 1] range in review_sentiment_v1.parquet"
        )
        assert aspect_df["aspect_sentiment"].between(-1.0, 1.0).all(), (
            "aspect_sentiment out of [-1, 1] range in aspect_sentiment_v1.parquet"
        )

    def test_shap_sum_consistency(self, xgb_model, feature_columns) -> None:
        """|sum(shap_values) + expected_value - margin_score| < 0.05 (see
        models/retention/shap_explain.py module docstring for why this
        check runs in margin space rather than probability space)."""
        import pandas as pd

        from models.retention.shap_explain import (
            SHAP_SUM_TOLERANCE,
            check_shap_sum_consistency,
            compute_shap_values,
            load_shap_explainer,
        )

        data_path = Path("data/feature_store/customer_features/rfm_behavioral_v2.parquet")
        if not data_path.exists():
            pytest.skip("Feature table not found — run data/pipelines/tabular_pipeline.py.")

        df = pd.read_parquet(data_path).dropna(subset=["churned"])
        sample = df.sample(n=min(10, len(df)), random_state=7)
        x_sample = sample[feature_columns].fillna(0).to_numpy(dtype=float)

        explainer = load_shap_explainer(xgb_model)

        for i in range(len(sample)):
            row = x_sample[i : i + 1]
            margin_score = float(xgb_model.predict(row, output_margin=True)[0])
            shap_values, expected_value = compute_shap_values(explainer, row)
            sum_check = check_shap_sum_consistency(shap_values, expected_value, margin_score)
            assert sum_check < SHAP_SUM_TOLERANCE, (
                f"SHAP sum-consistency check failed on row {i}: "
                f"{sum_check:.4f} >= {SHAP_SUM_TOLERANCE}"
            )

    def test_causal_integrity_no_future_sentiment(self) -> None:
        """
        Gate G8: reviews merged into retention features must predate the
        snapshot_date. Zero tolerance — even one row of future sentiment
        leaking into a churn feature is a data-leakage bug.
        """
        import pandas as pd

        review_path = TEXT_FEATURES_DIR / "review_sentiment_v1.parquet"
        aspect_path = TEXT_FEATURES_DIR / "aspect_sentiment_v1.parquet"
        if not review_path.exists() or not aspect_path.exists():
            pytest.skip("Synthetic sentiment data not generated — run "
                        "data/scripts/synthesize_demo_sentiment.py first.")

        snapshot_date = pd.Timestamp(SNAPSHOT_DATE_STR)
        review_df = pd.read_parquet(review_path)
        aspect_df = pd.read_parquet(aspect_path)

        future_reviews = review_df[pd.to_datetime(review_df["review_date"]) >= snapshot_date]
        future_aspects = aspect_df[pd.to_datetime(aspect_df["review_date"]) >= snapshot_date]

        assert len(future_reviews) == 0, (
            f"CAUSAL LEAKAGE: {len(future_reviews)} review rows dated on/after "
            f"snapshot_date ({SNAPSHOT_DATE_STR}) exist in the raw sentiment "
            "source — merge_sentiment_features() Gate G8 must filter these."
        )
        assert len(future_aspects) == 0, (
            f"CAUSAL LEAKAGE: {len(future_aspects)} aspect rows dated on/after "
            f"snapshot_date ({SNAPSHOT_DATE_STR}) exist in the raw sentiment source."
        )
