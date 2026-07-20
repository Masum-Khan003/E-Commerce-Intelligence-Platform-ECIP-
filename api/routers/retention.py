# api/routers/retention.py
# E-CIP v3.0 — Retention Scoring API Router
# Blueprint Section 12 — /v1/retention/score
#
# Response contract (Blueprint Section 05):
#   request_id, customer_id, churn_probability, risk_band,
#   recommended_action, top_risk_factors, churn_label_definition,
#   is_single_purchase_customer, model_version, calibration_method,
#   decision_threshold, inference_ms

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from models.retention.shap_explain import (
    compute_shap_values,
    load_shap_explainer,
    top_features_with_direction,
)
from models.retention.train import build_ensemble_prediction

# ─── Constants ────────────────────────────────────────────────────────────────

WEIGHTS_DIR = Path("models/retention/weights")
ARTIFACTS_DIR = Path("models/retention/artifacts")
REFERENCE_DIST_PATH = Path("data/reference_distributions/customer_features_ref_v1.json")

MODEL_VERSION = "retention_ensemble_v1.0.0"
CHURN_LABEL_DEFINITION = "no_purchase_90d"  # blueprint §21 — 90-day horizon
RISK_BAND_LOW = 0.3
RISK_BAND_HIGH = 0.6
TOP_RISK_FACTORS_N = 3

# Raw feature defaults — a customer with no purchase/review history at all.
# Real callers populate this from the customer feature store; these
# defaults only make the endpoint exercisable with a minimal payload.
FEATURE_DEFAULTS: dict[str, float] = {
    "frequency": 1.0,
    "monetary_value": 0.0,
    "recency_days": 0.0,
    "tenure_days": 0.0,
    "avg_order_value": 0.0,
    "recency_days_log": 0.0,
    "purchase_gap_cv": -1.0,
    "category_diversity": 1.0,
    "is_single_purchase": 1.0,
    "return_rate": 0.0,
    "purchase_trend": 0.0,
    "time_decay_weight": 0.0,
    "avg_sentiment_score": 0.0,
    "negative_review_count": 0.0,
    "last_review_sentiment": 0.0,
    "has_reviews": 0.0,
    "avg_battery_sentiment": 0.0,
    "avg_price_sentiment": 0.0,
    "avg_shipping_sentiment": 0.0,
}

router = APIRouter(prefix="/v1/retention", tags=["Retention Intelligence"])

# In-memory cache: request_id -> (raw feature row array, customer_id).
# Lets GET /v1/explain/shap/{request_id} regenerate a SHAP explanation for
# a prior score without the caller resubmitting features. Ephemeral by
# design (mirrors Grad-CAM's TTL-based lookup) — a restart clears it.
_EXPLANATION_CACHE: dict[str, tuple[Any, str]] = {}
_EXPLANATION_CACHE_MAX = 500


# ─── Request / response schemas ────────────────────────────────────────────────

class CustomerFeaturesRequest(BaseModel):
    customer_id: str
    frequency: float = FEATURE_DEFAULTS["frequency"]
    monetary_value: float = FEATURE_DEFAULTS["monetary_value"]
    recency_days: float = FEATURE_DEFAULTS["recency_days"]
    tenure_days: float = FEATURE_DEFAULTS["tenure_days"]
    avg_order_value: float = FEATURE_DEFAULTS["avg_order_value"]
    recency_days_log: float = FEATURE_DEFAULTS["recency_days_log"]
    purchase_gap_cv: float = FEATURE_DEFAULTS["purchase_gap_cv"]
    category_diversity: float = FEATURE_DEFAULTS["category_diversity"]
    is_single_purchase: float = FEATURE_DEFAULTS["is_single_purchase"]
    return_rate: float = FEATURE_DEFAULTS["return_rate"]
    purchase_trend: float = FEATURE_DEFAULTS["purchase_trend"]
    time_decay_weight: float = FEATURE_DEFAULTS["time_decay_weight"]
    avg_sentiment_score: float = Field(FEATURE_DEFAULTS["avg_sentiment_score"], ge=-1.0, le=1.0)
    negative_review_count: float = FEATURE_DEFAULTS["negative_review_count"]
    last_review_sentiment: float = Field(
        FEATURE_DEFAULTS["last_review_sentiment"], ge=-1.0, le=1.0
    )
    has_reviews: float = FEATURE_DEFAULTS["has_reviews"]
    avg_battery_sentiment: float = FEATURE_DEFAULTS["avg_battery_sentiment"]
    avg_price_sentiment: float = FEATURE_DEFAULTS["avg_price_sentiment"]
    avg_shipping_sentiment: float = FEATURE_DEFAULTS["avg_shipping_sentiment"]


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str


class RetentionScoreResponse(BaseModel):
    request_id: str
    customer_id: str
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_band: str
    recommended_action: str
    top_risk_factors: list[RiskFactor]
    churn_label_definition: str
    is_single_purchase_customer: bool
    model_version: str
    calibration_method: str
    decision_threshold: float
    inference_ms: int


# ─── Model registry ─────────────────────────────────────────────────────────────

_local_registry_cache: dict[str, Any] | None = None


def _load_local_retention_artifacts() -> dict[str, Any]:
    """
    Load retention artifacts directly from disk, cached after first call.
    Used when the FastAPI startup warm-up loader (api/main.py, Stage 4)
    hasn't populated a shared model_registry yet — retention's artifacts
    are small (joblib, no GPU) so eager local loading is cheap.
    """
    global _local_registry_cache
    if _local_registry_cache is not None:
        return _local_registry_cache

    import joblib

    registry: dict[str, Any] = {}

    xgb_path = WEIGHTS_DIR / "xgb_final.joblib"
    lgbm_path = WEIGHTS_DIR / "lgbm_final.joblib"
    calibrator_path = ARTIFACTS_DIR / "calibrator.joblib"
    scaler_path = Path("data/feature_store/artifacts/scaler_v1.joblib")

    if xgb_path.exists():
        registry["xgb_final"] = joblib.load(xgb_path)
    if lgbm_path.exists():
        registry["lgbm_final"] = joblib.load(lgbm_path)
    if calibrator_path.exists():
        registry["calibrator"] = joblib.load(calibrator_path)
    if scaler_path.exists():
        registry["scaler"] = joblib.load(scaler_path)

    feature_cols_path = ARTIFACTS_DIR / "feature_columns.json"
    if feature_cols_path.exists():
        registry["feature_columns"] = json.loads(feature_cols_path.read_text())

    calib_metrics_path = ARTIFACTS_DIR / "calibration_metrics.json"
    if calib_metrics_path.exists():
        registry["calibration_metrics"] = json.loads(calib_metrics_path.read_text())

    if REFERENCE_DIST_PATH.exists():
        registry["reference_distribution"] = json.loads(REFERENCE_DIST_PATH.read_text())

    if "xgb_final" in registry:
        registry["shap_explainer"] = load_shap_explainer(registry["xgb_final"])

    _local_registry_cache = registry
    return registry


def get_model_registry() -> dict[str, Any]:
    """
    Dependency returning the loaded retention model registry. Prefers the
    central api.main warm-up registry (Stage 4) if it has retention's keys,
    else falls back to a locally-loaded copy — mirrors
    api/routers/products.py's get_model_registry pattern.
    """
    try:
        from api.main import model_registry

        if all(k in model_registry for k in ("xgb_final", "lgbm_final", "calibrator")):
            registry: dict[str, Any] = model_registry
            return registry
    except ImportError:
        pass
    return _load_local_retention_artifacts()


# ─── Feature scaling (Fix #6 — training-serving consistency) ──────────────────

def build_scaled_feature_row(
    features: dict[str, float],
    feature_columns: list[str],
    scaler: Any,
) -> Any:
    """
    Build a model-ready feature row in `feature_columns` order, applying
    the SAME saved scaler used at training time — never re-fit at
    inference (Fix #6). Only the continuous columns tabular_pipeline.py
    scaled (NUMERIC_FEATURE_COLS) are transformed; binary flags
    (is_single_purchase, has_reviews) pass through untouched, exactly as
    fit_and_save_scaler() left them at training time.
    """
    import numpy as np

    from data.pipelines.tabular_pipeline import NUMERIC_FEATURE_COLS

    scale_cols = [c for c in NUMERIC_FEATURE_COLS if c in feature_columns]
    scaled_values = dict(features)

    if scaler is not None and scale_cols:
        sub = np.array([[features[c] for c in scale_cols]], dtype=float)
        scaled_sub = scaler.transform(sub)
        for col, val in zip(scale_cols, scaled_sub[0], strict=True):
            scaled_values[col] = float(val)

    row = np.array([[scaled_values[c] for c in feature_columns]], dtype=float)
    return row


def _risk_band(probability: float) -> str:
    if probability < RISK_BAND_LOW:
        return "LOW"
    if probability < RISK_BAND_HIGH:
        return "MEDIUM"
    return "HIGH"


def _cache_explanation(request_id: str, row: Any, customer_id: str) -> None:
    if len(_EXPLANATION_CACHE) >= _EXPLANATION_CACHE_MAX:
        oldest_key = next(iter(_EXPLANATION_CACHE))
        del _EXPLANATION_CACHE[oldest_key]
    _EXPLANATION_CACHE[request_id] = (row, customer_id)


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post(
    "/score",
    response_model=RetentionScoreResponse,
    summary="Score a customer's churn risk",
    description=(
        "Score a customer's 90-day churn probability from RFM/behavioural/"
        "sentiment features. Returns a calibrated probability, risk band, "
        "recommended action, and top-3 SHAP risk factors."
    ),
)
async def score_retention(
    request: CustomerFeaturesRequest,
    model_registry: dict[str, Any] = Depends(get_model_registry),
) -> RetentionScoreResponse:
    t0 = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    xgb_model = model_registry.get("xgb_final")
    lgbm_model = model_registry.get("lgbm_final")
    calibrator = model_registry.get("calibrator")
    feature_columns = model_registry.get("feature_columns")

    if xgb_model is None or lgbm_model is None or calibrator is None or not feature_columns:
        raise HTTPException(
            status_code=503,
            detail="Retention models not loaded — run models/retention/train.py "
                   "and models/retention/calibrate.py first.",
        )

    features = request.model_dump(exclude={"customer_id"})
    scaler = model_registry.get("scaler")
    row = build_scaled_feature_row(features, feature_columns, scaler)

    xgb_proba = float(xgb_model.predict_proba(row)[:, 1][0])
    lgbm_proba = float(lgbm_model.predict_proba(row)[:, 1][0])
    ensemble_proba = build_ensemble_prediction(xgb_proba, lgbm_proba)

    calib_metrics = model_registry.get("calibration_metrics", {})
    calibration_method = calib_metrics.get("calibration_method", "isotonic_regression")
    decision_threshold = float(calib_metrics.get("decision_threshold", 0.5))

    if calibration_method == "platt_scaling":
        calibrated = float(
            calibrator.predict_proba(np.array([[ensemble_proba]]))[0, 1]
        )
    else:
        calibrated = float(calibrator.predict(np.array([ensemble_proba]))[0])

    risk_band = _risk_band(calibrated)
    recommended_action = "RETENTION_OFFER" if calibrated >= decision_threshold else "NONE"

    explainer = model_registry.get("shap_explainer")
    top_risk_factors: list[RiskFactor] = []
    if explainer is not None:
        shap_values, _ = compute_shap_values(explainer, row)
        ranked = top_features_with_direction(
            shap_values,
            row[0],
            feature_columns,
            model_registry.get("reference_distribution"),
            top_n=TOP_RISK_FACTORS_N,
        )
        top_risk_factors = [
            RiskFactor(feature=f["feature"], shap_value=f["shap_value"], direction=f["direction"])
            for f in ranked
        ]

    _cache_explanation(request_id, row, request.customer_id)

    inference_ms = int((time.time() - t0) * 1000)

    return RetentionScoreResponse(
        request_id=request_id,
        customer_id=request.customer_id,
        churn_probability=round(calibrated, 4),
        risk_band=risk_band,
        recommended_action=recommended_action,
        top_risk_factors=top_risk_factors,
        churn_label_definition=CHURN_LABEL_DEFINITION,
        is_single_purchase_customer=bool(request.is_single_purchase),
        model_version=MODEL_VERSION,
        calibration_method=calibration_method,
        decision_threshold=decision_threshold,
        inference_ms=inference_ms,
    )


