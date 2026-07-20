# api/routers/explain.py
# E-CIP v3.0 — SHAP Explanation API Router
# Blueprint Section 11 — Fix #29
#
# GET /v1/explain/shap/{request_id} — always JSON (ShapExplanationResponse),
# never a dual-mode "JSON or image/png" endpoint. Grad-CAM lives on its own
# endpoint (already served from api/routers/products.py's gradcam_url).

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.routers.retention import _EXPLANATION_CACHE, MODEL_VERSION, get_model_registry
from api.schemas.explain import ShapExplanationResponse, ShapFeature
from models.retention.shap_explain import (
    TOP_N_FEATURES,
    check_shap_sum_consistency,
    compute_shap_values,
    top_features_with_direction,
)

router = APIRouter(prefix="/v1/explain", tags=["Explainability"])


@router.get(
    "/shap/{request_id}",
    response_model=ShapExplanationResponse,
    summary="Get the SHAP explanation for a prior retention score",
    description=(
        "Returns the full top-10 SHAP feature attribution for a previous "
        "POST /v1/retention/score call. request_id is only valid for "
        "requests scored since the API process last restarted."
    ),
)
async def get_shap_explanation(
    request_id: str,
    model_registry: dict[str, Any] = Depends(get_model_registry),
) -> ShapExplanationResponse:
    cached = _EXPLANATION_CACHE.get(request_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No explanation cached for request_id={request_id!r} — "
                   "either it never existed or the API has restarted since.",
        )
    row, customer_id = cached

    xgb_model = model_registry.get("xgb_final")
    explainer = model_registry.get("shap_explainer")
    feature_columns = model_registry.get("feature_columns")

    if xgb_model is None or explainer is None or not feature_columns:
        raise HTTPException(status_code=503, detail="Retention models not loaded.")

    churn_probability = float(xgb_model.predict_proba(row)[:, 1][0])
    margin_score = float(xgb_model.predict(row, output_margin=True)[0])

    shap_values, expected_value = compute_shap_values(explainer, row)
    sum_check = check_shap_sum_consistency(shap_values, expected_value, margin_score)

    ranked = top_features_with_direction(
        shap_values,
        row[0],
        feature_columns,
        model_registry.get("reference_distribution"),
        top_n=TOP_N_FEATURES,
    )

    return ShapExplanationResponse(
        request_id=request_id,
        customer_id=customer_id,
        churn_probability=churn_probability,
        expected_value=expected_value,
        top_features=[ShapFeature(**f) for f in ranked],
        shap_sum_check=sum_check,
        model_version=MODEL_VERSION,
        generated_at=datetime.now(UTC).isoformat(),
    )
