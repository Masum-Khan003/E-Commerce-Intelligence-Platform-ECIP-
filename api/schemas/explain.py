# api/schemas/explain.py
# E-CIP v3.0 — SHAP Explanation Response Schema
# Blueprint Section 11 — Fix #29
#
# Fix #29: ShapExplanationResponse is always JSON — never a dual-mode
# "JSON or image/png" endpoint like the ambiguous v2 spec. Grad-CAM (an
# actual image) lives on its own endpoint (/v1/explain/gradcam/{request_id}).

from __future__ import annotations

from pydantic import BaseModel, Field


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str = Field(description="'increases_churn' | 'decreases_churn'")
    feature_value: float = Field(description="Actual feature value, for reviewer context")
    percentile_in_training: float = Field(
        ge=0.0, le=100.0,
        description="Where this feature value falls in the training distribution",
    )


class ShapExplanationResponse(BaseModel):
    """Fix #29: always JSON, never dual-mode with image/png."""

    request_id: str
    customer_id: str
    churn_probability: float = Field(ge=0.0, le=1.0)
    expected_value: float = Field(description="SHAP base value (TreeExplainer.expected_value)")
    top_features: list[ShapFeature]
    shap_sum_check: float = Field(
        description="|sum(shap) + expected_value - prediction| — must be < 0.05"
    )
    model_version: str
    generated_at: str
