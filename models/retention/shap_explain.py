# models/retention/shap_explain.py
# E-CIP v3.0 — Retention SHAP Explanations
# Blueprint Section 11 — Explainability Layer
#
# TreeSHAP via shap.TreeExplainer on the final XGBoost model (the ensemble's
# LightGBM half is not explained separately — the blueprint's
# ShapExplanationResponse schema describes a single top-10 feature list per
# prediction, not a dual-model breakdown).
#
# SHAP values and expected_value are computed in the model's raw margin
# (log-odds) space, not probability space. TreeSHAP only guarantees exact
# additivity — sum(shap_values) + expected_value == prediction — in the
# space the trees natively output; for XGBoost that's the margin score.
# This installed shap/xgboost combination's tree_path_dependent perturbation
# (the only mode that doesn't require a background dataset) only supports
# model_output="raw"; forcing model_output="probability" requires
# feature_perturbation="interventional", which raises NotImplementedError
# against this XGBoost version's categorical-aware tree encoding. Rather
# than silently forcing incompatible options, this module does the
# sum-consistency check honestly in margin space and reports
# churn_probability separately via predict_proba() for the API contract —
# a nonlinear sigmoid transform doesn't have a well-defined per-feature
# decomposition, so there is no way to make the check hold exactly in
# probability space without approximation.
#
# Sum-consistency check: |sum(shap_values) + expected_value - margin_score| < 0.05.
# This is the sanity check that catches a broken SHAP wiring (wrong model
# loaded, mismatched feature order) before it ever reaches an API response.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SHAP_SUM_TOLERANCE = 0.05
TOP_N_FEATURES = 10


def load_shap_explainer(xgb_model: Any) -> Any:
    """
    Build a TreeSHAP explainer for the final XGBoost model in the model's
    native raw margin (log-odds) space — see module docstring for why
    probability-space output isn't available for this shap/xgboost pairing.
    """
    import shap

    return shap.TreeExplainer(xgb_model)


def compute_shap_values(
    explainer: Any,
    features_row: Any,
) -> tuple[Any, float]:
    """
    Compute SHAP values (margin space) for a single row (1 x n_features).
    Returns (shap_values for that row, expected_value).
    """
    shap_values = explainer.shap_values(features_row)
    expected_value = explainer.expected_value

    # shap can return a list (one array per class) for some model types —
    # XGBClassifier binary output is a single array, but guard defensively.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if isinstance(expected_value, list | np.ndarray):
        expected_value = (
            expected_value[-1] if hasattr(expected_value, "__len__") else expected_value
        )

    return shap_values[0], float(expected_value)


def check_shap_sum_consistency(
    shap_values: Any,
    expected_value: float,
    margin_score: float,
) -> float:
    """Return |sum(shap) + expected_value - margin_score| — must be < SHAP_SUM_TOLERANCE."""
    return float(abs(np.sum(shap_values) + expected_value - margin_score))


def top_features_with_direction(
    shap_values: Any,
    feature_values: Any,
    feature_names: list[str],
    training_reference: dict[str, dict[str, float]] | None = None,
    top_n: int = TOP_N_FEATURES,
) -> list[dict[str, Any]]:
    """
    Rank features by |shap_value| (margin-space contribution) and attach
    direction + training-set percentile context, matching
    api/schemas/explain.py's ShapFeature.
    """
    order = np.argsort(-np.abs(shap_values))[:top_n]

    results: list[dict[str, Any]] = []
    for idx in order:
        idx = int(idx)
        feature = feature_names[idx]
        value = float(shap_values[idx])
        raw_value = float(feature_values[idx])

        percentile = 50.0
        if training_reference and feature in training_reference:
            ref = training_reference[feature]
            spread = max(ref.get("p95", 0.0) - ref.get("p25", 0.0), 1e-9)
            percentile = float(
                np.clip((raw_value - ref.get("p25", 0.0)) / spread * 70.0 + 25.0, 0.0, 100.0)
            )

        results.append({
            "feature": feature,
            "shap_value": value,
            "direction": "increases_churn" if value > 0 else "decreases_churn",
            "feature_value": raw_value,
            "percentile_in_training": percentile,
        })

    return results


def explain_prediction(
    explainer: Any,
    xgb_model: Any,
    features_row: Any,
    feature_names: list[str],
    request_id: str,
    customer_id: str,
    model_version: str,
    training_reference: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    Full SHAP explanation for one prediction, shaped exactly like
    api/schemas/explain.py:ShapExplanationResponse.

    `explainer` should be built once (via load_shap_explainer) and reused
    across requests — rebuilding a TreeExplainer per call is wasteful.
    churn_probability comes from predict_proba(); the sum-consistency
    check is computed against the model's raw margin score instead (see
    module docstring) since that's what SHAP actually guarantees.
    """
    churn_probability = float(xgb_model.predict_proba(features_row)[0, 1])
    margin_score = float(xgb_model.predict(features_row, output_margin=True)[0])

    shap_values, expected_value = compute_shap_values(explainer, features_row)

    sum_check = check_shap_sum_consistency(shap_values, expected_value, margin_score)
    if sum_check >= SHAP_SUM_TOLERANCE:
        raise ValueError(
            f"SHAP sum-consistency check failed: {sum_check:.4f} >= {SHAP_SUM_TOLERANCE} "
            "— check model/feature-order wiring before trusting this explanation."
        )

    top_features = top_features_with_direction(
        shap_values, features_row[0], feature_names, training_reference
    )

    return {
        "request_id": request_id,
        "customer_id": customer_id,
        "churn_probability": churn_probability,
        "expected_value": expected_value,
        "top_features": top_features,
        "shap_sum_check": sum_check,
        "model_version": model_version,
        "generated_at": datetime.now(UTC).isoformat(),
    }


# ─── Standalone verification run ───────────────────────────────────────────────

def main() -> None:
    """
    Smoke-test SHAP wiring against the real trained models: load
    xgb_final.joblib, explain a handful of real customer rows, and verify
    the sum-consistency check passes on all of them.
    """
    import joblib

    weights_dir = Path("models/retention/weights")
    artifacts_dir = Path("models/retention/artifacts")
    data_path = Path("data/feature_store/customer_features/rfm_behavioral_v2.parquet")

    xgb_path = weights_dir / "xgb_final.joblib"
    if not xgb_path.exists() or not data_path.exists():
        print("  Trained model or feature table not found — run train.py first.")
        return

    import pandas as pd

    xgb_model = joblib.load(xgb_path)
    feature_cols = json.loads((artifacts_dir / "feature_columns.json").read_text())

    df = pd.read_parquet(data_path).dropna(subset=["churned"])
    exclude_cols = {"CustomerID", "churned", "last_order_date", "first_order_date"}
    feature_cols = [c for c in feature_cols if c not in exclude_cols]

    sample = df.sample(n=min(20, len(df)), random_state=42)
    x_sample = sample[feature_cols].fillna(0).to_numpy(dtype=float)

    explainer = load_shap_explainer(xgb_model)

    max_sum_check = 0.0
    result: dict[str, Any] = {}
    for i in range(len(sample)):
        result = explain_prediction(
            explainer,
            xgb_model,
            x_sample[i : i + 1],
            feature_cols,
            request_id=f"smoketest_{i}",
            customer_id=str(sample.iloc[i]["CustomerID"]),
            model_version="retention_ensemble_v1.0.0",
        )
        max_sum_check = max(max_sum_check, result["shap_sum_check"])

    print(f"  ✓ SHAP sum-consistency check passed on {len(sample)} samples "
          f"(max |sum_check| = {max_sum_check:.4f}, tolerance {SHAP_SUM_TOLERANCE})")
    print(f"  Example top feature: {result['top_features'][0]}")


if __name__ == "__main__":
    main()
