# models/retention/calibrate.py
# E-CIP v3.0 — Retention Probability Calibration
# Blueprint Section 05 — Calibration & Threshold Optimization
#
# Calibration method: Platt Scaling (logistic regression on model outputs)
# for n_calibration < 1000, Isotonic Regression for n_calibration >= 1000.
#
# "Calibration set must be held out from training and SMOTE — use a clean
# split." This module fits the calibrator on OUT-OF-FOLD predictions from
# the 5-fold CV in models/retention/train.py, not on a fresh holdout split.
# OOF predictions are, by construction, never seen by the model that
# produced them (each fold's validation rows are excluded from that fold's
# SMOTE + fit) — this satisfies the "held out from training and SMOTE"
# requirement without needing a second, model-diverging holdout that would
# leave the shipped production model (fit on all labeled rows in train.py)
# uncalibrated against its own training data.
#
# Threshold optimisation: sweep [0.3, 0.7], optimise F-beta (beta=2) to
# weight recall higher than precision — missing a churner costs more than
# a wasted retention offer.
#
# Usage:
#   python models/retention/calibrate.py
#   python models/retention/calibrate.py --data data/feature_store/customer_features/rfm_behavioral_v2.parquet

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.retention.train import (
    BEST_PARAMS_PATH,
    DEFAULT_LGBM_PARAMS,
    FEATURE_STORE_PATH,
    N_SPLITS,
    SEED,
    _oof_predictions_lgbm,
    _oof_predictions_xgb,
    build_ensemble_prediction,
    load_best_params,
    load_feature_table,
)

# ─── Constants ────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path("models/retention/artifacts")
MLFLOW_EXPERIMENT = "retention_classifier"

PLATT_N_THRESHOLD = 1000  # blueprint §05: n < 1000 -> Platt, else Isotonic
THRESHOLD_SWEEP = [round(0.30 + 0.01 * i, 2) for i in range(41)]  # 0.30..0.70
FBETA_BETA = 2.0
TARGET_ECE = 0.05


# ─── Calibration ────────────────────────────────────────────────────────────────

def fit_calibrator(oof_proba: Any, y: Any) -> tuple[Any, str]:
    """
    Fit Platt Scaling (1-D logistic regression) or Isotonic Regression on
    out-of-fold probabilities, per blueprint §05's sample-size rule.
    Returns (fitted calibrator, method name).
    """
    n = len(y)
    if n < PLATT_N_THRESHOLD:
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression()
        calibrator.fit(oof_proba.reshape(-1, 1), y)
        return calibrator, "platt_scaling"

    from sklearn.isotonic import IsotonicRegression

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_proba, y)
    return calibrator, "isotonic_regression"


def apply_calibrator(calibrator: Any, method: str, proba: Any) -> Any:
    """Apply a fitted calibrator to raw ensemble probabilities."""
    if method == "platt_scaling":
        result = calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
    else:
        result = calibrator.predict(proba)
    return result


def compute_ece(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    """Expected Calibration Error — mean |accuracy - confidence| weighted by bin size."""
    import numpy as np

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        bin_conf = float(y_prob[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def save_reliability_diagram(
    y_true: Any,
    proba_before: Any,
    proba_after: Any,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram (before vs after calibration) as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    frac_before, mean_before = calibration_curve(y_true, proba_before, n_bins=n_bins)
    frac_after, mean_after = calibration_curve(y_true, proba_after, n_bins=n_bins)

    ax.plot(mean_before, frac_before, marker="o", label="Before calibration")
    ax.plot(mean_after, frac_after, marker="o", label="After calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Retention Ensemble — Reliability Diagram")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Threshold optimisation ─────────────────────────────────────────────────────

def optimise_threshold(
    y_true: Any,
    y_prob: Any,
    thresholds: list[float] = THRESHOLD_SWEEP,
    beta: float = FBETA_BETA,
) -> dict[str, float]:
    """
    Sweep thresholds in [0.3, 0.7], pick the one maximising F-beta (beta=2)
    to weight recall over precision — a missed churner costs more than an
    unnecessary retention offer.
    """
    from sklearn.metrics import fbeta_score, precision_score, recall_score

    best_threshold = thresholds[0]
    best_fbeta = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        fbeta = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold
            best_precision = float(precision_score(y_true, y_pred, zero_division=0))
            best_recall = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "decision_threshold": best_threshold,
        "fbeta": best_fbeta,
        "precision_at_threshold": best_precision,
        "recall_at_threshold": best_recall,
    }


# ─── Full calibration pipeline ───────────────────────────────────────────────────

def run_calibration(
    data_path: Path = FEATURE_STORE_PATH,
    params_path: Path = BEST_PARAMS_PATH,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> dict[str, Any]:
    """
    Fit a calibrator on out-of-fold ensemble predictions, compute ECE
    before/after, optimise the decision threshold, and save all artifacts.
    """
    print("=" * 60)
    print("  E-CIP v3.0 — Retention Probability Calibration")
    print("  Blueprint Section 05 — Calibration & Threshold Optimization")
    print("=" * 60)

    if not data_path.exists():
        print(f"\n  Feature table not found: {data_path}")
        print("  Run data/pipelines/tabular_pipeline.py first.")
        return {}

    features_array, y, _feature_cols = load_feature_table(data_path)
    print(f"\n  Samples: {len(y):,}")

    xgb_params = load_best_params(params_path)
    lgbm_params = dict(DEFAULT_LGBM_PARAMS)

    print("\n  Generating out-of-fold ensemble predictions (SMOTE inside each fold)...")
    xgb_oof, _ = _oof_predictions_xgb(features_array, y, xgb_params, n_splits, seed)
    lgbm_oof, _ = _oof_predictions_lgbm(features_array, y, lgbm_params, n_splits, seed)
    ensemble_oof = build_ensemble_prediction(xgb_oof, lgbm_oof)

    # A calibrator fit AND evaluated on the same OOF set is optimistic —
    # Isotonic Regression in particular can overfit a monotonic step function
    # to its own fitting data and report a trivially perfect ECE. Split the
    # OOF pool itself into a fit half and a held-out eval half so the
    # reported ECE reflects genuine held-out calibration quality.
    from sklearn.model_selection import train_test_split

    fit_idx, eval_idx = train_test_split(
        range(len(y)), test_size=0.5, stratify=y, random_state=seed
    )
    proba_fit, y_fit = ensemble_oof[fit_idx], y[fit_idx]
    proba_eval, y_eval = ensemble_oof[eval_idx], y[eval_idx]

    calibrator, method = fit_calibrator(proba_fit, y_fit)
    print(f"  Calibration method: {method} (fit n={len(y_fit):,}, "
          f"eval n={len(y_eval):,}, threshold={PLATT_N_THRESHOLD})")

    calibrated_eval = apply_calibrator(calibrator, method, proba_eval)

    ece_before = compute_ece(y_eval, proba_eval)
    ece_after = compute_ece(y_eval, calibrated_eval)
    print(f"  ECE before calibration (held-out half): {ece_before:.4f}")
    print(f"  ECE after calibration  (held-out half): {ece_after:.4f} (target < {TARGET_ECE})")

    threshold_result = optimise_threshold(y_eval, calibrated_eval)
    print(f"\n  Optimal decision threshold: {threshold_result['decision_threshold']:.2f}")
    print(f"    F-beta (beta={FBETA_BETA}): {threshold_result['fbeta']:.4f}")
    print(f"    Precision: {threshold_result['precision_at_threshold']:.4f}")
    print(f"    Recall   : {threshold_result['recall_at_threshold']:.4f}")

    # Refit the calibrator on the FULL OOF pool for the artifact that ships —
    # the fit/eval split above was only to get an honest ECE estimate.
    calibrator, method = fit_calibrator(ensemble_oof, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    import joblib
    calibrator_path = ARTIFACTS_DIR / "calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)
    print(f"\n  ✓ Calibrator saved: {calibrator_path}")

    diagram_path = ARTIFACTS_DIR / "reliability_diagram.png"
    save_reliability_diagram(y_eval, proba_eval, calibrated_eval, diagram_path)
    print(f"  ✓ Reliability diagram saved: {diagram_path}")

    results: dict[str, Any] = {
        "calibration_method": method,
        "ece_before": ece_before,
        "ece_after": ece_after,
        **threshold_result,
        "n_calibration_samples": len(y),
    }
    results_path = ARTIFACTS_DIR / "calibration_metrics.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"  ✓ Calibration metrics saved: {results_path}")

    try:
        import mlflow
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="retention_calibration"):
            mlflow.log_params({"calibration_method": method})
            mlflow.log_metrics({
                k: v for k, v in results.items() if isinstance(v, int | float)
            })
            mlflow.log_artifact(str(calibrator_path))
            mlflow.log_artifact(str(diagram_path))
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("  Calibration complete.")
    print("  Next: models/retention/shap_explain.py")
    print("=" * 60 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Retention Probability Calibration"
    )
    parser.add_argument("--data", type=Path, default=FEATURE_STORE_PATH)
    parser.add_argument("--params", type=Path, default=BEST_PARAMS_PATH)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    run_calibration(data_path=args.data, params_path=args.params, n_splits=args.n_splits)


if __name__ == "__main__":
    main()
