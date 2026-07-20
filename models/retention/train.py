# models/retention/train.py
# E-CIP v3.0 — Retention Ensemble Training (XGBoost + LightGBM)
# Blueprint Section 10 — Critical Fix #10, Fix #19, Fix #21
#
# Fix #10: SMOTE applied INSIDE the CV loop, on the training fold only.
#          Resampling before the train/val split leaks synthetic
#          neighbours of validation-fold minority points into training —
#          this is the single most important invariant in this file.
# Fix #19: Hyperparameters come from the SQLite-persisted Optuna study
#          (mlops/optuna_search.py) via mlops/artifacts/best_retention_params.json.
# Fix #21: time_decay_lambda is a tabular_pipeline.py feature-engineering
#          parameter, not a model hyperparameter — stripped before it
#          reaches XGBClassifier/LGBMClassifier.
#
# Two-phase training:
#   Phase 1 — 5-fold stratified CV (SMOTE inside each fold) for an
#             honest, leakage-free estimate of generalisation AUC.
#   Phase 2 — Full retrain: refit on ALL labeled rows (SMOTE applied once,
#             on the whole training set) to produce the model that ships.
#             A dedicated calibration holdout is carved out in Phase 4
#             Week 12 (models/retention/calibrate.py) — this script does
#             not touch that split.
#
# Usage:
#   python models/retention/train.py
#   python models/retention/train.py --data data/feature_store/customer_features/rfm_behavioral_v1.parquet
#   python models/retention/train.py --n-splits 5 --xgb-weight 0.5

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_STORE_PATH = Path("data/feature_store/customer_features/rfm_behavioral_v1.parquet")
BEST_PARAMS_PATH = Path("mlops/artifacts/best_retention_params.json")
WEIGHTS_DIR = Path("models/retention/weights")
ARTIFACTS_DIR = Path("models/retention/artifacts")
MLFLOW_EXPERIMENT = "retention_classifier"

N_SPLITS = 5
SEED = 42
DEFAULT_XGB_WEIGHT = 0.5  # ensemble weight — LightGBM gets (1 - this)

# Columns that are identifiers/labels/dates, never model features
EXCLUDE_COLS = {
    "CustomerID", "churned", "last_order_date", "first_order_date",
}

# Keys in best_retention_params.json that are not XGBClassifier/LGBMClassifier
# constructor arguments — Fix #21 (time_decay_lambda belongs to the feature
# pipeline) plus bookkeeping fields written by mlops/optuna_search.py.
NON_MODEL_PARAM_KEYS = {"time_decay_lambda", "status", "best_cv_auc"}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
}

# Optuna does not yet have a dedicated LightGBM study runner wired into
# mlops/optuna_search.py's CLI (lgbm_objective exists but is unused) — use
# search-space midpoints as defaults until that's added.
DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbose": -1,
}


# ─── Feature table loading ─────────────────────────────────────────────────────

def load_feature_table(data_path: Path) -> Any:
    """Load the labeled customer feature table and split into X, y, columns."""
    import pandas as pd

    df = pd.read_parquet(data_path)
    if "churned" not in df.columns:
        raise ValueError(
            "'churned' column missing — run models/retention/churn_label_engineer.py "
            "and data/pipelines/tabular_pipeline.py --churn-labels first."
        )

    df_labeled = df.dropna(subset=["churned"]).copy()
    feature_cols = [c for c in df_labeled.columns if c not in EXCLUDE_COLS]

    features_array = df_labeled[feature_cols].fillna(0).to_numpy(dtype=float)
    y = df_labeled["churned"].astype(int).to_numpy()

    return features_array, y, feature_cols


# ─── Hyperparameters ────────────────────────────────────────────────────────────

def load_best_params(
    params_path: Path = BEST_PARAMS_PATH,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load Optuna-tuned hyperparameters (Fix #19), stripped of any key that
    is not a model constructor argument (Fix #21 — time_decay_lambda lives
    in the feature pipeline, not the model).

    Falls back to `defaults` if the artifact is missing or unreadable.
    """
    fallback = dict(defaults) if defaults is not None else dict(DEFAULT_XGB_PARAMS)

    if not params_path.exists():
        print(f"  No tuned params found at {params_path} — using defaults.")
        return fallback

    raw: dict[str, Any] = json.loads(params_path.read_text())
    cleaned = {k: v for k, v in raw.items() if k not in NON_MODEL_PARAM_KEYS}

    if not cleaned:
        print(f"  {params_path} contained no usable params — using defaults.")
        return fallback

    return cleaned


# ─── SMOTE (Fix #10) ────────────────────────────────────────────────────────────

def _smote_resample(
    x_train: Any,
    y_train: Any,
    seed: int = SEED,
) -> tuple[Any, Any]:
    """
    Apply SMOTE to a training fold ONLY. Never call this on data that
    includes a validation or test split (Critical Fix #10 — data leakage).

    k_neighbors is capped below the minority class count so SMOTE doesn't
    raise on small folds.
    """
    from imblearn.over_sampling import SMOTE

    minority_count = int(min((y_train == 0).sum(), (y_train == 1).sum()))
    k_neighbors = max(1, min(5, minority_count - 1))

    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    x_res, y_res = smote.fit_resample(x_train, y_train)
    return x_res, y_res


# ─── XGBoost — CV with SMOTE inside the loop ───────────────────────────────────

def _oof_predictions_xgb(
    features_array: Any,
    y: Any,
    params: dict[str, Any],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> tuple[Any, list[float]]:
    """
    Blueprint Critical Fix #10: SMOTE fit_resample is called INSIDE the
    per-fold loop, on X_train/y_train only — X_val/y_val are never touched
    by the resampler. Returns out-of-fold probabilities plus per-fold AUCs.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []

    model_params = {k: v for k, v in params.items() if k not in NON_MODEL_PARAM_KEYS}

    for train_idx, val_idx in skf.split(features_array, y):
        x_train, x_val = features_array[train_idx], features_array[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fix #10: resample the training fold only — never the full array.
        x_train_res, y_train_res = _smote_resample(x_train, y_train, seed=seed)

        model = XGBClassifier(**model_params)
        model.fit(x_train_res, y_train_res)

        val_proba = model.predict_proba(x_val)[:, 1]
        oof_proba[val_idx] = val_proba
        fold_aucs.append(float(roc_auc_score(y_val, val_proba)))

    return oof_proba, fold_aucs


def train_with_smote_cv(
    features_array: Any,
    y: Any,
    params: dict[str, Any],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    5-fold stratified CV for XGBoost with SMOTE applied inside each fold
    (Critical Fix #10). Returns (auc_mean, auc_std) across folds.

    Called by mlops/optuna_search.py:retention_objective — signature must
    stay (features_array, y, params) -> tuple[float, float].
    """
    import numpy as np

    _, fold_aucs = _oof_predictions_xgb(features_array, y, params, n_splits, seed)
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


# ─── LightGBM — CV with SMOTE inside the loop ──────────────────────────────────

def _oof_predictions_lgbm(
    features_array: Any,
    y: Any,
    params: dict[str, Any],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> tuple[Any, list[float]]:
    """LightGBM equivalent of _oof_predictions_xgb — same SMOTE-in-fold rule."""
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []

    model_params = {k: v for k, v in params.items() if k not in NON_MODEL_PARAM_KEYS}

    for train_idx, val_idx in skf.split(features_array, y):
        x_train, x_val = features_array[train_idx], features_array[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        x_train_res, y_train_res = _smote_resample(x_train, y_train, seed=seed)

        model = LGBMClassifier(**model_params)
        model.fit(x_train_res, y_train_res)

        val_proba = np.asarray(model.predict_proba(x_val))[:, 1]
        oof_proba[val_idx] = val_proba
        fold_aucs.append(float(roc_auc_score(y_val, val_proba)))

    return oof_proba, fold_aucs


def train_lgbm_cv(
    features_array: Any,
    y: Any,
    params: dict[str, Any],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    5-fold stratified CV for LightGBM with SMOTE applied inside each fold.
    Returns (auc_mean, auc_std). Same call contract as train_with_smote_cv,
    used by mlops/optuna_search.py:lgbm_objective.
    """
    import numpy as np

    _, fold_aucs = _oof_predictions_lgbm(features_array, y, params, n_splits, seed)
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


# ─── Ensemble ───────────────────────────────────────────────────────────────────

def build_ensemble_prediction(
    xgb_proba: Any,
    lgbm_proba: Any,
    xgb_weight: float = DEFAULT_XGB_WEIGHT,
) -> Any:
    """
    Weighted average of XGBoost + LightGBM churn probabilities.
    xgb_weight in [0, 1] — LightGBM gets (1 - xgb_weight).
    """
    if not 0.0 <= xgb_weight <= 1.0:
        raise ValueError(f"xgb_weight must be in [0, 1], got {xgb_weight}")
    return xgb_weight * xgb_proba + (1.0 - xgb_weight) * lgbm_proba


def evaluate_ensemble_cv(
    features_array: Any,
    y: Any,
    xgb_params: dict[str, Any],
    lgbm_params: dict[str, Any],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    xgb_weight: float = DEFAULT_XGB_WEIGHT,
) -> dict[str, float]:
    """
    Out-of-fold evaluation of the ensemble. Both base learners are run on
    the SAME StratifiedKFold split (identical seed/n_splits) so their
    out-of-fold probabilities line up row-for-row before averaging —
    this gives a leakage-free AUC estimate for the ensemble itself, not
    just each base learner individually.
    """
    from sklearn.metrics import roc_auc_score

    xgb_oof, xgb_fold_aucs = _oof_predictions_xgb(
        features_array, y, xgb_params, n_splits, seed
    )
    lgbm_oof, lgbm_fold_aucs = _oof_predictions_lgbm(
        features_array, y, lgbm_params, n_splits, seed
    )

    ensemble_oof = build_ensemble_prediction(xgb_oof, lgbm_oof, xgb_weight)

    import numpy as np

    return {
        "xgb_cv_auc_mean": float(np.mean(xgb_fold_aucs)),
        "xgb_cv_auc_std": float(np.std(xgb_fold_aucs)),
        "lgbm_cv_auc_mean": float(np.mean(lgbm_fold_aucs)),
        "lgbm_cv_auc_std": float(np.std(lgbm_fold_aucs)),
        "ensemble_cv_auc": float(roc_auc_score(y, ensemble_oof)),
        "xgb_weight": xgb_weight,
    }


# ─── Phase 2 — final full retrain ───────────────────────────────────────────────

def fit_final_xgboost(features_array: Any, y: Any, params: dict[str, Any], seed: int = SEED) -> Any:
    """
    Refit XGBoost on ALL labeled rows (Phase 2 — "full retrain on all
    folds"). SMOTE is applied once, on the full training set — this is
    the model that ships, not a CV artifact.
    """
    from xgboost import XGBClassifier

    model_params = {k: v for k, v in params.items() if k not in NON_MODEL_PARAM_KEYS}
    x_res, y_res = _smote_resample(features_array, y, seed=seed)
    model = XGBClassifier(**model_params)
    model.fit(x_res, y_res)
    return model


def fit_final_lgbm(features_array: Any, y: Any, params: dict[str, Any], seed: int = SEED) -> Any:
    """LightGBM equivalent of fit_final_xgboost."""
    from lightgbm import LGBMClassifier

    model_params = {k: v for k, v in params.items() if k not in NON_MODEL_PARAM_KEYS}
    x_res, y_res = _smote_resample(features_array, y, seed=seed)
    model = LGBMClassifier(**model_params)
    model.fit(x_res, y_res)
    return model


# ─── Full training pipeline ─────────────────────────────────────────────────────

def run_retention_training(
    data_path: Path = FEATURE_STORE_PATH,
    params_path: Path = BEST_PARAMS_PATH,
    n_splits: int = N_SPLITS,
    xgb_weight: float = DEFAULT_XGB_WEIGHT,
    seed: int = SEED,
) -> dict[str, Any]:
    """
    Full two-phase retention ensemble training:
      Phase 1 — SMOTE-in-fold CV for both base learners + ensemble OOF AUC.
      Phase 2 — full retrain of both base learners on all labeled rows.
    Logs everything to MLflow and saves models + metrics to disk.
    """
    print("=" * 60)
    print("  E-CIP v3.0 — Retention Ensemble Training")
    print("  Blueprint Section 10 — Critical Fix #10, Fix #19, Fix #21")
    print("=" * 60)

    if not data_path.exists():
        print(f"\n  Feature table not found: {data_path}")
        print("  Run data/pipelines/tabular_pipeline.py first.")
        return {}

    features_array, y, feature_cols = load_feature_table(data_path)
    print(f"\n  Samples   : {len(y):,}")
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Churn rate: {y.mean():.1%}")

    xgb_params = load_best_params(params_path, defaults=DEFAULT_XGB_PARAMS)
    lgbm_params = dict(DEFAULT_LGBM_PARAMS)
    print(f"\n  XGBoost params : {xgb_params}")
    print(f"  LightGBM params: {lgbm_params}")

    try:
        import mlflow
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except ImportError:
        mlflow = None  # type: ignore[assignment]

    with (mlflow.start_run(run_name="retention_ensemble_train") if mlflow else _NullContext()):
        if mlflow:
            mlflow.log_params({
                **{f"xgb_{k}": v for k, v in xgb_params.items()},
                **{f"lgbm_{k}": v for k, v in lgbm_params.items()},
                "n_splits": n_splits,
                "xgb_weight": xgb_weight,
                "n_samples": len(y),
                "n_features": len(feature_cols),
            })

        # ── Phase 1: SMOTE-in-fold CV ──────────────────────────────────────
        print("\n  Phase 1: 5-fold CV (SMOTE inside each fold)...")
        cv_metrics = evaluate_ensemble_cv(
            features_array, y, xgb_params, lgbm_params,
            n_splits=n_splits, seed=seed, xgb_weight=xgb_weight,
        )
        print(f"    XGBoost  CV AUC: {cv_metrics['xgb_cv_auc_mean']:.4f} "
              f"± {cv_metrics['xgb_cv_auc_std']:.4f}")
        print(f"    LightGBM CV AUC: {cv_metrics['lgbm_cv_auc_mean']:.4f} "
              f"± {cv_metrics['lgbm_cv_auc_std']:.4f}")
        print(f"    Ensemble CV AUC: {cv_metrics['ensemble_cv_auc']:.4f}")

        if mlflow:
            mlflow.log_metrics(cv_metrics)

        # ── Phase 2: full retrain on all folds ─────────────────────────────
        print("\n  Phase 2: full retrain on all labeled rows...")
        xgb_model = fit_final_xgboost(features_array, y, xgb_params, seed=seed)
        lgbm_model = fit_final_lgbm(features_array, y, lgbm_params, seed=seed)

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        import joblib
        xgb_path = WEIGHTS_DIR / "xgb_final.joblib"
        lgbm_path = WEIGHTS_DIR / "lgbm_final.joblib"
        joblib.dump(xgb_model, xgb_path)
        joblib.dump(lgbm_model, lgbm_path)
        print(f"    ✓ XGBoost final model saved : {xgb_path}")
        print(f"    ✓ LightGBM final model saved: {lgbm_path}")

        feature_cols_path = ARTIFACTS_DIR / "feature_columns.json"
        feature_cols_path.write_text(json.dumps(feature_cols, indent=2))

        results: dict[str, Any] = {
            **cv_metrics,
            "xgb_model_path": str(xgb_path),
            "lgbm_model_path": str(lgbm_path),
            "feature_columns": feature_cols,
            "n_samples": len(y),
            "churn_rate": float(y.mean()),
        }

        metrics_path = ARTIFACTS_DIR / "training_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2))
        print(f"    ✓ Metrics saved: {metrics_path}")

        if mlflow:
            mlflow.log_artifact(str(xgb_path))
            mlflow.log_artifact(str(lgbm_path))
            mlflow.log_artifact(str(metrics_path))

    print("\n" + "=" * 60)
    print("  Retention ensemble training complete.")
    print("  Next: models/retention/calibrate.py (Platt/Isotonic + holdout)")
    print("=" * 60 + "\n")

    return results


# ─── Null context manager (when MLflow not available) ──────────────────────────

class _NullContext:
    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ─── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Retention Ensemble Training (XGBoost + LightGBM)"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=FEATURE_STORE_PATH,
        help="Labeled customer feature table (.parquet)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=BEST_PARAMS_PATH,
        help="Optuna-tuned XGBoost hyperparameters (.json)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help=f"CV folds (default: {N_SPLITS})",
    )
    parser.add_argument(
        "--xgb-weight",
        type=float,
        default=DEFAULT_XGB_WEIGHT,
        help=f"Ensemble weight for XGBoost, LightGBM gets 1-weight (default: {DEFAULT_XGB_WEIGHT})",
    )
    args = parser.parse_args()

    run_retention_training(
        data_path=args.data,
        params_path=args.params,
        n_splits=args.n_splits,
        xgb_weight=args.xgb_weight,
    )


if __name__ == "__main__":
    main()
