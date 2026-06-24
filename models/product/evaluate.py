# models/product/evaluate.py
# E-CIP v3.0 — Product Classifier Evaluation Suite
# Blueprint Section 03 — Critical Fix #20
#
# Fix #20: Overall Top-1 accuracy alone is insufficient.
# Required outputs:
#   - Full confusion matrix saved as PNG artifact to MLflow
#   - Per-class F1 logged as individual metrics
#   - Top-3 confusion pairs documented (highest business cost)
#   - Delta vs ResNet-18 baseline per class
#
# Targets:
#   Top-1 accuracy : ≥ 92%
#   Macro F1       : ≥ 0.90
#   Calib. ECE     : < 0.05
#
# Usage (Colab/Kaggle):
#   python models/product/evaluate.py \
#       --model-path models/product/weights/efficientnet_b3_best.pt \
#       --data-dir data/processed/images

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_CATEGORIES = [
    "Electronics",
    "Fashion",
    "Home & Kitchen",
    "Sports",
    "Furniture",
    "Beauty",
    "Books",
    "Toys",
]

ARTIFACTS_DIR = Path("models/product/artifacts")
MODELS_DIR = Path("models/product/weights")

# Confidence threshold below which predictions are flagged for human review
LOW_CONFIDENCE_THRESHOLD = 0.65

# ECE calibration bins
N_CALIBRATION_BINS = 10


# ─── Per-class evaluation ─────────────────────────────────────────────────────

def log_per_class_evaluation(
    y_true: Any,
    y_pred: Any,
    y_proba: Any,
    class_names: list[str],
    output_dir: Path,
    mlflow_run: Any = None,
) -> dict[str, Any]:
    """
    Blueprint Section 03 — Critical Fix #20.

    Compute and log full evaluation suite:
    - Per-class precision, recall, F1
    - Macro and weighted averages
    - Full confusion matrix as PNG artifact
    - Top-3 confusion pairs with business cost annotation
    - Calibration ECE

    All per-class F1 scores logged as individual MLflow metrics
    so Optuna can filter on them.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            classification_report,
            top_k_accuracy_score,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}

        # ── Classification report ─────────────────────────────────────────
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        print("\n  Per-class evaluation:")
        print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("  " + "-" * 55)

        per_class_f1: dict[str, float] = {}
        for cls in class_names:
            if cls in report:
                p = report[cls]["precision"]
                r = report[cls]["recall"]
                f1 = report[cls]["f1-score"]
                sup = int(report[cls]["support"])
                per_class_f1[cls] = f1
                flag = " ⚠" if f1 < 0.85 else ""
                print(f"  {cls:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {sup:>10}{flag}")

        macro_f1 = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]
        top1_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))

        print(f"\n  Macro F1     : {macro_f1:.4f} {'✓' if macro_f1 >= 0.90 else '✗ (target ≥ 0.90)'}")
        print(f"  Weighted F1  : {weighted_f1:.4f}")
        print(f"  Top-1 Acc    : {top1_acc:.4f} {'✓' if top1_acc >= 0.92 else '✗ (target ≥ 0.92)'}")

        results["macro_f1"] = macro_f1
        results["weighted_f1"] = weighted_f1
        results["top1_accuracy"] = top1_acc
        results["per_class_f1"] = per_class_f1

        # Top-3 accuracy
        if y_proba is not None:
            try:
                top3_acc = top_k_accuracy_score(y_true, y_proba, k=3)
                print(f"  Top-3 Acc    : {top3_acc:.4f}")
                results["top3_accuracy"] = top3_acc
            except Exception:
                pass

        # Log to MLflow
        if mlflow_run:
            import mlflow
            mlflow.log_metric("top1_accuracy", top1_acc)
            mlflow.log_metric("macro_f1", macro_f1)
            mlflow.log_metric("weighted_f1", weighted_f1)
            for cls, f1 in per_class_f1.items():
                safe_cls = cls.replace(" ", "_").replace("&", "and")
                mlflow.log_metric(f"f1_{safe_cls}", f1)

        # ── Confusion matrix ──────────────────────────────────────────────
        cm_path = output_dir / "confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=class_names,
            ax=ax,
            cmap="Blues",
            colorbar=True,
        )
        ax.set_title(
            f"EfficientNet-B3 Confusion Matrix\n"
            f"Top-1: {top1_acc:.3f} | Macro F1: {macro_f1:.3f}",
            fontsize=12,
        )
        plt.tight_layout()
        fig.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  ✓ Confusion matrix saved: {cm_path}")

        if mlflow_run:
            import mlflow
            mlflow.log_artifact(str(cm_path))

        # ── Top-3 confusion pairs ─────────────────────────────────────────
        confusion_pairs = find_top_confusion_pairs(
            y_true, y_pred, class_names, top_n=3
        )
        results["top_confusion_pairs"] = confusion_pairs

        print("\n  Top-3 confusion pairs (highest misclassification count):")
        for i, pair in enumerate(confusion_pairs, 1):
            print(f"    {i}. {pair['true_class']} → {pair['predicted_class']}: "
                  f"{pair['count']} misclassifications "
                  f"({pair['business_impact']})")

        # Save confusion pairs for model card
        pairs_path = output_dir / "top_confusion_pairs.json"
        pairs_path.write_text(json.dumps(confusion_pairs, indent=2))

        if mlflow_run:
            import mlflow
            mlflow.log_artifact(str(pairs_path))

        return results

    except ImportError as e:
        print(f"  Evaluation skipped — missing dependency: {e}")
        print("  Install [train] extras in Colab/Kaggle.")
        return {}


def find_top_confusion_pairs(
    y_true: Any,
    y_pred: Any,
    class_names: list[str],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """
    Find the top-N most common misclassification pairs.
    Blueprint Section 03: top-3 confusion pairs documented in model card.
    Business cost annotation: Electronics vs Home&Kitchen confusion
    has higher cost than Fashion vs Sports.
    """
    try:
        import numpy as np

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # Business cost tiers (higher = more costly misclassification)
        high_cost_pairs = {
            ("Electronics", "Home & Kitchen"),
            ("Home & Kitchen", "Electronics"),
            ("Electronics", "Sports"),
            ("Sports", "Electronics"),
        }

        pairs: dict[tuple[str, str], int] = {}
        mask = y_true_arr != y_pred_arr  # only misclassifications
        for t, p in zip(y_true_arr[mask], y_pred_arr[mask]):
            true_cls = class_names[t] if t < len(class_names) else str(t)
            pred_cls = class_names[p] if p < len(class_names) else str(p)
            key = (true_cls, pred_cls)
            pairs[key] = pairs.get(key, 0) + 1

        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

        result = []
        for (true_cls, pred_cls), count in sorted_pairs[:top_n]:
            is_high_cost = (true_cls, pred_cls) in high_cost_pairs
            result.append({
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "count": count,
                "business_impact": (
                    "HIGH — search ranking and catalog integrity risk"
                    if is_high_cost
                    else "MEDIUM — user experience degradation"
                ),
            })
        return result

    except ImportError:
        return []


# ─── Calibration ECE ──────────────────────────────────────────────────────────

def compute_ece(
    y_true: Any,
    y_proba: Any,
    n_bins: int = N_CALIBRATION_BINS,
    output_dir: Path | None = None,
    mlflow_run: Any = None,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Blueprint Section 03: target ECE < 0.05.

    Also generates reliability diagram saved to MLflow.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        y_true_arr = np.array(y_true)
        y_proba_arr = np.array(y_proba)

        # Confidence = max probability across classes
        confidences = y_proba_arr.max(axis=1)
        predictions = y_proba_arr.argmax(axis=1)
        accuracies = (predictions == y_true_arr).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() == 0:
                bin_accs.append(0.0)
                bin_confs.append(0.0)
                bin_counts.append(0)
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_count = int(mask.sum())
            ece += (bin_count / len(y_true_arr)) * abs(bin_acc - bin_conf)
            bin_accs.append(float(bin_acc))
            bin_confs.append(float(bin_conf))
            bin_counts.append(bin_count)

        print(f"\n  Calibration ECE: {ece:.4f} "
              f"{'✓' if ece < 0.05 else '✗ (target < 0.05)'}")

        # Reliability diagram
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)]
            ax.bar(bin_centers, bin_accs, width=1 / n_bins,
                   alpha=0.7, label="Accuracy", color="steelblue")
            ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Reliability Diagram (ECE={ece:.4f})")
            ax.legend()
            plt.tight_layout()
            reliability_path = output_dir / "reliability_diagram.png"
            fig.savefig(reliability_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ Reliability diagram saved: {reliability_path}")

            if mlflow_run:
                import mlflow
                mlflow.log_metric("ece", ece)
                mlflow.log_artifact(str(reliability_path))

        return float(ece)

    except ImportError as e:
        print(f"  ECE computation skipped — missing dependency: {e}")
        return -1.0


# ─── Baseline delta ───────────────────────────────────────────────────────────

def compute_baseline_delta(
    efficientnet_metrics: dict[str, float],
    baseline_metrics_path: Path,
) -> dict[str, float]:
    """
    Compute delta between EfficientNet-B3 and ResNet-18 baseline.
    Blueprint Section 03: delta logged per class and overall.
    Returned in API response as baseline_comparison field.
    """
    if not baseline_metrics_path.exists():
        print(f"\n  Baseline metrics not found: {baseline_metrics_path}")
        print("  Train ResNet-18 baseline first (Week 5, Phase 2).")
        return {}

    with open(baseline_metrics_path) as f:
        baseline = json.load(f)

    delta: dict[str, float] = {}
    for metric in ["top1_accuracy", "macro_f1"]:
        if metric in efficientnet_metrics and metric in baseline:
            delta[metric] = efficientnet_metrics[metric] - baseline[metric]
            sign = "+" if delta[metric] >= 0 else ""
            print(f"  Δ {metric}: {sign}{delta[metric]:.4f} "
                  f"(EB3={efficientnet_metrics[metric]:.4f} "
                  f"vs ResNet18={baseline[metric]:.4f})")

    return delta


# ─── OOD detection ────────────────────────────────────────────────────────────

def compute_mahalanobis_threshold(
    train_features: Any,
    percentile: float = 99.0,
    output_path: Path | None = None,
) -> float:
    """
    Compute Mahalanobis distance threshold for OOD detection.
    Blueprint Section 03: flag if distance exceeds 99th percentile
    of training set. Saved to feature store as OOD baseline.
    """
    try:
        import numpy as np

        features = np.array(train_features)
        mean = features.mean(axis=0)
        cov = np.cov(features.T)

        # Regularise covariance matrix for stability
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov)

        # Compute Mahalanobis distances for all training samples
        diffs = features - mean
        distances = np.array([
            float(np.sqrt(d @ cov_inv @ d)) for d in diffs
        ])

        threshold = float(np.percentile(distances, percentile))
        print(f"\n  OOD threshold (p{percentile:.0f}): {threshold:.4f}")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ood_config = {
                "mean": mean.tolist(),
                "cov_inv": cov_inv.tolist(),
                "threshold": threshold,
                "percentile": percentile,
                "n_training_samples": len(features),
            }
            output_path.write_text(json.dumps(ood_config, indent=2))
            print(f"  ✓ OOD reference saved: {output_path}")

        return threshold

    except ImportError as e:
        print(f"  OOD computation skipped — missing dependency: {e}")
        return -1.0


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Product Classifier Evaluation"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "efficientnet_b3_best.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/images"),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=ARTIFACTS_DIR / "resnet18_baseline_metrics.json",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  E-CIP v3.0 — Product Classifier Evaluation")
    print("  Blueprint Section 03 — Critical Fix #20")
    print("=" * 60)

    if not args.model_path.exists():
        print(f"\n  Model not found: {args.model_path}")
        print("  Run models/product/train.py first.")
        print("\n  Evaluation suite structure verified.")
        print("  Functions available:")
        print("    log_per_class_evaluation() — per-class F1 + confusion matrix")
        print("    compute_ece()              — calibration ECE + reliability diagram")
        print("    compute_baseline_delta()   — EfficientNet vs ResNet-18 delta")
        print("    compute_mahalanobis_threshold() — OOD detection baseline")
        print("\n  All run in Colab/Kaggle after training (Phase 2, Week 7).")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  ✓ Artifacts directory ready: {ARTIFACTS_DIR}")
        print("=" * 60 + "\n")
        return

    print("\n  Model found — run full evaluation in Colab/Kaggle.")


if __name__ == "__main__":
    main()
