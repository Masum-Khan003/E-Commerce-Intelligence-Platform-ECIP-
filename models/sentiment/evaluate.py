# models/sentiment/evaluate.py
# E-CIP v3.0 — Sentiment Model Evaluation Suite
# Blueprint Section 04 — Fix #32
#
# Fix #32: Domain-shift evaluation required.
# DistilBERT trained on Electronics + Fashion.
# Must evaluate on 4 out-of-domain categories before deployment.
# Categories with F1 < 0.78 flagged as out-of-scope in model card.
#
# Evaluation targets:
#   Macro F1        : ≥ 0.88
#   Negative Recall : ≥ 0.85
#   Neutral Prec    : ≥ 0.75
#   Inference p95   : < 50ms
#
# Subgroup analysis (Blueprint Section 18):
#   - F1 by review length (short/medium/long)
#   - F1 by star rating (1★, 3★, 5★)
#   - F1 by category (in-domain vs out-of-domain)
#   - Delta vs SetFit baseline per sentiment class
#
# Usage:
#   python models/sentiment/evaluate.py
#   python models/sentiment/evaluate.py --model-path models/sentiment/weights/distilbert_sentiment_best.pt

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

MODELS_DIR = Path("models/sentiment/weights")
ARTIFACTS_DIR = Path("models/sentiment/artifacts")
TOKENIZER_ARTIFACT_DIR = Path("data/feature_store/artifacts/tokenizer_v1")

# Domain-shift evaluation categories (Fix #32)
OUT_OF_DOMAIN_CATEGORIES = [
    "Home & Kitchen",
    "Sports",
    "Toys",
    "Beauty",
]
OOD_F1_FLAG_THRESHOLD = 0.78  # categories below this flagged in model card

# Review length bands for subgroup analysis
LENGTH_BANDS = {
    "short": (0, 50),       # < 50 tokens
    "medium": (50, 300),    # 50–300 tokens
    "long": (300, 99999),   # > 300 tokens
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_test_split(csv_path: Path) -> tuple[list[str], list[int], list[dict]]:
    """
    Load test split CSV.
    Returns (texts, labels, metadata) where metadata has rating, source, char_length.
    """
    texts: list[str] = []
    labels: list[int] = []
    metadata: list[dict] = []

    if not csv_path.exists():
        return texts, labels, metadata

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label_str = row.get("label", "").strip().lower()
            if not text or label_str not in LABEL_MAP:
                continue
            texts.append(text)
            labels.append(LABEL_MAP[label_str])
            metadata.append({
                "rating": int(row.get("rating", 3)),
                "source": row.get("source", "unknown"),
                "char_length": int(row.get("char_length", len(text))),
            })

    return texts, labels, metadata


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference_batch(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
    batch_size: int = 32,
) -> tuple[list[int], list[float], float]:
    """
    Run batched inference on texts.
    Returns (predictions, confidences, p95_latency_ms).
    """
    try:
        import torch
        import torch.nn.functional as functional

        from models.sentiment.finetune import head_tail_tokenize

        model.eval()
        all_preds: list[int] = []
        all_confs: list[float] = []
        latencies: list[float] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            t0 = time.time()

            # Tokenize batch with head+tail truncation
            encodings = [
                head_tail_tokenize(text, tokenizer) for text in batch_texts
            ]
            input_ids = torch.tensor(
                [e["input_ids"] for e in encodings], dtype=torch.long
            ).to(device)
            attention_mask = torch.tensor(
                [e["attention_mask"] for e in encodings], dtype=torch.long
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probs = functional.softmax(outputs.logits, dim=-1)
                preds = probs.argmax(dim=-1)
                confs = probs.max(dim=-1).values

            latencies.append((time.time() - t0) / len(batch_texts) * 1000)
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

        # P95 latency
        latencies_sorted = sorted(latencies)
        p95_idx = int(len(latencies_sorted) * 0.95)
        p95_latency = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]

        return all_preds, all_confs, p95_latency

    except ImportError as e:
        print(f"  Inference skipped — missing dependency: {e}")
        return [], [], 0.0


# ─── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_core(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] = LABEL_NAMES,
) -> dict[str, Any]:
    """
    Compute core evaluation metrics.
    Blueprint Section 04 targets:
        Macro F1 ≥ 0.88, Negative Recall ≥ 0.85, Neutral Precision ≥ 0.75
    """
    try:
        from sklearn.metrics import (
            classification_report,
            f1_score,
        )

        report = classification_report(
            y_true, y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        neg_recall = report.get("negative", {}).get("recall", 0.0)
        neu_precision = report.get("neutral", {}).get("precision", 0.0)

        print("\n  Core evaluation metrics:")
        print(f"  {'Metric':<25} {'Value':>8} {'Target':>10} {'Status':>8}")
        print("  " + "-" * 55)

        checks = [
            ("Macro F1", macro_f1, 0.88, "≥ 0.88"),
            ("Negative Recall", neg_recall, 0.85, "≥ 0.85"),
            ("Neutral Precision", neu_precision, 0.75, "≥ 0.75"),
        ]
        for name, value, target, target_str in checks:
            status = "✓" if value >= target else "✗"
            print(f"  {name:<25} {value:>8.4f} {target_str:>10} {status:>8}")

        print("\n  Per-class breakdown:")
        for label in label_names:
            if label in report:
                p = report[label]["precision"]
                r = report[label]["recall"]
                f1 = report[label]["f1-score"]
                print(f"  {label:<12} P={p:.4f} R={r:.4f} F1={f1:.4f}")

        return {
            "macro_f1": round(macro_f1, 4),
            "negative_recall": round(neg_recall, 4),
            "neutral_precision": round(neu_precision, 4),
            "negative_f1": round(
                report.get("negative", {}).get("f1-score", 0.0), 4
            ),
            "neutral_f1": round(
                report.get("neutral", {}).get("f1-score", 0.0), 4
            ),
            "positive_f1": round(
                report.get("positive", {}).get("f1-score", 0.0), 4
            ),
            "n_samples": len(y_true),
        }

    except ImportError as e:
        print(f"  Evaluation skipped — missing dependency: {e}")
        return {}


# ─── Subgroup evaluation ──────────────────────────────────────────────────────

def evaluate_by_length(
    y_true: list[int],
    y_pred: list[int],
    metadata: list[dict],
) -> dict[str, Any]:
    """
    Blueprint Section 18 — Subgroup: F1 by review length.
    Short: < 50 tokens | Medium: 50–300 | Long: > 300
    Approximated by character length / 4.
    """
    try:
        from sklearn.metrics import f1_score

        results: dict[str, Any] = {}
        print("\n  Subgroup — Review length:")

        for band_name, (min_len, max_len) in LENGTH_BANDS.items():
            indices = [
                i for i, m in enumerate(metadata)
                if min_len <= m["char_length"] // 4 < max_len
            ]
            if len(indices) < 10:
                print(f"  {band_name:<10}: insufficient samples ({len(indices)})")
                continue

            band_true = [y_true[i] for i in indices]
            band_pred = [y_pred[i] for i in indices]
            f1 = f1_score(band_true, band_pred, average="macro", zero_division=0)
            results[f"f1_{band_name}"] = round(f1, 4)
            print(f"  {band_name:<10}: F1={f1:.4f} (n={len(indices)})")

        return results

    except ImportError:
        return {}


def evaluate_by_rating(
    y_true: list[int],
    y_pred: list[int],
    metadata: list[dict],
) -> dict[str, Any]:
    """
    Blueprint Section 18 — Subgroup: F1 by star rating (1★, 3★, 5★).
    """
    try:
        from sklearn.metrics import f1_score

        results: dict[str, Any] = {}
        print("\n  Subgroup — Star rating:")

        for rating in [1, 3, 5]:
            indices = [i for i, m in enumerate(metadata) if m["rating"] == rating]
            if len(indices) < 10:
                continue
            band_true = [y_true[i] for i in indices]
            band_pred = [y_pred[i] for i in indices]
            f1 = f1_score(band_true, band_pred, average="macro", zero_division=0)
            results[f"f1_rating_{rating}star"] = round(f1, 4)
            print(f"  {rating}★  : F1={f1:.4f} (n={len(indices)})")

        return results

    except ImportError:
        return {}


# ─── Domain-shift evaluation — Fix #32 ───────────────────────────────────────

def evaluate_domain_shift(
    model: Any,
    tokenizer: Any,
    device: Any,
    ood_samples_dir: Path,
    output_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    """
    Blueprint Section 04 — Fix #32.

    Evaluate DistilBERT on 100-review samples from 4 out-of-domain categories.
    Categories where F1 < 0.78 are flagged as out-of-scope in model card.

    OOD categories: Home & Kitchen, Sports, Toys, Beauty
    (Model trained on Electronics + Fashion only)
    """
    print("\n  Domain-shift evaluation (Fix #32):")
    print(f"  OOD categories: {OUT_OF_DOMAIN_CATEGORIES}")
    print(f"  Flag threshold: F1 < {OOD_F1_FLAG_THRESHOLD}")

    results: dict[str, Any] = {}
    flagged: list[str] = []

    for category in OUT_OF_DOMAIN_CATEGORIES:
        safe_name = category.lower().replace(" ", "_").replace("&", "and")
        sample_path = ood_samples_dir / f"domain_shift_{safe_name}_100.csv"

        if not sample_path.exists():
            print(f"  {category}: sample not found at {sample_path}")
            print("    Create 100 manually labeled reviews and save to above path.")
            results[f"ood_f1_{safe_name}"] = None
            continue

        texts, labels, _ = load_test_split(sample_path)
        if len(texts) < 10:
            print(f"  {category}: insufficient samples ({len(texts)})")
            continue

        preds, _, _ = run_inference_batch(model, tokenizer, texts, device)
        if not preds:
            continue

        try:
            from sklearn.metrics import f1_score
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            results[f"ood_f1_{safe_name}"] = round(f1, 4)

            flag = " ⚠ FLAGGED" if f1 < OOD_F1_FLAG_THRESHOLD else " ✓"
            print(f"  {category:<20}: F1={f1:.4f}{flag}")

            if f1 < OOD_F1_FLAG_THRESHOLD:
                flagged.append(category)

        except ImportError:
            pass

    results["flagged_ood_categories"] = flagged
    if flagged:
        print(f"\n  Categories flagged as out-of-scope: {flagged}")
        print("  Document in model card Known Limitations section.")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    ood_path = output_dir / "domain_shift_results.json"
    ood_path.write_text(json.dumps(results, indent=2))
    print(f"\n  ✓ Domain-shift results saved: {ood_path}")

    return results


# ─── Baseline delta ───────────────────────────────────────────────────────────

def compute_baseline_delta(
    distilbert_metrics: dict[str, Any],
    baseline_path: Path = ARTIFACTS_DIR / "setfit_baseline_metrics.json",
) -> dict[str, float]:
    """
    Compute F1 delta between DistilBERT and SetFit baseline.
    Blueprint Section 18: delta documented per sentiment class.
    """
    if not baseline_path.exists():
        print(f"\n  SetFit baseline metrics not found: {baseline_path}")
        print("  Run models/sentiment/setfit_baseline.py first.")
        return {}

    with open(baseline_path) as f:
        baseline = json.load(f)

    deltas: dict[str, float] = {}
    print("\n  Delta vs SetFit baseline:")

    for label in ["negative", "neutral", "positive"]:
        db_f1 = distilbert_metrics.get(f"{label}_f1", 0.0)
        sf_f1 = baseline.get(f"{label}_f1", 0.0)
        delta = db_f1 - sf_f1
        deltas[f"delta_{label}_f1"] = round(delta, 4)
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<12}: DistilBERT={db_f1:.4f} "
              f"SetFit={sf_f1:.4f} Δ={sign}{delta:.4f}")

    macro_delta = (
        distilbert_metrics.get("macro_f1", 0.0)
        - baseline.get("macro_f1", 0.0)
    )
    deltas["delta_macro_f1"] = round(macro_delta, 4)
    print(f"  {'macro':<12}: Δ={'+' if macro_delta >= 0 else ''}{macro_delta:.4f}")

    return deltas


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_dir: Path,
    macro_f1: float,
) -> None:
    """Save confusion matrix PNG to artifacts dir and log to MLflow."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=LABEL_NAMES,
            ax=ax,
            cmap="Blues",
        )
        ax.set_title(
            f"DistilBERT Sentiment — Confusion Matrix\n"
            f"Macro F1: {macro_f1:.3f}",
            fontsize=12,
        )
        plt.tight_layout()
        cm_path = output_dir / "sentiment_confusion_matrix.png"
        fig.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Confusion matrix saved: {cm_path}")

        try:
            import mlflow
            mlflow.log_artifact(str(cm_path))
        except ImportError:
            pass

    except ImportError as e:
        print(f"  Confusion matrix skipped — missing dependency: {e}")


# ─── Full evaluation pipeline ─────────────────────────────────────────────────

def run_evaluation(
    model_path: Path,
    test_csv: Path,
    ood_samples_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Full evaluation pipeline:
    1. Load model + tokenizer from artifacts
    2. Core metrics (Macro F1, Negative Recall, Neutral Precision)
    3. Subgroup analysis (length, rating)
    4. Domain-shift evaluation (Fix #32)
    5. Baseline delta vs SetFit
    6. Confusion matrix artifact
    7. Log all to MLflow
    """
    print("=" * 60)
    print("  E-CIP v3.0 — Sentiment Evaluation Suite")
    print("  Blueprint Section 04 — Fix #32")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification

        from models.sentiment.finetune import load_tokenizer_from_artifact

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        # Load tokenizer from artifact (Fix #6)
        tokenizer = load_tokenizer_from_artifact(TOKENIZER_ARTIFACT_DIR)
        if tokenizer is None:
            return {}

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        model = model.to(device)
        print(f"  ✓ Model loaded: {model_path}")

    except ImportError as e:
        print(f"  Model load skipped — missing dependency: {e}")
        return {}

    # Load test data
    print(f"\n  Loading test data: {test_csv}")
    texts, labels, metadata = load_test_split(test_csv)
    if not texts:
        print("  No test data found.")
        return {}
    print(f"  Test samples: {len(texts):,}")

    # Inference
    print("\n  Running inference...")
    preds, confs, p95_ms = run_inference_batch(model, tokenizer, texts, device)
    print(f"  Inference p95: {p95_ms:.1f}ms "
          f"{'✓' if p95_ms < 50 else '✗ (target < 50ms)'}")

    # Core metrics
    core_metrics = evaluate_core(labels, preds)
    core_metrics["inference_p95_ms"] = round(p95_ms, 1)

    # Subgroup analysis
    length_metrics = evaluate_by_length(labels, preds, metadata)
    rating_metrics = evaluate_by_rating(labels, preds, metadata)

    # Domain-shift (Fix #32)
    ood_metrics: dict[str, Any] = {}
    if ood_samples_dir and ood_samples_dir.exists():
        ood_metrics = evaluate_domain_shift(
            model, tokenizer, device, ood_samples_dir
        )

    # Baseline delta
    delta_metrics = compute_baseline_delta(core_metrics)

    # Confusion matrix
    save_confusion_matrix(
        labels, preds, ARTIFACTS_DIR,
        core_metrics.get("macro_f1", 0.0),
    )

    # Combine all metrics
    all_metrics: dict[str, Any] = {
        **core_metrics,
        **length_metrics,
        **rating_metrics,
        **ood_metrics,
        **delta_metrics,
    }

    # Save combined results
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = ARTIFACTS_DIR / "distilbert_evaluation_metrics.json"
    results_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\n  ✓ Full evaluation metrics saved: {results_path}")

    # Log to MLflow
    try:
        import mlflow
        with mlflow.start_run(run_name="distilbert_sentiment_eval"):
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            mlflow.log_artifact(str(results_path))
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("  Evaluation complete.")
    macro_f1 = core_metrics.get("macro_f1", 0.0)
    print(f"  Final Macro F1 : {macro_f1:.4f} "
          f"{'✓ TARGET MET' if macro_f1 >= 0.88 else '✗ Below target (≥ 0.88)'}")
    print("=" * 60 + "\n")

    return all_metrics


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Sentiment Evaluation Suite"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "distilbert_sentiment_best.pt",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/processed/reviews/test_reviews.csv"),
    )
    parser.add_argument(
        "--ood-dir",
        type=Path,
        default=Path("data/samples/domain_shift_100"),
        help="Directory with OOD sample CSVs for domain-shift evaluation",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  E-CIP v3.0 — Sentiment Evaluation Suite")
    print("  Blueprint Section 04 — Fix #32")
    print("=" * 60)

    if not args.model_path.exists():
        print(f"\n  Model not found: {args.model_path}")
        print("  Run models/sentiment/finetune.py first.")
        print("\n  Evaluation suite structure verified.")
        print("  Functions available:")
        print("    evaluate_core()          — Macro F1, Negative Recall, Neutral Prec")
        print("    evaluate_by_length()     — F1 by review length band")
        print("    evaluate_by_rating()     — F1 by star rating (1★, 3★, 5★)")
        print("    evaluate_domain_shift()  — OOD category evaluation Fix #32")
        print("    compute_baseline_delta() — Delta vs SetFit baseline")
        print("    save_confusion_matrix()  — PNG artifact to MLflow")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  ✓ Artifacts directory ready: {ARTIFACTS_DIR}")
        print("  Run in Colab/Kaggle after training (Phase 3, Week 9).")
        print("=" * 60 + "\n")
        return

    run_evaluation(
        model_path=args.model_path,
        test_csv=args.test_csv,
        ood_samples_dir=args.ood_dir,
    )


if __name__ == "__main__":
    main()
