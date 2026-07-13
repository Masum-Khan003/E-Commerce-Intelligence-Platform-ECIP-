# models/sentiment/absa_pipeline.py
# E-CIP v3.0 — Aspect-Based Sentiment Analysis Pipeline
# Blueprint Section 22 — Critical Fix #3
#
# Critical Fix #3: NER-based ABSA is impossible — Amazon Reviews 2023
# has no token-level NER annotations. Zero-shot NLI via DeBERTa-v3-small
# requires NO annotations and outperforms keyword-based approaches.
#
# Model: cross-encoder/nli-deberta-v3-small (HuggingFace)
# Method: Textual entailment — pose each aspect as NLI hypothesis
# Aspects: Battery, Display, Shipping, Build, Price, Support
# Confidence threshold: 0.70 (only label aspects above this)
#
# Evaluation: SemEval-2014 Task 4 (laptop domain)
# Target: F1 > 0.72 on laptop domain
#
# Output feeds Module 3 retention features:
#   avg_battery_sentiment, avg_shipping_sentiment, avg_price_sentiment
#
# Usage:
#   python models/sentiment/absa_pipeline.py
#   python models/sentiment/absa_pipeline.py --text "Battery dies in 2 hours"
#   python models/sentiment/absa_pipeline.py --eval-semeval

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-small"
CONF_THRESHOLD = 0.70
SENTIMENTS = ["positive", "negative", "neutral"]
ARTIFACTS_DIR = Path("models/sentiment/artifacts")

# Aspect hypothesis templates
# Each template is filled with sentiment label for NLI scoring
ASPECT_TEMPLATES: dict[str, str] = {
    "Battery": (
        "This review expresses {sentiment} sentiment about the battery life."
    ),
    "Display": (
        "This review expresses {sentiment} sentiment about the screen or display."
    ),
    "Shipping": (
        "This review expresses {sentiment} sentiment about delivery or shipping."
    ),
    "Build": (
        "This review expresses {sentiment} sentiment about build quality or durability."
    ),
    "Price": (
        "This review expresses {sentiment} sentiment about price or value for money."
    ),
    "Support": (
        "This review expresses {sentiment} sentiment about customer support or service."
    ),
}

# Module 3 feature mapping — which aspects feed retention features
RETENTION_FEATURE_MAP: dict[str, str] = {
    "Battery": "avg_battery_sentiment",
    "Shipping": "avg_shipping_sentiment",
    "Price": "avg_price_sentiment",
}


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class ABSAPipeline:
    """
    Zero-shot NLI Aspect-Based Sentiment Analysis.

    Blueprint Section 22 — Critical Fix #3:
    No NER annotations required. Uses textual entailment to determine
    sentiment polarity for each aspect independently.

    For each review:
        For each aspect in ASPECT_TEMPLATES:
            Score P(entailment) for positive/negative/neutral hypotheses
            → Select highest-scoring sentiment if confidence ≥ CONF_THRESHOLD
            → Skip aspect if all confidences below threshold (not mentioned)
    """

    def __init__(
        self,
        model_id: str = NLI_MODEL_ID,
        conf_threshold: float = CONF_THRESHOLD,
        device: int = -1,  # -1 = CPU, 0 = first GPU
    ) -> None:
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.device = device
        self._pipeline: Any = None

    def load(self) -> bool:
        """Load the NLI pipeline. Returns True if successful."""
        try:
            from transformers import pipeline

            print(f"  Loading NLI model: {self.model_id}")
            print("  (CPU inference — ~100ms per review)")
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_id,
                device=self.device,
            )
            print("  ✓ ABSA pipeline loaded")
            return True

        except ImportError:
            print("  transformers not installed.")
            print("  Install [train] extras for full ABSA pipeline.")
            return False
        except Exception as e:
            print(f"  Pipeline load failed: {e}")
            return False

    def extract_aspects(self, review_text: str) -> list[dict[str, Any]]:
        """
        Extract aspect sentiments from a single review.

        Blueprint Section 22 — Fix #3:
        Zero-shot NLI ABSA. No NER labels needed.
        Returns only aspects with confidence ≥ CONF_THRESHOLD.

        Args:
            review_text: Raw (cleaned) review text

        Returns:
            List of dicts: {aspect, sentiment, score, method}
        """
        if self._pipeline is None:
            return []

        results: list[dict[str, Any]] = []

        for aspect, template in ASPECT_TEMPLATES.items():
            # Build hypothesis for each sentiment
            hypotheses = [
                template.format(sentiment=s) for s in SENTIMENTS
            ]

            try:
                # NLI scoring: P(entailment | premise=review, hypothesis=template)
                output = self._pipeline(
                    review_text,
                    hypotheses,
                    multi_label=False,
                )

                scores: list[float] = output["scores"]
                labels: list[str] = output["labels"]

                # Match scores back to our sentiment order
                sentiment_scores: dict[str, float] = {}
                for label, score in zip(labels, scores):
                    for sentiment in SENTIMENTS:
                        if sentiment in label.lower():
                            sentiment_scores[sentiment] = score
                            break

                best_sentiment = max(sentiment_scores, key=sentiment_scores.get)  # type: ignore[arg-type]
                best_score = sentiment_scores[best_sentiment]

                # Only label if confidence exceeds threshold
                if best_score >= self.conf_threshold:
                    results.append({
                        "aspect": aspect,
                        "sentiment": best_sentiment.capitalize(),
                        "score": round(best_score, 4),
                        "method": "zero_shot_nli",
                        "model": self.model_id,
                    })

            except Exception as e:
                print(f"  ABSA error for aspect {aspect}: {e}")
                continue

        return results

    def extract_batch(
        self,
        reviews: list[str],
        show_progress: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """
        Extract aspect sentiments from a batch of reviews.
        Returns list of aspect lists (one per review).
        """
        all_results = []
        for i, review in enumerate(reviews):
            if show_progress and i % 100 == 0:
                print(f"\r  Processing review {i}/{len(reviews)}...", end="", flush=True)
            all_results.append(self.extract_aspects(review))

        if show_progress:
            print()
        return all_results

    def sentiment_to_score(self, sentiment: str) -> float:
        """
        Convert sentiment label to continuous score for retention features.
        Positive → +1.0, Neutral → 0.0, Negative → -1.0
        """
        mapping = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
        return mapping.get(sentiment, 0.0)

    def build_retention_features(
        self,
        aspect_results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Aggregate aspect sentiments into Module 3 retention features.
        Blueprint Section 05 — Fix #47: aspect features feed retention model.

        Returns dict with avg_{aspect}_sentiment for each mapped aspect.
        """
        features: dict[str, float] = {
            feat: 0.0 for feat in RETENTION_FEATURE_MAP.values()
        }

        aspect_scores: dict[str, list[float]] = {
            aspect: [] for aspect in RETENTION_FEATURE_MAP
        }

        for result in aspect_results:
            aspect = result["aspect"]
            if aspect in RETENTION_FEATURE_MAP:
                score = self.sentiment_to_score(result["sentiment"])
                aspect_scores[aspect].append(score)

        for aspect, feat_name in RETENTION_FEATURE_MAP.items():
            scores = aspect_scores[aspect]
            features[feat_name] = sum(scores) / len(scores) if scores else 0.0

        return features


# ─── SemEval-2014 evaluation ──────────────────────────────────────────────────

def evaluate_on_semeval(
    pipeline: ABSAPipeline,
    semeval_path: Path,
    output_dir: Path = ARTIFACTS_DIR,
) -> dict[str, float]:
    """
    Evaluate ABSA pipeline on SemEval-2014 Task 4 (laptop domain).
    Blueprint Section 22: target F1 > 0.72 on laptop domain.

    SemEval-2014 format (XML):
        <sentence id="...">
            <text>...</text>
            <aspectTerms>
                <aspectTerm term="..." polarity="positive/negative/neutral/conflict"/>
            </aspectTerms>
        </sentence>
    """
    try:
        import xml.etree.ElementTree as ET

        from sklearn.metrics import classification_report, f1_score

        if not semeval_path.exists():
            print(f"  SemEval data not found: {semeval_path}")
            print("  Download from: https://alt.qcri.org/semeval2014/task4/")
            return {}

        # Parse SemEval XML
        tree = ET.parse(semeval_path)
        root = tree.getroot()

        true_labels: list[str] = []
        pred_labels: list[str] = []

        print(f"  Evaluating on: {semeval_path.name}")
        n_sentences = 0

        for sentence in root.findall(".//sentence"):
            text_elem = sentence.find("text")
            if text_elem is None or text_elem.text is None:
                continue

            text = text_elem.text.strip()
            aspect_terms = sentence.findall(".//aspectTerm")

            if not aspect_terms:
                continue

            # Get ABSA predictions for this sentence
            predictions = pipeline.extract_aspects(text)
            pred_sentiments = {
                p["aspect"].lower(): p["sentiment"].lower()
                for p in predictions
            }

            for aspect_term in aspect_terms:
                polarity = aspect_term.get("polarity", "")
                # Map SemEval polarities to our labels
                if polarity == "conflict":
                    continue  # skip conflict labels
                if polarity not in ("positive", "negative", "neutral"):
                    continue

                true_labels.append(polarity)

                # Find best matching aspect prediction
                term = (aspect_term.get("term") or "").lower()
                matched_sentiment = "neutral"  # default

                for aspect_key, sentiment in pred_sentiments.items():
                    if aspect_key in term or term in aspect_key:
                        matched_sentiment = sentiment
                        break

                pred_labels.append(matched_sentiment)

            n_sentences += 1

        if not true_labels:
            print("  No valid aspect labels found in SemEval data.")
            return {}

        # Compute metrics
        macro_f1 = f1_score(
            true_labels, pred_labels,
            average="macro",
            zero_division=0,
            labels=["positive", "negative", "neutral"],
        )
        report = classification_report(
            true_labels, pred_labels,
            labels=["positive", "negative", "neutral"],
            output_dict=True,
            zero_division=0,
        )

        print(f"\n  SemEval-2014 Results ({n_sentences} sentences):")
        print(f"  Macro F1   : {macro_f1:.4f} "
              f"{'✓' if macro_f1 >= 0.72 else '✗ (target > 0.72)'}")
        for label in ["positive", "negative", "neutral"]:
            if label in report:
                print(f"  {label.capitalize()} F1: {report[label]['f1-score']:.4f}")

        results = {
            "semeval_macro_f1": round(macro_f1, 4),
            "semeval_positive_f1": round(
                report.get("positive", {}).get("f1-score", 0.0), 4
            ),
            "semeval_negative_f1": round(
                report.get("negative", {}).get("f1-score", 0.0), 4
            ),
            "semeval_neutral_f1": round(
                report.get("neutral", {}).get("f1-score", 0.0), 4
            ),
            "n_sentences": n_sentences,
            "n_aspect_labels": len(true_labels),
        }

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "absa_semeval_metrics.json"
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\n  ✓ SemEval metrics saved: {results_path}")

        # Log to MLflow
        try:
            import mlflow
            with mlflow.start_run(run_name="absa_semeval_eval"):
                mlflow.log_param("model_id", NLI_MODEL_ID)
                mlflow.log_param("conf_threshold", CONF_THRESHOLD)
                for k, v in results.items():
                    if isinstance(v, float):
                        mlflow.log_metric(k, v)
        except ImportError:
            pass

        return results

    except ImportError as e:
        print(f"  Evaluation skipped — missing dependency: {e}")
        return {}


# ─── Demo inference ───────────────────────────────────────────────────────────

def demo_inference(text: str) -> None:
    """Run ABSA on a single review text and print results."""
    print(f"\n  Input: {text[:100]}{'...' if len(text) > 100 else ''}")
    print()

    absa = ABSAPipeline()
    loaded = absa.load()

    if not loaded:
        print("  Demo requires [train] extras — install in Colab/Kaggle.")
        return

    t0 = time.time()
    results = absa.extract_aspects(text)
    inference_ms = int((time.time() - t0) * 1000)

    if results:
        print(f"  Aspect sentiments ({inference_ms}ms):")
        for r in results:
            print(f"    {r['aspect']:<12} → {r['sentiment']:<10} "
                  f"(confidence: {r['score']:.3f})")

        retention_features = absa.build_retention_features(results)
        print("\n  Retention features:")
        for feat, score in retention_features.items():
            print(f"    {feat}: {score:+.2f}")
    else:
        print(f"  No aspects detected above threshold ({CONF_THRESHOLD})")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Zero-Shot NLI ABSA Pipeline"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single review text for demo inference",
    )
    parser.add_argument(
        "--eval-semeval",
        action="store_true",
        help="Evaluate on SemEval-2014 Task 4",
    )
    parser.add_argument(
        "--semeval-path",
        type=Path,
        default=Path("data/raw/semeval2014/Laptops_Train.xml"),
        help="Path to SemEval-2014 XML file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  E-CIP v3.0 — Zero-Shot NLI ABSA Pipeline")
    print("  Blueprint Section 22 — Critical Fix #3")
    print("=" * 60)

    if args.text:
        demo_inference(args.text)
        return

    if args.eval_semeval:
        absa = ABSAPipeline()
        if absa.load():
            evaluate_on_semeval(absa, args.semeval_path)
        return

    # Default: print pipeline specification
    print(f"\n  NLI Model    : {NLI_MODEL_ID}")
    print(f"  Threshold    : {CONF_THRESHOLD}")
    print(f"  Aspects      : {list(ASPECT_TEMPLATES.keys())}")
    print(f"  Sentiments   : {SENTIMENTS}")
    print("  Target F1    : > 0.72 (SemEval-2014 laptop domain)")

    print("\n  Retention feature mapping:")
    for aspect, feature in RETENTION_FEATURE_MAP.items():
        print(f"    {aspect:<12} → {feature}")

    print("\n  Example usage:")
    print('    python models/sentiment/absa_pipeline.py \\')
    print('        --text "Battery dies in 2 hours but display is gorgeous"')
    print()
    print('    python models/sentiment/absa_pipeline.py --eval-semeval')

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  ✓ Artifacts directory ready: {ARTIFACTS_DIR}")
    print("  Run in Colab/Kaggle after downloading SemEval data.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
