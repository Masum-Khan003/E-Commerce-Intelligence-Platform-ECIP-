# data/scripts/synthesize_demo_sentiment.py
# E-CIP v3.0 — SYNTHETIC Demo Sentiment Generator
#
# SYNTHETIC DATA — see models/retention/model_card.md "Known Limitations".
#
# UCI Online Retail II has no review text and no linkage to Amazon Reviews
# (Module 2's training corpus) — there is no real customer-level sentiment
# signal available for this dataset. Rather than leaving the retention
# feature table's sentiment columns as neutral-prior placeholders
# (build_sentiment_placeholders() in tabular_pipeline.py), this script
# generates a seeded, clearly-labeled synthetic review-sentiment dataset so
# that:
#   1. The cross-module invariant test (negative sentiment -> higher churn)
#      has real, non-degenerate signal to exercise.
#   2. The SHAP explanation narrative for avg_sentiment_score has something
#      real to attribute direction to.
#
# This does NOT represent real customer feedback. A production deployment
# would consume live Module 2 (DistilBERT + ABSA) output keyed by a shared
# customer/order identifier — which does not exist for this dataset pairing.
#
# Output (review-level, one row per synthetic review):
#   data/feature_store/text_features/review_sentiment_v1.parquet
#     CustomerID, review_date, sentiment_score [-1, 1]
#   data/feature_store/text_features/aspect_sentiment_v1.parquet
#     CustomerID, review_date, aspect, aspect_sentiment [-1, 1]
#
# Usage:
#   python data/scripts/synthesize_demo_sentiment.py \
#       --churn-labels data/processed/tabular/churn_labels.parquet

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

SEED = 42
SNAPSHOT_DATE = pd.Timestamp("2010-11-30")
OBS_START = pd.Timestamp("2009-12-01")

TEXT_FEATURES_DIR = Path("data/feature_store/text_features")
ASPECTS = ["battery", "shipping", "price"]

# Not every customer leaves a review — mirrors real review-rate sparsity.
REVIEW_RATE = 0.55
MAX_REVIEWS_PER_CUSTOMER = 4

# Weak, documented correlation with churn: churned customers skew negative,
# retained customers skew positive. Noise dominates — this is a directional
# signal for testing, not a realistic effect size.
CHURN_SENTIMENT_MEAN = -0.35
RETAINED_SENTIMENT_MEAN = 0.25
SENTIMENT_STD = 0.45

ASPECT_MENTION_RATE = 0.4


def synthesize_reviews(
    churn_labels: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic review-level sentiment + aspect-sentiment rows.

    Only customers with `churned` known are given reviews, review_date is
    drawn from within the observation window (never after SNAPSHOT_DATE),
    so Gate G8 (review_date < snapshot_date) is satisfiable by construction
    for the vast majority of rows — merge_sentiment_features() still
    enforces the filter explicitly rather than trusting this.
    """
    rng = np.random.default_rng(seed)

    review_rows: list[dict[str, object]] = []
    aspect_rows: list[dict[str, object]] = []

    obs_span_days = (SNAPSHOT_DATE - OBS_START).days

    for _, row in churn_labels.iterrows():
        if rng.uniform() > REVIEW_RATE:
            continue

        customer_id = row["CustomerID"]
        mean_sentiment = (
            CHURN_SENTIMENT_MEAN if row["churned"] == 1 else RETAINED_SENTIMENT_MEAN
        )

        n_reviews = int(rng.integers(1, MAX_REVIEWS_PER_CUSTOMER + 1))
        for _ in range(n_reviews):
            offset_days = int(rng.integers(0, max(obs_span_days, 1)))
            review_date = OBS_START + pd.Timedelta(days=offset_days)
            sentiment_score = float(
                np.clip(rng.normal(mean_sentiment, SENTIMENT_STD), -1.0, 1.0)
            )

            review_rows.append({
                "CustomerID": customer_id,
                "review_date": review_date,
                "sentiment_score": sentiment_score,
            })

            for aspect in ASPECTS:
                if rng.uniform() > ASPECT_MENTION_RATE:
                    continue
                aspect_sentiment = float(
                    np.clip(rng.normal(mean_sentiment, SENTIMENT_STD), -1.0, 1.0)
                )
                aspect_rows.append({
                    "CustomerID": customer_id,
                    "review_date": review_date,
                    "aspect": aspect,
                    "aspect_sentiment": aspect_sentiment,
                })

    review_df = pd.DataFrame(review_rows)
    aspect_df = pd.DataFrame(aspect_rows)
    return review_df, aspect_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — SYNTHETIC demo sentiment generator (see model card)"
    )
    parser.add_argument(
        "--churn-labels",
        type=Path,
        default=Path("data/processed/tabular/churn_labels.parquet"),
        help="Path to engineered churn labels (from churn_label_engineer.py)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  E-CIP v3.0 — SYNTHETIC Demo Sentiment Generator")
    print("  NOT real customer feedback — see model card Known Limitations")
    print("=" * 60)

    if not args.churn_labels.exists():
        print(f"\n  Churn labels not found: {args.churn_labels}")
        print("  Run models/retention/churn_label_engineer.py first.")
        return

    churn_labels = pd.read_parquet(args.churn_labels)
    print(f"\n  Customers loaded: {len(churn_labels):,}")

    review_df, aspect_df = synthesize_reviews(churn_labels)
    print(f"  Synthetic reviews generated       : {len(review_df):,}")
    print(f"  Synthetic aspect mentions generated: {len(aspect_df):,}")

    TEXT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    review_path = TEXT_FEATURES_DIR / "review_sentiment_v1.parquet"
    aspect_path = TEXT_FEATURES_DIR / "aspect_sentiment_v1.parquet"
    review_df.to_parquet(review_path, index=False)
    aspect_df.to_parquet(aspect_path, index=False)

    print(f"\n  ✓ Review sentiment saved: {review_path}")
    print(f"  ✓ Aspect sentiment saved: {aspect_path}")
    print("\n" + "=" * 60)
    print("  Synthetic sentiment generation complete.")
    print("  Next: python data/pipelines/tabular_pipeline.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
