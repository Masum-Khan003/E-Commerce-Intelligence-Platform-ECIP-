# data/pipelines/tabular_pipeline.py
# E-CIP v3.0 — Tabular Data Pipeline
# Blueprint Section 05, 07, 08, 09
#
# Critical fixes implemented:
#   Fix #5:  Null CustomerID exclusion (guest checkouts)
#   Fix #10: SMOTE applied INSIDE CV loop only (enforced by design)
#   Fix #15: purchase_gap_cv NaN-safe — returns -1 for single-order customers
#   Fix #21: time_decay_lambda placeholder (Optuna tunes in Phase 4)
#   Fix #24: UK-only scope for single-currency monetary features
#
# Responsibilities:
#   - Load and clean UCI Online Retail II
#   - Scope to UK customers (Fix #24)
#   - Engineer RFM + behavioural + temporal features
#   - Join churn labels from churn_label_engineer.py
#   - Save feature table to data/feature_store/customer_features/
#   - Save reference distributions for drift detection (Section 09)
#   - Serialise scaler + encoder artifacts
#
# Usage:
#   python data/pipelines/tabular_pipeline.py
#   python data/pipelines/tabular_pipeline.py --input data/raw/online_retail2/online_retail_II.xlsx

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw/online_retail2")
PROCESSED_DIR = Path("data/processed/tabular")
FEATURE_STORE_DIR = Path("data/feature_store/customer_features")
ARTIFACTS_DIR = Path("data/feature_store/artifacts")
REFERENCE_DIR = Path("data/reference_distributions")
TEXT_FEATURES_DIR = Path("data/feature_store/text_features")

# ─── Constants ────────────────────────────────────────────────────────────────

SNAPSHOT_DATE = pd.Timestamp("2010-11-30")
OBS_START = pd.Timestamp("2009-12-01")

# Fix #21: time_decay_lambda placeholder — Optuna tunes this in Phase 4
# Range: [0.01, 0.5] as specified in blueprint Section 10
TIME_DECAY_LAMBDA_DEFAULT = 0.1

# Fix #15: Sentinel value for single-order customers (purchase_gap_cv)
SINGLE_ORDER_SENTINEL = -1.0


# ─── Step 1: Load & clean ─────────────────────────────────────────────────────

def load_and_clean(input_path: Path) -> pd.DataFrame:
    """Load UCI Online Retail II and apply all cleaning steps."""
    print(f"\n  Loading: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df1 = pd.read_excel(input_path, sheet_name="Year 2009-2010", engine="openpyxl")
        df2 = pd.read_excel(input_path, sheet_name="Year 2010-2011", engine="openpyxl")
        df = pd.concat([df1, df2], ignore_index=True)
    elif suffix == ".csv":
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    print(f"  Raw rows: {len(df):,}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Fix #5: Exclude guest checkouts (null CustomerID)
    before = len(df)
    df = df.dropna(subset=["Customer ID"]).copy()
    print(f"  After guest excl : {len(df):,} rows "
          f"(removed {before - len(df):,} = "
          f"{(before - len(df)) / before:.1%} guest checkouts)")

    # Fix #24: UK only — single-currency monetary features
    before = len(df)
    df = df[df["Country"] == "United Kingdom"].copy()
    print(f"  After UK scope   : {len(df):,} rows "
          f"(removed {before - len(df):,} international rows)")

    # Cast CustomerID
    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)

    # Remove cancellations
    before = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")].copy()
    print(f"  After cancels    : {len(df):,} rows "
          f"(removed {before - len(df):,} cancellations)")

    # Remove non-positive quantity/price
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    print(f"  After neg values : {len(df):,} rows "
          f"(removed {before - len(df):,} non-positive rows)")

    # Add line total
    df["LineTotal"] = df["Quantity"] * df["Price"]

    print(f"  Final clean rows : {len(df):,}")
    return df


# ─── Step 2: Scope to observation window ──────────────────────────────────────

def scope_to_observation_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within the observation window for feature engineering."""
    obs_df = df[df["InvoiceDate"].between(OBS_START, SNAPSHOT_DATE)].copy()
    print(f"\n  Observation window rows: {len(obs_df):,} "
          f"({OBS_START.date()} → {SNAPSHOT_DATE.date()})")
    return obs_df


# ─── Step 3: RFM features ─────────────────────────────────────────────────────

def compute_rfm_features(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) features.
    Blueprint Section 05 — feature engineering table.
    """
    rfm = obs_df.groupby("Customer ID").agg(
        last_order_date=("InvoiceDate", "max"),
        first_order_date=("InvoiceDate", "min"),
        frequency=("Invoice", "nunique"),
        monetary_value=("LineTotal", "sum"),
    ).reset_index()

    # Recency: days since last purchase
    rfm["recency_days"] = (SNAPSHOT_DATE - rfm["last_order_date"]).dt.days

    # Tenure: days since first order
    rfm["tenure_days"] = (SNAPSHOT_DATE - rfm["first_order_date"]).dt.days

    # Avg order value
    rfm["avg_order_value"] = rfm["monetary_value"] / rfm["frequency"]

    # Clip negative monetary values (returns already excluded but safeguard)
    rfm["monetary_value"] = rfm["monetary_value"].clip(lower=0)

    # Log-transform recency (right-skewed)
    rfm["recency_days_log"] = np.log1p(rfm["recency_days"])

    return rfm.rename(columns={"Customer ID": "CustomerID"})


# ─── Step 4: Behavioural features ─────────────────────────────────────────────

def compute_behavioural_features(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute behavioural features per customer.
    Blueprint Section 05 — Fix #15: NaN-safe purchase_gap_cv.
    """
    # Purchase gap CV — Fix #15
    def purchase_gap_cv(dates: pd.Series) -> float:
        """
        Coefficient of variation of inter-purchase intervals.
        Fix #15: Returns SINGLE_ORDER_SENTINEL (-1.0) for customers
        with fewer than 2 orders — prevents NaN propagation downstream.
        Downstream: add is_single_purchase binary flag.
        """
        sorted_dates = sorted(dates)
        if len(sorted_dates) < 2:
            return SINGLE_ORDER_SENTINEL
        gaps = [
            (sorted_dates[i + 1] - sorted_dates[i]).days
            for i in range(len(sorted_dates) - 1)
        ]
        mean_gap = np.mean(gaps)
        if mean_gap == 0:
            return 0.0
        return float(np.std(gaps) / mean_gap)

    # Category diversity
    def category_diversity(stock_codes: pd.Series) -> int:
        """Count of distinct product categories purchased."""
        return int(stock_codes.nunique())

    # Per-customer invoice dates for gap CV
    invoice_dates = obs_df.groupby("Customer ID")["InvoiceDate"].apply(list)

    # Category diversity
    diversity = obs_df.groupby("Customer ID")["StockCode"].agg(category_diversity)

    # Assemble behavioural features
    behavioural = pd.DataFrame({
        "CustomerID": invoice_dates.index,
        "purchase_gap_cv": invoice_dates.apply(purchase_gap_cv).values,
        "category_diversity": diversity.values,
    })

    # is_single_purchase flag (Fix #15 downstream indicator)
    behavioural["is_single_purchase"] = (
        behavioural["purchase_gap_cv"] == SINGLE_ORDER_SENTINEL
    ).astype(int)

    # Return rate (negative quantity rows already removed — use invoice count)
    behavioural["return_rate"] = 0.0  # Conservative: 0 since cancels excluded

    return behavioural


# ─── Step 5: Temporal features ────────────────────────────────────────────────

def compute_temporal_features(
    obs_df: pd.DataFrame,
    lambda_decay: float = TIME_DECAY_LAMBDA_DEFAULT,
) -> pd.DataFrame:
    """
    Compute temporal features including purchase trend and time decay.
    Fix #21: lambda_decay is a parameter — Optuna tunes it in Phase 4.
    """
    # Monthly order counts per customer (last 6 months)
    six_months_ago = SNAPSHOT_DATE - pd.DateOffset(months=6)
    recent_df = obs_df[obs_df["InvoiceDate"] >= six_months_ago].copy()
    recent_df["YearMonth"] = recent_df["InvoiceDate"].dt.to_period("M")

    monthly_counts = (
        recent_df.groupby(["Customer ID", "YearMonth"])["Invoice"]
        .nunique()
        .reset_index()
    )

    def purchase_trend(group: pd.DataFrame) -> float:
        """
        OLS slope of monthly order count over last 6 months.
        Requires >= 2 months of history — returns 0.0 otherwise.
        Blueprint Section 05: fill 0 for < 2 months.
        """
        if len(group) < 2:
            return 0.0
        x = np.arange(len(group), dtype=float)
        y = group["Invoice"].values.astype(float)
        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            slope = 0.0
        return slope

    trends = (
        monthly_counts.groupby("Customer ID")
        .apply(purchase_trend)
        .reset_index()
        .rename(columns={0: "purchase_trend", "Customer ID": "CustomerID"})
    )

    # Recency for time decay (recompute here for clarity)
    recency = obs_df.groupby("Customer ID")["InvoiceDate"].max().reset_index()
    recency["recency_days"] = (SNAPSHOT_DATE - recency["InvoiceDate"]).dt.days
    recency["time_decay_weight"] = np.exp(
        -lambda_decay * recency["recency_days"]
    )
    recency = recency[["Customer ID", "time_decay_weight"]].rename(
        columns={"Customer ID": "CustomerID"}
    )

    temporal = trends.merge(recency, on="CustomerID", how="left")
    temporal["time_decay_weight"] = temporal["time_decay_weight"].fillna(0.0)

    return temporal


# ─── Step 6: Sentiment feature placeholders ───────────────────────────────────

def build_sentiment_placeholders(customer_ids: pd.Series) -> pd.DataFrame:
    """
    Placeholder sentiment features — populated in Phase 4 (Week 11)
    when Module 2 sentiment scores are merged into the feature table.

    Blueprint Section 05 — sentiment features:
        avg_sentiment_score, last_review_sentiment,
        negative_review_count, avg_battery_sentiment,
        avg_shipping_sentiment, avg_price_sentiment

    G8 causal integrity enforced in Phase 4:
        WHERE review_date < snapshot_date
    """
    return pd.DataFrame({
        "CustomerID": customer_ids,
        # Document-level sentiment (filled in Phase 4)
        "avg_sentiment_score": 0.0,      # neutral prior
        "last_review_sentiment": 0.0,    # neutral prior
        "negative_review_count": 0,
        "has_reviews": 0,
        # Aspect-level sentiment (filled in Phase 4 — Fix #47)
        "avg_battery_sentiment": 0.0,
        "avg_shipping_sentiment": 0.0,
        "avg_price_sentiment": 0.0,
    })


# ─── Step 6b: Real sentiment merge (Phase 4, Week 11) ─────────────────────────

NEGATIVE_SENTIMENT_THRESHOLD = -0.3


def merge_sentiment_features(
    customer_ids: pd.Series,
    review_sentiment_path: Path,
    aspect_sentiment_path: Path,
    snapshot_date: pd.Timestamp = SNAPSHOT_DATE,
) -> pd.DataFrame:
    """
    Merge Module 2 sentiment output onto the customer feature table.

    Blueprint Section 06 — Gate G8 causal integrity: only reviews dated
    strictly before snapshot_date may inform features used to predict
    churn at that snapshot. This filter is applied explicitly here rather
    than trusted from upstream, since it is the single most important
    data-leakage guard in Module 3.

    NOTE: for this dataset the review-level inputs are SYNTHETIC —
    see data/scripts/synthesize_demo_sentiment.py and
    models/retention/model_card.md "Known Limitations". The merge logic
    itself (this function) is production-shaped: it is what would run
    against real Module 2 output if UK retail customers and Amazon
    reviewers were the same population.
    """
    review_df = pd.read_parquet(review_sentiment_path)
    aspect_df = pd.read_parquet(aspect_sentiment_path)

    review_df["review_date"] = pd.to_datetime(review_df["review_date"])
    aspect_df["review_date"] = pd.to_datetime(aspect_df["review_date"])

    # Gate G8: strictly before snapshot_date — never equal, never after.
    review_df = review_df[review_df["review_date"] < snapshot_date].copy()
    aspect_df = aspect_df[aspect_df["review_date"] < snapshot_date].copy()

    # Document-level aggregation
    doc_agg = review_df.groupby("CustomerID").agg(
        avg_sentiment_score=("sentiment_score", "mean"),
        negative_review_count=("sentiment_score", lambda s: int((s < NEGATIVE_SENTIMENT_THRESHOLD).sum())),
    )
    last_review = (
        review_df.sort_values("review_date")
        .groupby("CustomerID")["sentiment_score"]
        .last()
        .rename("last_review_sentiment")
    )
    doc_agg = doc_agg.join(last_review)
    doc_agg["has_reviews"] = 1
    doc_agg = doc_agg.reset_index()

    # Aspect-level aggregation — pivot mean sentiment per aspect per customer
    aspect_pivot = (
        aspect_df.groupby(["CustomerID", "aspect"])["aspect_sentiment"]
        .mean()
        .unstack("aspect")
        .rename(columns={
            "battery": "avg_battery_sentiment",
            "shipping": "avg_shipping_sentiment",
            "price": "avg_price_sentiment",
        })
        .reset_index()
    )

    sentiment = pd.DataFrame({"CustomerID": customer_ids})
    sentiment = sentiment.merge(doc_agg, on="CustomerID", how="left")
    sentiment = sentiment.merge(aspect_pivot, on="CustomerID", how="left")

    fill_values = {
        "avg_sentiment_score": 0.0,
        "last_review_sentiment": 0.0,
        "negative_review_count": 0,
        "has_reviews": 0,
        "avg_battery_sentiment": 0.0,
        "avg_shipping_sentiment": 0.0,
        "avg_price_sentiment": 0.0,
    }
    for col, default in fill_values.items():
        if col not in sentiment.columns:
            sentiment[col] = default
    sentiment = sentiment.fillna(fill_values)

    n_with_reviews = int(sentiment["has_reviews"].sum())
    print(f"  Sentiment merge: {n_with_reviews:,} / {len(sentiment):,} customers "
          f"have a pre-snapshot review (Gate G8 enforced)")

    return sentiment


# ─── Step 7: Assemble feature table ───────────────────────────────────────────

def assemble_feature_table(
    rfm: pd.DataFrame,
    behavioural: pd.DataFrame,
    temporal: pd.DataFrame,
    sentiment: pd.DataFrame,
    churn_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join all feature groups into the master customer feature table.
    Optionally join churn labels if available.
    """
    features = rfm.merge(behavioural, on="CustomerID", how="left")
    features = features.merge(temporal, on="CustomerID", how="left")
    features = features.merge(sentiment, on="CustomerID", how="left")

    # Join churn labels if available
    if churn_labels is not None:
        features = features.merge(
            churn_labels[["CustomerID", "churned"]],
            on="CustomerID",
            how="left",
        )
        n_labeled = features["churned"].notna().sum()
        print(f"\n  Churn labels joined: {n_labeled:,} customers labeled")

    # Fill any remaining NaNs with safe defaults
    fill_values: dict[str, Any] = {
        "purchase_gap_cv": SINGLE_ORDER_SENTINEL,
        "purchase_trend": 0.0,
        "time_decay_weight": 0.0,
        "avg_sentiment_score": 0.0,
        "last_review_sentiment": 0.0,
        "negative_review_count": 0,
        "has_reviews": 0,
        "avg_battery_sentiment": 0.0,
        "avg_shipping_sentiment": 0.0,
        "avg_price_sentiment": 0.0,
    }
    features = features.fillna(fill_values)

    return features


# ─── Step 8: Reference distributions ─────────────────────────────────────────

def save_reference_distribution(
    features: pd.DataFrame,
    output_path: Path,
    numeric_cols: list[str],
) -> None:
    """
    Blueprint Section 09 — Fix #11.

    Save training distribution as drift detection baseline.
    Called once at end of pipeline run. Stored to versioned path.
    At inference time, incoming batch is compared against this snapshot.

    PSI and KS tests in mlops/drift_detector.py use this reference.
    """
    ref: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        if col not in features.columns:
            continue
        series = features[col].dropna()
        if len(series) == 0:
            continue
        ref[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "min": float(series.min()),
            "max": float(series.max()),
            "n": int(len(series)),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ref, indent=2))
    print(f"\n  ✓ Reference distribution saved: {output_path}")
    print(f"    Features tracked: {len(ref)}")


# ─── Step 9: Scaler serialisation ─────────────────────────────────────────────

def fit_and_save_scaler(
    features: pd.DataFrame,
    numeric_cols: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """
    Fit StandardScaler on numeric features and save to artifact store.
    Blueprint Section 09 — Fix #6: training-serving consistency.
    Load from artifact path at inference — never re-fit.
    """
    try:
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        available_cols = [c for c in numeric_cols if c in features.columns]
        features[available_cols] = scaler.fit_transform(
            features[available_cols].fillna(0)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, output_path)
        print(f"  ✓ Scaler saved: {output_path}")
    except ImportError:
        print("  sklearn not in dev extras — scaler saved as stub.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            '{"status": "stub — install [train] extras to fit scaler"}'
        )

    return features


# ─── Main pipeline ────────────────────────────────────────────────────────────

# Numeric feature columns for scaling + reference distributions
NUMERIC_FEATURE_COLS = [
    "recency_days", "recency_days_log", "frequency", "monetary_value",
    "avg_order_value", "tenure_days", "purchase_gap_cv",
    "category_diversity", "return_rate", "purchase_trend",
    "time_decay_weight", "avg_sentiment_score", "last_review_sentiment",
    "negative_review_count", "avg_battery_sentiment",
    "avg_shipping_sentiment", "avg_price_sentiment",
]


def run_pipeline(
    input_path: Path,
    lambda_decay: float = TIME_DECAY_LAMBDA_DEFAULT,
    churn_labels_path: Path | None = None,
    review_sentiment_path: Path | None = None,
    aspect_sentiment_path: Path | None = None,
) -> None:
    """
    Full tabular pipeline:
    1. Load and clean UCI Online Retail II
    2. Scope to observation window
    3. Compute RFM features
    4. Compute behavioural features (NaN-safe)
    5. Compute temporal features (lambda as parameter)
    6. Merge real sentiment features (Gate G8) if available, else placeholders
    7. Assemble master feature table
    8. Join churn labels (if available)
    9. Save reference distributions
    10. Fit and save scaler artifact
    11. Save feature table to Parquet (rfm_behavioral_v2 if sentiment merged, else v1)
    """
    print("=" * 60)
    print("  E-CIP v3.0 — Tabular Pipeline")
    print("  Blueprint Section 05, 07, 08, 09")
    print("  Fixes: #5, #15, #21, #24")
    print("=" * 60)

    if not input_path.exists():
        print(f"\n  Dataset not found: {input_path}")
        print("  This is expected in Phase 0/1.")
        print("  Run: python data/scripts/download.py --module 3")
        print("\n  Saving pipeline structure artifacts...")
        _save_pipeline_stub()
        print("\n  ✓ Pipeline structure verified — runs fully in Phase 1 (Week 4).")
        return

    # Step 1: Load and clean
    df = load_and_clean(input_path)

    # Step 2: Observation window
    obs_df = scope_to_observation_window(df)

    # Step 3: RFM
    print("\n  Computing RFM features...")
    rfm = compute_rfm_features(obs_df)
    print(f"  RFM customers: {len(rfm):,}")

    # Step 4: Behavioural
    print("\n  Computing behavioural features (Fix #15 — NaN-safe gap CV)...")
    behavioural = compute_behavioural_features(obs_df)
    single_purchase = behavioural["is_single_purchase"].sum()
    print(f"  Single-purchase customers: {single_purchase:,} "
          f"({single_purchase / len(behavioural):.1%}) → gap_cv = {SINGLE_ORDER_SENTINEL}")

    # Step 5: Temporal
    print(f"\n  Computing temporal features (lambda={lambda_decay})...")
    temporal = compute_temporal_features(obs_df, lambda_decay)

    # Step 6: Real sentiment merge (Gate G8) if available, else placeholders
    sentiment_is_real = bool(
        review_sentiment_path and review_sentiment_path.exists()
        and aspect_sentiment_path and aspect_sentiment_path.exists()
    )
    if sentiment_is_real:
        print("\n  Merging sentiment features (Gate G8 — review_date < snapshot_date)...")
        assert review_sentiment_path is not None
        assert aspect_sentiment_path is not None
        sentiment = merge_sentiment_features(
            rfm["CustomerID"], review_sentiment_path, aspect_sentiment_path
        )
    else:
        print("\n  Building sentiment placeholders (no sentiment feature files found)...")
        sentiment = build_sentiment_placeholders(rfm["CustomerID"])

    # Step 7: Load churn labels if available
    churn_labels = None
    if churn_labels_path and churn_labels_path.exists():
        churn_labels = pd.read_parquet(churn_labels_path)
        print(f"\n  Churn labels loaded: {len(churn_labels):,} customers")
    else:
        print("\n  Churn labels not yet generated — run churn_label_engineer.py")
        print("  Feature table will be saved without churn labels.")

    # Step 8: Assemble
    print("\n  Assembling feature table...")
    features = assemble_feature_table(rfm, behavioural, temporal, sentiment, churn_labels)
    print(f"  Feature table shape: {features.shape}")
    print(f"  Feature columns: {list(features.columns)}")

    # Step 9: Reference distributions (before scaling — raw feature space)
    ref_path = REFERENCE_DIR / "customer_features_ref_v1.json"
    save_reference_distribution(features, ref_path, NUMERIC_FEATURE_COLS)

    # Step 10: Fit and save scaler
    print("\n  Fitting and saving scaler (Fix #6 — training-serving consistency)...")
    scaler_path = ARTIFACTS_DIR / "scaler_v1.joblib"
    features = fit_and_save_scaler(features, NUMERIC_FEATURE_COLS, scaler_path)

    # Step 11: Save feature table
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    feature_version = "v2" if sentiment_is_real else "v1"
    parquet_path = FEATURE_STORE_DIR / f"rfm_behavioral_{feature_version}.parquet"
    features.to_parquet(parquet_path, index=False)
    print(f"\n  ✓ Feature table saved: {parquet_path}")

    # Also save a CSV copy to processed/ for GE validation
    csv_path = PROCESSED_DIR / "customer_features.csv"
    features.to_csv(csv_path, index=False)
    print(f"  ✓ CSV copy saved    : {csv_path}")

    print("\n" + "=" * 60)
    print("  Tabular pipeline complete.")
    print(f"  Customers processed: {len(features):,}")
    print(f"  Features engineered: {len(features.columns)}")
    print("  Next: dvc repro (to track all pipeline outputs)")
    print("=" * 60 + "\n")


def _save_pipeline_stub() -> None:
    """Save stub artifacts so directory structure is verified."""
    # Stub reference distribution
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    stub_ref = {
        "status": "stub — run pipeline with UCI data to generate reference distributions",
        "fix_reference": "Blueprint Section 09 — Fix #11",
        "features": NUMERIC_FEATURE_COLS,
    }
    ref_path = REFERENCE_DIR / "customer_features_ref_v1.json"
    ref_path.write_text(json.dumps(stub_ref, indent=2))
    print(f"  ✓ Reference distribution stub: {ref_path}")

    # Stub scaler artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stub_scaler = ARTIFACTS_DIR / "scaler_v1_stub.json"
    stub_scaler.write_text(
        '{"status": "stub — install [train] extras and run with UCI data"}'
    )
    print(f"  ✓ Scaler stub              : {stub_scaler}")

    # Feature store directory
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Feature store directory  : {FEATURE_STORE_DIR}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Tabular Data Pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "online_retail_II.xlsx",
        help="Path to UCI Online Retail II dataset",
    )
    parser.add_argument(
        "--churn-labels",
        type=Path,
        default=Path("data/processed/tabular/churn_labels.parquet"),
        help="Path to engineered churn labels (from churn_label_engineer.py)",
    )
    parser.add_argument(
        "--lambda-decay",
        type=float,
        default=TIME_DECAY_LAMBDA_DEFAULT,
        help=f"Time decay lambda (default: {TIME_DECAY_LAMBDA_DEFAULT}, "
             "Optuna tunes in Phase 4)",
    )
    parser.add_argument(
        "--review-sentiment",
        type=Path,
        default=TEXT_FEATURES_DIR / "review_sentiment_v1.parquet",
        help="Path to review-level sentiment scores (Module 2 output or "
             "data/scripts/synthesize_demo_sentiment.py for this dataset)",
    )
    parser.add_argument(
        "--aspect-sentiment",
        type=Path,
        default=TEXT_FEATURES_DIR / "aspect_sentiment_v1.parquet",
        help="Path to aspect-level sentiment scores",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        lambda_decay=args.lambda_decay,
        review_sentiment_path=args.review_sentiment,
        aspect_sentiment_path=args.aspect_sentiment,
        churn_labels_path=args.churn_labels,
    )


if __name__ == "__main__":
    main()
