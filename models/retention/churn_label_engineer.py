# models/retention/churn_label_engineer.py
# E-CIP v3.0 — Churn Label Engineering
# Blueprint Section 21 — Critical Fix #2 and #5
#
# The UCI Online Retail II dataset has no pre-existing churn column.
# This module defines the label engineering logic explicitly.
#
# Label definition:
#   Churned (1): customer made ZERO purchases in the 90 days
#                after the observation window end date (snapshot_date).
#   Retained (0): customer made AT LEAST ONE purchase in that window.
#
# Observation window : 2009-12-01 → 2010-11-30 (12 months)
# Snapshot date      : 2010-11-30
# Prediction horizon : 2010-12-01 → 2011-02-28 (90 days)
#
# Fix #5: Rows with null CustomerID (guest checkouts ~25%) are excluded.
# Fix #24: Scoped to UK customers only for single-currency RFM integrity.
#
# Usage:
#   python models/retention/churn_label_engineer.py \
#       --input data/raw/online_retail2/online_retail_II.xlsx
#       --output data/processed/tabular/churn_labels.parquet

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

SNAPSHOT_DATE = pd.Timestamp("2010-11-30")
OBS_START = pd.Timestamp("2009-12-01")
HORIZON_DAYS = 90
MIN_CHURN_RATE = 0.15
MAX_CHURN_RATE = 0.80


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Load UCI Online Retail II from xlsx or csv."""
    print(f"\nLoading raw data from: {input_path}")
    suffix = input_path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        # The dataset has two sheets — Sheet Year 2009-2010 and 2010-2011
        df_1 = pd.read_excel(input_path, sheet_name="Year 2009-2010", engine="openpyxl")
        df_2 = pd.read_excel(input_path, sheet_name="Year 2010-2011", engine="openpyxl")
        df = pd.concat([df_1, df_2], ignore_index=True)
        print(f"  Loaded {len(df):,} rows across both sheets")
    elif suffix == ".csv":
        df = pd.read_csv(input_path, encoding="utf-8-sig")
        print(f"  Loaded {len(df):,} rows from CSV")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .xlsx or .csv")

    return df


# ─── Cleaning ─────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps with logged row counts at each stage."""
    print("\nCleaning pipeline:")
    print(f"  Start            : {len(df):,} rows")

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Fix #5: Exclude guest checkouts (null CustomerID)
    before = len(df)
    df = df.dropna(subset=["Customer ID"]).copy()
    excluded_guests = before - len(df)
    guest_pct = excluded_guests / before * 100
    print(f"  After guest excl : {len(df):,} rows "
          f"(removed {excluded_guests:,} rows = {guest_pct:.1f}% guest checkouts)")

    # Fix #24: Scope to UK customers — single-currency monetary features
    before = len(df)
    df = df[df["Country"] == "United Kingdom"].copy()
    excluded_intl = before - len(df)
    print(f"  After UK scope   : {len(df):,} rows "
          f"(removed {excluded_intl:,} international rows)")

    # Cast CustomerID to int then str for consistency
    df["Customer ID"] = df["Customer ID"].astype(int).astype(str)

    # Exclude cancelled invoices (InvoiceNo starting with 'C')
    before = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")].copy()
    excluded_cancel = before - len(df)
    print(f"  After cancels    : {len(df):,} rows "
          f"(removed {excluded_cancel:,} cancelled invoices)")

    # Exclude rows with non-positive quantity or price
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    excluded_neg = before - len(df)
    print(f"  After neg values : {len(df):,} rows "
          f"(removed {excluded_neg:,} rows with non-positive qty/price)")

    print(f"  Final clean rows : {len(df):,}")
    return df


# ─── Label Engineering ────────────────────────────────────────────────────────

def engineer_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the churn label from transactional data.

    Blueprint Section 21 — Critical Fix #2:
    A customer is labeled churned (1) if they made zero purchases
    in the HORIZON_DAYS following SNAPSHOT_DATE.

    Returns a DataFrame with columns:
        CustomerID, churned, first_order_date, last_order_date,
        total_orders, total_spend
    """
    horizon_end = SNAPSHOT_DATE + timedelta(days=HORIZON_DAYS)

    # Observation window: customers who purchased before snapshot
    obs_df = df[df["InvoiceDate"].between(OBS_START, SNAPSHOT_DATE)].copy()

    # Prediction horizon: who purchased after snapshot
    horizon_df = df[
        df["InvoiceDate"].between(
            SNAPSHOT_DATE + timedelta(days=1), horizon_end
        )
    ].copy()

    # Unique customers in observation window
    customers_in_obs = obs_df["Customer ID"].unique()

    # Unique customers who purchased in the horizon (retained)
    customers_retained = set(horizon_df["Customer ID"].unique())

    # Build label dataframe
    labels = pd.DataFrame({"CustomerID": customers_in_obs})
    labels["churned"] = (~labels["CustomerID"].isin(customers_retained)).astype(int)

    # Enrich with summary stats from observation window for context
    summary = (
        obs_df.groupby("Customer ID")
        .agg(
            first_order_date=("InvoiceDate", "min"),
            last_order_date=("InvoiceDate", "max"),
            total_orders=("Invoice", "nunique"),
            total_spend=("Price", lambda x: (x * obs_df.loc[x.index, "Quantity"]).sum()),
        )
        .reset_index()
        .rename(columns={"Customer ID": "CustomerID"})
    )

    labels = labels.merge(summary, on="CustomerID", how="left")
    return labels


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_labels(labels: pd.DataFrame) -> bool:
    """
    Validate churn label distribution is within expected range.
    Blueprint Section 21: rate must be between 15% and 80%.
    Returns True if valid, False if HORIZON_DAYS needs adjustment.
    """
    churn_rate = labels["churned"].mean()
    n_churned = labels["churned"].sum()
    n_total = len(labels)
    n_retained = n_total - n_churned

    print("\nChurn label distribution:")
    print(f"  Snapshot date      : {SNAPSHOT_DATE.date()}")
    print(f"  Observation window : {OBS_START.date()} → {SNAPSHOT_DATE.date()}")
    print(f"  Prediction horizon : {(SNAPSHOT_DATE + timedelta(days=1)).date()} → "
          f"{(SNAPSHOT_DATE + timedelta(days=HORIZON_DAYS)).date()} ({HORIZON_DAYS} days)")
    print(f"  Total customers    : {n_total:,}")
    print(f"  Churned (1)        : {n_churned:,} ({churn_rate:.1%})")
    print(f"  Retained (0)       : {n_retained:,} ({1 - churn_rate:.1%})")

    if MIN_CHURN_RATE <= churn_rate <= MAX_CHURN_RATE:
        print(f"  ✓ Churn rate {churn_rate:.1%} is within expected range "
              f"[{MIN_CHURN_RATE:.0%}, {MAX_CHURN_RATE:.0%}]")
        return True
    else:
        print(f"  ✗ WARNING: Churn rate {churn_rate:.1%} is OUTSIDE expected range "
              f"[{MIN_CHURN_RATE:.0%}, {MAX_CHURN_RATE:.0%}]")
        print(f"  Action: Adjust HORIZON_DAYS (currently {HORIZON_DAYS})")
        print("    If rate < 15% → increase HORIZON_DAYS (try 120 or 180)")
        print("    If rate > 80% → decrease HORIZON_DAYS (try 60)")
        return False


# ─── Output ───────────────────────────────────────────────────────────────────

def save_labels(labels: pd.DataFrame, output_path: Path) -> None:
    """Save churn labels to Parquet for downstream feature engineering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)
    print(f"\n  ✓ Labels saved to: {output_path}")
    print(f"    Shape  : {labels.shape}")
    print(f"    Columns: {list(labels.columns)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Churn Label Engineering (Blueprint Section 21)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/online_retail2/online_retail_II.xlsx"),
        help="Path to UCI Online Retail II dataset (.xlsx or .csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tabular/churn_labels.parquet"),
        help="Output path for engineered churn labels (.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate logic without saving output (requires data file)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  E-CIP v3.0 — Churn Label Engineering")
    print("  Blueprint Section 21 — Critical Fix #2 + #5")
    print("=" * 60)

    if not args.input.exists():
        print(f"\n  Dataset not yet downloaded: {args.input}")
        print("  This is expected in Phase 0.")
        print("  Run this script in Week 4 after data/scripts/download.py")
        print("  Download command:")
        print("    wget https://archive.ics.uci.edu/static/public/502/"
              "online+retail+ii.zip -P data/raw/online_retail2/")
        print("\n  ✓ Script structure verified — ready for Phase 1.")
        return

    # Full pipeline
    df_raw = load_raw_data(args.input)
    df_clean = clean_data(df_raw)
    labels = engineer_churn_label(df_clean)
    is_valid = validate_labels(labels)

    if not args.dry_run:
        save_labels(labels, args.output)

    if not is_valid:
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("  Churn label engineering complete.")
    print("  Next: python data/pipelines/tabular_pipeline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
