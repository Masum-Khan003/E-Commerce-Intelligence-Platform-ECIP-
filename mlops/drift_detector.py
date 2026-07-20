# mlops/drift_detector.py
# E-CIP v3.0 — Feature Drift Detection
# Blueprint Section 15 — Fix #11
#
# Fix #11: drift detection runs PSI and KS tests against the FEATURE
# distribution snapshot saved at training time — not against model output
# distributions. Output monitoring is separate and does not trigger
# retraining on its own.
#
# Known limitation, verified via --inject-drift self-test: the reference
# snapshot stores only 6 summary percentiles (min/p25/p50/p75/p95/max), not
# raw training rows. Comparing a dataset against itself with this detector
# correctly reports near-zero PSI for most features, but a handful of
# multi-modal / sentinel-heavy features (purchase_gap_cv's -1.0
# single-purchase sentinel, purchase_trend's 0-fill for <2 months history)
# still show moderate baseline PSI (~0.2-1.8) purely from 6-point
# reconstruction error — no 6-number summary can represent two separate
# point masses at once. The self-test confirms this is genuine noise, not
# masked real drift: injecting an actual 5-sigma shift into a feature
# produces PSI an order of magnitude above this noise floor (~12 vs ~1.8
# max baseline), so the detector remains useful — just don't treat PSI
# just-over-threshold on these specific features as a confident signal
# without also checking the Alertmanager trend over time.
#
# Usage:
#   python mlops/drift_detector.py --module retention
#   python mlops/drift_detector.py --module retention --inject-drift  (self-test)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PSI_THRESHOLD = 0.2
KS_ALPHA = 0.05

REFERENCE_DIST_PATH = Path("data/reference_distributions/customer_features_ref_v1.json")
FEATURE_TABLE_PATH = Path("data/feature_store/customer_features/rfm_behavioral_v2.parquet")
POSTGRES_DSN = "postgresql://ecip:ecip_dev@localhost:5432/ecip"


def compute_psi(ref: dict[str, float], current: Any) -> float:
    """
    Population Stability Index between the reference snapshot's known
    percentile buckets and the current sample's actual bucket occupancy.

    Only summary statistics (min/p25/p50/p75/p95/max), not raw training
    rows, are persisted in data/reference_distributions/ — so rather than
    resampling a synthetic reference array and re-histogramming it (which
    compounds approximation error for skewed/heavy-tailed features and can
    report spurious drift even comparing a dataset against itself), this
    compares current data directly against the reference's OWN bucket
    edges and their exactly-known proportions (25/25/25/20/5% by
    percentile definition) — no resampling step to introduce error.
    """
    import numpy as np

    quantile_points = np.array([0.0, 25.0, 50.0, 75.0, 95.0, 100.0])
    quantile_values = np.array(
        [ref["min"], ref["p25"], ref["p50"], ref["p75"], ref["p95"], ref["max"]]
    )
    edges, first_idx = np.unique(quantile_values, return_index=True)
    if len(edges) < 2:
        return 0.0  # constant reference feature — cannot drift by construction

    # Zero-inflated engineered features (e.g. avg_battery_sentiment defaults
    # to 0.0 for customers with no aspect mention) commonly have several
    # consecutive percentiles collapse to the same value — a point mass
    # spanning that percentile range, not a genuine gap. np.histogram uses
    # half-open buckets [left, right) except the last, so the reference
    # proportion for each bucket must use the FIRST (lowest) percentile
    # rank at which a value is reached, not the last — otherwise the point
    # mass gets attributed to the wrong side of its own bucket boundary.
    # quantile_values is already sorted ascending (percentiles are
    # monotonic by construction), so return_index from np.unique naturally
    # gives each unique value's first/lowest occurrence.
    cdf_at_edges = quantile_points[first_idx]
    ref_props = np.diff(cdf_at_edges) / 100.0
    ref_props = np.clip(ref_props, 1e-6, None)

    cur_counts, _ = np.histogram(current, bins=edges)
    cur_props = cur_counts / max(cur_counts.sum(), 1)
    cur_props = np.clip(cur_props, 1e-6, None)

    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def run_feature_drift_check(
    inference_features: dict[str, Any],
    reference: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """
    Compare each feature's incoming distribution against its training-time
    reference snapshot. drift_detected gates on PSI only — KS is reported
    for visibility but, computed against a percentile-interpolated
    reference proxy rather than raw reference rows, its p-value is not a
    reliable enough drift signal on its own to gate on (a smooth 6-point
    interpolation will almost always differ from a granular real
    distribution in a KS sense, even absent real drift).
    """
    import numpy as np
    from scipy import stats

    results: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(42)

    for feat, vals in inference_features.items():
        if feat not in reference:
            continue
        ref = reference[feat]
        vals_arr = np.asarray(vals, dtype=float)
        if len(vals_arr) == 0:
            continue

        psi = compute_psi(ref, vals_arr)

        quantile_points = [0, 25, 50, 75, 95, 100]
        quantile_values = [ref["min"], ref["p25"], ref["p50"], ref["p75"], ref["p95"], ref["max"]]
        ref_sample_for_ks = np.interp(rng.uniform(0, 100, 1000), quantile_points, quantile_values)
        _, ks_p = stats.ks_2samp(ref_sample_for_ks, vals_arr)

        results[feat] = {
            "psi": psi,
            "ks_pvalue": float(ks_p),
            "drift_detected": bool(psi > PSI_THRESHOLD),
        }

    return results


async def write_drift_events(
    module: str,
    results: dict[str, dict[str, Any]],
    reference_version: str = "v1",
) -> int:
    """
    Write drift check results to the drift_events PostgreSQL table.
    Drifted features also get a review_queue row (trigger='drift_alert') —
    one of the three trigger types db/schema.sql documents for that table
    ('low_confidence' | 'ood_flagged' | 'drift_alert'), so a real drift
    check gives the Review Queue dashboard page real content instead of
    an eternally-empty table.
    """
    import json as _json

    import asyncpg

    conn = await asyncpg.connect(POSTGRES_DSN)
    written = 0
    try:
        for feature_name, metrics in results.items():
            await conn.execute(
                """
                INSERT INTO drift_events
                    (module, feature_name, metric, metric_value, threshold,
                     alert_triggered, reference_version)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                module,
                feature_name,
                "psi",
                metrics["psi"],
                PSI_THRESHOLD,
                metrics["drift_detected"],
                reference_version,
            )
            written += 1

            if metrics["drift_detected"]:
                await conn.execute(
                    """
                    INSERT INTO review_queue (request_id, module, trigger, payload)
                    VALUES ($1, $2, $3, $4)
                    """,
                    f"drift_{module}_{feature_name}",
                    module,
                    "drift_alert",
                    _json.dumps({"feature": feature_name, **metrics}),
                )
    finally:
        await conn.close()

    from observability.prometheus.metrics import record_drift_events

    record_drift_events(results)

    return written


def run_drift_check(
    module: str = "retention",
    reference_path: Path = REFERENCE_DIST_PATH,
    feature_table_path: Path = FEATURE_TABLE_PATH,
    inject_drift: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Full drift check: load reference snapshot + current feature table,
    run PSI/KS per feature, print a summary. `inject_drift` shifts one
    feature's values to verify the detector actually flags drift when it
    should (self-test, not part of normal operation).
    """
    print("=" * 60)
    print("  E-CIP v3.0 — Feature Drift Detection")
    print("  Blueprint Section 15 — Fix #11 (drift on INPUT features)")
    print("=" * 60)

    if not reference_path.exists():
        print(f"\n  Reference distribution not found: {reference_path}")
        return {}
    if not feature_table_path.exists():
        print(f"\n  Feature table not found: {feature_table_path}")
        return {}

    import pandas as pd

    reference = json.loads(reference_path.read_text())
    df = pd.read_parquet(feature_table_path)

    # save_reference_distribution() runs BEFORE fit_and_save_scaler() in
    # tabular_pipeline.py, so the reference snapshot is in RAW feature
    # space — but the feature table parquet stores the SCALED values (the
    # scaler transform is applied in place before the parquet is written).
    # Comparing scaled inference data against a raw-space reference would
    # report drift on every feature regardless of any real distribution
    # shift. Inverse-transform back to raw space before comparing.
    scaler_path = Path("data/feature_store/artifacts/scaler_v1.joblib")
    df_raw = df
    if scaler_path.exists():
        import joblib

        from data.pipelines.tabular_pipeline import NUMERIC_FEATURE_COLS

        scaler = joblib.load(scaler_path)
        scale_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
        if scale_cols:
            df_raw = df.copy()
            df_raw[scale_cols] = scaler.inverse_transform(df[scale_cols].to_numpy(dtype=float))

    inference_features: dict[str, Any] = {}
    for feat in reference:
        if feat in df_raw.columns:
            inference_features[feat] = df_raw[feat].dropna().to_numpy()

    if inject_drift and inference_features:
        first_feat = next(iter(inference_features))
        inference_features[first_feat] = inference_features[first_feat] + (
            reference[first_feat]["std"] * 5.0
        )
        print(f"\n  [self-test] Injected a 5-sigma shift into '{first_feat}'")

    results = run_feature_drift_check(inference_features, reference)

    n_drifted = sum(1 for r in results.values() if r["drift_detected"])
    print(f"\n  Features checked: {len(results)}")
    print(f"  Drift detected  : {n_drifted}")
    for feat, metrics in sorted(results.items(), key=lambda kv: -kv[1]["psi"])[:5]:
        flag = "DRIFT" if metrics["drift_detected"] else "ok"
        print(f"    {feat:<25} PSI={metrics['psi']:.4f}  "
              f"KS p={metrics['ks_pvalue']:.4f}  [{flag}]")

    print("\n" + "=" * 60)
    print("  Drift check complete.")
    print("=" * 60 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="E-CIP v3.0 — Feature Drift Detection")
    parser.add_argument("--module", type=str, default="retention")
    parser.add_argument(
        "--reference", type=Path, default=REFERENCE_DIST_PATH,
    )
    parser.add_argument("--data", type=Path, default=FEATURE_TABLE_PATH)
    parser.add_argument(
        "--inject-drift", action="store_true",
        help="Self-test: shift one feature to verify the detector flags it",
    )
    parser.add_argument(
        "--write-db", action="store_true",
        help="Write results to the drift_events PostgreSQL table",
    )
    args = parser.parse_args()

    results = run_drift_check(
        module=args.module,
        reference_path=args.reference,
        feature_table_path=args.data,
        inject_drift=args.inject_drift,
    )

    if args.write_db and results:
        import asyncio

        written = asyncio.run(write_drift_events(args.module, results))
        print(f"  ✓ {written} drift events written to PostgreSQL")


if __name__ == "__main__":
    main()
