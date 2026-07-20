# tests/unit/test_tabular_pipeline.py
# E-CIP v3.0 — Unit tests for data/pipelines/tabular_pipeline.py

from __future__ import annotations

import numpy as np
import pandas as pd

from data.pipelines.tabular_pipeline import (
    SINGLE_ORDER_SENTINEL,
    compute_behavioural_features,
)


def _obs_row(customer_id: str, invoice: str, date: pd.Timestamp, stock_code: str = "A1") -> dict[str, object]:
    return {"Customer ID": customer_id, "Invoice": invoice, "InvoiceDate": date, "StockCode": stock_code}


class TestComputeBehaviouralFeatures:
    def test_single_order_customer_gets_sentinel_not_nan(self) -> None:
        """
        Fix #15: a customer with exactly one order has no inter-purchase
        gap to compute a coefficient of variation from — this must return
        the SINGLE_ORDER_SENTINEL (-1.0), never NaN, so downstream scaling/
        modeling doesn't silently propagate NaN through the feature table.
        """
        obs_df = pd.DataFrame([
            _obs_row("100", "1", pd.Timestamp("2010-01-01")),
        ])
        result = compute_behavioural_features(obs_df)
        row = result[result["CustomerID"] == "100"].iloc[0]

        assert row["purchase_gap_cv"] == SINGLE_ORDER_SENTINEL
        assert not np.isnan(row["purchase_gap_cv"])
        assert row["is_single_purchase"] == 1

    def test_multi_order_customer_gets_real_cv_not_sentinel(self) -> None:
        obs_df = pd.DataFrame([
            _obs_row("200", "1", pd.Timestamp("2010-01-01")),
            _obs_row("200", "2", pd.Timestamp("2010-01-15")),
            _obs_row("200", "3", pd.Timestamp("2010-02-01")),
        ])
        result = compute_behavioural_features(obs_df)
        row = result[result["CustomerID"] == "200"].iloc[0]

        assert row["purchase_gap_cv"] != SINGLE_ORDER_SENTINEL
        assert row["purchase_gap_cv"] >= 0.0
        assert row["is_single_purchase"] == 0

    def test_mixed_batch_flags_only_single_purchase_customers(self) -> None:
        obs_df = pd.DataFrame([
            _obs_row("100", "1", pd.Timestamp("2010-01-01")),  # single
            _obs_row("200", "1", pd.Timestamp("2010-01-01")),  # multi
            _obs_row("200", "2", pd.Timestamp("2010-01-15")),
        ])
        result = compute_behavioural_features(obs_df)
        single = result[result["CustomerID"] == "100"].iloc[0]
        multi = result[result["CustomerID"] == "200"].iloc[0]

        assert single["is_single_purchase"] == 1
        assert multi["is_single_purchase"] == 0

    def test_category_diversity_counts_distinct_stock_codes(self) -> None:
        obs_df = pd.DataFrame([
            _obs_row("300", "1", pd.Timestamp("2010-01-01"), stock_code="A1"),
            _obs_row("300", "1", pd.Timestamp("2010-01-01"), stock_code="A2"),
            _obs_row("300", "1", pd.Timestamp("2010-01-01"), stock_code="A1"),
        ])
        result = compute_behavioural_features(obs_df)
        row = result[result["CustomerID"] == "300"].iloc[0]
        assert row["category_diversity"] == 2
