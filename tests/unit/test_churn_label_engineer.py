# tests/unit/test_churn_label_engineer.py
# E-CIP v3.0 — Unit tests for models/retention/churn_label_engineer.py

from __future__ import annotations

import pandas as pd

from models.retention.churn_label_engineer import (
    HORIZON_DAYS,
    OBS_START,
    SNAPSHOT_DATE,
    clean_data,
    engineer_churn_label,
    validate_labels,
)


def _raw_row(
    customer_id: float | None,
    country: str,
    invoice: str,
    quantity: int,
    price: float,
    date: pd.Timestamp,
) -> dict[str, object]:
    return {
        "Invoice": invoice,
        "Customer ID": customer_id,
        "Country": country,
        "Quantity": quantity,
        "Price": price,
        "InvoiceDate": date,
        "StockCode": "A1",
    }


class TestCleanData:
    def test_excludes_guest_checkouts(self) -> None:
        """Fix #5: null CustomerID rows (guest checkouts) must be dropped."""
        df = pd.DataFrame([
            _raw_row(100.0, "United Kingdom", "1", 1, 5.0, OBS_START),
            _raw_row(None, "United Kingdom", "2", 1, 5.0, OBS_START),
        ])
        cleaned = clean_data(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["Customer ID"] == "100"

    def test_excludes_non_uk_customers(self) -> None:
        """Fix #24: only UK customers — avoids mixed-currency monetary_value."""
        df = pd.DataFrame([
            _raw_row(100.0, "United Kingdom", "1", 1, 5.0, OBS_START),
            _raw_row(101.0, "France", "2", 1, 5.0, OBS_START),
        ])
        cleaned = clean_data(df)
        assert len(cleaned) == 1
        assert (cleaned["Country"] == "United Kingdom").all()

    def test_excludes_cancelled_invoices(self) -> None:
        df = pd.DataFrame([
            _raw_row(100.0, "United Kingdom", "1", 1, 5.0, OBS_START),
            _raw_row(100.0, "United Kingdom", "C1", 1, 5.0, OBS_START),
        ])
        cleaned = clean_data(df)
        assert len(cleaned) == 1
        assert not cleaned["Invoice"].astype(str).str.startswith("C").any()

    def test_excludes_non_positive_quantity_or_price(self) -> None:
        df = pd.DataFrame([
            _raw_row(100.0, "United Kingdom", "1", 1, 5.0, OBS_START),
            _raw_row(100.0, "United Kingdom", "2", -1, 5.0, OBS_START),
            _raw_row(100.0, "United Kingdom", "3", 1, 0.0, OBS_START),
        ])
        cleaned = clean_data(df)
        assert len(cleaned) == 1


class TestEngineerChurnLabel:
    def test_customer_with_no_horizon_purchase_is_churned(self) -> None:
        df = pd.DataFrame([
            _raw_row(100.0, "United Kingdom", "1", 1, 5.0, OBS_START),
        ])
        df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
        labels = engineer_churn_label(df)
        row = labels[labels["CustomerID"] == "100"].iloc[0]
        assert row["churned"] == 1

    def test_customer_with_horizon_purchase_is_retained(self) -> None:
        horizon_date = SNAPSHOT_DATE + pd.Timedelta(days=HORIZON_DAYS - 1)
        df = pd.DataFrame([
            _raw_row(100.0, "United Kingdom", "1", 1, 5.0, OBS_START),
            _raw_row(100.0, "United Kingdom", "2", 1, 5.0, horizon_date),
        ])
        df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
        labels = engineer_churn_label(df)
        row = labels[labels["CustomerID"] == "100"].iloc[0]
        assert row["churned"] == 0

    def test_purchase_after_horizon_end_does_not_count_as_retained(self) -> None:
        """A purchase after the 90-day horizon must not retroactively retain the customer."""
        too_late = SNAPSHOT_DATE + pd.Timedelta(days=HORIZON_DAYS + 30)
        df = pd.DataFrame([
            _raw_row(100.0, "United Kingdom", "1", 1, 5.0, OBS_START),
            _raw_row(100.0, "United Kingdom", "2", 1, 5.0, too_late),
        ])
        df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
        labels = engineer_churn_label(df)
        row = labels[labels["CustomerID"] == "100"].iloc[0]
        assert row["churned"] == 1


class TestValidateLabels:
    def test_flags_out_of_range_churn_rate(self) -> None:
        labels = pd.DataFrame({"CustomerID": ["1", "2", "3", "4", "5"], "churned": [1, 1, 1, 1, 1]})
        assert validate_labels(labels) is False  # 100% churn — outside [15%, 80%]

    def test_accepts_in_range_churn_rate(self) -> None:
        labels = pd.DataFrame({
            "CustomerID": [str(i) for i in range(10)],
            "churned": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        })
        assert validate_labels(labels) is True  # 50% churn — within [15%, 80%]
