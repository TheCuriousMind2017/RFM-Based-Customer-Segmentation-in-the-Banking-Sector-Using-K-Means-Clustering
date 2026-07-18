"""Smoke test for stages 1-4 against synthetic data.

Run after generating synthetic data and executing the pipeline. Asserts the
invariants that downstream stages rely on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bank_rfm import schema as S  # noqa: E402


def main() -> None:
    rfm = pd.read_parquet(ROOT / "data/processed/rfm_customer.parquet")

    # one row per customer
    assert rfm[S.CUSTOMER_ID].is_unique, "CustomerID must be unique in RFM table"

    # core RFM well-formed
    assert (rfm[S.RECENCY] >= 0).all(), "Recency must be non-negative"
    assert (rfm[S.FREQUENCY] >= 1).all(), "Frequency must be >= 1"
    assert (rfm[S.MONETARY] > 0).all(), "Monetary must be positive"
    assert rfm[[S.RECENCY, S.FREQUENCY, S.MONETARY]].notna().all().all(), "no nulls in core RFM"

    # interim cleanliness
    clean = pd.read_parquet(ROOT / "data/interim/clean.parquet")
    assert clean[S.TRANSACTION_ID].is_unique, "Transaction IDs must be unique post-clean"
    assert (clean[S.TRANSACTION_AMOUNT] > 0).all(), "non-positive amounts must be dropped"
    assert pd.api.types.is_datetime64_any_dtype(clean[S.TRANSACTION_DATE]), "dates must be parsed"

    print(f"OK — {len(rfm)} customers, {len(clean)} clean transactions; all invariants hold.")


if __name__ == "__main__":
    main()
