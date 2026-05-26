#!/usr/bin/env python3
"""
Project: Bank Customer Segmentation using K-Means
Stage: 01 - Full Feature Engineering, Transformation & Normalization
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Loads banking data and enforces robust, optimized datetime parsing for mixed string layouts."""
    print(f"[*] Reading raw CSV from {filepath}...")
    df = pd.read_csv(filepath)

    # Using format="mixed" handles 2-digit and 4-digit years seamlessly,
    # while dayfirst=True safeguards native DD/MM/YYYY banking structures.
    df["TransactionDate"] = pd.to_datetime(
        df["TransactionDate"], format="mixed", dayfirst=True
    )
    return df


def engineer_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transactional rows into deterministic per-customer RFM features."""
    snapshot_date = df["TransactionDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            {
                "TransactionDate": lambda x: (snapshot_date - x.max()).days,
                "CustomerID": "count",
                "TransactionAmount (INR)": "sum",
            }
        )
        .rename(
            columns={
                "TransactionDate": "Recency",
                "CustomerID": "Frequency",
                "TransactionAmount (INR)": "Monetary",
            }
        )
    )
    return rfm


def transform_and_scale(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Treats right-skewness via logarithmic mapping and scales features for isotropic distance calculations."""
    print("[*] Applying log(x + 1) transform to compress heavy right-skewness...")
    # log1p handles zero-bounds safely and prevents -inf errors
    rfm_log = np.log1p(rfm_df)

    print("[*] Implementing StandardScaler (Zero Mean, Unit Variance)...")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(rfm_log)

    # Convert back to a structured DataFrame for downstream validation
    scaled_df = pd.DataFrame(
        scaled_array, index=rfm_df.index, columns=rfm_df.columns
    )
    return scaled_df


def main():
    raw_data_path = "data/bank_transactions.csv"
    output_dir = "data/processed"

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Ingest & Aggregate
        raw_df = load_and_validate_data(raw_data_path)
        rfm_raw_df = engineer_rfm_features(raw_df)

        # 2. Transform & Scale
        rfm_scaled_df = transform_and_scale(rfm_raw_df)

        # 3. Export both artifacts
        rfm_raw_df.to_csv(f"{output_dir}/rfm_raw.csv")
        rfm_scaled_df.to_csv(f"{output_dir}/rfm_scaled.csv")

        print(f"[+] Stage 01 Complete! Output files written to {output_dir}/")
        print("\n--- Transformed & Scaled Feature Preview ---")
        print(rfm_scaled_df.head())

    except Exception as e:
        print(f"\n[-] Pipeline Halted: {str(e)}")


if __name__ == "__main__":
    main()