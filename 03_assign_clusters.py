#!/usr/bin/env python3
"""
Project: Bank Customer Segmentation using K-Means
Stage: 03 - Final Model Fitting & Structural Cluster Assignment
"""

import os
import pandas as pd
from sklearn.cluster import KMeans


def main(k_clusters: int = 5):
    print(f"[*] Initializing final K-Means model execution (K={k_clusters})...")

    scaled_path = "data/processed/rfm_scaled.csv"
    raw_path = "data/processed/rfm_raw.csv"
    output_dir = "data/processed"

    if not os.path.exists(scaled_path) or not os.path.exists(raw_path):
        raise FileNotFoundError("[-] Missing preprocessing artifacts. Run '01_rfm_aggregation.py' first.")

    # 1. Load data matrices
    df_scaled = pd.read_csv(scaled_path, index_col="CustomerID")
    df_raw = pd.read_csv(raw_path, index_col="CustomerID")

    # 2. Fit final K-Means on scaled isotropic space
    print("    -> Fitting final K-Means model via k-means++ initialization...")
    kmeans = KMeans(n_clusters=k_clusters, init="k-means++", random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_scaled)

    # 3. Inject matching assignments into both data matrices
    df_scaled["Cluster"] = cluster_labels
    df_raw["Cluster"] = cluster_labels

    # 4. Export labeled datasets
    print("[*] Exporting structural cluster assignments...")
    df_raw.to_csv(f"{output_dir}/final_bank_segments.csv")
    df_scaled.to_csv(f"{output_dir}/rfm_scaled.csv")

    print(f"[+] Stage 03 Complete! Assigned clusters exported to {output_dir}/final_bank_segments.csv")


if __name__ == "__main__":
    try:
        main(k_clusters=5)
    except Exception as e:
        print(f"\n[-] Assignment Failed: {str(e)}")