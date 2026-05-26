#!/usr/bin/env python3
"""
Project: Bank Customer Segmentation using K-Means
Stage: 05 - Static 2D Projection Matrix for Paper Inclusion
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_scaled_dataset():
    """Loads datasets and reconstructs the labeled scaled matrix."""
    raw_path = "data/processed/final_bank_segments.csv"
    scaled_path = "data/processed/rfm_scaled.csv"

    if not os.path.exists(raw_path) or not os.path.exists(scaled_path):
        raise FileNotFoundError(
            "[-] Data artifacts missing. Please execute prior stages sequentially."
        )

    df_raw = pd.read_csv(raw_path, index_col="CustomerID")
    df_scaled = pd.read_csv(scaled_path, index_col="CustomerID")

    # Sync cluster tags cleanly
    df_scaled["Cluster"] = df_raw["Cluster"]
    return df_scaled


def main():
    try:
        print("[*] Engineering 2D projections and distribution pairplot...")
        df_scaled = load_scaled_dataset()

        sns.set_theme(style="ticks")
        df_plot = df_scaled.copy()
        df_plot["Cluster"] = df_plot["Cluster"].astype(str)

        # Subsample for swift static chart rendering
        if len(df_plot) > 10000:
            df_plot = df_plot.sample(n=10000, random_state=42)

        # Sort values to ensure consistent legend ordering
        df_plot = df_plot.sort_values("Cluster")

        g = sns.pairplot(
            df_plot,
            hue="Cluster",
            diag_kind="kde",
            palette="Set1",
            plot_kws={"alpha": 0.4, "s": 15, "edgecolor": "none"},
            diag_kws={"fill": True, "alpha": 0.3},
        )

        g.fig.suptitle(
            "RFM Multi-Dimensional Projections and Feature Distributions",
            y=1.02,
            fontsize=14,
            fontweight="bold",
        )

        # 1. Save static high-resolution asset for document insertion
        output_png = "rfm_2d_projections_matrix.png"
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        print(f"[+] Static publication-grade pairplot saved as: {output_png}")

        # 2. Render GUI pop-up window
        plt.show()

    except Exception as e:
        print(f"\n[-] Pairplot Processing Failed: {str(e)}")


if __name__ == "__main__":
    main()