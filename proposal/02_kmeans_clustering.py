#!/usr/bin/env python3
"""
Project: Bank Customer Segmentation using K-Means
Stage: 02 - Cluster Optimization & Mathematical Selection of K
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_scaled_data(filepath: str) -> pd.DataFrame:
    """Loads the preprocessed and scaled feature matrix."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[-] Scaled data missing at {filepath}. Run stage 01 first."
        )
    return pd.read_csv(filepath, index_col="CustomerID")


def evaluate_kmeans(scaled_df: pd.DataFrame, max_k: int = 6):
    """Calculates Inertia and Silhouette scores across a range of K values."""
    inertia_scores = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    print(
        f"[*] Evaluating K-Means configurations for K values between 2 and {max_k}..."
    )

    # Subsampling data for the Silhouette calculation if the dataset is massive
    # to prevent CPU execution bottlenecks while keeping statistical validity
    sample_df = (
        scaled_df.sample(n=50000, random_state=42)
        if len(scaled_df) > 50000
        else scaled_df
    )

    for k in k_range:
        print(f"    -> Computing metrics for K = {k}...")
        # n_init=10 runs K-Means with 10 different centroid seeds to avoid local minima
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_df)

        inertia_scores.append(kmeans.inertia_)

        # Calculate silhouette score on the consistent sample partition
        sample_labels = kmeans.predict(sample_df)
        sil_avg = silhouette_score(sample_df, sample_labels)
        silhouette_scores.append(sil_avg)

    return list(k_range), inertia_scores, silhouette_scores


def plot_metrics(k_values, inertia, silhouette):
    """Generates code-first dual plots for visual and mathematical analysis."""
    print("[*] Generating optimization diagnostic plots...")
    sns.set_theme(style="ticks")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Inertia (Elbow Method)
    color = "#1f77b4"
    ax1.set_xlabel("Number of Clusters (K)", fontweight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster WCSS)", color=color, fontweight="bold")
    ax1.plot(k_values, inertia, marker="o", color=color, linewidth=2, label="Inertia")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Instantiate a second axes that shares the same x-axis for Silhouette
    ax2 = ax1.twinx()
    color = "#d62728"
    ax2.set_ylabel("Average Silhouette Coefficient", color=color, fontweight="bold")
    ax2.plot(
        k_values,
        silhouette,
        marker="s",
        color=color,
        linewidth=2,
        linestyle="--",
        label="Silhouette Score",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(
        "K-Means Optimization: Evaluation of Clustering Metrics",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    fig.tight_layout()

    # Save to directory
    output_plot = "cluster_optimization_metrics.png"
    plt.savefig(output_plot, dpi=300)
    print(f"[+] Diagnostic evaluation plot saved cleanly as: {output_plot}")
    plt.show()


def main():
    scaled_data_path = "data/processed/rfm_scaled.csv"

    try:
        # 1. Load optimized features
        scaled_features = load_scaled_data(scaled_data_path)

        # 2. Evaluate K-Means configurations
        k_values, inertia, silhouette = evaluate_kmeans(scaled_features, max_k=6)

        # 3. Output numeric tables for your module write-up
        print("\n--- Structural Optimization Metrics Table ---")
        metrics_df = pd.DataFrame(
            {"K": k_values, "Inertia": inertia, "Silhouette Score": silhouette}
        ).set_index("K")
        print(metrics_df)

        # 4. Generate visual diagnosis
        plot_metrics(k_values, inertia, silhouette)

    except Exception as e:
        print(f"\n[-] Clustering Execution Halted: {str(e)}")


if __name__ == "__main__":
    main()