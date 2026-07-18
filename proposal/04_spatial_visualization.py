#!/usr/bin/env python3
"""
Project: Bank Customer Segmentation using K-Means
Stage: 04 - Interactive 3D Spatial Visualization (Raw vs. Scaled)
"""

import os
import pandas as pd
import plotly.express as px


def load_integrated_datasets():
    """Loads both the raw labeled data and scaled features for multi-aspect EDA."""
    raw_path = "data/processed/final_bank_segments.csv"
    scaled_path = "data/processed/rfm_scaled.csv"

    if not os.path.exists(raw_path) or not os.path.exists(scaled_path):
        raise FileNotFoundError(
            "[-] Data artifacts missing. Please execute stages 01, 02, and 03 first."
        )

    df_raw = pd.read_csv(raw_path, index_col="CustomerID")
    df_scaled = pd.read_csv(scaled_path, index_col="CustomerID")

    # Cast Cluster to string so Plotly treats it as a discrete categorical palette
    df_raw["Cluster"] = df_raw["Cluster"].astype(str)
    df_scaled["Cluster"] = df_scaled["Cluster"].astype(str)

    return df_raw, df_scaled


def generate_interactive_3d_plot(df: pd.DataFrame, space_type: str):
    """Generates an interactive, browser-based 3D scatter plot using Plotly."""
    print(f"[*] Compiling interactive 3D cluster space ({space_type})...")

    df_plot = df.copy()

    # Subsample if dataset causes browser rendering latency
    if len(df_plot) > 30000:
        print(f"    -> Subsampling 30,000 records for fluid 3D UI rotation...")
        df_plot = df_plot.sample(n=30000, random_state=42)

    # Apply pseudo-log rendering *only* if mapping raw data to handle the 500k skew
    log_setting = True if space_type == "Raw" else False

    fig = px.scatter_3d(
        df_plot,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Cluster",
        title=f"3D Vector Space Mapping: {space_type} Features",
        labels={
            "Recency": "Recency (R)",
            "Frequency": "Frequency (F)",
            "Monetary": "Monetary (M)",
        },
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        log_y=log_setting,
        log_z=log_setting
    )

    fig.update_traces(marker=dict(size=3, line=dict(width=0)))

    # Reverse Recency axis so recent users (0) face the front foreground
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title_text="Assigned Cluster"),
        scene=dict(
            xaxis=dict(autorange='reversed')
        )
    )

    # 1. Save file locally
    output_html = f"data/processed/3d_cluster_space_{space_type.lower()}.html"
    fig.write_html(output_html)
    print(f"[+] Interactive 3D visualization exported to: {output_html}")

    # 2. FORCE BROWSER POP-UP
    fig.show(renderer="browser")


def main():
    try:
        df_raw, df_scaled = load_integrated_datasets()

        # Plot Raw Feature Space (Will open first browser tab)
        generate_interactive_3d_plot(df_raw, space_type="Raw")

        # Plot Transformed/Scaled Feature Space (Will open second browser tab)
        generate_interactive_3d_plot(df_scaled, space_type="Scaled")

        print("\n[+] 3D EDA visualizations completed successfully with no blocking code.")
        print("[!] Interactive spaces are open in your browser tabs.")

    except Exception as e:
        print(f"\n[-] 3D Visualization Processing Failed: {str(e)}")


if __name__ == "__main__":
    main()
