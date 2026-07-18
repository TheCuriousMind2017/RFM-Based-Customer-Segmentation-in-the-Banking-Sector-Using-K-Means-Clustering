"""Stage 6 - Clustering sweep.

Fits K-Means across every (transform x K) cell and records the internal
scorecard on each cell: Inertia (elbow), sampled Silhouette, Calinski-Harabasz,
and Davies-Bouldin. Writes a tidy results table and metric-vs-K plots. The
sweep does not persist labels for every cell (refitting at the chosen point in
stage 7 is cheap and deterministic under a fixed seed); it produces the
evidence used to choose K.

Reporting rule (settled): Calinski-Harabasz and Davies-Bouldin are unbounded
and scale-dependent, so they compare K *within a fixed transform* only - never
to rank one transform against another. Silhouette (bounded) is the
cross-comparable internal index.

Run standalone after stage 5:
    python -m bank_rfm.cluster --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from . import schema as S
from .config import PipelineConfig
from .utils import get_logger, write_json

log = get_logger("bank_rfm.cluster")

SWEEP_CSV = "cluster_sweep.csv"
SWEEP_PLOT = "cluster_sweep.png"


def _load_transformed(cfg: PipelineConfig, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    fp = cfg.transformed_path / f"{name}.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"Transformed space '{name}' not found at {fp}. Run stage 5 first."
        )
    df = pd.read_parquet(fp)
    feats = [c for c in cfg.feature_columns if c in df.columns]
    return df[feats].to_numpy(dtype=float), df


def fit_kmeans(X: np.ndarray, k: int, cfg: PipelineConfig) -> KMeans:
    return KMeans(n_clusters=k, n_init=cfg.kmeans_n_init,
                  max_iter=cfg.kmeans_max_iter,
                  random_state=cfg.random_seed).fit(X)


def _cell_metrics(X: np.ndarray, labels: np.ndarray, inertia: float,
                  cfg: PipelineConfig) -> dict:
    n = X.shape[0]
    sample = min(cfg.silhouette_sample, n)
    sil = float(silhouette_score(X, labels, sample_size=sample,
                                 random_state=cfg.random_seed))
    return {
        "inertia": float(inertia),
        "silhouette": sil,
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def _plot_sweep(df: pd.DataFrame, cfg: PipelineConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        log.warning("matplotlib unavailable, skipping sweep plot: %s", e)
        return
    metrics = [("inertia", "Inertia (elbow)"),
               ("silhouette", "Silhouette (higher=better)"),
               ("calinski_harabasz", "Calinski-Harabasz (higher=better)"),
               ("davies_bouldin", "Davies-Bouldin (lower=better)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for name, g in df.groupby("transform"):
            ax.plot(g["k"], g[col], marker="o", ms=4, label=name)
        ax.set_title(title); ax.set_xlabel("k")
        ax.legend(fontsize=8)
    fig.suptitle("Stage 6 - clustering scorecard across (transform x K)")
    fig.tight_layout()
    fig.savefig(cfg.reports_path / SWEEP_PLOT, dpi=110)
    plt.close(fig)


def run(cfg: PipelineConfig) -> pd.DataFrame:
    cfg.reports_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in cfg.cluster_transformers:
        X, _ = _load_transformed(cfg, name)
        for k in range(cfg.k_min, cfg.k_max + 1):
            km = fit_kmeans(X, k, cfg)
            m = _cell_metrics(X, km.labels_, km.inertia_, cfg)
            rows.append({"transform": name, "k": k, **m})
            log.info("[%s | k=%d] sil=%.4f  CH=%.0f  DB=%.4f",
                     name, k, m["silhouette"], m["calinski_harabasz"],
                     m["davies_bouldin"])
    df = pd.DataFrame(rows)
    df.to_csv(cfg.reports_path / SWEEP_CSV, index=False)
    _plot_sweep(df, cfg)

    # Per-transform best K by silhouette (a starting point, not the final call).
    best = (df.loc[df.groupby("transform")["silhouette"].idxmax()]
              [["transform", "k", "silhouette"]]
              .to_dict("records"))
    write_json({"best_k_by_silhouette": best}, cfg.reports_path / "cluster_sweep_summary.json")
    log.info("Sweep done. Best-by-silhouette: %s", best)
    log.info("Results: %s", cfg.reports_path / SWEEP_CSV)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 6 - clustering sweep")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    if args.force:
        cfg.force = True
    run(cfg)


if __name__ == "__main__":
    main()
