"""Stage 5 - Transform / normalization (the swappable module).

The core lever for cluster geometry. Banking RFM is extremely right-skewed
(Monetary skew ~47, AccountBalance ~61), which is what produces wall-like
K-Means clusters after a plain log + standardize. This module exposes a
registry of transform strategies behind one interface, fits each to the R/F/M
matrix, and reports how close to Gaussian and how decorrelated each makes the
features. Lower mean |skew| and lower feature correlation are proxies for
rounder, more separable clusters - the evidence used to pick the transform for
stage 6.

Run standalone after stages 1-4:
    python -m bank_rfm.transform --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from . import schema as S
from .config import PipelineConfig
from .utils import get_logger, write_json

log = get_logger("bank_rfm.transform")

DIAG_NAME = "transform_diagnostics.json"
PLOT_NAME = "transform_comparison.png"


# --------------------------------------------------------------------------- #
# Custom steps
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to learned [low, high] percentiles. Not invertible
    (clipping loses information); inverse is identity on the clipped values."""

    def __init__(self, low=0.01, high=0.99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.low, axis=0)
        self.hi_ = np.nanquantile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float)


class RFMScorer(BaseEstimator, TransformerMixin):
    """Quantile-based 1-5 scoring per feature (canonical RFM scoring).

    Recency is scored inversely (more recent -> higher score). Ties are handled
    by ranking before binning, so a near-constant Frequency collapses into few
    distinct scores - which is itself the visible signature of F degeneracy.
    Not cleanly invertible (binning is lossy).
    """

    def __init__(self, feature_names=None, recency_index=0, n_bins=5):
        self.feature_names = feature_names
        self.recency_index = recency_index
        self.n_bins = n_bins

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        # store quantile edges per feature from rank-transformed values
        self.edges_ = []
        qs = np.linspace(0, 1, self.n_bins + 1)
        for j in range(X.shape[1]):
            r = stats.rankdata(X[:, j], method="average") / len(X[:, j])
            self.edges_.append(np.quantile(r, qs))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            r = stats.rankdata(X[:, j], method="average") / len(X[:, j])
            score = np.clip(np.digitize(r, self.edges_[j][1:-1], right=True) + 1,
                            1, self.n_bins)
            if j == self.recency_index:        # recent = better -> invert
                score = (self.n_bins + 1) - score
            out[:, j] = score
        return out

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float)  # lossy; scores are the interpretation


def _log_step():
    return FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_transformer(name: str, cfg: PipelineConfig, n_features: int):
    """Return (sklearn estimator, invertible: bool) for a registry name."""
    seed = cfg.random_seed
    n_q = 1000
    reg = {
        # log1p then z-score: the proposal baseline
        "log_standard": (Pipeline([("log", _log_step()), ("scale", StandardScaler())]), True),
        # power transforms target Gaussianity directly
        "power_yeojohnson": (PowerTransformer(method="yeo-johnson", standardize=True), True),
        "power_boxcox": (PowerTransformer(method="box-cox", standardize=True), True),
        # rank/quantile mapping is immune to tail magnitude -> strongest vs skew
        "quantile_normal": (QuantileTransformer(output_distribution="normal",
                                                n_quantiles=n_q, subsample=100_000,
                                                random_state=seed), True),
        "quantile_uniform": (QuantileTransformer(output_distribution="uniform",
                                                 n_quantiles=n_q, subsample=100_000,
                                                 random_state=seed), True),
        # outlier-resistant centering (does not fix skew on its own)
        "robust": (RobustScaler(), True),
        "log_robust": (Pipeline([("log", _log_step()), ("scale", RobustScaler())]), True),
        # plain min-max (matches some literature; weak on skew - a comparator)
        "minmax": (MinMaxScaler(), True),
        # canonical 1-5 RFM scoring (bounded ordinal)
        "rfm_score": (RFMScorer(recency_index=_recency_index(cfg)), False),
    }
    if name not in reg:
        raise ValueError(f"Unknown transformer '{name}'. Available: {sorted(reg)}")
    est, invertible = reg[name]
    if cfg.winsorize:
        est = Pipeline([("winsor", Winsorizer(*cfg.winsorize_limits)), ("t", est)])
        invertible = False
    return est, invertible


def _recency_index(cfg: PipelineConfig) -> int:
    try:
        return cfg.feature_columns.index(S.RECENCY)
    except ValueError:
        return 0


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _diagnose(Z: np.ndarray, features: list[str]) -> dict:
    per_feature = {}
    for j, f in enumerate(features):
        col = Z[:, j]
        per_feature[f] = {
            "skew": float(stats.skew(col)),
            "kurtosis_excess": float(stats.kurtosis(col)),
            "std": float(np.std(col)),
        }
    corr = np.corrcoef(Z, rowvar=False)
    off = corr[~np.eye(len(features), dtype=bool)]
    return {
        "per_feature": per_feature,
        "mean_abs_skew": float(np.mean([abs(v["skew"]) for v in per_feature.values()])),
        "mean_abs_kurtosis": float(np.mean([abs(v["kurtosis_excess"]) for v in per_feature.values()])),
        "max_abs_offdiag_corr": float(np.max(np.abs(off))) if off.size else 0.0,
    }


def _plot(results: dict, X: np.ndarray, transformed: dict, features: list[str],
          cfg: PipelineConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        log.warning("matplotlib unavailable, skipping comparison plot: %s", e)
        return
    # scatter the two most informative axes (Recency vs Monetary) per transform
    xi = features.index(S.RECENCY) if S.RECENCY in features else 0
    yi = features.index(S.MONETARY) if S.MONETARY in features else len(features) - 1
    names = list(transformed)
    rng = np.random.default_rng(cfg.random_seed)
    idx = rng.choice(X.shape[0], size=min(cfg.plot_sample, X.shape[0]), replace=False)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, nm in zip(axes, names):
        Z = transformed[nm][idx]
        ax.scatter(Z[:, xi], Z[:, yi], s=3, alpha=0.15, linewidths=0)
        ax.set_title(f"{nm}\nmean|skew|={results[nm]['mean_abs_skew']:.2f}", fontsize=9)
        ax.set_xlabel(features[xi]); ax.set_ylabel(features[yi])
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Transformed feature space (Recency vs Monetary)", fontsize=11)
    fig.tight_layout()
    fig.savefig(cfg.reports_path / PLOT_NAME, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(cfg: PipelineConfig, rfm_parquet: Path | None = None) -> dict:
    rfm_parquet = rfm_parquet or (cfg.processed_path / "rfm_customer.parquet")
    df = pd.read_parquet(rfm_parquet)
    features = list(cfg.feature_columns)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"feature_columns not in RFM table: {missing}")

    X = df[features].to_numpy(dtype=float)
    cfg.transformed_path.mkdir(parents=True, exist_ok=True)
    cfg.models_path.mkdir(parents=True, exist_ok=True)
    (cfg.models_path / "transformers").mkdir(parents=True, exist_ok=True)

    # raw-space baseline for reference
    raw_diag = _diagnose(X, features)
    results: dict = {"_features": features, "_raw_space": raw_diag, "transformers": {}}
    transformed_cache: dict = {}

    for name in cfg.transformers:
        out_pq = cfg.transformed_path / f"{name}.parquet"
        model_fp = cfg.models_path / "transformers" / f"{name}.joblib"
        if out_pq.exists() and model_fp.exists() and not cfg.force:
            log.info("[%s] cached, loading for diagnostics.", name)
            Z = pd.read_parquet(out_pq)[features].to_numpy(dtype=float)
        else:
            est, invertible = build_transformer(name, cfg, len(features))
            Z = est.fit_transform(X)
            joblib.dump({"estimator": est, "features": features,
                         "invertible": invertible}, model_fp)
            out = pd.DataFrame(Z, columns=features)
            out.insert(0, S.CUSTOMER_ID, df[S.CUSTOMER_ID].to_numpy())
            out.to_parquet(out_pq, index=False)

        diag = _diagnose(Z, features)
        results["transformers"][name] = diag
        transformed_cache[name] = Z
        log.info("[%s] mean|skew|=%.3f  mean|kurt|=%.1f  max|corr|=%.3f",
                 name, diag["mean_abs_skew"], diag["mean_abs_kurtosis"],
                 diag["max_abs_offdiag_corr"])

    # rank transforms by how symmetric they make the features (skew proxy)
    ranking = sorted(results["transformers"].items(),
                     key=lambda kv: kv[1]["mean_abs_skew"])
    results["ranking_by_mean_abs_skew"] = [
        {"transformer": n, "mean_abs_skew": round(d["mean_abs_skew"], 4)}
        for n, d in ranking
    ]

    write_json(results, cfg.reports_path / DIAG_NAME)
    _plot(results["transformers"], X, transformed_cache, features, cfg)
    best = results["ranking_by_mean_abs_skew"][0]["transformer"]
    log.info("Stage 5 done. Least-skew transform: %s. Diagnostics: %s",
             best, cfg.reports_path / DIAG_NAME)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 - transform sweep")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    if args.force:
        cfg.force = True
    run(cfg)


if __name__ == "__main__":
    main()
