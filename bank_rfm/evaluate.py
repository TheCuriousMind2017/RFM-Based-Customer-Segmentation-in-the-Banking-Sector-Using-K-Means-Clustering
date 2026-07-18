"""Stage 7 - Evaluation scorecard (deep diagnostics at the chosen operating point).

Reads the stage-6 sweep, selects K (config.selected_k, else auto by silhouette
within the primary transform), refits K-Means at the chosen point, and runs the
diagnostics that the per-cell sweep cannot:

  * Gap statistic (principled K selection) on the primary transform
  * GMM + BIC shape diagnostic - the actual test of whether the data is
    spherical (K-Means assumption) or elliptical: compares spherical/diag/full
    covariance. full << spherical => K-Means is mismodelling genuine structure
  * ARI / NMI vs a canonical RFM-scoring segmentation (does ML clustering add
    anything over textbook RFM scoring), reported with the Frequency-score
    spread so the F degeneracy is visible and honestly interpreted
  * per-feature contribution to separation (Frequency should be small)
  * leave-F-out ARI (refit on R+M; high agreement => F barely moves assignments)

Settled framing: the internal scorecard selects the operating point; GMM+BIC and
ARI are interpretive evidence and do not silently override that selection.

Run after stage 6:
    python -m bank_rfm.evaluate --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from . import schema as S
from .cluster import SWEEP_CSV, fit_kmeans
from .config import PipelineConfig
from .utils import get_logger, write_json

log = get_logger("bank_rfm.evaluate")

EVAL_NAME = "evaluation_report.json"
SELECTED_LABELS = "selected_labels.parquet"

# Standard RFM segment grid indexed by (R score, FM score), each 1-5.
# A recognised scheme used to turn 1-5 scores into named segments; serves as the
# canonical-RFM reference partition for ARI/NMI.
_SEGMENT_GRID = {
    (5, 5): "Champions", (5, 4): "Champions", (4, 5): "Champions", (4, 4): "Champions",
    (5, 3): "Loyal", (4, 3): "Loyal", (3, 5): "Loyal", (3, 4): "Loyal",
    (5, 2): "Potential Loyalist", (4, 2): "Potential Loyalist",
    (5, 1): "New", (4, 1): "Promising",
    (3, 3): "Need Attention", (3, 2): "Need Attention",
    (3, 1): "About to Sleep", (2, 1): "About to Sleep",
    (2, 5): "At Risk", (2, 4): "At Risk", (1, 5): "Cant Lose", (1, 4): "Cant Lose",
    (2, 3): "At Risk", (1, 3): "At Risk",
    (2, 2): "Hibernating", (1, 2): "Hibernating",
    (1, 1): "Lost",
}


def _load_space(cfg: PipelineConfig, name: str):
    df = pd.read_parquet(cfg.transformed_path / f"{name}.parquet")
    feats = [c for c in cfg.feature_columns if c in df.columns]
    return df[feats].to_numpy(dtype=float), df, feats


def _select_k(cfg: PipelineConfig, sweep: pd.DataFrame) -> int:
    if cfg.selected_k is not None:
        return int(cfg.selected_k)
    sub = sweep[sweep["transform"] == cfg.primary_transform]
    return int(sub.loc[sub["silhouette"].idxmax(), "k"])


def _subsample(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    if X.shape[0] <= n:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(X.shape[0], size=n, replace=False)]


def gap_statistic(X: np.ndarray, cfg: PipelineConfig) -> dict:
    Xs = _subsample(X, cfg.gap_subsample, cfg.random_seed)
    rng = np.random.default_rng(cfg.random_seed)
    lo, hi = Xs.min(0), Xs.max(0)
    ks = list(range(cfg.k_min, cfg.k_max + 1))
    gaps, sks = [], []
    for k in ks:
        wk = KMeans(k, n_init=5, random_state=cfg.random_seed).fit(Xs).inertia_
        ref = []
        for b in range(cfg.gap_b):
            Xb = rng.uniform(lo, hi, size=Xs.shape)
            ref.append(np.log(KMeans(k, n_init=5, random_state=cfg.random_seed + b).fit(Xb).inertia_))
        gaps.append(float(np.mean(ref) - np.log(wk)))
        sks.append(float(np.std(ref) * np.sqrt(1 + 1 / cfg.gap_b)))
    # Tibshirani rule: smallest k with gap[k] >= gap[k+1] - s[k+1]
    choice = ks[-1]
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            choice = ks[i]
            break
    return {"k": ks, "gap": gaps, "s_k": sks, "gap_choice": choice}


def gmm_shape_diagnostic(X: np.ndarray, k: int, cfg: PipelineConfig) -> dict:
    Xs = _subsample(X, cfg.eval_subsample, cfg.random_seed)
    bic = {}
    for cov in cfg.gmm_covariances:
        gm = GaussianMixture(n_components=k, covariance_type=cov,
                             random_state=cfg.random_seed, max_iter=200).fit(Xs)
        bic[cov] = float(gm.bic(Xs))
    best = min(bic, key=bic.get)
    spherical = bic.get("spherical")
    full = bic.get("full")
    verdict = "inconclusive"
    if spherical is not None and full is not None:
        # Lower BIC = better fit. If full beats spherical substantially, the
        # data has elliptical/anisotropic structure K-Means cannot capture.
        rel = (spherical - full) / abs(spherical)
        verdict = ("data favours elliptical structure (K-Means limited)"
                   if rel > 0.02 else
                   "spherical assumption adequate (K-Means appropriate)")
    return {"bic": bic, "best_covariance": best, "verdict": verdict}


def canonical_rfm_segments(cfg: PipelineConfig) -> tuple[np.ndarray, dict]:
    df = pd.read_parquet(cfg.transformed_path / f"{cfg.rfm_reference}.parquet")
    r = df[S.RECENCY].round().astype(int).clip(1, 5)
    f = df[S.FREQUENCY].round().astype(int).clip(1, 5)
    m = df[S.MONETARY].round().astype(int).clip(1, 5)
    fm = ((f + m) / 2).round().astype(int).clip(1, 5)
    seg = [_SEGMENT_GRID.get((ri, fmi), "Other") for ri, fmi in zip(r, fm)]
    f_spread = {int(k): int(v) for k, v in f.value_counts().sort_index().items()}
    return np.array(seg), {"frequency_score_distribution": f_spread}


def per_feature_contribution(km: KMeans, feats: list[str]) -> dict:
    spread = km.cluster_centers_.std(axis=0)
    total = spread.sum() or 1.0
    return {f: round(float(s / total), 4) for f, s in zip(feats, spread)}


def run(cfg: PipelineConfig) -> dict:
    sweep_fp = cfg.reports_path / SWEEP_CSV
    if not sweep_fp.exists():
        raise FileNotFoundError(f"{sweep_fp} not found. Run stage 6 first.")
    sweep = pd.read_csv(sweep_fp)
    k = _select_k(cfg, sweep)
    log.info("Selected operating point: transform=%s, k=%d", cfg.primary_transform, k)

    Xp, dfp, feats = _load_space(cfg, cfg.primary_transform)
    km = fit_kmeans(Xp, k, cfg)

    # --- persist selected labels for stage 8 ---
    cfg.clusters_path.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({S.CUSTOMER_ID: dfp[S.CUSTOMER_ID].to_numpy(), "cluster": km.labels_})
    out.to_parquet(cfg.clusters_path / SELECTED_LABELS, index=False)

    report: dict = {
        "operating_point": {"transform": cfg.primary_transform, "k": k},
        "cluster_sizes": {int(c): int(n) for c, n in
                          zip(*np.unique(km.labels_, return_counts=True))},
    }

    # --- gap statistic (primary transform) ---
    log.info("Computing gap statistic ...")
    report["gap_statistic"] = gap_statistic(Xp, cfg)

    # --- GMM + BIC shape diagnostic ---
    log.info("Running GMM+BIC shape diagnostic ...")
    report["gmm_shape"] = gmm_shape_diagnostic(Xp, k, cfg)

    # --- ARI / NMI vs canonical RFM scoring ---
    log.info("Comparing against canonical RFM scoring ...")
    seg, seg_meta = canonical_rfm_segments(cfg)
    report["rfm_agreement"] = {
        "ari": float(adjusted_rand_score(seg, km.labels_)),
        "nmi": float(normalized_mutual_info_score(seg, km.labels_)),
        **seg_meta,
    }

    # --- per-feature contribution ---
    report["feature_contribution"] = per_feature_contribution(km, feats)

    # --- leave-F-out ARI (Frequency limitation evidence) ---
    if S.FREQUENCY in feats:
        rm_idx = [i for i, f in enumerate(feats) if f != S.FREQUENCY]
        km_rm = fit_kmeans(Xp[:, rm_idx], k, cfg)
        report["leave_f_out"] = {
            "ari_full_vs_RM": float(adjusted_rand_score(km.labels_, km_rm.labels_)),
            "note": "high ARI => Frequency barely changes assignments",
        }

    # --- baseline + robustness cross-check (silhouette at same k) ---
    cross = {}
    for tag, name in [("baseline", cfg.baseline_transform),
                      ("robustness", cfg.robustness_transform)]:
        row = sweep[(sweep["transform"] == name) & (sweep["k"] == k)]
        if len(row):
            cross[name] = {"silhouette": float(row["silhouette"].iloc[0]), "role": tag}
    report["cross_transform_at_k"] = cross

    write_json(report, cfg.reports_path / EVAL_NAME)
    log.info("Stage 7 done. GMM verdict: %s | ARI vs RFM: %.3f | F contribution: %s",
             report["gmm_shape"]["verdict"], report["rfm_agreement"]["ari"],
             report["feature_contribution"].get(S.FREQUENCY))
    log.info("Report: %s", cfg.reports_path / EVAL_NAME)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7 - evaluation scorecard")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    if args.force:
        cfg.force = True
    run(cfg)


if __name__ == "__main__":
    main()
