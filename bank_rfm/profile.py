"""Stage 8 - Profiling and persistence.

Takes the selected cluster labels back to original RFM units (INR / days),
profiles each cluster, assigns a transparent data-driven persona name and a
suggested action, and persists the deliverables: a personas table, a
customer->segment assignment table, profile figures, and a single deployable
model bundle (fitted transformer + K-Means + persona map) that reproduces the
full inference path.

Persona naming is rule-based and auditable: clusters are ranked into Low/Mid/
High bands on Recency and Monetary (and a repeat-vs-one-time flag on Frequency),
then mapped to a name. The raw bands are written alongside the names so the
labelling can be checked or relabelled for the report.

Run after stage 7:
    python -m bank_rfm.profile --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import schema as S
from .cluster import fit_kmeans
from .config import PipelineConfig
from .evaluate import SELECTED_LABELS
from .utils import get_logger, write_json

log = get_logger("bank_rfm.profile")

PERSONAS_CSV = "cluster_personas.csv"
PROFILE_JSON = "profiling_report.json"
PROFILE_PLOT = "cluster_profiles.png"
SEGMENTS_PARQUET = "customer_segments.parquet"
BUNDLE_NAME = "final_model_bundle.joblib"
MANIFEST = "final_run_manifest.json"


def _bands(values: pd.Series, labels: list[str]) -> dict:
    """Rank cluster-level values into ordered bands (robust to ties)."""
    order = values.rank(method="first")
    edges = np.linspace(0, len(values), len(labels) + 1)
    out = {}
    for cl, rnk in order.items():
        idx = min(int(np.searchsorted(edges, rnk, side="left")) - 1, len(labels) - 1)
        out[cl] = labels[max(idx, 0)]
    return out


def assign_personas(profile: pd.DataFrame) -> dict:
    """Map each cluster to (persona, action, bands) from its R/F/M position.

    Recency: lower = more recent (better engagement).
    Monetary: higher = more valuable.
    Frequency: near-binary repeat flag on this data.
    """
    # Recency ascending -> first band is most recent.
    r_band = _bands(profile[S.RECENCY], ["recent", "mid", "dormant"])
    m_band = _bands(profile[S.MONETARY], ["low", "mid", "high"])
    f_repeat = (profile[S.FREQUENCY] > 1.5).to_dict()

    rules = {
        ("high", "recent"): ("High-Value Active", "Retain and upsell premium products"),
        ("high", "mid"): ("High-Value Core", "Deepen relationship, protect share of wallet"),
        ("high", "dormant"): ("High-Value At-Risk", "Priority win-back / relationship outreach"),
        ("mid", "recent"): ("Mainstream Active", "Cross-sell, grow share of wallet"),
        ("mid", "mid"): ("Mainstream Core", "Standard engagement and nurture"),
        ("mid", "dormant"): ("Mainstream Lapsing", "Re-engagement campaigns"),
        ("low", "recent"): ("Low-Value New/Active", "Onboard, encourage repeat use"),
        ("low", "mid"): ("Low-Value Core", "Low-cost automated engagement"),
        ("low", "dormant"): ("Low-Value Dormant", "Automated retention or sunset"),
    }
    out = {}
    for cl in profile.index:
        name, action = rules[(m_band[cl], r_band[cl])]
        out[cl] = {
            "persona": name,
            "action": action,
            "recency_band": r_band[cl],
            "monetary_band": m_band[cl],
            "frequency_profile": "repeat" if f_repeat[cl] else "mostly one-time",
        }
    return out


def _plot(profile: pd.DataFrame, sizes: pd.Series, cfg: PipelineConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        log.warning("matplotlib unavailable, skipping profile plot: %s", e)
        return
    feats = [S.RECENCY, S.FREQUENCY, S.MONETARY]
    Z = profile[feats].copy()
    Z = (Z - Z.mean()) / Z.std(ddof=0)  # z across clusters for comparability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    im = ax1.imshow(Z.values, cmap="coolwarm", aspect="auto", vmin=-1.8, vmax=1.8)
    ax1.set_xticks(range(len(feats))); ax1.set_xticklabels(feats)
    ax1.set_yticks(range(len(Z))); ax1.set_yticklabels([f"C{c}" for c in Z.index])
    ax1.set_title("Cluster profiles (z-scored across clusters)")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax1.text(j, i, f"{Z.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax1, shrink=0.8)
    ax2.bar([f"C{c}" for c in sizes.index], sizes.values)
    ax2.set_title("Cluster sizes"); ax2.set_ylabel("customers")
    fig.tight_layout()
    fig.savefig(cfg.reports_path / PROFILE_PLOT, dpi=110)
    plt.close(fig)


def run(cfg: PipelineConfig) -> dict:
    labels_fp = cfg.clusters_path / SELECTED_LABELS
    if not labels_fp.exists():
        raise FileNotFoundError(f"{labels_fp} not found. Run stage 7 first.")
    labels = pd.read_parquet(labels_fp)
    rfm = pd.read_parquet(cfg.processed_path / "rfm_customer.parquet")
    df = rfm.merge(labels, on=S.CUSTOMER_ID, how="inner")

    core = [S.RECENCY, S.FREQUENCY, S.MONETARY]
    extras = [c for c in [S.MONETARY_MEAN, S.TENURE, S.LAST_BALANCE, S.DOMINANT_HOUR]
              if cfg.profile_include_extras and c in df.columns]
    agg = {c: "median" for c in core + extras}
    profile = df.groupby("cluster").agg(agg)
    profile.columns = list(profile.columns)
    sizes = df.groupby("cluster").size().rename("size")
    profile["size"] = sizes
    profile["share"] = (sizes / len(df)).round(4)

    personas = assign_personas(profile)

    # --- personas table (original units + names) ---
    rows = []
    for cl in profile.index:
        p = personas[cl]
        rows.append({
            "cluster": int(cl), "persona": p["persona"], "action": p["action"],
            "size": int(profile.loc[cl, "size"]), "share": float(profile.loc[cl, "share"]),
            "median_recency_days": round(float(profile.loc[cl, S.RECENCY]), 1),
            "median_frequency": round(float(profile.loc[cl, S.FREQUENCY]), 2),
            "median_monetary_inr": round(float(profile.loc[cl, S.MONETARY]), 2),
            "recency_band": p["recency_band"], "monetary_band": p["monetary_band"],
            "frequency_profile": p["frequency_profile"],
        })
    personas_df = pd.DataFrame(rows).sort_values("median_monetary_inr", ascending=False)
    cfg.reports_path.mkdir(parents=True, exist_ok=True)
    personas_df.to_csv(cfg.reports_path / PERSONAS_CSV, index=False)

    # --- customer -> segment assignment ---
    persona_map = {int(cl): personas[cl]["persona"] for cl in profile.index}
    seg = df[[S.CUSTOMER_ID, "cluster"]].copy()
    seg["persona"] = seg["cluster"].map(persona_map)
    cfg.clusters_path.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(cfg.clusters_path / SEGMENTS_PARQUET, index=False)

    _plot(profile, sizes, cfg)

    # --- deployable model bundle (transformer + kmeans + persona map) ---
    tr_fp = cfg.models_path / "transformers" / f"{cfg.primary_transform}.joblib"
    bundle_meta = {}
    if tr_fp.exists():
        tr = joblib.load(tr_fp)
        feats = [c for c in cfg.feature_columns]
        Xt = pd.read_parquet(cfg.transformed_path / f"{cfg.primary_transform}.parquet")
        km = fit_kmeans(Xt[feats].to_numpy(float),
                        cfg.selected_k or int(profile.shape[0]), cfg)
        joblib.dump({"transformer": tr["estimator"], "kmeans": km, "features": feats,
                     "persona_map": persona_map, "transform_name": cfg.primary_transform,
                     "selected_k": cfg.selected_k, "random_seed": cfg.random_seed},
                    cfg.models_path / BUNDLE_NAME)
        bundle_meta = {"bundle": str(cfg.models_path / BUNDLE_NAME)}
    else:
        log.warning("Transformer %s not found; skipping model bundle.", tr_fp)

    report = {
        "operating_point": {"transform": cfg.primary_transform, "k": cfg.selected_k},
        "n_customers": int(len(df)),
        "personas": rows,
        "profile_medians": profile.round(3).reset_index().to_dict("records"),
    }
    write_json(report, cfg.reports_path / PROFILE_JSON)

    manifest = {
        "operating_point": report["operating_point"],
        "random_seed": cfg.random_seed,
        "outputs": {
            "personas_csv": str(cfg.reports_path / PERSONAS_CSV),
            "customer_segments": str(cfg.clusters_path / SEGMENTS_PARQUET),
            "profiling_report": str(cfg.reports_path / PROFILE_JSON),
            **bundle_meta,
        },
    }
    write_json(manifest, cfg.reports_path / MANIFEST)

    log.info("Stage 8 done. Personas:")
    for r in rows:
        log.info("  C%d %-22s n=%d  R=%.0fd F=%.2f M=Rs%.0f",
                 r["cluster"], r["persona"], r["size"],
                 r["median_recency_days"], r["median_frequency"], r["median_monetary_inr"])
    log.info("Deliverables in %s and %s", cfg.reports_path, cfg.clusters_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8 - profiling / personas")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    if args.force:
        cfg.force = True
    run(cfg)


if __name__ == "__main__":
    main()
