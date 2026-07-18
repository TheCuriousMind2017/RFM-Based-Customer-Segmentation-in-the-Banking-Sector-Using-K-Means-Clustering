"""Stage 3 - EDA gate.

Produces diagnostics that inform (and can block) the modeling plan:
  * distribution shape: skewness / kurtosis of Monetary-type fields
  * outlier quantification (IQR rule, beyond 1.5x and 3x)
  * missingness summary on the cleaned table
  * the key gate: transactions-per-customer distribution and the share of
    low-frequency customers (the Frequency-degeneracy check)

By default this REPORTS and FLAGS only. Auto-filtering the customer base is a
deliberate manual decision (cfg.eda_auto_filter). A machine-readable
eda_summary.json is written for auditing and for the paper's EDA section.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import schema as S
from .config import PipelineConfig
from .utils import get_logger, write_json

log = get_logger("bank_rfm.eda")

SUMMARY_NAME = "eda_summary.json"


def _dist_stats(x: pd.Series) -> dict:
    x = x.dropna().astype(float)
    if len(x) < 3:
        return {"n": int(len(x))}
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    out = {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "skew": float(stats.skew(x)),
        "kurtosis_excess": float(stats.kurtosis(x)),  # excess (normal -> 0)
    }
    if iqr > 0:
        out["outliers_1_5_iqr"] = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
        out["outliers_3_iqr"] = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())
        out["outlier_pct_1_5_iqr"] = round(100 * out["outliers_1_5_iqr"] / len(x), 2)
    return out


def _make_figures(df: pd.DataFrame, freq: pd.Series, reports_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        log.warning("matplotlib unavailable, skipping figures: %s", e)
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    amt = df[S.TRANSACTION_AMOUNT].dropna().astype(float)
    axes[0, 0].hist(amt, bins=80)
    axes[0, 0].set_title("Transaction amount (raw)")
    axes[0, 1].hist(np.log1p(amt), bins=80)
    axes[0, 1].set_title("Transaction amount (log1p)")
    axes[1, 0].boxplot(df[S.CUST_ACCOUNT_BALANCE].dropna().astype(float), vert=False)
    axes[1, 0].set_title("Account balance")
    axes[1, 1].hist(freq.clip(upper=freq.quantile(0.99)), bins=range(1, 12))
    axes[1, 1].set_title("Transactions per customer")
    fig.tight_layout()
    fig.savefig(reports_dir / "eda_distributions.png", dpi=110)
    plt.close(fig)


def run(cfg: PipelineConfig, clean_parquet: Path) -> dict:
    df = pd.read_parquet(clean_parquet)
    cfg.reports_path.mkdir(parents=True, exist_ok=True)

    freq = df.groupby(S.CUSTOMER_ID)[S.TRANSACTION_ID].count()
    n_customers = int(freq.shape[0])
    low = int((freq < cfg.min_frequency).sum())
    share_low = (low / n_customers) if n_customers else 0.0
    share_single = float((freq == 1).mean()) if n_customers else 0.0

    summary = {
        "n_transactions": int(len(df)),
        "n_customers": n_customers,
        "distribution": {
            S.TRANSACTION_AMOUNT: _dist_stats(df[S.TRANSACTION_AMOUNT]),
            S.CUST_ACCOUNT_BALANCE: _dist_stats(df[S.CUST_ACCOUNT_BALANCE]),
        },
        "frequency": {
            "mean_txn_per_customer": round(float(freq.mean()), 3),
            "median_txn_per_customer": float(freq.median()),
            "max_txn_per_customer": int(freq.max()),
            "share_single_transaction": round(share_single, 4),
            "share_below_min_frequency": round(share_low, 4),
            "min_frequency_threshold": cfg.min_frequency,
            "quantiles": {q: float(freq.quantile(q)) for q in (0.5, 0.9, 0.99)},
        },
        "missing_clean": {
            c: int(df[c].isna().sum())
            for c in [S.CUST_GENDER, S.CUST_LOCATION, S.CUSTOMER_DOB, S.AGE]
            if c in df.columns
        },
    }

    # --- the gate ---
    degeneracy = share_low > cfg.degeneracy_warn_threshold
    summary["gate"] = {
        "frequency_degeneracy_flag": bool(degeneracy),
        "auto_filter_enabled": bool(cfg.eda_auto_filter),
        "recommendation": (
            "Frequency is near-degenerate: most customers transact rarely, so "
            "the F dimension carries little separating signal. Decide before "
            "modeling whether to (a) keep low-frequency customers, (b) filter to "
            ">= min_frequency, or (c) substitute/augment F with another feature."
            if degeneracy else
            "Frequency spread is adequate for RFM clustering."
        ),
    }
    if degeneracy:
        log.warning("FREQUENCY-DEGENERACY GATE: %.1f%% of customers have < %d "
                    "transactions (single-txn share %.1f%%).",
                    100 * share_low, cfg.min_frequency, 100 * share_single)

    if cfg.make_figures:
        _make_figures(df, freq, cfg.reports_path)

    write_json(summary, cfg.reports_path / SUMMARY_NAME)
    log.info("EDA gate written: %s customers, mean freq %.2f.",
             n_customers, summary["frequency"]["mean_txn_per_customer"])
    return summary
