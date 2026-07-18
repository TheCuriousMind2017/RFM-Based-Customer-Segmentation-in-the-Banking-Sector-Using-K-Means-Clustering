"""Stage 4 - RFM feature engineering.

Aggregates the cleaned transaction table to one row per customer and writes the
RFM modeling table consumed by stage 5+. Snapshot date for Recency defaults to
max(TransactionDate) + 1 day (deterministic). Optional non-RFM features are
carried along for later experimentation but are not part of core R/F/M.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema as S
from .config import PipelineConfig
from .utils import get_logger, write_json

log = get_logger("bank_rfm.features")

RFM_NAME = "rfm_customer.parquet"
REPORT_NAME = "rfm_report.json"


def _dominant_hour(s: pd.Series):
    m = s.dropna()
    if m.empty:
        return pd.NA
    return int(m.mode().iloc[0])


def run(cfg: PipelineConfig, clean_parquet: Path) -> Path:
    out = cfg.processed_path / RFM_NAME
    if out.exists() and not cfg.force:
        log.info("RFM table already present, skipping (use force to redo).")
        return out

    df = pd.read_parquet(clean_parquet)

    if cfg.snapshot_date:
        snapshot = pd.Timestamp(cfg.snapshot_date)
    else:
        snapshot = df[S.TRANSACTION_DATE].max() + pd.Timedelta(days=1)

    g = df.groupby(S.CUSTOMER_ID)
    rfm = pd.DataFrame({
        S.RECENCY: (snapshot - g[S.TRANSACTION_DATE].max()).dt.days,
        S.FREQUENCY: g[S.TRANSACTION_ID].count(),
        S.MONETARY: g[S.TRANSACTION_AMOUNT].sum(),
        S.MONETARY_MEAN: g[S.TRANSACTION_AMOUNT].mean(),
    })

    if cfg.include_extra_features:
        rfm[S.TENURE] = (g[S.TRANSACTION_DATE].max() - g[S.TRANSACTION_DATE].min()).dt.days
        rfm[S.LAST_BALANCE] = g[S.CUST_ACCOUNT_BALANCE].last()
        rfm[S.DOMINANT_HOUR] = g[S.TRANSACTION_HOUR].apply(_dominant_hour)

    rfm = rfm.reset_index()

    # --- validation: core RFM must be well-formed ---
    problems = {
        "recency_negative": int((rfm[S.RECENCY] < 0).sum()),
        "frequency_lt_1": int((rfm[S.FREQUENCY] < 1).sum()),
        "monetary_nonpositive": int((rfm[S.MONETARY] <= 0).sum()),
        "any_null_core": int(rfm[[S.RECENCY, S.FREQUENCY, S.MONETARY]].isna().any(axis=1).sum()),
    }
    if any(problems.values()):
        log.warning("RFM validation found issues: %s", problems)

    cfg.processed_path.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(out, index=False)

    report = {
        "snapshot_date": snapshot,
        "n_customers": int(len(rfm)),
        "validation": problems,
        "summary": {
            col: {
                "mean": float(rfm[col].mean()),
                "median": float(rfm[col].median()),
                "min": float(rfm[col].min()),
                "max": float(rfm[col].max()),
            }
            for col in [S.RECENCY, S.FREQUENCY, S.MONETARY]
        },
    }
    write_json(report, cfg.reports_path / REPORT_NAME)
    log.info("RFM table written: %s customers | snapshot=%s.",
             len(rfm), snapshot.date())
    return out
