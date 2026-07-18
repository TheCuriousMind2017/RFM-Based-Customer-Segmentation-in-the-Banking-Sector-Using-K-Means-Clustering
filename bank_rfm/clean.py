"""Stage 2 - Clean and validate.

Loads the raw snapshot, enforces the expected schema, parses dates/times,
repairs the known DOB anomalies (sentinel years + 2-digit-year century
rollover), applies an explicit missing-value policy, drops rows that cannot
yield RFM, and writes an interim parquet plus a cleaning report that accounts
for every dropped row by reason.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema as S
from .config import PipelineConfig
from .utils import get_logger, write_json

log = get_logger("bank_rfm.clean")

CLEAN_NAME = "clean.parquet"
REPORT_NAME = "clean_report.json"


def _to_datetime_fast(s: pd.Series, formats=("%d/%m/%y", "%d/%m/%Y", "%m/%d/%Y")) -> pd.Series:
    """Vectorized date parse: try explicit formats first (fast), then fall back
    to flexible inference only for the residual rows that still failed. Avoids
    the per-element dateutil path on the full 1M-row column.
    """
    raw = s.astype("string")
    out = pd.to_datetime(pd.Series(pd.NaT, index=s.index))
    remaining = raw.notna()
    for fmt in formats:
        if not remaining.any():
            break
        parsed = pd.to_datetime(raw[remaining], format=fmt, errors="coerce")
        ok = parsed.notna()
        out.loc[parsed.index[ok]] = parsed[ok]
        remaining.loc[parsed.index[ok]] = False
    if remaining.any():  # last resort for genuinely mixed leftovers
        out.loc[remaining] = pd.to_datetime(raw[remaining], dayfirst=True, errors="coerce")
    return out


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in S.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw data is missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _fix_dob(raw: pd.Series, anchor: pd.Timestamp, min_year: int) -> pd.Series:
    """Parse DOB, roll back impossible future births, null out sentinels."""
    dob = _to_datetime_fast(raw)
    # 2-digit years can land in the future (e.g. '1/1/65' -> 2065): roll back 100y.
    future = dob.notna() & (dob > anchor)
    dob.loc[future] = dob.loc[future] - pd.DateOffset(years=100)
    # Sentinel / implausible births (e.g. 1800) -> missing.
    dob.loc[dob.notna() & (dob.dt.year < min_year)] = pd.NaT
    return dob


def run(cfg: PipelineConfig, raw_csv: Path) -> Path:
    out = cfg.interim_path / CLEAN_NAME
    report_path = cfg.reports_path / REPORT_NAME
    if out.exists() and not cfg.force:
        log.info("Clean output already present, skipping (use force to redo).")
        return out

    df = pd.read_csv(raw_csv, dtype=str)  # read as str; we coerce explicitly
    _validate_schema(df)
    n_raw = len(df)
    report: dict = {"n_raw": n_raw, "dropped": {}, "imputed": {}, "repaired": {}}

    # --- numeric coercion ---
    df[S.TRANSACTION_AMOUNT] = pd.to_numeric(df[S.TRANSACTION_AMOUNT], errors="coerce")
    df[S.CUST_ACCOUNT_BALANCE] = pd.to_numeric(df[S.CUST_ACCOUNT_BALANCE], errors="coerce")
    df[S.TRANSACTION_TIME] = pd.to_numeric(df[S.TRANSACTION_TIME], errors="coerce")

    # --- dates ---
    df[S.TRANSACTION_DATE] = _to_datetime_fast(df[S.TRANSACTION_DATE])
    # Temporal anchor (deterministic): latest valid transaction date.
    anchor = df[S.TRANSACTION_DATE].max()
    report["temporal_anchor"] = anchor

    dob_raw_missing = df[S.CUSTOMER_DOB].isna().sum()
    df[S.CUSTOMER_DOB] = _fix_dob(df[S.CUSTOMER_DOB], anchor, cfg.min_birth_year)
    df[S.DOB_VALID] = df[S.CUSTOMER_DOB].notna()
    report["repaired"]["dob_now_missing"] = int(df[S.CUSTOMER_DOB].isna().sum())
    report["repaired"]["dob_raw_missing"] = int(dob_raw_missing)

    # --- transaction hour from HHMMSS ---
    hour = (df[S.TRANSACTION_TIME] // 10000)
    df[S.TRANSACTION_HOUR] = hour.where((hour >= 0) & (hour <= 23))

    # --- derived age (optional carry; not an RFM core field) ---
    df[S.AGE] = ((anchor - df[S.CUSTOMER_DOB]).dt.days // 365)
    df.loc[(df[S.AGE] < 0) | (df[S.AGE] > 120), S.AGE] = pd.NA

    # --- missing-value report (pre-policy) ---
    report["missing_raw"] = {
        c: int(df[c].isna().sum())
        for c in [S.CUSTOMER_ID, S.CUST_GENDER, S.CUST_LOCATION,
                  S.CUST_ACCOUNT_BALANCE, S.TRANSACTION_DATE, S.TRANSACTION_AMOUNT]
    }

    # --- drop rows lacking RFM-core fields ---
    before = len(df)
    df = df.dropna(subset=S.RFM_REQUIRED)
    report["dropped"]["missing_rfm_core"] = before - len(df)

    # --- drop non-positive amounts (config) ---
    if cfg.drop_nonpositive_amount:
        before = len(df)
        df = df[df[S.TRANSACTION_AMOUNT] > 0]
        report["dropped"]["nonpositive_amount"] = before - len(df)

    # --- duplicate transaction IDs ---
    before = len(df)
    df = df.drop_duplicates(subset=[S.TRANSACTION_ID])
    report["dropped"]["duplicate_txn_id"] = before - len(df)

    # --- impute non-core categoricals (keep the row) ---
    if cfg.impute_gender_mode and df[S.CUST_GENDER].isna().any():
        mode = df[S.CUST_GENDER].mode(dropna=True)
        if len(mode):
            n = int(df[S.CUST_GENDER].isna().sum())
            df[S.CUST_GENDER] = df[S.CUST_GENDER].fillna(mode.iloc[0])
            report["imputed"]["gender_mode"] = n
    if df[S.CUST_LOCATION].isna().any():
        n = int(df[S.CUST_LOCATION].isna().sum())
        df[S.CUST_LOCATION] = df[S.CUST_LOCATION].fillna(cfg.impute_location_token)
        report["imputed"]["location_token"] = n
    if cfg.fill_balance_zero and df[S.CUST_ACCOUNT_BALANCE].isna().any():
        n = int(df[S.CUST_ACCOUNT_BALANCE].isna().sum())
        df[S.CUST_ACCOUNT_BALANCE] = df[S.CUST_ACCOUNT_BALANCE].fillna(0.0)
        report["imputed"]["balance_zero"] = n

    report["n_clean"] = len(df)
    report["retained_fraction"] = round(len(df) / n_raw, 4) if n_raw else 0.0

    cfg.interim_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_json(report, report_path)
    log.info("Cleaned %s -> %s rows (%.1f%% retained).",
             n_raw, len(df), 100 * report["retained_fraction"])
    return out
