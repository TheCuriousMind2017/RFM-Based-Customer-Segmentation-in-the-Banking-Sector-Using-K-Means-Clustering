"""Generate a synthetic CSV that mirrors the Kaggle bank-transactions schema
and its known quirks, so stages 2-4 can be validated without network access.

Quirks reproduced:
  * column names verbatim, incl. "TransactionAmount (INR)"
  * TransactionTime as integer HHMMSS
  * heavy right-skew (log-normal) amounts and balances
  * ~1.3 transactions/customer (Frequency near-degeneracy)
  * sentinel DOBs (1/1/1800) and 2-digit-year future rollovers
  * scattered missing values, a few non-positive amounts, a duplicate id
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def make(n_customers: int = 20000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Most customers transact once; a tail transacts more -> mean ~1.3.
    n_txn = rng.choice([1, 2, 3, 5, 12], size=n_customers, p=[0.78, 0.12, 0.06, 0.03, 0.01])
    cust_ids = np.array([f"C{100000+i}" for i in range(n_customers)])

    rows = []
    base = pd.Timestamp("2016-08-01")
    for cid, k in zip(cust_ids, n_txn):
        for _ in range(int(k)):
            day = base + pd.Timedelta(days=int(rng.integers(0, 92)))
            amt = float(np.expm1(rng.normal(6.0, 1.3)))           # log-normal, skewed
            bal = float(np.expm1(rng.normal(9.0, 1.6)))
            hhmmss = int(rng.integers(0, 24) * 10000 + rng.integers(0, 60) * 100 + rng.integers(0, 60))
            rows.append({
                "TransactionID": f"T{len(rows):08d}",
                "CustomerID": cid,
                "CustomerDOB": _rand_dob(rng),
                "CustGender": rng.choice(["M", "F", np.nan], p=[0.7, 0.29, 0.01]),
                "CustLocation": rng.choice(["MUMBAI", "DELHI", "BANGALORE", np.nan], p=[0.4, 0.3, 0.29, 0.01]),
                "CustAccountBalance": bal,
                "TransactionDate": day.strftime("%d/%m/%y"),
                "TransactionTime": hhmmss,
                "TransactionAmount (INR)": round(amt, 2),
            })

    df = pd.DataFrame(rows)
    # Inject defects.
    idx = rng.choice(df.index, size=max(5, len(df) // 500), replace=False)
    df.loc[idx[: len(idx) // 3], "TransactionAmount (INR)"] = 0.0          # non-positive
    df.loc[idx[len(idx) // 3: 2 * len(idx) // 3], "CustAccountBalance"] = np.nan
    df.loc[idx[2 * len(idx) // 3:], "CustomerDOB"] = "1/1/1800"            # sentinel DOB
    if len(df) > 1:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)             # duplicate id
    return df


def _rand_dob(rng) -> str:
    roll = rng.random()
    if roll < 0.03:
        return "1/1/1800"                       # sentinel
    if roll < 0.06:
        return f"1/1/{rng.integers(63, 69):02d}"  # 2-digit -> future rollover
    y = int(rng.integers(1955, 1999))
    return f"{rng.integers(1,28)}/{rng.integers(1,12)}/{y}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/bank_transactions.csv")
    ap.add_argument("--customers", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = make(args.customers, args.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
