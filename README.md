# Credit Card Fraud Detection — Shared Data Pipeline

Reproducible base pipeline for our ML module group project. It ingests the
Kaggle Sparkov dataset, runs deterministic base preprocessing, and produces a
frozen **60 / 20 / 20** train/validation/test split that every team member loads
to build their own models — no re-cleaning, no leakage, identical data for all
seven models.

This repo deliberately stops **before** feature engineering. Encoding, scaling,
and imputation are each member's job, fitted on the training split only.

## Pipeline at a glance

```
Kaggle (kartik2112/fraud-detection) (https://www.kaggle.com/datasets/kartik2112/fraud-detection)
        │  ingest.py        download (kagglehub / CLI) or use cached CSVs
        ▼
   clean_raw()              global dedup on trans_num  (before the split!)
        ▼
   stratified 60/20/20      split.py  (stratify on is_fraud; temporal mode optional)
        ▼
   derive_features()        timestamp→hour/dayofweek/month, age, haversine distance,
        │                   drop PII/identifiers, keep the feature contract
        ▼
data/processed/  train.parquet  val.parquet  test.parquet  metadata.json
        ▼
   data_access.load_splits()    ← every member imports this
```

## Quickstart

```bash
pip install -r requirements.txt

# one-time Kaggle auth (for downloading): place kaggle.json in ~/.kaggle/
#   or export KAGGLE_USERNAME=... KAGGLE_KEY=...
# token: kaggle.com → Settings → API → "Create New Token"

python build_dataset.py            # builds data/processed/*.parquet
# python build_dataset.py --mode temporal   # time-ordered split instead
```

Already have `fraudTrain.csv` / `fraudTest.csv`? Drop them in `data/raw/` and the
download step is skipped automatically.

## How each member builds their models

```python
from data_access import load_splits, feature_groups

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits(as_xy=True)
cols = feature_groups()
# cols == {'numeric': [...], 'categorical_low_card': [...],
#          'categorical_high_card': [...], 'target': 'is_fraud'}
```

Then apply **your branch** and train. See `examples/member_template.py` for a
runnable skeleton that fits transformers on train only.

| Owner | Models | Branch |
|---|---|---|
| Agathiyan | Logistic Regression, Linear SVM | A — one-hot + standardise |
| Kiran | Decision Tree, Random Forest | B — label-encode + native scales |
| Pattanasavich | Isolation Forest, XGBoost, FT-Transformer | B / C |

## Feature contract

**Target:** `is_fraud`

**Numeric:** `amt`, `city_pop`, `lat`, `long`, `merch_lat`, `merch_long`,
`hour`, `dayofweek`, `month`, `age`, `distance_km`

**Categorical (low-card, one-hot friendly):** `category`, `gender`, `state`

**Categorical (high-card — ~700 merchants, ~500 jobs):** `merchant`, `job`
→ use target/frequency encoding for linear models, label-encoding for trees,
embeddings for FT-Transformer. Do **not** blindly one-hot these.

**Dropped** (identifiers / PII / leakage / redundant): `cc_num`, `first`, `last`,
`street`, `city`, `zip`, `trans_num`, `dob`, `trans_date_trans_time`,
`unix_time`, `Unnamed: 0`. `cc_num` is dropped on purpose — with 1000 simulated
customers it would let a model memorise card holders instead of learning fraud.

## Ground rules (so results stay comparable)

1. **Never re-split or re-clean.** Load the frozen parquet files via `data_access`.
2. **Fit on train only.** Every imputer / encoder / scaler is fitted on `X_train`,
   then `.transform`-ed onto val and test.
3. **Tune on validation, report on test.** The test split is touched once, at the
   end, for the final comparison.
4. **Primary metric: PR-AUC** (average precision). Select your decision threshold
   on validation to minimise missed fraud, then report the test confusion matrix.
5. **One seed.** Change `SEED` in `config.py` only by team agreement.

## Layout

```
config.py            seed, split ratios, paths, feature contract
build_dataset.py     entry point: ingest → clean → split → derive → save
data_access.py       what members import (load_splits, feature_groups, metadata)
src/
  ingest.py          Kaggle download / local cache
  preprocess.py      clean_raw + derive_features (deterministic, no fitted stats)
  split.py           stratified / temporal 60-20-20
examples/
  member_template.py copyable starting point
```
