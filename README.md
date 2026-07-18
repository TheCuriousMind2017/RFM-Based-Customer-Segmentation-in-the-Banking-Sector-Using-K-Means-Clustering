# RFM-Based Customer Segmentation in the Banking Sector Using K-Means Clustering

Reproducible, config-driven pipeline that turns the Kaggle *Bank Customer
Segmentation* dataset (1,048,567 transactions from an Indian retail bank,
883,660 customers) into validated customer segments. All eight stages are
implemented and run end to end from a single command.

This repository accompanies the research paper of the same title
(Advanced Research Methodology, University of Europe for Applied Sciences).

```
Stage 1  ingest    -> data/raw/bank_transactions.csv  + raw_manifest.json (sha256)
Stage 2  clean     -> data/interim/clean.parquet       + reports/clean_report.json
Stage 3  eda gate  -> reports/eda_summary.json (+ eda_distributions.png)
Stage 4  features  -> data/processed/rfm_customer.parquet + reports/rfm_report.json
Stage 5  transform -> data/processed/transformed/<name>.parquet + models/transformers/<name>.joblib
                     + reports/transform_diagnostics.json (+ transform_comparison.png)
Stage 6  cluster   -> reports/cluster_sweep.csv (+ cluster_sweep.png) - internal metrics per (transform x K)
Stage 7  evaluate  -> reports/evaluation_report.json + data/processed/clusters/selected_labels.parquet
Stage 8  profile   -> reports/cluster_personas.csv + data/processed/clusters/customer_segments.parquet
                     + models/final_model_bundle.joblib (+ cluster_profiles.png)
```

## Headline findings

**Normalization is not a formality.** The common log-and-standardize step
damaged the Recency feature, which was already close to symmetric. Taking its
logarithm produced a heavy left tail (skewness near -3.7) and a detached group
of points, the structure that yields stretched, wall-like clusters. A
per-feature Yeo-Johnson power transform avoided this and cut mean absolute
skewness from about 15 in the raw space to 0.64.

**Five clusters, agreed by four measures.** The inertia elbow, the
Calinski-Harabasz index, the Davies-Bouldin index, and the gap statistic all
pointed to K = 5. (The Silhouette coefficient peaked at K = 2, but a
two-cluster split only separates active from inactive customers and is too
coarse to be a segmentation, so it was set aside deliberately.) The resulting
segments are balanced, each holding 16 to 22 percent of customers.

**Frequency collapses on single-snapshot data.** About 84 percent of customers
transacted exactly once in the three-month window, so Frequency was nearly
constant. Removing it entirely and clustering on Recency and Monetary alone
reproduced the full three-feature result at an Adjusted Rand Index of 0.52.
On this kind of data the three-factor RFM model behaves much like a two-factor
recency-and-value model, with Frequency acting only as a repeat-customer flag.
This is a property of the data rather than a flaw in the method, and it is
reported rather than hidden.

## Repository layout

```
bank_rfm/     the eight-stage pipeline package
tests/        synthetic-data generator and smoke test
reports/      figures and JSON summaries produced by the pipeline
proposal/     earlier proposal-stage scripts (see note below)
config.yaml   single source of configuration for every stage
```

### About `proposal/`

These scripts are the proposal-stage implementation, kept deliberately rather
than deleted. They use the log-and-standardize normalization that the final
paper examines and argues against, so they are the concrete "before" case for
the normalization finding above. They are not part of the final pipeline and
are not maintained.

### A note on data and references

The raw dataset is not committed. Stage 1 downloads it from Kaggle
(`shivamb/bank-customer-segmentation`), or you can supply a local copy; see
the offline instructions below. The reference papers cited in the write-up are
likewise not redistributed here, as they are published under copyright; they
are cited with DOIs in the paper itself.

## Install

```bash
pip install -r requirements.txt
```

## Run

Configure Kaggle credentials (`~/.kaggle/kaggle.json`, chmod 600) and run.

**One command (reproduce everything, stages 1-8):**

```bash
python -m bank_rfm.run_all --config config.yaml        # full pipeline
python -m bank_rfm.run_all --config config.yaml --force # recompute all
python -m bank_rfm.run_all --config config.yaml --from 5 # resume from a phase
```

With `primary_transform` and `selected_k` locked in config, this reproduces
the whole analysis and writes `reports/pipeline_manifest.json` (per-phase timing
+ headline results). `--from` accepts 1, 5, 6, 7, or 8.

**Or stage by stage (for the checkpoints):**

```bash
python -m bank_rfm.pipeline --config config.yaml      # stages 1-4
python -m bank_rfm.transform --config config.yaml     # stage 5 (transform sweep)
python -m bank_rfm.cluster --config config.yaml       # stage 6 (clustering sweep)
python -m bank_rfm.evaluate --config config.yaml      # stage 7 (deep evaluation at chosen K)
python -m bank_rfm.profile --config config.yaml       # stage 8 (personas + model bundle)
```

To run fully offline, download `bank_transactions.csv` yourself and set
`local_csv` in `config.yaml` — stage 1 then copies that file into an immutable
snapshot instead of downloading. `--force` recomputes all stages.

## Design notes

- **One config object** (`config.yaml` → `PipelineConfig`) drives every stage;
  unknown keys are rejected so typos fail loudly. A `run_manifest.json` records
  the config, seed, and raw-data sha256 for each run.
- **Every stage is idempotent** — existing outputs are reused unless `--force`.
- **Stage 2** repairs the dataset's known DOB anomalies (sentinel years like
  1800 and 2-digit-year century rollovers), decodes `TransactionTime` (HHMMSS)
  to an hour, drops only rows missing RFM-core fields, and accounts for every
  dropped/imputed row in `clean_report.json`.
- **Stage 3 is a gate, not just a chapter.** It quantifies skew/outliers and —
  critically — the transactions-per-customer distribution. If most customers
  transact rarely (Frequency degeneracy), it raises a flag and a recommendation
  but does **not** auto-filter; that is a manual decision
  (`eda_auto_filter: false` by default).
- **Stage 4** aggregates to one row per customer (Recency / Frequency /
  Monetary, plus optional Tenure / LastAccountBalance / DominantHour) using a
  deterministic snapshot date (`max(TransactionDate) + 1 day`).
- **Stage 5** is the swappable normalization module. A registry of strategies
  (`log_standard`, `power_yeojohnson`, `power_boxcox`, `quantile_normal`,
  `quantile_uniform`, `robust`, `log_robust`, `minmax`, `rfm_score`) is fitted
  to the R/F/M matrix behind one interface. It saves each transformed space and
  its fitted transformer, and writes `transform_diagnostics.json` comparing
  per-feature skew/kurtosis and feature correlation (lower = rounder, more
  separable clusters). An optional `winsorize` pre-step caps outliers. Swapping
  the active transform later is a one-line config change.
- **Stage 6** sweeps K-Means across every (transform x K) cell and records the
  internal scorecard per cell: Inertia, sampled Silhouette, Calinski-Harabasz,
  Davies-Bouldin. CH/DB are unbounded and compare K *within* a transform only;
  Silhouette is the cross-comparable index. Output is a tidy results table and
  metric-vs-K plots used to choose K.
- **Stage 7** runs the deeper diagnostics at the chosen operating point: the
  gap statistic (principled K selection), a GMM+BIC shape diagnostic
  (spherical vs diag vs full covariance - the actual test of whether K-Means'
  spherical assumption holds), ARI/NMI against a canonical RFM-scoring
  segmentation (reported with the Frequency-score spread), per-feature
  contribution, and a leave-F-out ARI. It persists the selected cluster labels
  for stage 8. Set `selected_k` in config to fix K, or leave null to auto-pick
  by silhouette within the primary transform.
- **Stage 8** profiles the selected clusters in original RFM units (INR/days),
  assigns a transparent rule-based persona name and action to each (clusters
  ranked into Low/Mid/High bands on Recency and Monetary, with a repeat flag on
  Frequency), and persists the deliverables: `cluster_personas.csv`, a
  `customer_segments.parquet` assignment table, profile figures, and a single
  `final_model_bundle.joblib` (fitted transformer + K-Means + persona map) that
  reproduces the full raw-RFM -> persona inference path.

## Validate without the real data

```bash
python tests/make_synthetic.py --out data/raw_src/bank_transactions.csv
# set local_csv: "data/raw_src/bank_transactions.csv" in a config, then:
python -m bank_rfm.pipeline --config test_config.yaml --force
python tests/test_smoke.py
```

The synthetic generator mirrors the real schema and its quirks (heavy right
skew, ~1.3 transactions/customer, sentinel DOBs), so the Frequency-degeneracy
gate fires just as it does on the real dataset.
