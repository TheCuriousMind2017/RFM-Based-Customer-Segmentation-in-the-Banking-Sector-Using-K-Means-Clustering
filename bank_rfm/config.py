"""Configuration for the pipeline.

A single config object drives every stage so a run is fully reproducible from
one file. Stage 5+ (transform / cluster / evaluate) fields are intentionally
omitted here; they will extend this dataclass when those stages are built.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PipelineConfig:
    # --- Source ---
    dataset_slug: str = "shivamb/bank-customer-segmentation"
    # If set, stage 1 uses this CSV instead of downloading (reproducibility /
    # offline / testing). Absolute or relative to project_root.
    local_csv: Optional[str] = None

    # --- Paths (relative to project_root unless absolute) ---
    project_root: str = "."
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    reports_dir: str = "reports"

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Stage 2: cleaning policy ---
    # Birth years below this are treated as sentinels (e.g. 1800) -> DOB missing.
    min_birth_year: int = 1920
    # Drop transactions whose amount is <= 0 (non-purchase / reversal noise).
    drop_nonpositive_amount: bool = True
    # Impute missing categorical fields rather than dropping the row.
    impute_gender_mode: bool = True
    impute_location_token: str = "UNKNOWN"
    # Missing account balance -> 0.0 (balance is not an RFM-core field).
    fill_balance_zero: bool = True

    # --- Stage 3: EDA gate ---
    # A customer with fewer than this many transactions is "low-frequency".
    min_frequency: int = 2
    # If the share of low-frequency customers exceeds this, the gate flags a
    # frequency-degeneracy warning in the report.
    degeneracy_warn_threshold: float = 0.5
    # Default is to REPORT and FLAG only. Auto-filtering the customer base is a
    # deliberate manual decision (see project notes).
    eda_auto_filter: bool = False
    make_figures: bool = True

    # --- Stage 4: RFM feature engineering ---
    # Snapshot date for Recency. None -> max(TransactionDate) + 1 day (deterministic).
    snapshot_date: Optional[str] = None
    # Carry optional non-RFM features onto the customer table for later stages.
    include_extra_features: bool = True

    # --- Stage 5: transform / normalization (swappable) ---
    # Features clustered on. Title commits to full RFM, so all three by default.
    feature_columns: list = field(default_factory=lambda: ["Recency", "Frequency", "Monetary"])
    # Sweep set: every named transform is fitted and diagnosed for comparison.
    transformers: list = field(default_factory=lambda: [
        "log_standard", "power_yeojohnson", "quantile_normal",
        "log_robust", "minmax", "rfm_score",
    ])
    # Optional outlier pre-step applied before the transform (per feature clip).
    winsorize: bool = False
    winsorize_limits: tuple = (0.01, 0.99)
    transformed_dir: str = "data/processed/transformed"
    models_dir: str = "models"
    plot_sample: int = 20000  # points drawn in the comparison scatter

    # --- Stage 6: clustering sweep ---
    # Transformed spaces to cluster on (rfm_score is excluded - it is the ARI
    # reference in stage 7, not a clustering candidate).
    cluster_transformers: list = field(default_factory=lambda: [
        "power_yeojohnson", "log_standard", "quantile_normal",
    ])
    k_min: int = 2
    k_max: int = 10
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    silhouette_sample: int = 20000   # sampled silhouette for O(n^2) control
    clusters_dir: str = "data/processed/clusters"

    # --- Stage 7: evaluation scorecard ---
    primary_transform: str = "power_yeojohnson"
    baseline_transform: str = "log_standard"
    robustness_transform: str = "quantile_normal"
    rfm_reference: str = "rfm_score"   # canonical RFM scoring for ARI/NMI
    selected_k: int | None = None      # None -> auto-pick by silhouette (primary)
    gap_b: int = 10                    # reference sets for the gap statistic
    gap_subsample: int = 50000
    eval_subsample: int = 50000        # subsample for heavier diagnostics
    gmm_covariances: list = field(default_factory=lambda: ["spherical", "diag", "full"])

    # --- Stage 8: profiling / persistence ---
    profile_include_extras: bool = True   # add Tenure/Balance/Hour to profiles

    # --- Orchestration ---
    force: bool = False  # recompute even if stage outputs already exist

    # ---- helpers ----
    def root(self) -> Path:
        return Path(self.project_root).expanduser().resolve()

    def _resolve(self, p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else self.root() / path

    @property
    def raw_path(self) -> Path:
        return self._resolve(self.raw_dir)

    @property
    def interim_path(self) -> Path:
        return self._resolve(self.interim_dir)

    @property
    def processed_path(self) -> Path:
        return self._resolve(self.processed_dir)

    @property
    def reports_path(self) -> Path:
        return self._resolve(self.reports_dir)

    @property
    def transformed_path(self) -> Path:
        return self._resolve(self.transformed_dir)

    @property
    def models_path(self) -> Path:
        return self._resolve(self.models_dir)

    @property
    def clusters_path(self) -> Path:
        return self._resolve(self.clusters_dir)

    @property
    def local_csv_path(self) -> Optional[Path]:
        return self._resolve(self.local_csv) if self.local_csv else None

    def make_dirs(self) -> None:
        for p in (self.raw_path, self.interim_path, self.processed_path, self.reports_path):
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)
