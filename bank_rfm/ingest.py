"""Stage 1 - Ingest.

Acquire the raw CSV (download from Kaggle or copy a provided local file),
write an immutable snapshot into data/raw/, and record a manifest with a
SHA-256 fingerprint so downstream runs are reproducible and auditable.

Kaggle download requires credentials: place kaggle.json at ~/.kaggle/kaggle.json
(chmod 600) or set KAGGLE_USERNAME / KAGGLE_KEY. To run fully offline, set
`local_csv` in the config to a CSV you have already downloaded.
"""
from __future__ import annotations

import datetime as dt
import shutil
import zipfile
from pathlib import Path

from .config import PipelineConfig
from .utils import get_logger, sha256_file, write_json

log = get_logger("bank_rfm.ingest")

RAW_SNAPSHOT_NAME = "bank_transactions.csv"
MANIFEST_NAME = "raw_manifest.json"


def _download_from_kaggle(slug: str, dest: Path) -> Path:
    """Download + unzip a Kaggle dataset into `dest`. Returns the CSV path."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            "The 'kaggle' package is required to download. Install it "
            "(`pip install kaggle`) and configure credentials, or set "
            "config.local_csv to use an already-downloaded CSV."
        ) from e

    api = KaggleApi()
    api.authenticate()
    log.info("Downloading Kaggle dataset '%s' ...", slug)
    api.dataset_download_files(slug, path=str(dest), unzip=False)

    zips = list(dest.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()

    csvs = sorted(dest.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found after download in {dest}")
    return csvs[0]


def run(cfg: PipelineConfig) -> Path:
    """Produce data/raw/bank_transactions.csv + raw_manifest.json.

    Idempotent: if the snapshot and a matching manifest already exist and
    cfg.force is False, the download/copy step is skipped.
    """
    cfg.make_dirs()
    snapshot = cfg.raw_path / RAW_SNAPSHOT_NAME
    manifest_path = cfg.raw_path / MANIFEST_NAME

    if snapshot.exists() and manifest_path.exists() and not cfg.force:
        log.info("Raw snapshot already present, skipping ingest (use force to redo).")
        return snapshot

    # Acquire source CSV -> temp location, then freeze into the snapshot path.
    if cfg.local_csv_path is not None:
        src = cfg.local_csv_path
        if not src.exists():
            raise FileNotFoundError(f"local_csv not found: {src}")
        log.info("Using local CSV: %s", src)
        source_desc = f"local:{src}"
    else:
        src = _download_from_kaggle(cfg.dataset_slug, cfg.raw_path)
        source_desc = f"kaggle:{cfg.dataset_slug}"

    if src.resolve() != snapshot.resolve():
        shutil.copyfile(src, snapshot)

    digest = sha256_file(snapshot)
    # Cheap row count without loading into pandas.
    with open(snapshot, "r", errors="replace") as f:
        n_lines = sum(1 for _ in f)
    n_rows = max(n_lines - 1, 0)  # minus header

    manifest = {
        "source": source_desc,
        "snapshot": str(snapshot),
        "sha256": digest,
        "n_rows": n_rows,
        "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(manifest, manifest_path)
    log.info("Raw snapshot frozen: %s rows | sha256=%s", n_rows, digest[:12])
    return snapshot
