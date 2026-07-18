"""Orchestrator for stages 1-4.

Runs ingest -> clean -> eda -> features in order, each step idempotent, and
writes a run manifest (config snapshot, seed, raw fingerprint, row counts) for
reproducibility.

CLI:
    python -m bank_rfm.pipeline --config config.yaml
    python -m bank_rfm.pipeline --config config.yaml --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from . import clean, eda, features, ingest
from .config import PipelineConfig
from .utils import get_logger, write_json

log = get_logger("bank_rfm.pipeline")

RUN_MANIFEST = "run_manifest.json"


def run(cfg: PipelineConfig) -> dict:
    started = dt.datetime.now()
    cfg.make_dirs()
    log.info("=== Stage 1: ingest ===")
    raw_csv = ingest.run(cfg)

    log.info("=== Stage 2: clean ===")
    clean_pq = clean.run(cfg, raw_csv)

    log.info("=== Stage 3: eda gate ===")
    eda_summary = eda.run(cfg, clean_pq)

    log.info("=== Stage 4: rfm features ===")
    rfm_pq = features.run(cfg, clean_pq)

    raw_manifest_path = cfg.raw_path / ingest.MANIFEST_NAME
    raw_manifest = json.loads(raw_manifest_path.read_text()) if raw_manifest_path.exists() else {}

    manifest = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "random_seed": cfg.random_seed,
        "config": cfg.to_dict(),
        "raw": raw_manifest,
        "outputs": {
            "clean": str(clean_pq),
            "rfm": str(rfm_pq),
        },
        "eda_gate": eda_summary.get("gate", {}),
        "n_customers": eda_summary.get("n_customers"),
    }
    write_json(manifest, cfg.reports_path / RUN_MANIFEST)
    log.info("=== Done. Run manifest: %s ===", cfg.reports_path / RUN_MANIFEST)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Banking RFM pipeline (stages 1-4)")
    ap.add_argument("--config", default="config.yaml", help="Path to config YAML")
    ap.add_argument("--force", action="store_true", help="Recompute all stages")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    if args.force:
        cfg.force = True
    run(cfg)


if __name__ == "__main__":
    main()
