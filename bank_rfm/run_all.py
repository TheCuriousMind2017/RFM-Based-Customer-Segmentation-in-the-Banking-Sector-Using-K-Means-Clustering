"""Unified runner - reproduce the whole pipeline (stages 1-8) in one command.

With the decisions locked in config (primary_transform, selected_k), this runs
ingest -> clean -> eda -> features -> transform -> cluster -> evaluate -> profile
end to end and writes a consolidated manifest with per-phase timing and the
headline results. This is the entry point the report's reproducibility section
references.

    python -m bank_rfm.run_all --config config.yaml            # full run
    python -m bank_rfm.run_all --config config.yaml --force     # recompute all
    python -m bank_rfm.run_all --config config.yaml --from 5    # resume at stage 5

Phases (numbered by their first stage):
    1  ingest+clean+eda+features   5  transform
    6  cluster                     7  evaluate     8  profile
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from . import cluster, evaluate, pipeline, profile, transform
from .config import PipelineConfig
from .utils import get_logger, write_json

log = get_logger("bank_rfm.run_all")

# phase id -> (label, callable taking cfg)
PHASES = [
    (1, "ingest+clean+eda+features", lambda cfg: pipeline.run(cfg)),
    (5, "transform", lambda cfg: transform.run(cfg)),
    (6, "cluster", lambda cfg: cluster.run(cfg)),
    (7, "evaluate", lambda cfg: evaluate.run(cfg)),
    (8, "profile", lambda cfg: profile.run(cfg)),
]

MANIFEST = "pipeline_manifest.json"


def _headline(cfg: PipelineConfig) -> dict:
    """Pull a few key results from stage reports if present."""
    out = {}
    rp = cfg.reports_path
    for fname, keys in [
        ("eda_summary.json", ["n_customers"]),
        ("evaluation_report.json", ["operating_point", "gmm_shape", "rfm_agreement"]),
    ]:
        fp = rp / fname
        if fp.exists():
            data = json.loads(fp.read_text())
            for k in keys:
                if k in data:
                    out[k] = data[k]
    personas_fp = rp / "cluster_personas.csv"
    if personas_fp.exists():
        out["personas_csv"] = str(personas_fp)
    return out


def run(cfg: PipelineConfig, start_phase: int = 1) -> dict:
    started = dt.datetime.now()
    timings = []
    for pid, label, fn in PHASES:
        if pid < start_phase:
            log.info("--- skipping phase %d (%s) [--from %d] ---", pid, label, start_phase)
            continue
        log.info("=========== PHASE %d: %s ===========", pid, label)
        t0 = time.time()
        fn(cfg)
        dt_s = round(time.time() - t0, 1)
        timings.append({"phase": pid, "label": label, "seconds": dt_s})
        log.info("--- phase %d done in %.1fs ---", pid, dt_s)

    manifest = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "total_seconds": round(sum(t["seconds"] for t in timings), 1),
        "random_seed": cfg.random_seed,
        "primary_transform": cfg.primary_transform,
        "selected_k": cfg.selected_k,
        "phase_timings": timings,
        "config": cfg.to_dict(),
        "headline_results": _headline(cfg),
    }
    write_json(manifest, cfg.reports_path / MANIFEST)
    log.info("=========== PIPELINE COMPLETE in %.1fs ===========", manifest["total_seconds"])
    log.info("Consolidated manifest: %s", cfg.reports_path / MANIFEST)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full RFM pipeline (stages 1-8)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="recompute all stages")
    ap.add_argument("--from", dest="from_phase", type=int, default=1,
                    choices=[1, 5, 6, 7, 8], help="resume from a phase id")
    args = ap.parse_args()
    cfg = PipelineConfig.from_yaml(args.config) if Path(args.config).exists() else PipelineConfig()
    if args.force:
        cfg.force = True
    run(cfg, start_phase=args.from_phase)


if __name__ == "__main__":
    main()
