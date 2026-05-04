"""Evaluation results writer — stores fold outputs to a structured
artifact directory.

Directory layout per run:
  artifacts/runs/{tier}_{hostname}_{YYYYMMDD_HHMMSS}/
    run_metadata.json          — written immediately on run start
    fold_a/
      calibration_summary.json — thresholds per primitive per scheme
    fold_b/
      gate_verdicts.jsonl      — one JSON line per (prompt, gate) pair
      fold_b_summary.json      — ABR, FPR, ASR aggregates
    fold_c/
      gate_verdicts.jsonl
      fold_c_summary.json
    fold_d/
      drift_results.json       — ADWIN latency and recovery metrics
    fold_e/
      bcn2_records.jsonl       — BCN-2 paired records
    hypothesis_results.json    — H1-H5 accept/reject (written last)
    manifest.json              — full reproducibility manifest

The run directory name encodes tier + device + time so that results
from different devices or different runs never overwrite each other.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path

from cliffguard.eval.runner import FoldResult
from cliffguard.types import Tier


def make_run_dir(
    artifacts_dir: Path,
    tier: Tier,
) -> Path:
    """Create and return the unique run directory:
      artifacts_dir/runs/{tier.value}_{hostname}_{YYYYMMDD_HHMMSS}/
    Uses UTC timestamp. Creates all parent directories.
    hostname is socket.gethostname() truncated to 20 chars,
    with spaces replaced by underscores.
    Returns the created Path."""
    hostname = socket.gethostname().replace(" ", "_")[:20]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dir_name = f"{tier.value}_{hostname}_{ts}"
    run_dir = artifacts_dir / "runs" / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_metadata(
    run_dir: Path,
    tier: Tier,
    schemes: list[str],
    git_hash: str,
    hardware_description: str = "unspecified",
    config_path: str | None = None,
) -> Path:
    """Write run_metadata.json to run_dir immediately on run start.
    Fields: tier, hostname, timestamp_utc, schemes, git_hash,
    hardware_description, config_path (or null).
    Returns the written path.
    This file is written FIRST — before any fold executes —
    so that partial runs are identifiable."""
    metadata: dict[str, object] = {
        "tier": tier.value,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schemes": schemes,
        "git_hash": git_hash,
        "hardware_description": hardware_description,
        "config_path": config_path,
    }
    out_path = run_dir / "run_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return out_path


def _serialize_fold_result(result: FoldResult) -> dict[str, object]:
    """Serialize a FoldResult to a dict, including computed properties."""
    return {
        "fold_name": result.fold_name,
        "tier": result.tier.value,
        "scheme": result.scheme.value,
        "n_prompts": result.n_prompts,
        "n_blocked": result.n_blocked,
        "n_passed": result.n_passed,
        "fpr": result.fpr,
        "asr": result.asr,
        "notes": result.notes,
        "tpr": result.tpr,
        "abr": result.abr,
    }


def write_fold_result(
    run_dir: Path,
    fold_name: str,
    results: list[FoldResult],
    extra: dict[str, object] | None = None,
) -> Path:
    """Write fold summary JSON to run_dir/{fold_name}/
      {fold_name}_summary.json
    Fields: fold_name, n_results, results (list of serialised
    FoldResult dicts), extra (any additional data).
    FoldResult serialisation: all fields including computed
    properties tpr and abr.
    Creates the fold subdirectory. Returns the written path."""
    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "fold_name": fold_name,
        "n_results": len(results),
        "results": [_serialize_fold_result(r) for r in results],
        "extra": extra,
    }
    out_path = fold_dir / f"{fold_name}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def write_gate_verdicts(
    run_dir: Path,
    fold_name: str,
    verdicts: list[dict[str, object]],
) -> Path:
    """Write gate verdicts as JSONL to
      run_dir/{fold_name}/gate_verdicts.jsonl
    One JSON line per verdict dict. Appends if file exists
    (allows incremental writes during a long fold).
    Returns the written path."""
    fold_dir = run_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    out_path = fold_dir / "gate_verdicts.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        for verdict in verdicts:
            f.write(json.dumps(verdict) + "\n")
    return out_path


def write_calibration_summary(
    run_dir: Path,
    thresholds_by_primitive_scheme: dict[str, dict[str, float]],
) -> Path:
    """Write Fold A calibration summary to
      run_dir/fold_a/calibration_summary.json
    Fields: thresholds_by_primitive_scheme
      {primitive_name: {scheme_name: tau_q}}.
    Returns the written path."""
    fold_a_dir = run_dir / "fold_a"
    fold_a_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "thresholds_by_primitive_scheme": thresholds_by_primitive_scheme,
    }
    out_path = fold_a_dir / "calibration_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def write_drift_results(
    run_dir: Path,
    drift_data: dict[str, object],
) -> Path:
    """Write Fold D drift simulation results to
      run_dir/fold_d/drift_results.json
    drift_data is the dict returned by run_drift_simulation().
    Returns the written path."""
    fold_d_dir = run_dir / "fold_d"
    fold_d_dir.mkdir(parents=True, exist_ok=True)
    out_path = fold_d_dir / "drift_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(drift_data, f, indent=2)
    return out_path


def write_hypothesis_results(
    run_dir: Path,
    results: dict[str, object],
) -> Path:
    """Write H1-H5 hypothesis test results to
      run_dir/hypothesis_results.json
    Written LAST — only call after all folds complete.
    results keys: h1_accepted, h1_summary, h2_ks_stat, h2_p,
    h3_ks_stat, h3_p, h4_stat, h4_p, h4_accepted, h5_tier_c_p,
    h5_tier_c_plus_p, h5_tier_c_accepted, h5_tier_c_plus_accepted.
    Adds written_at (UTC ISO timestamp) automatically.
    Returns the written path."""
    payload: dict[str, object] = dict(results)
    payload["written_at"] = datetime.now(timezone.utc).isoformat()
    out_path = run_dir / "hypothesis_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def write_manifest(
    run_dir: Path,
    manifest: dict[str, object],
) -> Path:
    """Write the reproducibility manifest to
      run_dir/manifest.json
    manifest is the dict from cliffguard.eval.repro.build_manifest().
    Returns the written path."""
    out_path = run_dir / "manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path


def list_runs(
    artifacts_dir: Path,
) -> list[Path]:
    """Return sorted list of run directories under artifacts_dir/runs/.
    Sorted by directory name (chronological because name starts
    with timestamp). Returns empty list if no runs exist."""
    runs_dir = artifacts_dir / "runs"
    if not runs_dir.exists():
        return []
    return sorted(
        [p for p in runs_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )


def load_run_metadata(run_dir: Path) -> dict[str, object]:
    """Load and return run_metadata.json from run_dir.
    Raises FileNotFoundError if not present."""
    path = run_dir / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"run_metadata.json not found in {run_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def load_hypothesis_results(run_dir: Path) -> dict[str, object]:
    """Load and return hypothesis_results.json from run_dir.
    Raises FileNotFoundError if fold is incomplete."""
    path = run_dir / "hypothesis_results.json"
    if not path.exists():
        raise FileNotFoundError(f"hypothesis_results.json not found in {run_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]
