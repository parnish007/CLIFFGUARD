"""Helper module for the CLIFFGUARD Colab notebook.

Public API:
  banner()                            — print repo + GPU + Drive status
  ensure_repo_cwd()                   — assert/cd into /content/CLIFFGUARD
  symlink_datasets_from_drive(...)    — link /content/drive/.../datasets → data/
  choose_model()                      — VRAM-based model + scheme picker
  run_fold_a_with_checkpoint(config)  — Fold A with per-scheme checkpointing
  run_fold_b_with_checkpoint(config)  — Fold B with per-scheme checkpointing
  sync_artifacts_to_drive(...)        — copy artifacts/ → Drive
  assemble_fold_b(out_dir)            — pull AdvBench + JBB into fold_b/
  verify_fold_a_complete(config, ...) — guard before running Fold B
  fold_isolation_audit()              — check Fold A / B SHA-256 disjointness

All checkpoint files use the same shape so a killed-then-restarted session
re-reads the checkpoint and skips completed (model, scheme) pairs.

Colab-specific imports (`google.colab`) are guarded — the module imports
cleanly outside Colab so it can be used from a regular GPU host or from
`scripts/colab_run.py`.

Notes on the live-mode behavioural metric (`Δ_B-cliff`):
  Phase B uses a PROBE-RM margin-threshold proxy in
  `cliffguard/eval/five_fold_live.py:243-250`, NOT the StrongREJECT +
  Llama-Guard-3-8B judge stack required by the paper §11.3. The proxy
  is correctly typed but is a behavioural surrogate. A real-judge
  implementation is tracked in `development.md` as Phase C work.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/cliffguard")
DEFAULT_LOCAL_DATA = Path("data")
DEFAULT_LOCAL_ARTIFACTS = Path("artifacts")
DEFAULT_REPO_ROOT = Path("/content/CLIFFGUARD")


# --------------------------------------------------------------------------
# Working-directory discipline
# --------------------------------------------------------------------------

def ensure_repo_cwd(repo_root: Path = DEFAULT_REPO_ROOT) -> Path:
    """`chdir` into the repo root and assert the expected layout.

    Many cells call `!python scripts/...` or open `data/folds/...` with
    relative paths. If a previous `%cd` or runtime restart moved CWD,
    those calls silently fail. Call this at the start of every code cell
    that touches the filesystem to guarantee CWD == repo root.
    """
    if not repo_root.exists():
        raise RuntimeError(
            f"Repo not found at {repo_root}. Run the clone cell first."
        )
    os.chdir(repo_root)
    expected = ["cliffguard", "scripts", "notebooks", "docs", "tests"]
    missing = [d for d in expected if not (repo_root / d).is_dir()]
    if missing:
        raise RuntimeError(
            f"{repo_root} is not a CLIFFGUARD checkout (missing: {missing})."
        )
    return repo_root


# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------

def _git_short_sha(repo_root: Path = DEFAULT_REPO_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
            cwd=str(repo_root) if repo_root.exists() else None,
        )
        return out.stdout.strip() or "unknown"
    except FileNotFoundError:
        return "unknown"


def _gpu_info() -> tuple[str, float, float]:
    """Return (device_name, free_gb, total_gb). (CPU, 0, 0) when no CUDA."""
    try:
        import torch
    except ImportError:
        return ("CPU (torch not installed)", 0.0, 0.0)
    if not torch.cuda.is_available():
        return ("CPU", 0.0, 0.0)
    name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return (name, free_bytes / (1024**3), total_bytes / (1024**3))


def banner(
    drive_root: Path = DEFAULT_DRIVE_ROOT,
    repo_root: Path = DEFAULT_REPO_ROOT,
) -> None:
    """Print a one-screen status banner for the notebook."""
    name, free_gb, total_gb = _gpu_info()
    print("=" * 72)
    print("CLIFFGUARD - Colab session")
    print("=" * 72)
    print(f"  Repo HEAD:    {_git_short_sha(repo_root)}")
    print(f"  Repo root:    {repo_root}  (exists: {repo_root.exists()})")
    print(f"  CWD:          {Path.cwd()}")
    print(f"  Host:         {socket.gethostname()}")
    print(f"  GPU:          {name}")
    print(f"  VRAM:         {free_gb:.2f} GB free / {total_gb:.2f} GB total")
    print(f"  Drive root:   {drive_root}  (exists: {drive_root.exists()})")
    print(f"  Time (UTC):   {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)


# --------------------------------------------------------------------------
# GPU memory hygiene
# --------------------------------------------------------------------------

def torch_cleanup() -> None:
    """Aggressively free GPU memory between schemes.

    `del adapter` alone leaves cached allocator blocks on a T4 — enough
    to make the next scheme's load fail with an opaque "weight conversion"
    error or an OOM. Call this between every `(model, scheme)` pair.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Drive symlinks
# --------------------------------------------------------------------------

def symlink_datasets_from_drive(
    drive_root: Path = DEFAULT_DRIVE_ROOT,
    local_data_dir: Path = DEFAULT_LOCAL_DATA,
) -> None:
    """Make `data/` shadow `drive_root/datasets/` so HuggingFace and the
    fold loaders write to Drive automatically.

    Always resolves `local_data_dir` against the current CWD — call
    `ensure_repo_cwd()` first if you want it to land at
    `<repo>/data/` (the expected location).
    """
    drive_datasets = drive_root / "datasets"
    drive_datasets.mkdir(parents=True, exist_ok=True)
    # Ensure the fold subtree exists on Drive so download_fold_a.py finds it.
    (drive_datasets / "folds" / "fold_a").mkdir(parents=True, exist_ok=True)
    (drive_datasets / "folds" / "fold_b").mkdir(parents=True, exist_ok=True)

    if local_data_dir.is_symlink():
        existing_target = local_data_dir.resolve()
        if existing_target == drive_datasets.resolve():
            print(f"[symlink_datasets] already linked: {local_data_dir} -> {drive_datasets}")
            return
        local_data_dir.unlink()
    elif local_data_dir.exists():
        print(
            f"[symlink_datasets] {local_data_dir} exists and is not a symlink - "
            "leaving it alone. Move its contents to "
            f"{drive_datasets} manually if you want Drive persistence."
        )
        return

    local_data_dir.symlink_to(drive_datasets, target_is_directory=True)
    print(f"[symlink_datasets] linked {local_data_dir} -> {drive_datasets}")


# --------------------------------------------------------------------------
# Model picker
# --------------------------------------------------------------------------

def choose_model() -> dict[str, Any]:
    """Return a dict describing the run configuration:
       {model_id, layer, schemes, est_runtime_min, vram_gb}.

    Selection logic (free VRAM in GB):
      >= 35  -> Llama-3.1-8B-Instruct, layer 20, [FP16, NF4, AWQ_INT4]
      >= 14  -> Llama-3.2-3B-Instruct, layer 14, [FP16, NF4]
      >=  8  -> Llama-3.2-1B-Instruct, layer  8, [FP16, NF4]
      <   8  -> Qwen2.5-0.5B-Instruct, layer  6, [FP16]   (CPU-degraded)

    Memory rule of thumb: ~2 GB per 1B params in FP16; ~0.5 GB per 1B
    in NF4. These thresholds leave headroom for KV cache + activations.
    """
    _name, free_gb, _total_gb = _gpu_info()

    if free_gb >= 35.0:
        return {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "layer": 20,
            "schemes": ["FP16", "NF4", "AWQ_INT4"],
            "est_runtime_min": 30,
            "vram_gb": free_gb,
        }
    if free_gb >= 14.0:
        return {
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "layer": 14,
            "schemes": ["FP16", "NF4"],
            "est_runtime_min": 25,
            "vram_gb": free_gb,
        }
    if free_gb >= 8.0:
        return {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "layer": 8,
            "schemes": ["FP16", "NF4"],
            "est_runtime_min": 15,
            "vram_gb": free_gb,
        }
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "layer": 6,
        "schemes": ["FP16"],
        "est_runtime_min": 30,
        "vram_gb": free_gb,
    }


# --------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------

def _make_run_id(fold: str) -> str:
    host = socket.gethostname().replace(" ", "_")[:20]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{fold}_colab-{host}_{ts}"


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {}
    with checkpoint_path.open(encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def _save_checkpoint(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    payload["last_updated"] = datetime.now(timezone.utc).isoformat()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(checkpoint_path)


def _find_existing_run_dir(
    artifacts_dir: Path,
    fold_letter: str,
    model_id: str,
) -> Path | None:
    """Locate an in-progress run for (fold, model) by inspecting checkpoints.
    Returns the run dir, or None if no resumable run exists.
    """
    runs = artifacts_dir / "runs"
    if not runs.exists():
        return None
    for d in sorted(runs.iterdir()):
        cp = d / f"fold_{fold_letter.lower()}" / "checkpoint.json"
        if not cp.exists():
            continue
        try:
            payload = _load_checkpoint(cp)
        except json.JSONDecodeError:
            continue
        if payload.get("model_id") == model_id:
            return d
    return None


# --------------------------------------------------------------------------
# Fold A: load from disk (avoid rerunning calibration in Fold B)
# --------------------------------------------------------------------------

def _accumulate_calibration(fold_a_dir: Path) -> None:
    """Merge the just-written calibration_summary.json into calibration_all.json.

    live_execute_fold_a overwrites calibration_summary.json for every scheme,
    so per-scheme thresholds are lost. This function accumulates all thresholds
    into a persistent calibration_all.json so Fold B sees every scheme.
    Called in run_fold_a_with_checkpoint after each scheme completes.
    """
    summary_path = fold_a_dir / "calibration_summary.json"
    all_path = fold_a_dir / "calibration_all.json"

    if not summary_path.exists():
        return

    with summary_path.open(encoding="utf-8") as f:
        new_data = json.load(f)

    if all_path.exists():
        with all_path.open(encoding="utf-8") as f:
            merged = json.load(f)
    else:
        merged = {
            "primitive": new_data.get("primitive", "PROBE-RM"),
            "fpr_target": new_data.get("fpr_target"),
            "thresholds": {},
            "model_id": new_data.get("model_id"),
            "layer": new_data.get("layer"),
        }

    merged["thresholds"].update(new_data.get("thresholds", {}))

    tmp = all_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    tmp.replace(all_path)


def _load_fold_a_results_from_disk(run_dir: Path, model_id: str) -> Any:
    """Reconstruct a `FoldAResults` dataclass from saved `.npz` and
    `calibration_all.json` (preferred) or `calibration_summary.json` (fallback).
    Used by `run_fold_b_with_checkpoint` so Fold B does not re-run calibration.
    """
    import numpy as np  # local import: keep helper import-cheap

    from cliffguard.eval.five_fold_orchestrator import FoldAResults
    from cliffguard.eval.threshold_calibrator import CalibrationTable
    from cliffguard.types import QuantScheme

    family_key = model_id.split("/")[-1]
    fold_a_dir = run_dir / "fold_a"

    # calibration_all.json accumulates thresholds across all schemes.
    # calibration_summary.json is overwritten per-scheme — use only as fallback.
    all_path = fold_a_dir / "calibration_all.json"
    summary_path = fold_a_dir / "calibration_summary.json"
    cal_path = all_path if all_path.exists() else summary_path

    if not cal_path.exists():
        raise FileNotFoundError(
            f"calibration_all.json and calibration_summary.json both missing in "
            f"{fold_a_dir}. Run run_fold_a_with_checkpoint() first."
        )
    with cal_path.open(encoding="utf-8") as f:
        cal_json = json.load(f)

    refusal_directions: dict[str, Any] = {}
    for npz in fold_a_dir.glob(f"r_hat_{family_key}_*.npz"):
        scheme_str = npz.stem[len(f"r_hat_{family_key}_"):]
        arr = np.load(npz)
        # Saved by calibrate_refusal_direction as the "direction" key.
        if "direction" in arr.files:
            refusal_directions[f"{family_key}:{scheme_str}"] = arr["direction"]
        else:
            # Fallback: first array
            refusal_directions[f"{family_key}:{scheme_str}"] = arr[arr.files[0]]

    thresholds = {
        QuantScheme(s): float(v)
        for s, v in cal_json.get("thresholds", {}).items()
    }
    cal_table = CalibrationTable(
        primitive=cal_json.get("primitive", "PROBE-RM"),
        thresholds=thresholds,
    )

    return FoldAResults(
        refusal_directions=refusal_directions,
        calibration_tables={"PROBE-RM": cal_table},
        kenlm_paths={},
        fp16_behavior=[],
        fold_a_hashes=frozenset(),
    )


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def verify_fold_a_complete(
    config: dict[str, Any],
    artifacts_dir: Path = DEFAULT_LOCAL_ARTIFACTS,
) -> Path:
    """Raise `RuntimeError` if Fold A's checkpoint does not include every
    scheme in `config['schemes']`. Run BEFORE `run_fold_b_with_checkpoint`
    so a partially-completed Fold A cannot silently produce a degenerate
    Fold B result (e.g. `cliff(FP16, FP16) = 0`).

    Returns the validated Fold A run directory.
    """
    model_id = config["model_id"]
    requested = set(config["schemes"])

    run_dir = _find_existing_run_dir(artifacts_dir, "a", model_id)
    if run_dir is None:
        raise RuntimeError(
            "No Fold A run directory found. Call run_fold_a_with_checkpoint() first."
        )

    cp = _load_checkpoint(run_dir / "fold_a" / "checkpoint.json")
    completed = set(cp.get("completed_schemes", []))
    missing = requested - completed

    if missing:
        raise RuntimeError(
            f"Fold A incomplete: schemes {sorted(missing)} have not finished. "
            f"completed={sorted(completed)}, requested={sorted(requested)}.\n"
            "Re-run run_fold_a_with_checkpoint(config) to finish them, OR "
            "narrow config['schemes'] to the completed set.\n"
            "Cliff measurement on a single scheme is degenerate and will "
            "silently return zero cliff."
        )

    if len(requested) < 2:
        raise RuntimeError(
            "Fold B needs >= 2 schemes (cliff is scheme-vs-scheme). "
            f"config['schemes'] = {sorted(requested)} has only "
            f"{len(requested)} scheme."
        )
    print(f"[guard] Fold A complete for {sorted(completed & requested)} - OK to run Fold B")
    return run_dir


def fold_isolation_audit(
    fold_a_dir: Path = Path("data/folds/fold_a"),
    fold_b_dir: Path = Path("data/folds/fold_b"),
) -> None:
    """Run `cliffguard.eval.folds.fold_isolation_check` over the two
    assembled corpora and print the result. Raises `AssertionError` on
    overlap — Fold B prompts must not appear in Fold A.
    """
    from cliffguard.eval.folds import (
        Fold, _load_jsonl_fold, fold_isolation_check,
    )

    fa = _load_jsonl_fold(
        fold_a_dir,
        [p.name for p in fold_a_dir.glob("*.jsonl")],
        Fold.A, "benign",
    )
    fb = _load_jsonl_fold(
        fold_b_dir,
        [p.name for p in fold_b_dir.glob("*.jsonl")],
        Fold.B, "harmful_test",
    )
    sets = fold_isolation_check({Fold.A: fa, Fold.B: fb})
    print(f"[fold_isolation] Fold A: {len(sets[Fold.A])} unique prompts")
    print(f"[fold_isolation] Fold B: {len(sets[Fold.B])} unique prompts")
    print("[fold_isolation] OK - no overlap between Fold A and Fold B.")


# --------------------------------------------------------------------------
# Fold A driver with checkpointing
# --------------------------------------------------------------------------

def run_fold_a_with_checkpoint(
    config: dict[str, Any],
    artifacts_dir: Path = DEFAULT_LOCAL_ARTIFACTS,
    drive_root: Path = DEFAULT_DRIVE_ROOT,
    fold_a_dir: Path = Path("data/folds/fold_a"),
) -> Path:
    """Wrap `cliffguard.eval.five_fold_live.live_execute_fold_a`, but
    checkpoint after each scheme so the cell can be re-run safely.

    `config` is the dict returned by `choose_model()`.

    Between schemes, runs `torch_cleanup()` to free the VRAM the previous
    scheme's model occupied — without this, a T4 will reject the next
    NF4 load with a "weight conversion" error.
    """
    from cliffguard.eval.five_fold_live import live_execute_fold_a
    from cliffguard.eval.five_fold_orchestrator import (
        FiveFoldOrchestrator,
        OrchestratorConfig,
    )
    from cliffguard.types import QuantScheme, Tier

    model_id = config["model_id"]
    layer = config["layer"]
    requested_schemes = [QuantScheme(s) for s in config["schemes"]]

    existing = _find_existing_run_dir(artifacts_dir, "a", model_id)
    if existing is not None:
        run_dir = existing
        checkpoint_path = run_dir / "fold_a" / "checkpoint.json"
        checkpoint = _load_checkpoint(checkpoint_path)
        completed = set(checkpoint.get("completed_schemes", []))
        print(f"[fold_a] resuming run {run_dir.name}; completed: {sorted(completed)}")
    else:
        run_id = _make_run_id("A")
        run_dir = artifacts_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "fold_a").mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "fold_a" / "checkpoint.json"
        completed = set()
        checkpoint = {
            "run_id": run_id,
            "fold": "A",
            "model_id": model_id,
            "layer": layer,
            "completed_schemes": [],
            "pending_schemes": [s.value for s in requested_schemes],
        }
        _save_checkpoint(checkpoint_path, checkpoint)
        print(f"[fold_a] new run dir: {run_dir}")

    pending = [s for s in requested_schemes if s.value not in completed]
    if not pending:
        print("[fold_a] all schemes already complete - nothing to do.")
        return run_dir

    for scheme in pending:
        torch_cleanup()  # free residue from previous scheme BEFORE loading next
        single_orch = FiveFoldOrchestrator(
            OrchestratorConfig(
                data_dir=Path("data/"),
                artifacts_dir=artifacts_dir,
                schemes=[scheme],
                tiers=[Tier.A],
            )
        )
        single_orch.make_run_dir = lambda: run_dir  # type: ignore[assignment]

        print(f"[fold_a] running scheme {scheme.value} ...")
        try:
            live_execute_fold_a(
                orch=single_orch,
                model_id=model_id,
                layer=layer,
                fold_a_dir=fold_a_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[fold_a] FAILED scheme={scheme.value}: {type(exc).__name__}: {exc}")
            print(f"[fold_a] checkpoint NOT advanced; rerun the cell to retry.")
            torch_cleanup()
            raise

        completed.add(scheme.value)
        checkpoint["completed_schemes"] = sorted(completed)
        checkpoint["pending_schemes"] = [s.value for s in requested_schemes if s.value not in completed]
        _save_checkpoint(checkpoint_path, checkpoint)
        _accumulate_calibration(run_dir / "fold_a")
        torch_cleanup()

        try:
            sync_artifacts_to_drive(artifacts_dir=artifacts_dir, drive_root=drive_root)
        except Exception as exc:  # noqa: BLE001
            print(f"[fold_a] WARNING: Drive sync failed: {exc}")

    print(f"[fold_a] complete. Artifacts in {run_dir}")
    return run_dir


# --------------------------------------------------------------------------
# Fold B driver with checkpointing
# --------------------------------------------------------------------------

def run_fold_b_with_checkpoint(
    config: dict[str, Any],
    artifacts_dir: Path = DEFAULT_LOCAL_ARTIFACTS,
    drive_root: Path = DEFAULT_DRIVE_ROOT,
    fold_b_dir: Path = Path("data/folds/fold_b"),
) -> Path:
    """Run Fold B reusing the Fold A directions saved on disk.

    Fold A's raw data dir is intentionally not needed here — Fold B
    reads the saved `r_hat_*.npz` and `calibration_summary.json` from
    `artifacts/runs/<run_id>/fold_a/`, never re-loading the raw prompts.

    Pre-conditions (checked here):
      - `verify_fold_a_complete(config)` would pass (all schemes done).
      - `data/folds/fold_b/*.jsonl` exists.
      - `len(config['schemes']) >= 2`  (cliff is scheme-vs-scheme).
    """
    from cliffguard.eval.five_fold_live import live_execute_fold_b
    from cliffguard.eval.five_fold_orchestrator import (
        FiveFoldOrchestrator,
        OrchestratorConfig,
    )
    from cliffguard.types import QuantScheme, Tier

    model_id = config["model_id"]
    layer = config["layer"]
    schemes = [QuantScheme(s) for s in config["schemes"]]

    # GUARD 1: Fold A must be complete for every requested scheme.
    run_dir = verify_fold_a_complete(config, artifacts_dir=artifacts_dir)

    # GUARD 2: Fold B corpus must exist.
    if not fold_b_dir.exists() or not any(fold_b_dir.glob("*.jsonl")):
        raise RuntimeError(
            f"Fold B corpus not found at {fold_b_dir}; call assemble_fold_b() first."
        )

    # Skip if already done.
    fold_b_checkpoint = run_dir / "fold_b" / "checkpoint.json"
    existing_cp = _load_checkpoint(fold_b_checkpoint)
    if set(existing_cp.get("completed_schemes", [])) >= {s.value for s in schemes}:
        print(f"[fold_b] already complete for {sorted(existing_cp['completed_schemes'])}.")
        return run_dir
    fold_b_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # GUARD 3: load FoldAResults from DISK, do not re-calibrate.
    print(f"[fold_b] loading Fold A directions from {run_dir / 'fold_a'} ...")
    fa = _load_fold_a_results_from_disk(run_dir, model_id)

    orch = FiveFoldOrchestrator(
        OrchestratorConfig(
            data_dir=Path("data/"),
            artifacts_dir=artifacts_dir,
            schemes=schemes,
            tiers=[Tier.A],
        )
    )
    orch.make_run_dir = lambda: run_dir  # type: ignore[assignment]
    orch.fold_a_results = fa

    torch_cleanup()
    out = live_execute_fold_b(orch, model_id=model_id, layer=layer, fold_b_dir=fold_b_dir)
    with (run_dir / "fold_b" / "cliff_results.json").open("w") as f:
        json.dump(out, f, indent=2)
    _save_checkpoint(fold_b_checkpoint, {
        "run_id": run_dir.name,
        "fold": "B",
        "model_id": model_id,
        "completed_schemes": [s.value for s in schemes],
        "pending_schemes": [],
    })
    torch_cleanup()

    try:
        sync_artifacts_to_drive(artifacts_dir=artifacts_dir, drive_root=drive_root)
    except Exception as exc:  # noqa: BLE001
        print(f"[fold_b] WARNING: Drive sync failed: {exc}")

    print(f"[fold_b] complete. Cliff results in {run_dir / 'fold_b'}")
    return run_dir


# --------------------------------------------------------------------------
# Drive sync
# --------------------------------------------------------------------------

def sync_artifacts_to_drive(
    artifacts_dir: Path = DEFAULT_LOCAL_ARTIFACTS,
    drive_root: Path = DEFAULT_DRIVE_ROOT,
) -> None:
    """Copy `artifacts_dir/runs/` into `drive_root/results/` so a session
    disconnect does not lose finished work.

    Idempotent: skips files that exist with the same size at the
    destination. Uses `shutil.copy2` (preserves mtimes).
    """
    if not artifacts_dir.exists():
        print(f"[sync] {artifacts_dir} does not exist - nothing to sync.")
        return
    drive_results = drive_root / "results"
    drive_results.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    src_runs = artifacts_dir / "runs"
    if not src_runs.exists():
        return
    for src in src_runs.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(src_runs)
        dst = drive_results / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)
        n_copied += 1
    print(f"[sync] copied {n_copied} new/changed files -> {drive_results}")


# --------------------------------------------------------------------------
# Fold B corpus assembly
# --------------------------------------------------------------------------

def assemble_fold_b(
    out_dir: Path = Path("data/folds/fold_b"),
) -> None:
    """Pull `walledai/AdvBench` + `JailbreakBench/JBB-Behaviors` into
    `out_dir`, one JSONL file each. Idempotent: skips downloads if both
    files already exist.

    File shape matches `cliffguard.eval.folds._load_jsonl_fold`:
        {"prompt": "<text>", "source": "<dataset-name>"}

    NOTE: This is a subset of the paper §12.6 Fold B spec — JBB + AdvBench
    are present, HarmBench + ArtPrompt are not. This is acceptable for
    Phase B reproduction at the scale this notebook targets; a full
    paper-spec corpus needs additional sources.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    adv_path = out_dir / "advbench.jsonl"
    jbb_path = out_dir / "jailbreakbench.jsonl"

    if adv_path.exists() and jbb_path.exists():
        print(f"[assemble_fold_b] already present in {out_dir}")
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets library not installed. Run: pip install datasets"
        ) from exc

    if not adv_path.exists():
        print("[assemble_fold_b] downloading walledai/AdvBench ...")
        ad = load_dataset("walledai/AdvBench", split="train")
        with adv_path.open("w", encoding="utf-8") as f:
            for row in ad:
                prompt = row.get("prompt") or row.get("goal") or row.get("query")
                if not prompt:
                    continue
                f.write(json.dumps({"prompt": str(prompt), "source": "advbench"}) + "\n")
        print(f"[assemble_fold_b]   wrote {adv_path}")

    if not jbb_path.exists():
        print("[assemble_fold_b] downloading JailbreakBench/JBB-Behaviors ...")
        try:
            jb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        except Exception:
            jb = load_dataset("JailbreakBench/JBB-Behaviors", split="harmful")
        with jbb_path.open("w", encoding="utf-8") as f:
            for row in jb:
                prompt = row.get("Goal") or row.get("goal") or row.get("prompt")
                if not prompt:
                    continue
                f.write(json.dumps({"prompt": str(prompt), "source": "jbb"}) + "\n")
        print(f"[assemble_fold_b]   wrote {jbb_path}")


# --------------------------------------------------------------------------
# Smoke entry
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("colab_helper.py loaded. Smoke check:")
    banner()
    print()
    print("choose_model() ->", json.dumps(choose_model(), indent=2))
    sys.exit(0)
