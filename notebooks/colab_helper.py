"""Helper module for the CLIFFGUARD Colab notebook.

Public API:
  banner()                            — print repo + GPU + Drive status
  symlink_datasets_from_drive(...)    — link /content/drive/.../datasets → data/
  choose_model()                      — VRAM-based model + scheme picker
  run_fold_a_with_checkpoint(config)  — Fold A with per-scheme checkpointing
  run_fold_b_with_checkpoint(config)  — Fold B with per-scheme checkpointing
  sync_artifacts_to_drive(...)        — copy artifacts/ → Drive
  assemble_fold_b(out_dir)            — pull AdvBench + JBB into fold_b/

All checkpoint files use the same shape so a killed-then-restarted session
re-reads the checkpoint and skips completed (model, scheme) pairs.

Colab-specific imports (`google.colab`) are guarded — the module imports
cleanly outside Colab so it can be used from a regular GPU host or from
`scripts/colab_run.py`.
"""

from __future__ import annotations

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


# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------

def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
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


def banner(drive_root: Path = DEFAULT_DRIVE_ROOT) -> None:
    """Print a one-screen status banner for the notebook."""
    name, free_gb, total_gb = _gpu_info()
    print("=" * 72)
    print("CLIFFGUARD — Colab session")
    print("=" * 72)
    print(f"  Repo HEAD:    {_git_short_sha()}")
    print(f"  Host:         {socket.gethostname()}")
    print(f"  GPU:          {name}")
    print(f"  VRAM:         {free_gb:.2f} GB free / {total_gb:.2f} GB total")
    print(f"  Drive root:   {drive_root}  (exists: {drive_root.exists()})")
    print(f"  Time (UTC):   {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)


# --------------------------------------------------------------------------
# Drive symlinks
# --------------------------------------------------------------------------

def symlink_datasets_from_drive(
    drive_root: Path = DEFAULT_DRIVE_ROOT,
    local_data_dir: Path = DEFAULT_LOCAL_DATA,
) -> None:
    """Make `data/` shadow `drive_root/datasets/` so HuggingFace and the
    fold loaders write to Drive automatically.

    If `local_data_dir` already exists and is not a symlink, it is left
    alone (we don't want to delete user data). In that case the user must
    manually move existing files into Drive.
    """
    drive_datasets = drive_root / "datasets"
    drive_datasets.mkdir(parents=True, exist_ok=True)

    if local_data_dir.is_symlink():
        existing_target = local_data_dir.resolve()
        if existing_target == drive_datasets.resolve():
            print(f"[symlink_datasets] already linked: {local_data_dir} → {drive_datasets}")
            return
        local_data_dir.unlink()
    elif local_data_dir.exists():
        print(
            f"[symlink_datasets] {local_data_dir} exists and is not a symlink — "
            "leaving it alone. Move its contents to "
            f"{drive_datasets} manually if you want Drive persistence."
        )
        return

    local_data_dir.symlink_to(drive_datasets, target_is_directory=True)
    print(f"[symlink_datasets] linked {local_data_dir} → {drive_datasets}")


# --------------------------------------------------------------------------
# Model picker
# --------------------------------------------------------------------------

def choose_model() -> dict[str, Any]:
    """Return a dict describing the run configuration:
       {model_id, layer, schemes, est_runtime_min, vram_gb}.

    Selection logic (free VRAM in GB):
      >= 35  → Llama-3.1-8B-Instruct, layer 20, [FP16, NF4, AWQ_INT4]
      >= 14  → Llama-3.2-3B-Instruct, layer 14, [FP16, NF4]
      >=  8  → Llama-3.2-1B-Instruct, layer  8, [FP16, NF4]
      <   8  → Qwen2.5-0.5B-Instruct, layer  6, [FP16]   (CPU-degraded)

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
        if payload.get("model_id") == model_id and payload.get("pending_schemes"):
            return d
    return None


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

    Returns the run directory path.
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
        print("[fold_a] all schemes already complete — nothing to do.")
        return run_dir

    # Run one scheme at a time so a kill costs only one scheme.
    for scheme in pending:
        single_orch = FiveFoldOrchestrator(
            OrchestratorConfig(
                data_dir=Path("data/"),
                artifacts_dir=artifacts_dir,
                schemes=[scheme],
                tiers=[Tier.A],
            )
        )
        single_orch._run_dir_override = run_dir  # type: ignore[attr-defined]
        # `live_execute_fold_a` calls orch.make_run_dir(); we patch it to
        # reuse our checkpoint-aware directory.
        original_make_run_dir = single_orch.make_run_dir
        single_orch.make_run_dir = lambda: run_dir  # type: ignore[assignment, return-value]

        print(f"[fold_a] running scheme {scheme.value} ...")
        try:
            live_execute_fold_a(
                orch=single_orch,
                model_id=model_id,
                layer=layer,
                fold_a_dir=fold_a_dir,
            )
        finally:
            single_orch.make_run_dir = original_make_run_dir  # type: ignore[assignment]

        completed.add(scheme.value)
        checkpoint["completed_schemes"] = sorted(completed)
        checkpoint["pending_schemes"] = [s.value for s in requested_schemes if s.value not in completed]
        _save_checkpoint(checkpoint_path, checkpoint)

        # Sync after every scheme — bound the loss window to one scheme.
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
    """Run Fold B reusing the Fold A directions stored in the latest
    matching run directory.

    Pre-conditions:
      - run_fold_a_with_checkpoint(config) has produced an
        artifacts/runs/<run_id>/fold_a/ directory.
      - data/folds/fold_b/*.jsonl exists (use assemble_fold_b()).
    """
    from cliffguard.eval.five_fold_live import live_execute_fold_b, live_execute_fold_a
    from cliffguard.eval.five_fold_orchestrator import (
        FiveFoldOrchestrator,
        OrchestratorConfig,
    )
    from cliffguard.types import QuantScheme, Tier

    model_id = config["model_id"]
    layer = config["layer"]
    schemes = [QuantScheme(s) for s in config["schemes"]]

    run_dir = _find_existing_run_dir(artifacts_dir, "a", model_id)
    if run_dir is None:
        raise RuntimeError(
            "Fold B needs Fold A first; run run_fold_a_with_checkpoint() and rerun."
        )

    if not fold_b_dir.exists() or not any(fold_b_dir.glob("*.jsonl")):
        raise RuntimeError(
            f"Fold B corpus not found at {fold_b_dir}; call assemble_fold_b() first."
        )

    orch = FiveFoldOrchestrator(
        OrchestratorConfig(
            data_dir=Path("data/"),
            artifacts_dir=artifacts_dir,
            schemes=schemes,
            tiers=[Tier.A],
        )
    )
    orch.make_run_dir = lambda: run_dir  # type: ignore[assignment]

    # Reuse the Fold A artifacts on disk by replaying calibration.
    print("[fold_b] reloading Fold A directions ...")
    fa = live_execute_fold_a(
        orch=orch, model_id=model_id, layer=layer,
        fold_a_dir=Path("data/folds/fold_a"),
    )
    orch.fold_a_results = fa

    fold_b_checkpoint = run_dir / "fold_b" / "checkpoint.json"
    fold_b_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing_cp = _load_checkpoint(fold_b_checkpoint)
    if existing_cp.get("completed_schemes") == [s.value for s in schemes]:
        print("[fold_b] already complete.")
        return run_dir

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
        print(f"[sync] {artifacts_dir} does not exist — nothing to sync.")
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
    print(f"[sync] copied {n_copied} new/changed files → {drive_results}")


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
