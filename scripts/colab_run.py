"""CLI alternative to the Colab notebook — same workflow in one file.

Mirrors `scripts/run_evaluation_3050.py` and reuses the same live-mode
orchestrator. The difference from `run_evaluation_3050.py` is that this
script knows about Drive paths and per-scheme checkpointing through
`notebooks/colab_helper.py`.

Usage on a Colab terminal cell (or any GPU host with a `data/` directory):

    python scripts/colab_run.py --tier A --schemes FP16 NF4
    python scripts/colab_run.py --tier A --fold-a-only
    python scripts/colab_run.py --tier A --skip-download

Behaviour:
  1. (Colab only) mount Drive at /content/drive/MyDrive/cliffguard if not mounted.
  2. Symlink data/ → Drive datasets/ so corpora persist across sessions.
  3. Auto-pick model + schemes from free VRAM, or honour --model/--schemes.
  4. Download Fold A corpus (unless --skip-download).
  5. Run Fold A with per-scheme checkpointing.
  6. Optionally run Fold B if data/folds/fold_b/ exists.
  7. Sync artifacts back to Drive.
  8. Print a one-line summary to stdout.

The script is idempotent: re-running it after a kill resumes from the
last completed scheme without rerunning finished work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))


def _mount_drive_if_colab() -> bool:
    """Return True if we are on Colab and Drive mounted successfully."""
    try:
        from google.colab import drive as colab_drive  # type: ignore[import-not-found]
    except ImportError:
        return False
    if Path("/content/drive/MyDrive").exists():
        return True
    print("[colab_run] mounting Google Drive ...")
    colab_drive.mount("/content/drive")
    return Path("/content/drive/MyDrive").exists()


def _maybe_download_fold_a(max_prompts: int | None) -> None:
    fold_a_dir = Path("data/folds/fold_a")
    if fold_a_dir.exists() and any(fold_a_dir.glob("*.jsonl")):
        print(f"[colab_run] Fold A already present at {fold_a_dir}")
        return
    print("[colab_run] downloading Fold A ...")
    from scripts.download_fold_a import main as download_main  # type: ignore[import-not-found]

    argv = ["--download"]
    if max_prompts is not None:
        argv += ["--max", str(max_prompts)]
    rc = download_main(argv)
    if rc != 0:
        raise SystemExit(f"download_fold_a returned non-zero exit {rc}")


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    import colab_helper as ch  # noqa: E402 — sys.path adjusted above

    config = ch.choose_model()
    if args.model:
        config["model_id"] = args.model
    if args.schemes:
        config["schemes"] = list(args.schemes)
    return config  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CLIFFGUARD Colab CLI — Drive-checkpointed Fold A/B runner."
    )
    parser.add_argument("--tier", choices=["A", "B", "C", "C+"], default="A")
    parser.add_argument("--schemes", nargs="*", default=None,
                        help="Override scheme list (e.g. FP16 NF4). Default: auto from VRAM.")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model id. Default: auto from VRAM.")
    parser.add_argument("--fold-a-only", action="store_true",
                        help="Skip Fold B even if its corpus is present.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Fold A corpus download (use existing data/).")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Cap prompts per category in Fold A download.")
    parser.add_argument("--drive-root", type=Path,
                        default=Path("/content/drive/MyDrive/cliffguard"),
                        help="Persistent Drive root (Colab) or any local path.")
    args = parser.parse_args(argv)

    if args.tier != "A":
        print(f"[colab_run] --tier {args.tier} is reserved; only tier A is supported "
              "by the Colab live runner today.")
        return 2

    import colab_helper as ch  # noqa: E402 — sys.path adjusted above

    is_colab = _mount_drive_if_colab()
    if is_colab:
        ch.symlink_datasets_from_drive(drive_root=args.drive_root)

    if not args.skip_download:
        _maybe_download_fold_a(args.max_prompts)

    config = _resolve_config(args)
    print("[colab_run] config:")
    print(json.dumps(config, indent=2))

    run_dir = ch.run_fold_a_with_checkpoint(
        config=config,
        drive_root=args.drive_root,
    )
    summary: dict[str, Any] = {"fold_a_run_dir": str(run_dir)}

    fold_b_dir = Path("data/folds/fold_b")
    if not args.fold_a_only and fold_b_dir.exists() and any(fold_b_dir.glob("*.jsonl")):
        run_dir = ch.run_fold_b_with_checkpoint(
            config=config,
            drive_root=args.drive_root,
            fold_b_dir=fold_b_dir,
        )
        summary["fold_b_run_dir"] = str(run_dir)
    elif not args.fold_a_only:
        print("[colab_run] No Fold B corpus found; skipping. "
              "Call colab_helper.assemble_fold_b() to fetch it.")

    ch.sync_artifacts_to_drive(drive_root=args.drive_root)
    print("\n[colab_run] === Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
