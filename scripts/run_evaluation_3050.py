"""Turnkey CLIFFGUARD evaluation for an RTX 3050 (4-8 GB VRAM).

What this does end-to-end:
  1. Verifies CUDA is available and prints GPU info.
  2. Checks that data/folds/fold_a/ exists; if missing, prompts the user
     to run scripts/download_fold_a.py --download.
  3. Picks a model size appropriate to detected VRAM:
       - >= 7 GB free  → Llama-3.2-3B-Instruct full precision + NF4 + Q3_K_M
       -  4-7 GB free  → Llama-3.2-1B-Instruct FP16 + NF4 + Q3_K_M
       -  < 4 GB free  → Qwen2.5-0.5B-Instruct (Tier C-style minimal run)
  4. Runs Fold A calibration via cliffguard.eval.five_fold_live.
  5. Runs Fold B cliff measurement if a Fold B corpus directory is present.
  6. Writes a one-page summary to artifacts/runs/<run_id>/summary_3050.md.

Usage:
  uv run python scripts/run_evaluation_3050.py --model auto
  uv run python scripts/run_evaluation_3050.py --model meta-llama/Llama-3.2-3B-Instruct --layer 14

Models downloaded automatically by transformers on first run; ensure you've
run `huggingface-cli login` if any model is gated (Llama family is gated on HF).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def detect_vram_gb() -> float:
    """Return free GPU VRAM in GB, or 0.0 if CUDA unavailable."""
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


def choose_model(vram_gb: float) -> tuple[str, int]:
    """Return (model_id, recommended_layer)."""
    if vram_gb >= 7.0:
        # 3B in NF4 ≈ 1.6 GB; FP16 ≈ 5.7 GB. Both fit.
        return ("meta-llama/Llama-3.2-3B-Instruct", 14)
    if vram_gb >= 4.0:
        # 1B FP16 ≈ 2.5 GB; NF4 ≈ 800 MB.
        return ("meta-llama/Llama-3.2-1B-Instruct", 8)
    # Last resort: 0.5B, runs anywhere.
    return ("Qwen/Qwen2.5-0.5B-Instruct", 6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Turnkey CLIFFGUARD evaluation for RTX 3050."
    )
    parser.add_argument("--model", default="auto",
                        help="HuggingFace model id, or 'auto' for VRAM-based pick.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Residual-stream layer to hook (default: auto-picked).")
    parser.add_argument("--fold-b-dir", type=Path, default=None,
                        help="Optional Fold B corpus dir (e.g. data/folds/fold_b/).")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip GPU / dataset preflight checks.")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------
    if not args.skip_checks:
        try:
            import torch
        except ImportError:
            print("torch not installed. Run: uv sync --extra gpu", file=sys.stderr)
            return 2
        if not torch.cuda.is_available():
            print("CUDA not available. CLIFFGUARD requires a CUDA GPU.", file=sys.stderr)
            return 2

        device_name = torch.cuda.get_device_name(0)
        free_gb = detect_vram_gb()
        print(f"[3050] GPU: {device_name}")
        print(f"[3050] Free VRAM: {free_gb:.2f} GB")

        fold_a_dir = Path("data/folds/fold_a")
        if not fold_a_dir.exists() or not any(fold_a_dir.glob("*.jsonl")):
            print(
                f"\nFold A corpus not found at {fold_a_dir}.\n"
                "Run this first:\n"
                "  uv run python scripts/download_fold_a.py --download\n",
                file=sys.stderr,
            )
            return 3

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    if args.model == "auto":
        free_gb = detect_vram_gb()
        model_id, default_layer = choose_model(free_gb)
        print(f"[3050] Auto-selected model: {model_id} (free VRAM {free_gb:.2f} GB)")
    else:
        model_id = args.model
        default_layer = 14
        print(f"[3050] Using model: {model_id}")

    layer = args.layer if args.layer is not None else default_layer
    print(f"[3050] Hooking layer {layer}")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    from cliffguard.eval.five_fold_orchestrator import (
        FiveFoldOrchestrator,
        OrchestratorConfig,
    )
    from cliffguard.eval.five_fold_live import live_run_all
    from cliffguard.types import QuantScheme, Tier

    # For a 3050 we measure FP16 vs NF4 vs (optional) Q3_K_M.
    # GGUF Q3_K_M requires llama.cpp; we leave that for a separate run.
    config = OrchestratorConfig(
        data_dir=Path("data/"),
        artifacts_dir=Path("artifacts/"),
        schemes=[QuantScheme.FP16, QuantScheme.NF4],
        tiers=[Tier.A],
    )
    orch = FiveFoldOrchestrator(config)

    try:
        summary = live_run_all(
            orch,
            model_id=model_id,
            layer=layer,
            fold_a_dir=Path("data/folds/fold_a"),
            fold_b_dir=args.fold_b_dir,
        )
    except RuntimeError as exc:
        print(f"[3050] ERROR: {exc}", file=sys.stderr)
        return 4

    print("\n[3050] === Summary ===")
    import json as _json
    print(_json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
