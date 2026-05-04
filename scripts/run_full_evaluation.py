"""Full evaluation entry-point — see blueprint §12.

Parses command-line arguments and invokes FiveFoldOrchestrator for
the requested folds. Intended for a GPU host (Tier A or Tier B) with
the gpu extra installed.

Usage:
  uv run python scripts/run_full_evaluation.py \\
    --tier A \\
    --schemes FP16 NF4 GGUF_Q4_K_M GGUF_Q3_K_M \\
    --folds A B C D E \\
    --data-dir data/ \\
    --artifacts-dir artifacts/results/ \\
    --fpr-target 0.05

All folds raise NotImplementedError in Phase A (scaffolding mode).
The script exits with code 0 after printing the config summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cliffguard.eval.five_fold_orchestrator import FiveFoldOrchestrator, OrchestratorConfig
from cliffguard.types import QuantScheme, Tier

_VALID_TIERS: dict[str, Tier] = {t.value: t for t in Tier}
_VALID_FOLDS: list[str] = ["A", "B", "C", "D", "E"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the argument parser.

    Arguments:
      --config:        Optional JSON config file (pre-populates OrchestratorConfig).
      --tier:          One of A, B, C, C_PLUS (required).
      --schemes:       One or more QuantScheme names (required).
      --folds:         One or more of A, B, C, D, E (default: all five).
      --data-dir:      Path (default: data/).
      --artifacts-dir: Path (default: artifacts/results/).
      --fpr-target:    Float in (0, 1) (default: 0.05).
      --dry-run:       Print config and exit without running folds.
    """
    parser = argparse.ArgumentParser(
        prog="run_full_evaluation",
        description="CLIFFGUARD five-fold evaluation runner (blueprint §12).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional JSON config file to pre-populate OrchestratorConfig.",
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=list(_VALID_TIERS.keys()),
        help="Hardware deployment tier (A, B, C, C_PLUS).",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        required=True,
        metavar="SCHEME",
        help="One or more QuantScheme names (e.g. FP16 GGUF_Q3_K_M).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=_VALID_FOLDS,
        default=list(_VALID_FOLDS),
        metavar="FOLD",
        help="Folds to run: A B C D E (default: all five).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/"),
        metavar="PATH",
        help="Root directory for corpus data (default: data/).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/results/"),
        metavar="PATH",
        help="Directory for evaluation outputs (default: artifacts/results/).",
    )
    parser.add_argument(
        "--fpr-target",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Target false-positive rate (default: 0.05).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation config and exit without running folds.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> OrchestratorConfig:
    """Convert parsed args to an OrchestratorConfig."""
    tier = _VALID_TIERS[args.tier]
    schemes = [QuantScheme.from_string(s) for s in args.schemes]
    return OrchestratorConfig(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        schemes=schemes,
        tiers=[tier],
        fpr_target=args.fpr_target,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (always 0 in Phase A).

    If --dry-run: print config summary and return 0.
    Otherwise: attempt each requested fold, catch NotImplementedError
    (Phase A stub) or RuntimeError (prerequisite not met), print a
    clear message, and continue to the next fold. Always returns 0.
    """
    args = parse_args(argv)
    config = build_config(args)

    print("[CLIFFGUARD] Evaluation config:")
    print(f"  tier:          {args.tier}")
    print(f"  schemes:       {[s.name for s in config.schemes]}")
    print(f"  folds:         {args.folds}")
    print(f"  fpr_target:    {config.fpr_target}")
    print(f"  data_dir:      {config.data_dir}")
    print(f"  artifacts_dir: {config.artifacts_dir}")

    if args.dry_run:
        print("[CLIFFGUARD] --dry-run: exiting without running folds.")
        return 0

    orchestrator = FiveFoldOrchestrator(config)

    _fold_methods = {
        "A": orchestrator.execute_fold_a,
        "B": orchestrator.execute_fold_b,
        "C": orchestrator.execute_fold_c,
        "D": orchestrator.execute_fold_d,
        "E": orchestrator.execute_fold_e,
    }

    for fold in args.folds:
        print(f"[CLIFFGUARD] Running Fold {fold}...")
        try:
            _fold_methods[fold]()
            print(f"[CLIFFGUARD] Fold {fold}: complete.")
        except NotImplementedError as exc:
            print(
                f"[CLIFFGUARD] Fold {fold} skipped (Phase A): {exc}\n"
                "             Phase B wiring required for real evaluation."
            )
        except RuntimeError as exc:
            print(f"[CLIFFGUARD] Fold {fold} skipped (prerequisite): {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
