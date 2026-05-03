"""Full evaluation entry-point — see blueprint §12.8.

Parses command-line arguments and invokes the EvaluationRunner for
the requested folds. This script is intended to be run on a GPU host
(Tier A or Tier B hardware) with the gpu extra installed. It does not
perform any inference on the development machine.

Usage:
  uv run python scripts/run_full_evaluation.py \\
    --tier A \\
    --schemes FP16 NF4 GGUF_Q4_K_M GGUF_Q3_K_M \\
    --folds A B C \\
    --data-dir data/ \\
    --output-dir artifacts/results/ \\
    --fpr-target 0.05

All folds raise NotImplementedError in Phase A (scaffolding mode).
The script exits with code 0 after printing the plan summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cliffguard.types import QuantScheme, Tier, ThreatModel
from cliffguard.eval.runner import EvaluationPlan, EvaluationRunner

_VALID_TIERS: dict[str, Tier] = {t.value: t for t in Tier}
_VALID_FOLDS: frozenset[str] = frozenset({"A", "B", "C"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the argument parser.
    Arguments:
      --tier: one of A, B, C, C_PLUS (required)
      --schemes: one or more QuantScheme names (required)
      --folds: one or more of A, B, C (default: A B C)
      --data-dir: Path (default: data/)
      --output-dir: Path (default: artifacts/results/)
      --fpr-target: float in (0, 1) (default: 0.05)
      --n-calibration: int (default: 2000)
      --n-attack: int (default: 500)
      --dry-run: flag, if set print plan and exit without running"""
    parser = argparse.ArgumentParser(
        prog="run_full_evaluation",
        description="CLIFFGUARD five-fold evaluation runner (blueprint §12.8).",
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
        choices=sorted(_VALID_FOLDS),
        default=["A", "B", "C"],
        metavar="FOLD",
        help="Folds to run: A, B, C (default: A B C).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/"),
        metavar="PATH",
        help="Root directory for corpus data (default: data/).",
    )
    parser.add_argument(
        "--output-dir",
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
        "--n-calibration",
        type=int,
        default=2000,
        metavar="INT",
        help="Calibration corpus size (default: 2000, blueprint §12.2 minimum).",
    )
    parser.add_argument(
        "--n-attack",
        type=int,
        default=500,
        metavar="INT",
        help="Attack prompts per adversary per scheme (default: 500).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation plan and exit without running folds.",
    )
    return parser.parse_args(argv)


def build_plan(args: argparse.Namespace) -> EvaluationPlan:
    """Convert parsed args to an EvaluationPlan.
    Converts tier string to Tier enum.
    Converts scheme strings to QuantScheme enum via from_string().
    Uses all nine ThreatModel adversaries (A1–A9) by default."""
    tier = _VALID_TIERS[args.tier]
    schemes = [QuantScheme.from_string(s) for s in args.schemes]
    adversaries = list(ThreatModel)
    return EvaluationPlan(
        schemes=schemes,
        tiers=[tier],
        adversaries=adversaries,
        fpr_target=args.fpr_target,
        n_calibration=args.n_calibration,
        n_attack=args.n_attack,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 = success).
    If --dry-run: print the plan summary and return 0.
    Otherwise: attempt each requested fold via runner.execute_fold_*,
    catch NotImplementedError (Phase A), print a clear message that
    Phase B wiring is required, and return 0 (not an error in Phase A).
    Print runner.summary() at the end."""
    args = parse_args(argv)
    plan = build_plan(args)

    print("[CLIFFGUARD] Evaluation plan:")
    print(f"  tier:          {args.tier}")
    print(f"  schemes:       {[s.name for s in plan.schemes]}")
    print(f"  folds:         {args.folds}")
    print(f"  fpr_target:    {plan.fpr_target}")
    print(f"  n_calibration: {plan.n_calibration}")
    print(f"  n_attack:      {plan.n_attack}")
    print(f"  data_dir:      {args.data_dir}")
    print(f"  output_dir:    {args.output_dir}")

    if args.dry_run:
        print("[CLIFFGUARD] --dry-run: exiting without running folds.")
        return 0

    runner = EvaluationRunner(plan)

    _fold_methods = {
        "A": runner.execute_fold_a,
        "B": runner.execute_fold_b,
        "C": runner.execute_fold_c,
    }

    for fold in args.folds:
        print(f"[CLIFFGUARD] Running Fold {fold}...")
        try:
            results = _fold_methods[fold]()
            runner.results.extend(results)
            print(f"[CLIFFGUARD] Fold {fold}: {len(results)} results.")
        except NotImplementedError as exc:
            print(
                f"[CLIFFGUARD] Fold {fold} skipped (Phase A): {exc}\n"
                f"             Phase B wiring required for real evaluation."
            )

    summary = runner.summary()
    print("[CLIFFGUARD] Summary:")
    for key, val in summary.items():
        print(f"  {key}: {val:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
