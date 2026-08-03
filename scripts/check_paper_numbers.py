"""Verify that numbers written in the manuscript prose match the measurements.

Tables and figures are generated, so they cannot drift. Inline prose is typed by
hand and can, which makes it the weakest link in the paper's provenance chain.
Rounding a bound the wrong way is enough to turn a true statement false: the
simultaneous upper bound is 4.6194%, so "below 4.6%" is an overstatement of
precision that happens to also be wrong.

This script recomputes each quoted quantity from docs/paper/review_stats.json and
asserts the manuscript contains the correctly rounded string. It is not a general
number extractor -- it checks a curated list of the load-bearing claims, and a
new claim in the prose needs a new entry here.

Exit status is non-zero if any check fails, so this belongs in CI beside the
tests.

Usage:
  python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, NamedTuple


class Check(NamedTuple):
    """One quoted quantity, its source, and the string the prose must contain."""

    label: str
    expected: Callable[[dict[str, Any]], str]
    description: str


def _max_simultaneous(stats: dict[str, Any]) -> str:
    value = max(r["upper95_simultaneous"]
                for block in stats["transitions"].values()
                for r in block["rows"])
    return f"{100 * value:.2f}"


def _kappa_pooled(stats: dict[str, Any]) -> str:
    return f"{stats['drift']['_pooled']['estimate']:.2f}"


def _kappa_ci(stats: dict[str, Any]) -> str:
    pooled = stats["drift"]["_pooled"]
    return f"[{pooled['ci_low']:.2f}, {pooled['ci_high']:.2f}]"


def _max_transition_rate(stats: dict[str, Any]) -> str:
    value = max(r["rate_itt"] for block in stats["transitions"].values()
                for r in block["rows"])
    return f"{100 * value:.1f}"


def _probe_band_low(stats: dict[str, Any]) -> str:
    """Lowest frozen-probe retention over 4.5--8.5 bits, across all models."""
    values = [r["retained_mean"] for block in stats["probe"].values()
              for r in block["rows"] if 4.5 <= r["bits"] <= 8.5]
    return f"{100 * min(values):.0f}"


def _gsm8k_relative(stats: dict[str, Any]) -> str:
    """Qwen2.5-3B's 4.5-bit accuracy as a percentage of its own baseline."""
    block = stats["gsm8k"]["Qwen2.5-3B"]
    row = next(r for r in block["rows"] if r["bits"] == 4.5)
    return f"{100 * row['accuracy'] / block['fp16_accuracy']:.0f}"


def _anchor_error(model: str) -> Callable[[dict[str, Any]], str]:
    def inner(stats: dict[str, Any]) -> str:
        return f"{stats['drift'][model]['anchor_error_pp']:.1f}"
    return inner


def _top_rung_gap(model: str) -> Callable[[dict[str, Any]], str]:
    def inner(stats: dict[str, Any]) -> str:
        return f"{stats['drift'][model]['top_rung_minus_fp16_pp']:.1f}"
    return inner


CHECKS: tuple[Check, ...] = (
    Check("simultaneous upper bound", _max_simultaneous,
          "max over all model x rung cells of the Bonferroni-adjusted "
          "one-sided exact upper bound"),
    Check("pooled kappa", _kappa_pooled, "pooled drift coefficient"),
    Check("pooled kappa CI", _kappa_ci, "cluster-bootstrap interval for kappa"),
    Check("max transition rate", _max_transition_rate,
          "largest refusal-to-compliance rate over all prompts"),
    Check("probe band floor", _probe_band_low,
          "lowest frozen-probe retention over 4.5--8.5 bits"),
    Check("GSM8K relative accuracy", _gsm8k_relative,
          "Qwen2.5-3B 4.5-bit accuracy as a fraction of its own baseline"),
    Check("Qwen anchor error", _anchor_error("Qwen2.5-3B"),
          "points by which the band fit undershoots FP16"),
    Check("Phi anchor error", _anchor_error("Phi-3.5-mini"),
          "points by which the band fit undershoots FP16"),
    Check("Qwen top-rung gap", _top_rung_gap("Qwen2.5-3B"),
          "8.5-bit refusal rate minus FP16"),
    Check("Phi top-rung gap", _top_rung_gap("Phi-3.5-mini"),
          "8.5-bit refusal rate minus FP16"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", type=Path,
                    default=Path("docs/paper/cliff_artifact.tex"))
    ap.add_argument("--stats", type=Path,
                    default=Path("docs/paper/review_stats.json"))
    args = ap.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    text = args.tex.read_text(encoding="utf-8")

    failures: list[str] = []
    print(f"{'quantity':28s} {'expected':18s} status")
    print("-" * 62)
    for check in CHECKS:
        expected = check.expected(stats)
        present = expected in text
        print(f"{check.label:28s} {expected:18s} {'ok' if present else 'MISSING'}")
        if not present:
            failures.append(
                f"  {check.label}: manuscript does not contain {expected!r}\n"
                f"    ({check.description})")

    if failures:
        print("\nprose disagrees with the measurements:")
        print("\n".join(failures))
        return 1
    print(f"\nall {len(CHECKS)} quoted quantities match {args.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
