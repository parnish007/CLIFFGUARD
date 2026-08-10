"""Refit the frozen refusal probe on the corrected scorer's labels.

The probe section argues that the probe must be trained on the same labels the
behavioural arm reports, because a probe fitted on one target and compared
against behaviour measured on another cannot establish a dissociation between
them -- it can only show that two different questions have two different
answers. That argument is right, and correcting the label scorer broke it: the
behavioural arm now reports corrected labels while the probe was still fitted
on the original scorer's.

So the probe is refitted here on the corrected full-precision labels. Nothing
about the probe changes -- same layer, same difference-in-means direction, same
held-out split protocol, same seed, same replicate count. Only the definition of
which prompts count as refusals moves, which is exactly the quantity the
correction changed: 409 refusals become 353 on Qwen2.5-3B and 435 become 385 on
Phi-3.5-mini.

That "same protocol" is load-bearing enough to be enforced structurally rather
than described. This script does not reimplement retention; it calls the same
`probe_retention` the published table came from, twice, with only the scorer
argument differing. An earlier version did reimplement it, taking the ratio of
two mean d' values where the published estimator takes the mean of the
per-split ratios, and the two disagree by 42 points at Qwen2.5-3B's 3.5-bit
rung -- a difference that has nothing to do with the label correction this
script exists to measure.

The legacy column is therefore a check as well as a baseline: it should
reproduce `review_stats.json` cell for cell, and `--assert-baseline` fails if it
does not.

This runs on a CPU. The activations are already stored per scheme, the labels
are already graded, and a d' is a projection and two moments.

Usage:
  python scripts/refit_probe_corrected.py
  python scripts/refit_probe_corrected.py --splits 200 --assert-baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.review_reanalysis import probe_retention

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "artifacts" / "runs"

PATTERNS = ("*colab-behavioural-qwen3b", "*colab-behavioural-phi35")
# "first-token-legacy" is what the published probe was fitted on; "letter" is
# the corrected scorer the behavioural arm now reports.
PUBLISHED, CORRECTED = "first-token-legacy", "letter"
# The published table's replicate count. Retention percentiles are a dispersion
# measure over splits, so a different count is a different quantity.
SPLITS = 200
ORDER = ("FP16", "RTN_8B", "RTN_7B", "RTN_6B", "RTN_5B", "RTN_4B", "RTN_3B",
         "RTN_2B")


def find(pattern: str) -> Path:
    hits = sorted(RUNS.glob(pattern))
    if not hits:
        raise SystemExit(f"no run matching {pattern}")
    return hits[-1]


def rows_by_scheme(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["scheme"]: r for r in block["rows"]}


def check_baseline(legacy: dict[str, Any], published: dict[str, Any],
                   tolerance: float) -> list[str]:
    """Every legacy cell against the published one, to the stated tolerance.

    The point of the corrected refit is that one thing changed. If the arm that
    changed nothing does not reproduce, then something else changed too and the
    difference between the two columns is not the label definition.
    """
    problems: list[str] = []
    for model, block in published.items():
        if model not in legacy:
            problems.append(f"{model}: not refitted")
            continue
        mine, theirs = rows_by_scheme(legacy[model]), rows_by_scheme(block)
        if legacy[model]["n_positive"] != block["n_positive"]:
            problems.append(
                f"{model}: {legacy[model]['n_positive']} positives against the "
                f"published {block['n_positive']}")
        for scheme, row in theirs.items():
            if scheme not in mine:
                problems.append(f"{model}/{scheme}: missing")
                continue
            delta = abs(mine[scheme]["retained_mean"] - row["retained_mean"])
            if delta > tolerance:
                problems.append(
                    f"{model}/{scheme}: retention "
                    f"{100 * mine[scheme]['retained_mean']:.1f}% against the "
                    f"published {100 * row['retained_mean']:.1f}%")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / "paper" / "probe_corrected.json")
    ap.add_argument("--splits", type=int, default=SPLITS)
    ap.add_argument("--assert-baseline", action="store_true",
                    help="fail unless the legacy column reproduces "
                         "review_stats.json")
    ap.add_argument("--tolerance", type=float, default=1e-9,
                    help="allowed absolute drift in retention against the "
                         "published table; the seed is fixed, so the default "
                         "is exact equality")
    args = ap.parse_args()

    runs = [find(p) for p in PATTERNS]
    scorers = {}
    for scorer in (PUBLISHED, CORRECTED):
        print(f"[{scorer}] {args.splits} replicates")
        scorers[scorer] = probe_retention(runs, args.splits, "judge",
                                          scorer=scorer)

    published_path = REPO / "docs" / "paper" / "review_stats.json"
    published = json.loads(published_path.read_text(encoding="utf-8"))["probe"]
    problems = check_baseline(scorers[PUBLISHED], published, args.tolerance)

    out: dict[str, Any] = {
        "protocol": {
            "splits": args.splits,
            "estimator": ("mean over splits of d'(rung)/d'(FP16), the same "
                          "probe_retention the published table uses"),
            "direction": ("difference in means, fitted on one half of the "
                          "prompts and scored on the disjoint other half"),
            "note": ("Only the label definition differs between the two scorer "
                     "blocks. Layer, direction estimator, split count and seed "
                     "are identical, so any change in retention is the target "
                     "moving."),
            "baseline_reproduces": not problems,
            "baseline_problems": problems,
        },
        "scorers": scorers,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {args.out}\n")

    models = sorted({m for block in scorers.values() for m in block})
    for model in models:
        print(f"=== {model}")
        print(f"  {'scheme':8s} {'original':>20s} {'corrected':>20s}")
        blocks = {s: scorers[s].get(model) for s in (PUBLISHED, CORRECTED)}
        for scheme in ORDER:
            cells = ""
            for scorer in (PUBLISHED, CORRECTED):
                block = blocks[scorer]
                row = rows_by_scheme(block).get(scheme) if block else None
                cells += (f"{100 * row['retained_mean']:9.1f}%"
                          f" [{100 * row['retained_ci_low']:4.0f},"
                          f"{100 * row['retained_ci_high']:4.0f}]"
                          if row else f"{'--':>20s}")
            print(f"  {scheme:8s} {cells}")
        for scorer in (PUBLISHED, CORRECTED):
            block = blocks[scorer]
            if block:
                print(f"    {scorer:20s} positives {block['n_positive']}/"
                      f"{block['n_positive'] + block['n_negative']}, "
                      f"FP16 d' {block['fp16_absolute_dprime']:.3f}")

    if problems:
        print("\nthe legacy column does NOT reproduce the published table:")
        for problem in problems:
            print(f"  {problem}")
        if args.assert_baseline:
            return 1
    else:
        print("\nthe legacy column reproduces review_stats.json exactly, so the "
              "difference between the two columns is the label definition and "
              "nothing else")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
