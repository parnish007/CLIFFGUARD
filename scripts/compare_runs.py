"""Compare hypothesis results across multiple CLIFFGUARD evaluation runs.

Loads hypothesis_results.json from each run directory and produces
a comparison table showing for each hypothesis whether it was
accepted in each run, and the key metric values (p-value, statistic).

Useful for:
  - Comparing results before vs after a code change.
  - Comparing results across hardware tiers.
  - Verifying reproducibility across devices.

Output: JSON comparison file + printed table to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.results_writer import (  # noqa: E402
    load_hypothesis_results,
    load_run_metadata,
)

_HYPOTHESES: list[str] = ["H1", "H2", "H3", "H4", "H5"]

# Keys used to determine accept/reject status for each hypothesis.
# H2/H3 have no dedicated accepted key in the schema; callers may add one.
_ACCEPTED_KEYS: dict[str, str] = {
    "H1": "h1_accepted",
    "H2": "h2_accepted",
    "H3": "h3_accepted",
    "H4": "h4_accepted",
    "H5": "h5_tier_c_accepted",
}


def compare_runs(run_dirs: list[Path]) -> dict[str, object]:
    """Load metadata and hypothesis results from each run_dir.
    Returns a comparison dict with:
      runs: list of {run_dir, tier, hostname, timestamp_utc,
                     schemes, git_hash, hypothesis_results}
      hypotheses: [H1, H2, H3, H4, H5]
      agreement: {H1: all_same, H2: all_same, ...}
        where all_same=True means all runs agree on accept/reject.
    Skips runs where hypothesis_results.json is missing (partial
    run) and logs a warning to stderr."""
    runs: list[dict[str, object]] = []
    for run_dir in run_dirs:
        try:
            hyp: dict[str, object] = load_hypothesis_results(run_dir)
        except FileNotFoundError:
            print(
                f"WARNING: hypothesis_results.json missing in {run_dir} — skipping",
                file=sys.stderr,
            )
            continue
        try:
            meta: dict[str, object] = load_run_metadata(run_dir)
        except FileNotFoundError:
            meta = {}

        runs.append(
            {
                "run_dir": str(run_dir),
                "tier": meta.get("tier", "UNKNOWN"),
                "hostname": meta.get("hostname", "UNKNOWN"),
                "timestamp_utc": meta.get("timestamp_utc", "UNKNOWN"),
                "schemes": meta.get("schemes", []),
                "git_hash": meta.get("git_hash", "UNKNOWN"),
                "hypothesis_results": hyp,
            }
        )

    agreement: dict[str, object] = {}
    for h in _HYPOTHESES:
        key = _ACCEPTED_KEYS.get(h, "")
        values: list[object] = []
        for r in runs:
            hyp_r = r.get("hypothesis_results")
            if isinstance(hyp_r, dict) and key in hyp_r:
                values.append(hyp_r[key])
        if len(values) >= 2:
            agreement[h] = all(v == values[0] for v in values[1:])
        else:
            agreement[h] = None

    return {
        "runs": runs,
        "hypotheses": _HYPOTHESES,
        "agreement": agreement,
    }


def print_comparison_table(comparison: dict[str, object]) -> None:
    """Print a readable table to stdout:
    Run | Tier | Host | H1 | H2 | H3 | H4 | H5
    One row per run. H columns show ACCEPT/REJECT/MISSING."""
    runs_raw = comparison.get("runs", [])
    if not isinstance(runs_raw, list) or not runs_raw:
        print("No completed runs to compare.")
        return

    h_header = "  ".join(f"{h:<7}" for h in _HYPOTHESES)
    header = f"{'Run':<35} {'Tier':<6} {'Host':<15}  {h_header}"
    print(header)
    print("-" * len(header))

    for run in runs_raw:
        if not isinstance(run, dict):
            continue
        hyp = run.get("hypothesis_results")
        h_values: list[str] = []
        for h in _HYPOTHESES:
            key = _ACCEPTED_KEYS.get(h, "")
            if isinstance(hyp, dict):
                val = hyp.get(key)
                if val is None:
                    h_values.append("MISSING")
                elif val:
                    h_values.append("ACCEPT")
                else:
                    h_values.append("REJECT")
            else:
                h_values.append("MISSING")

        run_name = Path(str(run.get("run_dir", ""))).name or "?"
        tier = str(run.get("tier", "?"))
        host = str(run.get("hostname", "?"))[:15]
        h_str = "  ".join(f"{v:<7}" for v in h_values)
        print(f"{run_name:<35} {tier:<6} {host:<15}  {h_str}")


def main(argv: list[str] | None = None) -> int:
    """Parse --runs (one or more paths), --output (optional).
    Call compare_runs, print table, optionally save JSON.
    Return 0 on success."""
    parser = argparse.ArgumentParser(
        description="Compare hypothesis results across CLIFFGUARD evaluation runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="RUN_DIR",
        help="One or more run directories to compare.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="OUTPUT_FILE",
        help="Optional output JSON file for the comparison.",
    )
    args = parser.parse_args(argv)

    run_dirs = [Path(r) for r in args.runs]
    comparison = compare_runs(run_dirs)
    print_comparison_table(comparison)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
