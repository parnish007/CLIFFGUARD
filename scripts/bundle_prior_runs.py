"""Package the five published runs so a Colab session can re-grade them.

Round 3's cheapest measurement re-grades the runs behind the published numbers
under the corrected label scorer. That needs those run directories present in
the session, and `data/` and `artifacts/` are both gitignored, so a fresh clone
has neither.

This builds the smallest bundle that makes the step possible. Two things it
deliberately leaves out:

  * `activations/`, which is 63-95 MB per behavioural run and is read only by
    the probe arm. Re-grading never touches it, and including it would turn a
    5 MB upload into 160 MB.
  * every scheme except full precision and the 4.5-bit rung on the behavioural
    runs, since those are the only two the re-grade scores.

What remains is a few megabytes of prompts, completions, NLL and manifests --
small enough to drop into Drive by hand, which is the point. The alternative is
publishing model completions to harmful prompts in a public repository, and
that is a decision worth making on purpose rather than as a side effect of
wanting a file to be reachable from Colab.

Usage:
  python scripts/bundle_prior_runs.py
  python scripts/bundle_prior_runs.py --out prior_runs.zip --all-schemes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]

# The five runs behind the published numbers, and the schemes the re-grade
# actually scores. XSTest is graded at full precision only; the behavioural
# runs carry the FP16-versus-4.5-bit comparison.
WANTED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("*colab-behavioural-qwen3b", ("FP16", "RTN_4B")),
    ("*colab-behavioural-phi35", ("FP16", "RTN_4B")),
    ("*lab-qwen3b-xstest", ("FP16",)),
    ("*lab-phi35-xstest", ("FP16",)),
    ("*lab-smol17-xstest", ("FP16",)),
)

# Everything the graders read, beyond the per-scheme completions.
ALWAYS = ("manifest.json", "results/prompts.json", "results/completion_nll.json")


def wanted_members(run: Path, schemes: tuple[str, ...],
                   all_schemes: bool) -> list[Path]:
    """Files to carry, as paths relative to the repository root."""
    members: list[Path] = []
    for rel in ALWAYS:
        path = run / rel
        if path.is_file():
            members.append(path)
    for path in sorted((run / "results").glob("completions_*.json")):
        scheme = path.stem.removeprefix("completions_")
        if all_schemes or scheme in schemes:
            members.append(path)
    # Generation token ids when a run has them. Absent on everything generated
    # before they were stored, which is every run this script targets today --
    # carried anyway so the same command keeps working once they exist.
    for path in sorted((run / "results").glob("tokens_*.json")):
        scheme = path.stem.removeprefix("tokens_")
        if all_schemes or scheme in schemes:
            members.append(path)
    return members


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=REPO / "artifacts" / "runs")
    ap.add_argument("--out", type=Path, default=REPO / "prior_runs.zip")
    ap.add_argument("--all-schemes", action="store_true",
                    help="carry every rung rather than only the graded ones")
    args = ap.parse_args()

    if not args.runs.is_dir():
        print(f"no run directory at {args.runs}", file=sys.stderr)
        return 1

    selected: list[tuple[Path, list[Path]]] = []
    missing: list[str] = []
    for pattern, schemes in WANTED:
        hits = sorted(args.runs.glob(pattern))
        if not hits:
            missing.append(pattern)
            continue
        run = hits[-1]
        members = wanted_members(run, schemes, args.all_schemes)
        if not members:
            missing.append(f"{pattern} (no readable results)")
            continue
        selected.append((run, members))

    if not selected:
        print("nothing to bundle", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for run, members in selected:
            for path in members:
                # Stored relative to the repository root, so unzipping at the
                # root -- or into Drive's artifacts/ -- lands each file exactly
                # where the graders look for it.
                arc = path.relative_to(REPO).as_posix()
                zf.write(path, arc)
                total += path.stat().st_size
            print(f"  {run.name}: {len(members)} files")

    size = args.out.stat().st_size
    print(f"\nwrote {args.out}  ({size / 1e6:.1f} MB compressed, "
          f"{total / 1e6:.1f} MB raw)")
    if missing:
        print(f"\nNOT FOUND: {missing}")
        print("The re-grade step will skip these and say so. It is the only "
              "step that needs them; everything else in round 3 generates its "
              "own data.")

    manifest_note = {
        "purpose": "prior runs needed by the round-3 re-grade step",
        "runs": [run.name for run, _ in selected],
        "excludes": ["activations/", "schemes other than FP16 and RTN_4B"],
    }
    print("\ncontents:")
    print(json.dumps(manifest_note, indent=2))
    print("\nUpload this zip to Drive at  MyDrive/cliffguard/prior_runs.zip")
    print("The notebook unpacks it before preflight if it is there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
