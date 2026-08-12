"""Merge a Colab round archive into artifacts/runs without losing what is there.

A round exports the run directories it touched, and those directories already
exist locally: the archive is the same runs with more gradings in them. Most of
what it carries is new per-prompt cache files, which are named by the
fingerprint of the grading policy and so cannot collide with a different
measurement.

The summary files can, and do. `judge_classification.json` and
`completion_taxonomy.json` are written fresh by every grader invocation and are
not namespaced, so the copy in the archive describes whichever grading ran last
-- round 4's canonical letter re-grade -- while the local copy describes the
original-scorer measurement the manuscript's published numbers come from.
Copying the tree over would replace the second with the first and leave no
record that it had happened, and the two are not interchangeable: the whole
point of the round is that they disagree.

So summaries are written to a scorer-suffixed name instead, and the incumbent is
left alone. Everything else is copied only when it is genuinely new or
byte-identical; anything else stops the import and is reported, because a
differing file that is not a summary means the archive and the workspace
disagree about a measurement and that is not something to resolve by overwriting.

Usage:
  python scripts/import_round_archive.py notebooks/cliffguard_round4_*.zip
  python scripts/import_round_archive.py <zip> --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Written by a grader on every invocation and overwritten in place, so the
# archive's copy is the last grading rather than the grading a caller wants.
SUMMARIES = {"judge_classification.json", "completion_taxonomy.json"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scorer_of(path: Path) -> str:
    """The scorer a summary describes, for naming its preserved copy.

    Read from the file rather than assumed from the round, because a round can
    run more than one scorer and the archive does not say which finished last.
    """
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    scoring = blob.get("scoring")
    if not isinstance(scoring, str) or not scoring:
        # Absent means the identity predates the key, which is the original
        # first-token grading. Naming it explicitly beats "unknown".
        return "first-token-legacy"
    order = blob.get("letter_order")
    labels = blob.get("labels")
    if isinstance(order, list) and isinstance(labels, list) and order != labels:
        # A permuted grading is a different instrument from the canonical one
        # and must not land on the same filename as it. Positions rather than
        # initials: the five-way labels include DEFLECT and DISCLAIM, which
        # share a first letter, so an initials suffix collides on the two
        # permutations that swap exactly those and silently overwrites one
        # grading with the other.
        position = {name: i for i, name in enumerate(labels)}
        if all(name in position for name in order):
            return f"{scoring}-" + "".join(str(position[n]) for n in order)
        return f"{scoring}-permuted"
    return scoring


Pair = tuple[Path, Path]


def plan(src_runs: Path, dst_runs: Path
         ) -> tuple[list[Pair], list[Pair], list[Path]]:
    """(copy, preserve, conflict) -- decided before anything is written."""
    copy: list[Pair] = []
    preserve: list[Pair] = []
    conflict: list[Path] = []
    for source in sorted(src_runs.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(src_runs)
        target = dst_runs / rel
        if not target.exists():
            copy.append((source, target))
        elif digest(source) == digest(target):
            continue
        elif source.name in SUMMARIES:
            suffix = scorer_of(source)
            kept = target.with_name(f"{target.stem}_{suffix}.json")
            # An existing preserved summary is itself a measurement, and this
            # one is not necessarily the same measurement -- two rounds can
            # both grade under `letter` and differ in batch size, window, or
            # which rungs they covered. Overwriting would destroy the first
            # with no record, so it is a conflict like any other and stops the
            # import. Re-importing the SAME archive is unaffected: the
            # byte-identical branch above returns before reaching here.
            if kept.exists() and digest(source) != digest(kept):
                conflict.append(rel)
                continue
            if any(kept == other for _, other in preserve):
                conflict.append(rel)
                continue
            preserve.append((source, kept))
        else:
            conflict.append(rel)
    return copy, preserve, conflict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"no such archive: {args.archive}")
    args.runs.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        src_runs = root / "artifacts" / "runs"
        if not src_runs.is_dir():
            raise SystemExit(
                f"{args.archive} has no artifacts/runs/; it is not a round archive")

        copy, preserve, conflict = plan(src_runs, args.runs)

        if conflict:
            print("REFUSING TO IMPORT: these files differ and are not summaries.")
            print("The archive and the workspace disagree about a measurement;")
            print("work out which is right before importing either.")
            for rel in conflict:
                print(f"  {rel}")
            return 1

        loose = sorted(p for p in src_runs.glob("*.json") if p.is_file())
        print(f"{len(copy)} new file(s), {len(preserve)} summary file(s) "
              f"preserved under a scorer-suffixed name")
        for _, target in preserve:
            print(f"  summary -> {target.relative_to(args.runs)}")
        if loose:
            print(f"  ({len(loose)} run-level result file(s) including "
                  f"{', '.join(p.name for p in loose[:3])}...)")

        if args.dry_run:
            print("\n--dry-run: nothing written")
            return 0

        for source, target in copy + preserve:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    print(f"\nimported {len(copy) + len(preserve)} file(s) into {args.runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
