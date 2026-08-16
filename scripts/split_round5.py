"""Derive per-part Round 5 notebooks from the one master notebook.

Round 5 runs six steps in one session. On a free Colab runtime that is longer
than the session lasts, so it hits the wall, and hitting the wall repeatedly is
worse than slow: each disconnect costs whatever was in flight.

The steps do not depend on each other. Step 0 re-grades stored text, step 3
generates without grading, and the notebook's own comments say steps 2--4
"answer questions of their own" and need nothing from step 1. So the session can
be cut anywhere, and the only thing that has to survive the cut is the run
directories --- which already mirror to Drive after every step, and which
`completed_run` re-reads on the next session rather than trusting a flag.

This emits two notebooks from the master rather than forking it by hand. Forking
would double every environment, restore, preflight and helper cell, and those
are exactly the cells that have already drifted once in this project. One
source, two derived artifacts, and a `RUN_STEPS` set at the top of each so the
split is visible rather than implied by which cells were deleted.

**Both parts must use the same Google Drive account.** Resume works by finding
the previous part's run directories under `artifacts/runs`; a different Drive is
an empty Drive, and an empty Drive means part B regenerates what part A already
produced.

Usage:
  python scripts/split_round5.py
  python scripts/split_round5.py --check     # verify the patch points, write nothing
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

MASTER = Path("notebooks/colab_round5.ipynb")

# Which cell carries which step, and the exact guard line to gate. Verified
# before anything is written: a guard that has moved fails the build rather
# than silently producing a notebook where a step cannot be switched off.
STEP_CELLS: dict[str, tuple[int, str]] = {
    "step0": (10, "if not (SESSION_OK):"),
    "step1": (12, "if STEP1_BLOCKER:"),
    "step2": (14, "if not (SESSION_OK):"),
    "step3": (16, "if not (SESSION_OK):"),
    "step4": (18, "if not (SESSION_OK):"),
    "step5": (20, "if not (SESSION_OK and PREREG_OK):"),
}

CONFIG_CELL = 8

# Names a gated cell must still bind, because a later cell reads them. Step 1's
# `run_dir_1` is the one the master calls out by name: a skipped step 1 that
# left it undefined took the verdict cell down with a NameError, which is a
# correctly skipped step destroying the rest of the session.
STEP_LEFTOVERS: dict[str, tuple[str, ...]] = {
    "step1": ("run_dir_1 = None",
              "STEP_NOTES['deployed_3b'] = {'skipped': True, "
              "'other_part': True,",
              "                             'reason': 'runs in the other part "
              "of round 5'}"),
}

# The split. A is the pair that answers the two objections the manuscript most
# needs answered and that between them need no new deployed checkpoint: the
# option-assignment audit, which re-grades stored text, and the batch-size
# isolation, which generates for twelve minutes and grades nothing. B is the
# generation-heavy remainder.
PARTS: dict[str, tuple[set[str], str]] = {
    "a": ({"step0", "step3"},
          "option-assignment invariance, and the batch-size "
          "isolation. Re-grades stored completions under every letter "
          "assignment for the three-way scorer, and generates one short "
          "twelve-minute comparison. No deployed checkpoint is downloaded and "
          "no quantization backend is needed, so this part runs even if the "
          "AWQ and GPTQ wheels do not install."),
    "b": ({"step1", "step2", "step4", "step5"},
          "deployed quantizers, the generation window, sampled "
          "decoding, and scale. Every step here generates, so this is the "
          "expensive half. Run it AFTER part A, on the SAME Drive account: "
          "part A's run directories are what let this session skip anything "
          "already finished."),
}

SELECTOR = '''
# ---------------------------------------------------------------------------
# Which steps this notebook runs. Set by scripts/split_round5.py; edit it here
# if you want a different cut. A step not in RUN_STEPS is skipped and SAYS it
# was skipped, so a partial session cannot be mistaken for a complete one.
#
# The six steps are independent, and every run directory mirrors to Drive as
# soon as it is finished. So the parts can run in any order and across as many
# sessions as it takes -- PROVIDED every session mounts the SAME Drive account.
# A different account is an empty Drive, and an empty Drive silently regenerates
# work that was already done.
RUN_STEPS = {run_steps!r}


def want(step):
    """True if this notebook is the one that runs `step`."""
    if step in RUN_STEPS:
        return True
    print(f'SKIPPED: {{step}} belongs to the other part of round 5 '
          f'(this notebook runs {{sorted(RUN_STEPS)}}).')
    return False
# ---------------------------------------------------------------------------
'''


def load() -> dict:
    if not MASTER.is_file():
        raise SystemExit(f"{MASTER} not found")
    return json.loads(MASTER.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def check(nb: dict) -> None:
    """Every patch point must exist before anything is written."""
    for step, (idx, guard) in STEP_CELLS.items():
        if idx >= len(nb["cells"]):
            raise SystemExit(f"{step}: cell {idx} is past the end of the master")
        src = cell_source(nb["cells"][idx])
        if guard not in src:
            raise SystemExit(
                f"{step}: guard {guard!r} not found in cell {idx}. The master "
                "notebook changed -- fix STEP_CELLS.")
    if "SAMPLE_TEMPERATURE" not in cell_source(nb["cells"][CONFIG_CELL]):
        raise SystemExit(
            f"cell {CONFIG_CELL} does not look like the configuration cell")
    covered = set().union(*(steps for steps, _ in PARTS.values()))
    if covered != set(STEP_CELLS):
        raise SystemExit(
            f"the parts do not cover every step: missing "
            f"{sorted(set(STEP_CELLS) - covered)}")
    overlap = PARTS["a"][0] & PARTS["b"][0]
    if overlap:
        raise SystemExit(f"a step is in both parts: {sorted(overlap)}")


def build_part(nb: dict, part: str) -> dict:
    steps, blurb = PARTS[part]
    out = copy.deepcopy(nb)

    cfg = out["cells"][CONFIG_CELL]
    set_source(cfg, cell_source(cfg).rstrip("\n") + "\n"
               + SELECTOR.format(run_steps=sorted(steps)))

    for step, (idx, guard) in STEP_CELLS.items():
        cell = out["cells"][idx]
        src = cell_source(cell)
        if step in steps:
            continue
        # Gate rather than delete. A deleted cell leaves the notebook's own
        # readout referring to a step that produced nothing, and the export
        # cell then reports a gap it cannot explain.
        # The gate gets its OWN branch rather than being folded into the
        # guard below it. Folding it in reused a body written for a different
        # reason: step 1's records returncode 1 and `blocked: True`, so part A
        # reported the deployed-quantizer step as FAILED rather than as not its
        # job, and the export listed it under INCOMPLETE. The others print
        # "preflight failed", which was simply untrue and sends whoever reads
        # it looking for a preflight problem.
        #
        # `want()` has already printed why. This branch only has to leave
        # behind what later cells read: for step 1 that is `run_dir_1`, and its
        # absence is exactly the NameError the master's own comment warns
        # about.
        leftovers = STEP_LEFTOVERS.get(step, ())
        body = "".join(f"    {line}\n" for line in leftovers) or "    pass\n"
        patched = src.replace(
            guard, f"if not want({step!r}):\n{body}el{guard.lstrip()}", 1)
        if patched == src:
            raise SystemExit(f"failed to patch guard for {step}")
        set_source(cell, patched)

    title = out["cells"][0]
    head = cell_source(title)
    marker = f"\n\n> **Round 5{part}.** {blurb}\n"
    set_source(title, head.split("\n", 1)[0] + f" — part {part}" + marker
               + ("\n" + head.split("\n", 1)[1] if "\n" in head else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    nb = load()
    check(nb)
    if args.check:
        for part, (steps, _) in PARTS.items():
            print(f"round 5{part}: {sorted(steps)}")
        print("all patch points resolve against the master")
        return 0

    for part in PARTS:
        out = MASTER.with_name(f"colab_round5{part}.ipynb")
        out.write_text(json.dumps(build_part(nb, part), indent=1,
                                  ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}  (runs {sorted(PARTS[part][0])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
