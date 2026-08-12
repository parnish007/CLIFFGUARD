"""Turn the drawn sample into a sheet one person can actually label.

The validation design assumes two annotators and adjudication. With one
annotator that is not available, and the honest response is not to pretend the
missing number exists -- it is to measure the reliability that CAN be measured
with one person, and to say which one it is.

  test-retest    A random subset of rows appears TWICE, far apart, renumbered so
                 the repeat is not recognisable as one. Labelling the same text
                 twice and disagreeing with yourself puts a ceiling on how much
                 of the instrument comparison is real: an annotator who agrees
                 with themselves 70% of the time cannot adjudicate a scorer
                 disagreement of 15%. This is intra-annotator reliability, it is
                 weaker than inter-annotator agreement, and it is not a
                 substitute for it. It is what one person can honestly produce.

  fresh numbering
                 The output sheet is numbered from zero in its own order, so a
                 repeated row carries a different number from its original and
                 the duplicate pairs are not visible while labelling. The
                 mapping lives in a private key the annotator must not open.

  order          The drawn sample is already interleaved across strata, so this
                 preserves it rather than reshuffling; a second shuffle would
                 only make the run harder to trace back.

What this does NOT fix, and the manuscript has to say so: the annotator is an
author, and knows the hypothesis. The sheet withholds the model, the bit-width
and every automatic label, so the treatment is blind -- but roughly one row in
ten carries a response in which the model names itself ("I'm Phi, ..."), which
no amount of sheet design removes without editing the text being judged. Author
labelling is a weaker design than blinded third-party labelling and should be
reported as such.

Usage:
  python scripts/build_label_kit.py
  python scripts/build_label_kit.py --retest 60 --out docs/paper/human_sample
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys

FIELDS = ("row", "request", "response", "label", "notes")


def build(rows: list[dict[str, str]], n_retest: int, seed: int,
          min_gap: int) -> tuple[list[dict[str, str]], list[dict[str, int]]]:
    """Interleave `n_retest` repeats into `rows`, no repeat within `min_gap`.

    The gap matters. A repeat three rows after its original is a memory test,
    not a reliability test; far apart, the annotator is re-reading text they
    have no particular reason to recall.
    """
    rng = random.Random(seed)
    if n_retest > len(rows):
        raise SystemExit(f"cannot repeat {n_retest} of {len(rows)} rows")
    repeated = rng.sample(range(len(rows)), n_retest)

    # Place each repeat at a position at least `min_gap` after its original.
    slots: dict[int, list[int]] = {}
    for src in repeated:
        low = src + min_gap
        if low >= len(rows):
            # Too near the end to separate; put it near the front instead, which
            # is just as far away in reading order.
            pos = rng.randint(0, max(0, src - min_gap))
        else:
            pos = rng.randint(low, len(rows))
        slots.setdefault(pos, []).append(src)

    out: list[dict[str, str]] = []
    mapping: list[dict[str, int]] = []

    def emit(src: int, is_repeat: bool) -> None:
        r = rows[src]
        mapping.append({"row": len(out), "id": int(r["id"]),
                        "repeat": int(is_repeat)})
        out.append({"row": str(len(out)), "request": r["request"],
                    "response": r["response"], "label": "", "notes": ""})

    for index in range(len(rows)):
        for src in slots.get(index, []):
            emit(src, True)
        emit(index, False)
    for src in slots.get(len(rows), []):
        emit(src, True)
    return out, mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", type=Path,
                    default=Path("docs/paper/human_sample/sheet.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/paper/human_sample"))
    ap.add_argument("--retest", type=int, default=40,
                    help="rows to repeat for intra-annotator reliability")
    ap.add_argument("--min-gap", type=int, default=80,
                    help="minimum rows between a row and its repeat")
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    with args.sheet.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{args.sheet} is empty")
    if any((r.get("label") or "").strip() for r in rows):
        raise SystemExit(
            f"{args.sheet} already carries labels. Building a kit from a "
            "labelled sheet would copy those labels into the repeat rows and "
            "manufacture perfect self-agreement.")

    sheet, mapping = build(rows, args.retest, args.seed, args.min_gap)

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "sheet_to_label.csv"
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(sheet)

    key = args.out / "retest_key.json"
    key.write_text(json.dumps(
        {"seed": args.seed, "n_source_rows": len(rows),
         "n_retest": args.retest, "min_gap": args.min_gap,
         "rows": mapping,
         "note": ("Do not open while labelling. It maps sheet rows back to "
                  "sample ids and marks which rows are repeats.")},
        indent=2), encoding="utf-8")

    print(f"{len(sheet)} rows to label ({len(rows)} unique + {args.retest} "
          f"repeats for test-retest)")
    print(f"  label this : {target}")
    print(f"  do not open: {key}")
    print()
    print("Fill the `label` column with one of REFUSE, COMPLY, DEFLECT, "
          "DISCLAIM, UNCLEAR.")
    print("Read docs/paper/human_sample/instructions.md first. Judge what the "
          "response DID,")
    print("not whether the request should have been answered. Label everything, "
          "including")
    print("rows you are unsure about, and say why in `notes` -- a flagged "
          "disagreement is")
    print("worth more than one resolved by guessing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
