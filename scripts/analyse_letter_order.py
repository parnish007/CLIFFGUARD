"""Does the corrected scorer's answer depend on which letter carries which class?

The corrected scorer replaced label words with single-token letters, which
removes the tokenization asymmetry the manuscript is about. It does not remove
every property of the scoring construction. Multiple-choice prompting is known
to carry position and symbol preferences, and a judge that leans toward A --
whatever A happens to say -- would produce a grading that looks commensurable
and is not.

That is testable, and cheaply: hold the five descriptions fixed, permute which
letter each is offered under, re-grade the same stored text, and map the
verdicts back to classes. If the classes barely move, the letters are carrying
no signal and the corrected scorer measures what it claims to. If they move, we
have found a second measurement effect, and the paper is about measurement
effects.

What this reports, per (run, scheme):

  agreement        share of completions given the same CLASS under the
                   canonical assignment and the permutation
  per-class delta  how the class counts move, class by class
  letter bias      how often each LETTER won, canonical against permuted. This
                   is the diagnostic that separates the two explanations: a
                   judge with no letter preference produces different letter
                   frequencies under a permutation (because the classes moved),
                   while a judge that always answers "A" produces the SAME
                   letter frequency and different classes.

It deliberately reports no p-value. The comparison is a paired re-grading of
identical text under a deterministic scorer, so the sampling model a test would
need is not the one that generated the difference; the quantity of interest is
the size of the movement, not whether it is distinguishable from zero.

Usage:
  python scripts/analyse_letter_order.py artifacts/runs/<run> \\
      --orders "COMPLY,UNCLEAR,REFUSE" --orders "UNCLEAR,REFUSE,COMPLY"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval import scorer_caches


def _load(path: Path) -> list[str]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, dict):
        return list(blob["verdicts"])
    return list(blob)


def gradings(run: Path, order: Sequence[str] | None, judge: str,
             five_way: bool, batch_size: int = 4,
             completion_chars: int = 2000) -> dict[str, list[str]]:
    """Per-scheme verdicts for one letter assignment, or {} if not graded.

    Resolution is by recomputed fingerprint rather than by filename pattern, so
    a file is only read when the hash of the identity we are asking for equals
    the name it is stored under. A permutation that was never run therefore
    returns nothing instead of quietly matching the canonical grading.
    """
    if five_way:
        # The batch size is part of the taxonomy fingerprint, so passing the
        # wrong one reports "not graded" for a grading that is on disk. It has
        # to match what the grader was run with, not what is convenient here.
        found = scorer_caches.resolve_taxonomy(run, judge=judge,
                                               letter_order=order,
                                               batch_size=batch_size,
                                               completion_chars=completion_chars)
        digest, prefix = found.get("letter"), "taxonomy"
    else:
        # 2000, not 600: the window the GRADERS are run with. It is part of
    # the fingerprint only when it actually truncates something, so with today's
    # stored completions (longest 465 characters) any value at or above that is
    # equivalent -- but a 256-token run reaches ~1300 characters, and then 600
    # and 2000 are different gradings. Passing what the grader was invoked with
    # keeps this correct when that happens rather than when someone notices.
        found = scorer_caches.resolve(run, judge=judge,
                                      completion_chars=completion_chars,
                                      letter_order=order)
        digest, prefix = found.get("letter"), "judge"
    if not digest:
        return {}
    out: dict[str, list[str]] = {}
    for path in sorted(run.glob(f"results/{prefix}_{digest}_*.json")):
        out[path.stem.split("_", 2)[-1]] = _load(path)
    return out


def compare(canonical: list[str], permuted: list[str],
            letters: Sequence[str], canonical_order: Sequence[str],
            permuted_order: Sequence[str]) -> dict[str, Any]:
    """One scheme, one permutation.

    `letter_of` inverts each assignment, so the letter frequencies can be
    compared as well as the class frequencies. Those two together are what
    distinguish "the judge moved its answer with the meaning" from "the judge
    kept answering the same letter".
    """
    if len(canonical) != len(permuted):
        raise SystemExit(
            f"{len(canonical)} canonical verdicts against {len(permuted)} "
            "permuted; these are not gradings of the same completions")
    agree = sum(a == b for a, b in zip(canonical, permuted))
    canon_letter = {name: letters[k] for k, name in enumerate(canonical_order)}
    perm_letter = {name: letters[k] for k, name in enumerate(permuted_order)}
    return {
        "n": len(canonical),
        "agreement": agree / len(canonical) if canonical else 0.0,
        "moved": len(canonical) - agree,
        "classes": {
            "canonical": dict(Counter(canonical)),
            "permuted": dict(Counter(permuted)),
        },
        "letters": {
            "canonical": dict(Counter(canon_letter[v] for v in canonical)),
            "permuted": dict(Counter(perm_letter[v] for v in permuted)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--orders", action="append", default=[],
                    help="comma-separated permutation of the class names. Repeat "
                         "for several. Without any, the canonical grading is "
                         "listed and nothing is compared.")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="the batch size the grading was RUN with. Part of the "
                         "five-way cache fingerprint, so a mismatch reports a "
                         "grading that exists as missing. Round 3 and round 4 "
                         "both use 4.")
    ap.add_argument("--completion-chars", type=int, default=2000,
                    help="the window the grading was RUN with. Part of the "
                         "five-way fingerprint unconditionally and of the "
                         "three-way one when it truncates, so a mismatch "
                         "reports a grading that exists as missing -- which "
                         "reads as an experiment that never ran.")
    ap.add_argument("--five-way", action="store_true",
                    help="read the taxonomy grader's caches rather than the "
                         "three-way judge's.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    module = (scorer_caches._taxonomy_module() if args.five_way
              else scorer_caches._judge_module())
    labels, letters = module.LABELS, module.LETTERS
    orders = [tuple(o.split(",")) for o in args.orders]
    for order in orders:
        # Fail before loading anything: a typo here would otherwise resolve to
        # no fingerprint and be reported as "not graded", which reads as a
        # missing experiment rather than as a bad argument.
        module.letter_template(order)

    report: dict[str, Any] = {"judge": args.judge,
                              "five_way": bool(args.five_way),
                              "batch_size": args.batch_size,
                              "completion_chars": args.completion_chars,
                              "labels": list(labels),
                              "canonical_order": list(labels),
                              "runs": {}}
    worst = 1.0
    for run in args.runs:
        canonical = gradings(run, None, args.judge, args.five_way,
                             args.batch_size, args.completion_chars)
        if not canonical:
            print(f"{run.name}: no canonical letter grading; skipping")
            continue
        block: dict[str, Any] = {"schemes": sorted(canonical), "orders": {}}
        for order in orders:
            permuted = gradings(run, order, args.judge, args.five_way,
                                args.batch_size, args.completion_chars)
            shared = sorted(set(canonical) & set(permuted))
            if not shared:
                print(f"{run.name}: {','.join(order)} not graded")
                continue
            rows = {s: compare(canonical[s], permuted[s], letters, labels, order)
                    for s in shared}
            block["orders"][",".join(order)] = rows
            for scheme, row in rows.items():
                worst = min(worst, row["agreement"])
                print(f"{run.name[-34:]:34s} {scheme:7s} "
                      f"{','.join(order):34s} "
                      f"agree {100 * row['agreement']:5.1f}%  "
                      f"moved {row['moved']:3d}/{row['n']}")
        report["runs"][run.name] = block

    if not any(b["orders"] for b in report["runs"].values()):
        print("\nNo permuted gradings found. Run the grader with --letter-order "
              "first; this script reads caches and never grades.")
        return 1

    report["worst_agreement"] = worst
    print(f"\nworst class agreement across every scheme and permutation: "
          f"{100 * worst:.1f}%")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
