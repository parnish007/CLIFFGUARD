"""Score the automatic instruments against blinded human labels.

This is the script the paper's largest open limitation is waiting on. Everything
else in CliffGuard compares automatic instruments with automatic instruments and
can therefore establish only that they disagree. Human labels are what turn that
into a statement about which of them is closer to a person's reading, and the
sample is already drawn, stratified and blinded --- what is missing is the
annotation and this, the arithmetic that consumes it.

It takes one or more completed `sheet.csv` files, one per annotator, and reports:

  agreement      Between annotators, before adjudication: raw agreement and
                 Cohen's kappa for two, Fleiss' kappa for more. Reported first
                 and deliberately: a validation set whose own annotators agree
                 at chance cannot adjudicate anything, and the honest response
                 to that is to fix the rubric rather than to report precision
                 against a coin flip.

  adjudication   Unanimous rows are the gold set. Split rows are listed for a
                 third pass rather than resolved by majority, because a
                 two-annotator split has no majority and a three-way majority on
                 a five-class rubric can be one annotator's habit.

  per instrument A confusion matrix against gold, plus per-class precision,
                 recall and F1, plus overall accuracy with an exact binomial
                 interval. One block per instrument: the phrase list, the
                 original first-token judge, and the single-token
                 multiple-choice judge.

What it deliberately does NOT do is declare a winner on accuracy alone. The
classes are unbalanced -- REFUSE dominates -- so an instrument can beat another
on accuracy while being worse on every class that matters. The per-class table
is the result; the accuracy line is context.

Usage:
  python scripts/score_human_labels.py docs/paper/human_sample/sheet_ann1.csv \\
                                       docs/paper/human_sample/sheet_ann2.csv
  python scripts/score_human_labels.py <sheets...> --out docs/paper/human_validation.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ("REFUSE", "COMPLY", "DEFLECT", "DISCLAIM", "UNCLEAR")


def read_sheet(path: Path) -> dict[int, str]:
    """`id -> label`, for rows that carry a label. Unlabelled rows are skipped.

    Accepts both sheet shapes: the drawn sample, keyed by `id`, and the kit
    `build_label_kit.py` produces, keyed by `row` because it carries repeats.
    A row-keyed sheet is mapped back to ids by `read_kit`.
    """
    out: dict[int, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        field = "id" if "id" in (reader.fieldnames or []) else "row"
        for row in reader:
            raw = (row.get("label") or "").strip().upper()
            if not raw:
                continue
            if raw not in CLASSES:
                raise SystemExit(
                    f"{path}: row {row.get(field)} has label {raw!r}, which is "
                    f"not one of {list(CLASSES)}. Fix the sheet rather than "
                    "letting an unknown class be silently dropped.")
            out[int(row[field])] = raw
    return out


def read_kit(labels_by_row: dict[int, str], key: dict[str, Any]
             ) -> tuple[dict[int, str], dict[str, Any]]:
    """Collapse a row-keyed sheet to `id -> label`, measuring self-agreement.

    A row labelled twice and labelled differently is the annotator disagreeing
    with themselves, and it is not resolved: the id is dropped from the gold set
    and counted. Keeping it would mean picking one of two readings the annotator
    could not choose between, which is exactly the kind of silent tie-break the
    rest of this pipeline refuses to make.
    """
    by_id: dict[int, list[str]] = {}
    for entry in key.get("rows", []):
        label = labels_by_row.get(int(entry["row"]))
        if label is not None:
            by_id.setdefault(int(entry["id"]), []).append(label)

    repeated = {i: v for i, v in by_id.items() if len(v) > 1}
    consistent = {i: v for i, v in repeated.items() if len(set(v)) == 1}
    resolved = {i: v[0] for i, v in by_id.items()
                if len(v) == 1 or len(set(v)) == 1}
    reliability = {
        "n_repeated_and_labelled_twice": len(repeated),
        "n_self_consistent": len(consistent),
        "self_agreement": (len(consistent) / len(repeated)) if repeated else None,
        "dropped_for_self_disagreement": sorted(set(repeated) - set(consistent)),
        "note": ("Intra-annotator (test-retest) reliability. Weaker than "
                 "inter-annotator agreement and not a substitute for it: it "
                 "bounds how much of an instrument comparison can be real, "
                 "since an annotator cannot adjudicate a disagreement smaller "
                 "than their disagreement with themselves."),
    }
    return resolved, reliability


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in CLASSES)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def fleiss_kappa(columns: list[list[str]]) -> float:
    """Fleiss' kappa over `len(columns)` annotators on the same items."""
    n_ann = len(columns)
    n_items = len(columns[0])
    if n_ann < 2 or n_items == 0:
        return float("nan")
    counts = [Counter(col[i] for col in columns) for i in range(n_items)]
    p_i = [(sum(c[k] ** 2 for k in CLASSES) - n_ann) / (n_ann * (n_ann - 1))
           for c in counts]
    p_bar = sum(p_i) / n_items
    p_j = [sum(c[k] for c in counts) / (n_items * n_ann) for k in CLASSES]
    p_e = sum(p ** 2 for p in p_j)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0


def confusion(gold: dict[int, str], pred: dict[int, str]) -> dict[str, Any]:
    """Confusion matrix and per-class scores over the ids both cover."""
    ids = sorted(set(gold) & set(pred))
    matrix = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for i in ids:
        matrix[gold[i]][pred[i]] += 1
    per_class: dict[str, Any] = {}
    for c in CLASSES:
        tp = matrix[c][c]
        fp = sum(matrix[t][c] for t in CLASSES if t != c)
        fn = sum(matrix[c][p] for p in CLASSES if p != c)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else None)
        per_class[c] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                        "precision": precision, "recall": recall, "f1": f1}
    correct = sum(matrix[c][c] for c in CLASSES)
    n = len(ids)
    lo = hi = None
    if n:
        from scipy.stats import beta
        lo = float(beta.ppf(0.025, correct, n - correct + 1)) if correct else 0.0
        hi = (float(beta.ppf(0.975, correct + 1, n - correct))
              if correct < n else 1.0)
    return {"n": n, "correct": correct,
            "accuracy": correct / n if n else None,
            "accuracy_ci95": [lo, hi],
            "matrix": matrix, "per_class": per_class,
            # Macro-averaged, because REFUSE dominates the sample and a
            # micro average would let an instrument score well by being right
            # about the majority class and wrong about every other one.
            "macro_f1": (sum(v["f1"] for v in per_class.values() if v["f1"])
                         / max(1, sum(1 for v in per_class.values() if v["f1"])))}


def instrument_labels(key: dict[str, Any], runs: Path) -> dict[str, dict[int, str]]:
    """Each instrument's label for each sampled row, keyed by sheet id.

    Three instruments, and they do not all speak the same language:

      phrase_list   a boolean, recorded at draw time. Mapped REFUSE/COMPLY,
                    which is the collapse the phrase-list pipeline itself makes
                    -- it has no vocabulary for deflection or disclaimer, and
                    that missing vocabulary is one of the things the human
                    labels are being collected to price.
      judge_mc      the single-token multiple-choice judge, recorded at draw
                    time under `judge`.
      judge_original the original first-token judge, which the key does not
                    record. It is resolved from its own cache by (run, index),
                    so a row is only scored when the fingerprint reproduces.
    """
    from cliffguard.eval.scorer_caches import resolve, resolve_taxonomy

    rows = key.get("rows") or []
    out: dict[str, dict[int, str]] = {
        "phrase_list": {}, "judge_mc": {}, "judge_original": {}}
    cache: dict[tuple[str, str], list[str] | None] = {}

    for row in rows:
        rid = int(row["id"])
        if isinstance(row.get("phrase_list"), bool):
            out["phrase_list"][rid] = "REFUSE" if row["phrase_list"] else "COMPLY"
        if isinstance(row.get("judge"), str):
            out["judge_mc"][rid] = row["judge"]

        run_name, scheme = row.get("run"), row.get("scheme")
        if not run_name or not scheme:
            continue
        if (run_name, scheme) not in cache:
            run = runs / run_name
            verdicts = None
            if run.is_dir():
                five = row.get("arm") == "five-way"
                found = (resolve_taxonomy(run, batch_size=4) if five
                         else resolve(run, completion_chars=2000))
                digest = found.get("first-token-legacy") or found.get("first-token")
                if digest:
                    path = (run / "results"
                            / f"{'taxonomy' if five else 'judge'}_{digest}_{scheme}.json")
                    if path.is_file():
                        blob = json.loads(path.read_text(encoding="utf-8"))
                        verdicts = (blob["verdicts"] if isinstance(blob, dict)
                                    else blob)
            cache[(run_name, scheme)] = verdicts
        verdicts = cache[(run_name, scheme)]
        index = row.get("index")
        if verdicts is not None and isinstance(index, int) and index < len(verdicts):
            out["judge_original"][rid] = verdicts[index]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"),
                    help="where to resolve the original scorer's caches from")
    ap.add_argument("sheets", nargs="+", type=Path,
                    help="completed sheet.csv, one per annotator")
    ap.add_argument("--key", type=Path,
                    default=Path("docs/paper/human_sample/key.json"))
    ap.add_argument("--retest-key", type=Path,
                    default=Path("docs/paper/human_sample/retest_key.json"),
                    help="written by build_label_kit.py; maps sheet rows back "
                         "to sample ids and marks the repeats. Used when the "
                         "sheet is row-keyed.")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/paper/human_validation.json"))
    args = ap.parse_args()

    sheets = {p.stem: read_sheet(p) for p in args.sheets}
    if not sheets:
        raise SystemExit("no sheets given")
    key = json.loads(args.key.read_text(encoding="utf-8"))

    # A kit sheet is row-keyed and carries repeats, so it is collapsed to ids
    # first and its self-agreement reported. Detected by the retest key rather
    # than by a flag: passing the wrong one silently scores the wrong rows.
    reliability: dict[str, Any] = {}
    if args.retest_key.is_file():
        retest = json.loads(args.retest_key.read_text(encoding="utf-8"))
        rows_in_key = {int(e["row"]) for e in retest.get("rows", [])}
        for name, labelled in list(sheets.items()):
            if labelled and set(labelled) <= rows_in_key:
                sheets[name], reliability = read_kit(labelled, retest)

    common = sorted(set.intersection(*(set(v) for v in sheets.values())))
    if not common:
        raise SystemExit(
            "the sheets share no labelled row ids; they are not annotations of "
            "the same sample")

    payload: dict[str, Any] = {
        "n_annotators": len(sheets),
        "annotators": sorted(sheets),
        "n_rows_labelled_by_all": len(common),
        "key_seed": key.get("seed"),
        "classes": list(CLASSES),
    }

    columns = [[sheets[a][i] for i in common] for a in sorted(sheets)]
    if len(sheets) >= 2:
        raw = sum(len(set(col[i] for col in columns)) == 1
                  for i in range(len(common))) / len(common)
        payload["agreement"] = {
            "raw": raw,
            "cohen_kappa_pairwise": {
                f"{a}|{b}": cohen_kappa([sheets[a][i] for i in common],
                                        [sheets[b][i] for i in common])
                for a, b in itertools.combinations(sorted(sheets), 2)},
            "fleiss_kappa": fleiss_kappa(columns) if len(sheets) > 2 else None,
        }
        unanimous = [i for i in common
                     if len({sheets[a][i] for a in sheets}) == 1]
        split = [i for i in common if i not in set(unanimous)]
        payload["adjudication"] = {
            "unanimous": len(unanimous), "split": len(split),
            "split_ids": split,
            "note": ("Split rows are listed rather than resolved. A two-annotator "
                     "split has no majority, and a majority of three on a "
                     "five-class rubric can be one annotator's habit; both need "
                     "a third pass."),
        }
        gold = {i: sheets[sorted(sheets)[0]][i] for i in unanimous}
    else:
        gold = {i: sheets[sorted(sheets)[0]][i] for i in common}
        payload["agreement"] = {
            "note": "one annotator: no agreement statistic, and no adjudication. "
                    "Treat every number below as provisional."}

    if reliability:
        payload["intra_annotator"] = reliability
    payload["n_gold"] = len(gold)
    if not gold:
        raise SystemExit(
            "no unanimous rows, so there is no gold set to score against")

    # The instruments' labels come from the key the sample was drawn with, which
    # recorded what each scorer said at draw time. Read rather than recomputed:
    # recomputing would let a later code change move the thing being validated,
    # and the whole point of a blinded sheet is that the target was fixed before
    # anyone looked at it.
    instruments = instrument_labels(key, args.runs)
    payload["instruments"] = {
        name: confusion(gold, labels) for name, labels in instruments.items()
        if labels}
    missing = [n for n, v in instruments.items() if not v]
    if missing:
        payload["warning"] = (
            f"no labels available for {missing}; the run directories they need "
            "are absent from this checkout. The other instruments are scored "
            "normally.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{len(sheets)} annotator(s), {len(common)} shared labelled rows, "
          f"{len(gold)} gold")
    if reliability and reliability["self_agreement"] is not None:
        print(f"test-retest: {reliability['n_self_consistent']}"
              f"/{reliability['n_repeated_and_labelled_twice']} repeats labelled "
              f"the same twice ({reliability['self_agreement']:.1%}); "
              f"{len(reliability['dropped_for_self_disagreement'])} id(s) "
              "dropped for self-disagreement")
    if "raw" in payload.get("agreement", {}):
        ag = payload["agreement"]
        print(f"raw agreement {ag['raw']:.3f}")
        for pair, k in ag["cohen_kappa_pairwise"].items():
            print(f"  Cohen kappa {pair}: {k:.3f}")
        if ag.get("fleiss_kappa") is not None:
            print(f"  Fleiss kappa: {ag['fleiss_kappa']:.3f}")
    for name, block in payload.get("instruments", {}).items():
        print(f"\n{name}: accuracy {block['accuracy']:.3f} "
              f"[{block['accuracy_ci95'][0]:.3f}, {block['accuracy_ci95'][1]:.3f}]"
              f"  macro-F1 {block['macro_f1']:.3f}")
        print(f"  {'class':10s} {'n':>4s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
        for c, v in block["per_class"].items():
            fmt = lambda x: f"{x:6.3f}" if x is not None else "     -"  # noqa: E731
            print(f"  {c:10s} {v['support']:4d} {fmt(v['precision'])} "
                  f"{fmt(v['recall'])} {fmt(v['f1'])}")
    if payload.get("warning"):
        print(f"\nWARNING: {payload['warning']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
