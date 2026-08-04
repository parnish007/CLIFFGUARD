"""Does the headline survive a judge from a different family?

The paper reports that quantization increases the judge's refusal class. That
rests on one grader. This script takes the verdicts written by
judge_via_api.py and asks three questions of them:

  1. How often does the second judge agree with the 7B one, and on what? A
     single agreement percentage hides the shape of the disagreement, so we
     report the full confusion matrix and per-class recall.

  2. Does the result reproduce? The paired transition counts and the exact
     McNemar test are recomputed using the second judge's labels alone. If the
     direction reverses or the significance evaporates, the finding was a
     property of the grader and the paper must say so.

  3. Where they disagree, who is right? We cannot answer that without human
     labels, but we can hand a reviewer the disagreements sorted for
     inspection, which is what turns "unvalidated" into "here is the evidence".

Agreement is computed only on completions both judges scored, which is the
composite gate's admitted set.

Usage:
  python scripts/compare_judges.py
  python scripts/compare_judges.py --out docs/paper/judge_agreement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_paper_data import MODEL_LABELS, bits_of
from scripts.reanalyse_runs import analyse, load_run
from scripts.review_reanalysis import exact_mcnemar, holm
from scripts.run_behavioural_ladder import is_degenerate

STATES = ("REFUSE", "COMPLY", "UNCLEAR", "UNPARSED")


def discover(run_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Second-judge verdicts, keyed by judge tag then scheme."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for f in sorted((run_dir / "results").glob("judge_api_*.json")):
        stem = f.stem[len("judge_api_"):]
        tag, _, scheme = stem.rpartition("_")
        # Scheme names are FP16 or RTN_<n>B; rpartition splits one token too
        # early for the RTN form, so put it back.
        if scheme.endswith("B") and tag.endswith("_RTN"):
            tag, scheme = tag[: -len("_RTN")], f"RTN_{scheme}"
        out.setdefault(tag, {})[scheme] = json.loads(
            f.read_text(encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/paper/judge_agreement.json"))
    ap.add_argument("--examples", type=int, default=40,
                    help="disagreements to save for inspection")
    args = ap.parse_args()

    report: dict[str, Any] = {}
    any_found = False

    for run_dir in sorted(args.runs.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run is None or not run["judge_raw"]:
            continue
        model = MODEL_LABELS.get(run["manifest"].get("model_id", "?"))
        if model is None or len(run["completions"]["FP16"]) < 150:
            continue
        found = discover(run_dir)
        if not found:
            continue
        any_found = True
        threshold = analyse(run)["threshold"]

        for tag, per_scheme in found.items():
            block = report.setdefault(tag, {}).setdefault(model, {})
            matrix = {a: {b: 0 for b in STATES} for a in STATES}
            examples: list[dict[str, Any]] = []

            # ---- agreement -------------------------------------------------
            for scheme, verdicts in per_scheme.items():
                texts = run["completions"][scheme]
                for key, theirs in verdicts.items():
                    i = int(key)
                    ours = run["judge_raw"][scheme][i]
                    if ours not in STATES or theirs not in STATES:
                        continue
                    matrix[ours][theirs] += 1
                    if ours != theirs and len(examples) < args.examples:
                        examples.append({
                            "scheme": scheme, "index": i,
                            "seven_b": ours, "second_judge": theirs,
                            "completion": texts[i][:220],
                        })
            total = sum(matrix[a][b] for a in STATES for b in STATES)
            agree = sum(matrix[a][a] for a in STATES)
            block["n_compared"] = total
            block["agreement"] = (agree / total) if total else None
            block["confusion"] = matrix
            block["recall_by_class"] = {
                a: (matrix[a][a] / sum(matrix[a].values()))
                if sum(matrix[a].values()) else None for a in STATES}
            block["disagreement_examples"] = examples

            # ---- does the finding reproduce? -------------------------------
            if "FP16" not in per_scheme:
                block["note"] = ("FP16 not graded by this judge; the paired "
                                 "comparison cannot be recomputed")
                continue

            completions: dict[str, list[str]] = run["completions"]
            nll: dict[str, Any] = run["nll"]

            def label(scheme: str, i: int,
                      _per: dict[str, dict[str, str]] = per_scheme,
                      _c: dict[str, list[str]] = completions,
                      _n: dict[str, Any] = nll, _t: float = threshold) -> str:
                text, value = _c[scheme][i], float(_n[scheme][i])
                if is_degenerate(text, value, _t):
                    return "degenerate"
                verdict = _per[scheme].get(str(i))
                return {"REFUSE": "refusal", "COMPLY": "compliance"}.get(
                    verdict or "", "unclear")

            n = len(run["completions"]["FP16"])
            rows, raw_p = [], []
            for scheme in sorted(per_scheme, key=lambda s: -bits_of(s)):
                if scheme == "FP16":
                    continue
                to_comply = to_refuse = 0
                for i in range(n):
                    base, cur = label("FP16", i), label(scheme, i)
                    to_comply += base == "refusal" and cur == "compliance"
                    to_refuse += base == "compliance" and cur == "refusal"
                p = exact_mcnemar(to_comply, to_refuse)
                rows.append({"scheme": scheme, "bits": bits_of(scheme),
                             "to_compliance": to_comply, "to_refusal": to_refuse,
                             "mcnemar_p": p})
                raw_p.append(p)
            for row, adj in zip(rows, holm(raw_p)):
                row["mcnemar_p_holm"] = adj
            block["reproduction"] = rows

    if not any_found:
        print("No second-judge verdicts found. Run scripts/judge_via_api.py first.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for tag, models in report.items():
        print(f"\n=== {tag} vs the 7B judge ===")
        for model, block in models.items():
            pct = block["agreement"]
            print(f"{model}: {block['n_compared']} compared, "
                  f"agreement {100 * pct:.1f}%" if pct is not None else model)
            for state, recall in block["recall_by_class"].items():
                if recall is not None:
                    print(f"    7B said {state:9s} -> second judge agrees "
                          f"{100 * recall:5.1f}%")
            for row in block.get("reproduction", []):
                direction = ("toward refusal" if row["to_refusal"] > row["to_compliance"]
                             else "toward compliance")
                print(f"    {row['bits']:.1f} bits: "
                      f"{row['to_refusal']} newly refusing vs "
                      f"{row['to_compliance']} newly complying, {direction}, "
                      f"p_holm={row['mcnemar_p_holm']:.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
