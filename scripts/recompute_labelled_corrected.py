"""The five-way claims, recomputed under the corrected label scorer.

Section 5.2 argues that one refusal-phrase list reads three model families at
wildly different coverage against a five-way judge that places all three in a
declining class at nearly the same rate, and that the two label sets nest --
precision 1.000, so the phrase list never flags anything the judge does not
also call declining. Both claims are statements about the judge's labels, and
correcting the label scorer moved 38 to 55 of every 300 five-way verdicts.

So neither claim can be inherited. This recomputes the coverage ratio, the
precision, the nesting property and the declining totals under both scorers, on
the same completions, so the paper can say which of them survived rather than
assuming they did.

Only full precision is recomputed, because only full precision has been
re-graded under the corrected scorer. The seven quantized rungs still exist
only under the original one, and this script says so rather than quietly
reporting a partial table as a whole one.

Usage:
  python scripts/recompute_labelled_corrected.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.scorer_caches import resolve_taxonomy, unresolved_taxonomy
from scripts.classify_completion_taxonomy import gate_mask
from scripts.classify_completion_taxonomy import resolve as resolve_classes
from scripts.classify_completions_judge import has_refusal_marker
from scripts.reanalyse_runs import load_run

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "artifacts" / "runs"

MODELS = (
    ("Qwen2.5-3B", "*lab-qwen3b-xstest"),
    ("Phi-3.5-mini", "*lab-phi35-xstest"),
    ("SmolLM2-1.7B", "*lab-smol17-xstest"),
)
DECLINING = ("refusal", "deflection", "disclaimer")
SCORERS = ("first-token", "letter")


def find(pattern: str) -> Path:
    hits = sorted(RUNS.glob(pattern))
    if not hits:
        raise SystemExit(f"no run matching {pattern}")
    return hits[-1]


def measure(labels: np.ndarray, completions: list[str],
            harm: list[str]) -> dict[str, Any]:
    """Coverage, precision and nesting for one label vector.

    Precision and the miss count are the quantities the paper argues from,
    because they do not presuppose the judge is right: they say the two label
    sets nest and that the phrase list's shortfall runs one way. Recall is
    reported beside them and is the number that does assume a standard.
    """
    declining = {i for i, lab in enumerate(labels) if lab in DECLINING}
    flagged = {i for i, text in enumerate(completions) if has_refusal_marker(text)}
    n = len(labels)
    tp = len(flagged & declining)
    fp = len(flagged - declining)
    fn = len(declining - flagged)
    tn = n - tp - fp - fn
    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    harm_arr = np.asarray(harm)
    return {
        "n_prompts": n,
        "marker_flagged": len(flagged),
        "judge_declining": len(declining),
        "recall": tp / len(declining) if declining else None,
        "precision": tp / len(flagged) if flagged else None,
        "missed_by_marker": fn,
        "flagged_but_not_declining": fp,
        "kappa": ((observed - expected) / (1 - expected)
                  if expected < 1 else None),
        "harmful_compliance": int(
            ((harm_arr == "harmful") & (labels == "compliance")).sum()),
        "class_counts": {c: int((labels == c).sum())
                         for c in sorted(set(labels.tolist()))},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / "paper" / "labelled_corrected.json")
    args = ap.parse_args()

    out: dict[str, Any] = {
        "scope": ("full precision only. The seven quantized rungs have not been "
                  "re-graded under the corrected scorer, so every rung-level "
                  "five-way result in the manuscript remains an original-scorer "
                  "result and is labelled as one."),
        "models": {},
    }

    for model, pattern in MODELS:
        run_dir = find(pattern)
        modes = resolve_taxonomy(run_dir)
        results = run_dir / "results"
        stored = json.loads((results / "prompts.json").read_text(encoding="utf-8"))
        harm = stored["harm_label"]
        summary = json.loads(
            (results / "completion_taxonomy.json").read_text(encoding="utf-8"))
        threshold = float(summary["degeneracy_threshold"])

        block: dict[str, Any] = {
            "run": run_dir.name,
            "fingerprints": modes,
            "unresolved": sorted(unresolved_taxonomy(run_dir, modes)),
            "scorers": {},
        }
        for scorer in SCORERS:
            digest = modes.get(scorer)
            if digest is None:
                continue
            path = results / f"taxonomy_{digest}_FP16.json"
            if not path.is_file():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            verdicts = raw.get("verdicts") if isinstance(raw, dict) else raw
            loaded = load_run(run_dir, scorer=f"collapsed{digest}")
            if loaded is None:
                continue
            completions = loaded["completions"]["FP16"]
            gradable = gate_mask(completions, loaded["nll"]["FP16"], threshold)
            labels = resolve_classes(list(verdicts), gradable)
            block["scorers"][scorer] = measure(labels, completions, harm)
        out["models"][model] = block

    # The coverage ratio the paper leads with: the spread in how much of each
    # family's declining behaviour one fixed phrase list detects.
    for scorer in SCORERS:
        recalls = {m: b["scorers"][scorer]["recall"]
                   for m, b in out["models"].items()
                   if scorer in b["scorers"]
                   and b["scorers"][scorer]["recall"] is not None}
        if len(recalls) >= 2 and min(recalls.values()) > 0:
            out.setdefault("coverage_spread", {})[scorer] = {
                "recall": recalls,
                "ratio_max_over_min": max(recalls.values()) / min(recalls.values()),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}\n")

    print(f"{'model':14s} {'scorer':12s} {'declining':>9s} {'marker':>7s} "
          f"{'recall':>7s} {'prec':>6s} {'missed':>7s} {'harm-comply':>11s}")
    print("-" * 82)
    for model, block in out["models"].items():
        for scorer, m in block["scorers"].items():
            recall = "--" if m["recall"] is None else f"{100 * m['recall']:.1f}%"
            precision = "--" if m["precision"] is None else f"{m['precision']:.3f}"
            print(f"{model:14s} {scorer:12s} {m['judge_declining']:9d} "
                  f"{m['marker_flagged']:7d} {recall:>7s} {precision:>6s} "
                  f"{m['missed_by_marker']:7d} {m['harmful_compliance']:11d}")
    print()
    for scorer, spread in out.get("coverage_spread", {}).items():
        print(f"{scorer:12s} coverage spread "
              f"{spread['ratio_max_over_min']:.1f}x  "
              f"({', '.join(f'{k} {100 * v:.1f}%' for k, v in spread['recall'].items())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
