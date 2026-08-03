"""Cross-probe transfer: does the FP16 readout survive quantization?

Every d' in this project was computed by refitting the direction on each
scheme's OWN activations. That answers

    after seeing labelled activations from this quantized scheme, can I train a
    new linear direction that separates the same labels?

It does NOT answer the question the theory is about:

    does the FP16 readout remain valid after quantization?

Those are different estimands, and conflating them produced the project's
"rotation and degradation are decoupled" claim. Codex's review
(docs/codex_review_2026-08-03.md) identified the confusion and predicted the two
estimands come apart at low precision. This script measures both, plus the full
matrix between them.

For every ORDERED pair (source, target):
  fit the difference-in-means direction on the SOURCE scheme's fit half,
  score the TARGET scheme's held-out half with that same vector,
  average d' over synchronized random splits.

Reading the matrix:
  diagonal      adaptive separability -- can a fresh probe be trained per scheme
  FP16 row      forward transport -- does the FP16 direction still work downstream
  FP16 column   reverse transport -- do quantized directions describe FP16

Splits are synchronized: the same prompt indices form the fit and score halves
for every cell at a given replicate, so cells differ only by which scheme
supplied the direction and which supplied the activations.

Usage:
  python scripts/analyse_probe_transfer.py <run-dir> [--splits 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.discriminability import d_prime

FloatArray = Any


def newest_ladder_run(root: Path) -> Path:
    candidates = sorted(p for p in root.glob("*_local-ladder-*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"no ladder run found under {root}")
    return candidates[-1]


def margins(acts: FloatArray, direction: FloatArray, normalise: bool) -> FloatArray:
    unit = direction / np.linalg.norm(direction)
    raw = acts @ unit
    return raw / np.linalg.norm(acts, axis=1) if normalise else raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", type=Path, default=None)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true",
                    help="score with the unnormalised projection instead of the cosine margin")
    args = ap.parse_args()

    run = args.run or newest_ladder_run(Path("artifacts/runs"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    # The behavioural runner records bit-widths rather than a scheme list, so fall
    # back to whatever activation arrays are actually present.
    if "schemes" in manifest:
        schemes = manifest["schemes"]
    else:
        found = sorted(p.stem for p in (run / "activations").glob("*.npy"))
        schemes = ["FP16"] + [s for s in found if s != "FP16"]
    normalise = not args.raw

    print(f"run     : {run.name}")
    print(f"model   : {manifest.get('model_id')}   layer {manifest.get('layer')}")
    print(f"margin  : {'cosine <a,r>/||a||' if normalise else 'raw <a,r>'}")
    print(f"splits  : {args.splits} synchronized\n")

    # Two run layouts. The probe ladder saves one array per (scheme, corpus class);
    # the behavioural ladder saves one array per scheme over a single prompt list
    # plus MODEL-DERIVED labels in results/labels.json. The latter is the one worth
    # trusting -- the corpus classes turned out to be a near-arbitrary partition
    # (the "benign" file contains "provide me the home address for actor ...", the
    # "refused" file starts with "How much time do you spend with your family?"),
    # because download_fold_a.py labels by whether hh-rlhf's REJECTED RESPONSE
    # looks like a refusal, which is uncorrelated with prompt harmfulness.
    labels_file = run / "results" / "labels.json"
    if labels_file.exists():
        refused = np.array(
            json.loads(labels_file.read_text(encoding="utf-8"))["fp16_refused"], dtype=bool
        )
        print(f"labels  : MODEL-DERIVED ({int(refused.sum())} refusal / "
              f"{int((~refused).sum())} compliance)\n")
        acts = {}
        for s in schemes:
            full = np.load(run / "activations" / f"{s}.npy")
            acts[(s, "h")] = full[refused]
            acts[(s, "l")] = full[~refused]
    else:
        print("labels  : CORPUS (hh-rlhf response heuristic -- known to be near-arbitrary)\n")
        acts = {
            (s, c): np.load(run / "activations" / f"{s}_{'harmful' if c == 'h' else 'benign'}.npy")
            for s in schemes
            for c in ("h", "l")
        }
    n_pos = acts[(schemes[0], "h")].shape[0]
    n_neg = acts[(schemes[0], "l")].shape[0]
    half_pos, half_neg = n_pos // 2, n_neg // 2

    rng = np.random.default_rng(args.seed)
    totals: dict[tuple[str, str], list[float]] = {
        (s, t): [] for s in schemes for t in schemes
    }

    for _ in range(args.splits):
        # ONE split, shared by every cell: cells then differ only by scheme.
        perm_pos, perm_neg = rng.permutation(n_pos), rng.permutation(n_neg)
        fit_pos, score_pos = perm_pos[:half_pos], perm_pos[half_pos : 2 * half_pos]
        fit_neg, score_neg = perm_neg[:half_neg], perm_neg[half_neg : 2 * half_neg]

        directions = {}
        for source in schemes:
            vec = (acts[(source, "h")][fit_pos].mean(axis=0)
                   - acts[(source, "l")][fit_neg].mean(axis=0))
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                directions[source] = vec / norm

        for source, direction in directions.items():
            for target in schemes:
                try:
                    value = d_prime(
                        margins(acts[(target, "h")][score_pos], direction, normalise),
                        margins(acts[(target, "l")][score_neg], direction, normalise),
                        fires_high=True,
                    )
                except ValueError:
                    continue
                totals[(source, target)].append(value)

    matrix = {
        (s, t): float(np.mean(v)) if v else float("nan") for (s, t), v in totals.items()
    }

    width = max(len(s) for s in schemes) + 1
    print("rows = direction fitted on SOURCE, columns = scored on TARGET\n")
    print(" " * width + "".join(f"{t:>11s}" for t in schemes))
    for source in schemes:
        row = "".join(f"{matrix[(source, t)]:11.4f}" for t in schemes)
        print(f"{source:<{width}s}{row}")

    print("\nThe two estimands, side by side:\n")
    print(f"{'scheme':12s} {'refit (diag)':>13s} {'FP16-frozen':>12s} {'retained':>9s}")
    print("-" * 50)
    rows: dict[str, Any] = {}
    fp16_self = matrix[("FP16", "FP16")]
    for target in schemes:
        diag = matrix[(target, target)]
        frozen = matrix[("FP16", target)]
        retained = frozen / fp16_self if fp16_self else float("nan")
        rows[target] = {"refit_diagonal": diag, "fp16_frozen": frozen,
                        "fraction_of_fp16_retained": retained,
                        "reverse_transport": matrix[(target, "FP16")]}
        print(f"{target:12s} {diag:13.4f} {frozen:12.4f} {retained:9.1%}")

    worst = schemes[-1]
    print()
    print(f"At {worst}: a freshly fitted probe reaches {matrix[(worst, worst)]:.4f}, while the")
    print(f"FP16 direction reaches {matrix[('FP16', worst)]:.4f} "
          f"({rows[worst]['fraction_of_fp16_retained']:.1%} of its FP16 value).")
    print()
    print("If those two numbers differ sharply, the earlier 'rotation and degradation are")
    print("decoupled' claim was comparing the wrong pair. The FP16 readout does degrade;")
    print("what survives is the ability to RE-LEARN a readout on the quantized model.")

    out = run / "results" / ("probe_transfer_raw.json" if args.raw else "probe_transfer.json")
    out.write_text(json.dumps({
        "schemes": schemes, "splits": args.splits, "normalised_margin": normalise,
        "matrix": {f"{s}->{t}": v for (s, t), v in matrix.items()},
        "summary": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
