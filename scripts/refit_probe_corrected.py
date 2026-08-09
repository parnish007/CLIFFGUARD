"""Refit the frozen refusal probe on the corrected scorer's labels.

The probe section argues that the probe must be trained on the same labels the
behavioural arm reports, because a probe fitted on one target and compared
against behaviour measured on another cannot establish a dissociation between
them -- it can only show that two different questions have two different
answers. That argument is right, and correcting the label scorer broke it: the
behavioural arm now reports corrected labels while the probe was still fitted
on the original scorer's.

So the probe is refitted here on the corrected full-precision labels. Nothing
about the probe changes -- same layer, same difference-in-means direction, same
held-out split protocol, same seed. Only the definition of which prompts count
as refusals moves, which is exactly the quantity the correction changed: 409
refusals become 353 on Qwen2.5-3B and 435 become 385 on Phi-3.5-mini.

This runs on a CPU. The activations are already stored per scheme, the labels
are already graded, and a d' is a projection and two moments.

Usage:
  python scripts/refit_probe_corrected.py
  python scripts/refit_probe_corrected.py --out docs/paper/probe_corrected.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.discriminability import held_out_d_prime
from scripts.reanalyse_runs import load_run
from scripts.review_reanalysis import label_matrix

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "artifacts" / "runs"

MODELS = (
    ("Qwen2.5-3B", "*colab-behavioural-qwen3b"),
    ("Phi-3.5-mini", "*colab-behavioural-phi35"),
)
# The scorers to compare. "first-token-legacy" is what the published probe was
# fitted on; "letter" is the corrected one the behavioural arm now reports.
SCORERS = ("first-token-legacy", "letter")
SPLITS, SEED = 20, 0


def find(pattern: str) -> Path:
    hits = sorted(RUNS.glob(pattern))
    if not hits:
        raise SystemExit(f"no run matching {pattern}")
    return hits[-1]


def fp16_refused(run_dir: Path, scorer: str) -> np.ndarray:
    """The probe's positive class: prompts full precision refused.

    Read through the same label_matrix the behavioural arm uses, so the probe's
    target and the reported behaviour cannot drift apart again.
    """
    loaded = load_run(run_dir, scorer=scorer)
    if loaded is None:
        raise SystemExit(f"{run_dir.name}: not loadable")
    return label_matrix(loaded, "composite")["FP16"] == "refusal"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / "paper" / "probe_corrected.json")
    ap.add_argument("--splits", type=int, default=SPLITS)
    args = ap.parse_args()

    out: dict[str, Any] = {
        "protocol": {
            "splits": args.splits, "seed": SEED,
            "direction": "difference in means, fitted on one half, scored on the other",
            "note": ("Only the label definition differs between the two scorer "
                     "blocks. Layer, direction estimator, split count and seed "
                     "are identical, so any change in d' is the target moving."),
        },
        "models": {},
    }

    for model, pattern in MODELS:
        run_dir = find(pattern)
        activations_dir = run_dir / "activations"
        if not activations_dir.is_dir():
            print(f"[skip] {model}: no activations stored")
            continue

        block: dict[str, Any] = {"run": run_dir.name, "scorers": {}}
        for scorer in SCORERS:
            try:
                refused = fp16_refused(run_dir, scorer)
            except SystemExit as error:
                print(f"[skip] {model} / {scorer}: {error}")
                continue

            per_scheme: dict[str, Any] = {}
            for path in sorted(activations_dir.glob("*.npy")):
                scheme = path.stem
                acts = np.load(path)
                if acts.shape[0] != len(refused):
                    per_scheme[scheme] = {
                        "note": (f"{acts.shape[0]} activation rows against "
                                 f"{len(refused)} labels; not paired")}
                    continue
                positive, negative = acts[refused], acts[~refused]
                if len(positive) < 4 or len(negative) < 4:
                    per_scheme[scheme] = {"note": "too few in one class"}
                    continue
                mean, sd = held_out_d_prime(positive, negative,
                                            n_splits=args.splits,
                                            fires_high=True, seed=SEED)
                per_scheme[scheme] = {"d_prime": mean, "sd": sd}

            fp16 = per_scheme.get("FP16", {}).get("d_prime")
            # Retention against this scorer's OWN full-precision d'. Comparing
            # a corrected rung against the original scorer's baseline would mix
            # the two targets, which is the error this script exists to undo.
            for scheme, entry in per_scheme.items():
                if "d_prime" in entry and fp16:
                    entry["retained_vs_fp16"] = entry["d_prime"] / fp16

            block["scorers"][scorer] = {
                "n_refused": int(refused.sum()),
                "n_prompts": int(len(refused)),
                "schemes": per_scheme,
            }
        out["models"][model] = block

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}\n")

    order = ["FP16", "RTN_8B", "RTN_7B", "RTN_6B", "RTN_5B", "RTN_4B",
             "RTN_3B", "RTN_2B"]
    for model, block in out["models"].items():
        print(f"=== {model}")
        header = "  scheme     " + "".join(
            f"{s.replace('first-token-legacy', 'original'):>22s}"
            for s in block["scorers"])
        print(header)
        for scheme in order:
            cells = ""
            for scorer, data in block["scorers"].items():
                entry = data["schemes"].get(scheme, {})
                if "d_prime" in entry:
                    cells += (f"{entry['d_prime']:+8.3f}"
                              f" ({entry.get('retained_vs_fp16', float('nan')):5.0%})"
                              .rjust(22))
                else:
                    cells += f"{'--':>22s}"
            print(f"  {scheme:10s}{cells}")
        for scorer, data in block["scorers"].items():
            print(f"    {scorer:20s} positives {data['n_refused']}/"
                  f"{data['n_prompts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
