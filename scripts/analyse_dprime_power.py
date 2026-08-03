"""How large a d' change could this design actually have detected?

"d' did not move" is only informative if the design could have seen it move. The
ladder run reports a FP16 -> 2-bit change of 0.0215 d' units against a per-scheme
standard deviation of 0.085, and stops there. That is a qualitative statement
dressed as a quantitative one: without a minimum detectable effect, a flat curve
is indistinguishable from a blind instrument.

This script turns the null into a bounded claim of the form

    "degradation larger than X d' units is ruled out at 80 % power"

by paired bootstrap over prompts. Prompt indices are resampled ONCE per replicate
and applied to every scheme, because the schemes score the identical prompt list
and their estimator errors are strongly correlated. Resampling them independently
would inflate the variance of the difference and overstate the MDE -- the same
mistake that made an earlier version of the Stage 0 gate powerless.

Usage:
  python scripts/analyse_dprime_power.py                 # newest ladder run
  python scripts/analyse_dprime_power.py <run-dir> --bootstrap 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.discriminability import d_prime, held_out_d_prime

FloatArray = Any


def newest_ladder_run(root: Path) -> Path:
    candidates = sorted(p for p in root.glob("*_local-ladder-*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"no ladder run found under {root}")
    return candidates[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--splits", type=int, default=12,
                    help="held-out splits per bootstrap replicate (kept low; the outer "
                         "bootstrap already averages over split noise)")
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = args.run or newest_ladder_run(Path("artifacts/runs"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if "schemes" in manifest:
        schemes = manifest["schemes"]
    else:
        found = sorted(p.stem for p in (run / "activations").glob("*.npy"))
        schemes = ["FP16"] + [s for s in found if s != "FP16"]
    baseline = "FP16"
    print(f"run    : {run.name}")
    print(f"model  : {manifest.get('model_id')}   layer {manifest.get('layer')}   "
          f"n={manifest.get('n_per_class')}/class")
    print(f"design : {args.bootstrap} paired bootstrap replicates, "
          f"{args.power:.0%} power, alpha={args.alpha}")
    print()

    # Prefer MODEL-DERIVED labels when the run has them. The hh-rlhf corpus
    # partition was shown to agree with the model's own behaviour only 52.4% of
    # the time, so a power analysis against it measures sensitivity to a random
    # partition. See docs/corrections_2026-08-03.md.
    labels_file = run / "results" / "labels.json"
    if labels_file.exists():
        refused = np.array(
            json.loads(labels_file.read_text(encoding="utf-8"))["fp16_refused"], dtype=bool
        )
        print(f"labels : MODEL-DERIVED ({int(refused.sum())} refusal / "
              f"{int((~refused).sum())} compliance)")
        acts = {}
        for s in schemes:
            full = np.load(run / "activations" / f"{s}.npy")
            acts[(s, "h")] = full[refused]
            acts[(s, "l")] = full[~refused]
    else:
        print("labels : CORPUS (hh-rlhf response heuristic -- near-arbitrary; see corrections)")
        acts = {
            (s, c): np.load(run / "activations" / f"{s}_{'harmful' if c == 'h' else 'benign'}.npy")
            for s in schemes
            for c in ("h", "l")
        }
    n_pos = acts[(baseline, "h")].shape[0]
    n_neg = acts[(baseline, "l")].shape[0]

    # (z_{1-alpha/2} + z_{power}) is the standard two-sided normal-approximation
    # multiplier for a minimum detectable effect.
    multiplier = float(norm.ppf(1.0 - args.alpha / 2.0) + norm.ppf(args.power))

    rng = np.random.default_rng(args.seed)
    diffs: dict[str, list[float]] = {s: [] for s in schemes if s != baseline}
    point: dict[str, float] = {}

    def paired_d_prime(scheme: str, fit_p: FloatArray, fit_n: FloatArray,
                       score_p: FloatArray, score_n: FloatArray) -> float:
        direction = (acts[(scheme, "h")][fit_p].mean(axis=0)
                     - acts[(scheme, "l")][fit_n].mean(axis=0))
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("zero-norm direction")
        unit = direction / norm

        def margin(rows: FloatArray) -> FloatArray:
            return (rows @ unit) / np.linalg.norm(rows, axis=1)

        return d_prime(
            margin(acts[(scheme, "h")][score_p]),
            margin(acts[(scheme, "l")][score_n]),
            fires_high=True,
        )

    for replicate in range(args.bootstrap):
        # Split into disjoint fit/score halves FIRST, then bootstrap within each
        # half separately.
        #
        # The obvious ordering -- resample with replacement, then half-split --
        # is wrong, and it was the first version of this script. Duplicate copies
        # of the same original prompt land in both halves, so the direction is
        # fitted on prompts it is then scored against. That leaks training data
        # into test and makes the MDE optimistic. Found by adversarial review;
        # the numbers from the old version are not quoted anywhere.
        perm_pos, perm_neg = rng.permutation(n_pos), rng.permutation(n_neg)
        half_pos, half_neg = n_pos // 2, n_neg // 2
        fit_pos, score_pos = perm_pos[:half_pos], perm_pos[half_pos : 2 * half_pos]
        fit_neg, score_neg = perm_neg[:half_neg], perm_neg[half_neg : 2 * half_neg]

        boot_fit_pos = fit_pos[rng.integers(0, len(fit_pos), len(fit_pos))]
        boot_fit_neg = fit_neg[rng.integers(0, len(fit_neg), len(fit_neg))]
        boot_score_pos = score_pos[rng.integers(0, len(score_pos), len(score_pos))]
        boot_score_neg = score_neg[rng.integers(0, len(score_neg), len(score_neg))]

        try:
            base_d = paired_d_prime(
                baseline, boot_fit_pos, boot_fit_neg, boot_score_pos, boot_score_neg
            )
        except ValueError:
            continue
        for scheme in diffs:
            try:
                scheme_d = paired_d_prime(
                    scheme, boot_fit_pos, boot_fit_neg, boot_score_pos, boot_score_neg
                )
            except ValueError:
                continue
            diffs[scheme].append(base_d - scheme_d)
        if (replicate + 1) % 50 == 0:
            print(f"  {replicate + 1}/{args.bootstrap} replicates", flush=True)

    # Point estimates on the real (unresampled) data.
    base_point, _ = held_out_d_prime(
        acts[(baseline, "h")], acts[(baseline, "l")],
        n_splits=50, fires_high=True, seed=args.seed,
    )
    for scheme in diffs:
        scheme_point, _ = held_out_d_prime(
            acts[(scheme, "h")], acts[(scheme, "l")],
            n_splits=50, fires_high=True, seed=args.seed,
        )
        point[scheme] = base_point - scheme_point

    print(f"\nd'_0 (FP16) = {base_point:.4f}\n")
    header = (f"{'scheme':12s} {'observed':>9s} {'sd(diff)':>9s} {'MDE':>7s} "
              f"{'MDE/d0':>7s} {'detectable?':>12s}")
    print(header)
    print("-" * len(header))

    rows: dict[str, Any] = {}
    for scheme in schemes:
        if scheme == baseline or not diffs[scheme]:
            continue
        sd = float(np.std(diffs[scheme], ddof=1))
        mde = multiplier * sd
        observed = point[scheme]
        detectable = abs(observed) >= mde
        rows[scheme] = {
            "observed_drop": observed, "sd_of_difference": sd,
            "mde_at_power": mde, "mde_as_fraction_of_d0": mde / base_point,
            "observed_is_detectable": bool(detectable),
            "n_replicates": len(diffs[scheme]),
        }
        print(f"{scheme:12s} {observed:+9.4f} {sd:9.4f} {mde:7.4f} "
              f"{mde / base_point:7.1%} {'yes' if detectable else 'no':>12s}")

    worst = max(rows, key=lambda s: rows[s]["mde_at_power"]) if rows else None
    if worst:
        mde = rows[worst]["mde_at_power"]
        print(f"\nLargest MDE across schemes: {mde:.4f} d' units "
              f"({mde / base_point:.1%} of d'_0), at {worst}.")
        detected = [s for s, r in rows.items() if r["observed_is_detectable"]]
        null_only = [s for s, r in rows.items() if not r["observed_is_detectable"]]
        print()
        print("How to state this honestly:")
        if detected:
            print(f"  DETECTED degradation at: {', '.join(sorted(detected))}. For these the")
            print("  observed drop exceeds the minimum detectable effect, so it is a positive")
            print("  finding, not a null.")
        for scheme in sorted(null_only):
            row = rows[scheme]
            print(f"  {scheme}: no detected change; degradation above "
                  f"{row['mde_at_power']:.3f} d' units "
                  f"({row['mde_as_fraction_of_d0']:.0%} of d'_0) is ruled out at "
                  f"{args.power:.0%} power.")
        print()
        print("  Each bound applies to THIS READOUT only. A tight bound on probe")
        print("  discriminability is not a bound on behaviour: the same run shows")
        print("  coherent unsafe flips rising to 48% at 3 bits while the probe still")
        print("  retains 82% of d'_0.")

    out = run / "results" / "dprime_power.json"
    out.write_text(json.dumps({
        "baseline": baseline, "d_prime_0": base_point,
        "bootstrap_replicates": args.bootstrap, "splits_per_replicate": args.splits,
        "power": args.power, "alpha": args.alpha, "multiplier": multiplier,
        "schemes": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
