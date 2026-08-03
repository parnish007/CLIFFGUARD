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

from cliffguard.eval.discriminability import held_out_d_prime

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
    schemes = manifest["schemes"]
    baseline = "FP16"
    print(f"run    : {run.name}")
    print(f"model  : {manifest.get('model_id')}   layer {manifest.get('layer')}   "
          f"n={manifest.get('n_per_class')}/class")
    print(f"design : {args.bootstrap} paired bootstrap replicates, "
          f"{args.power:.0%} power, alpha={args.alpha}")
    print()

    acts = {
        (s, cls): np.load(run / "activations" / f"{s}_{'harmful' if cls == 'h' else 'benign'}.npy")
        for s in schemes
        for cls in ("h", "l")
    }
    n_pos = acts[(baseline, "h")].shape[0]
    n_neg = acts[(baseline, "l")].shape[0]

    # (z_{1-alpha/2} + z_{power}) is the standard two-sided normal-approximation
    # multiplier for a minimum detectable effect.
    multiplier = float(norm.ppf(1.0 - args.alpha / 2.0) + norm.ppf(args.power))

    rng = np.random.default_rng(args.seed)
    diffs: dict[str, list[float]] = {s: [] for s in schemes if s != baseline}
    point: dict[str, float] = {}

    for replicate in range(args.bootstrap):
        # ONE resample, applied to every scheme: the schemes share prompts.
        idx_pos = rng.integers(0, n_pos, n_pos)
        idx_neg = rng.integers(0, n_neg, n_neg)
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            base_d, _ = held_out_d_prime(
                acts[(baseline, "h")][idx_pos], acts[(baseline, "l")][idx_neg],
                n_splits=args.splits, fires_high=True, seed=seed,
            )
        except ValueError:
            continue
        for scheme in diffs:
            try:
                scheme_d, _ = held_out_d_prime(
                    acts[(scheme, "h")][idx_pos], acts[(scheme, "l")][idx_neg],
                    n_splits=args.splits, fires_high=True, seed=seed,
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
        print()
        print("How to state the null honestly:")
        print(f"  A degradation larger than {mde:.3f} d' units ({mde / base_point:.0%} of the")
        print(f"  full-precision value) would have been detected at {args.power:.0%} power.")
        print("  The observed changes are all smaller than that, so the claim is bounded:")
        print("  quantization does not cost more than that much discriminability ON THIS")
        print("  READOUT. It is NOT a claim that nothing changed.")

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
