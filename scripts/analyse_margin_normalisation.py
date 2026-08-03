"""Is the flat d' curve an artifact of the norm-normalised margin?

Theorem 1 says quantization inflates variance along the behavioural direction and
that this inflation IS the degradation. The ladder run measured eta growing 7400x
with essentially no change in d', which refutes that mechanism as stated.

One candidate explanation is structural rather than behavioural. The margin used
throughout is

    m(x) = <a(x), r> / ||a(x)||,

which is norm-normalised. If quantization inflates ||a(x)|| and the component
along r roughly in proportion, the ratio absorbs the inflation and d' cannot see
it -- the readout would be blind to exactly the effect Theorem 1 predicts.

This script tests that directly from the saved activations of a completed run.
It costs no GPU time. It compares, per scheme:

  * d' on the normalised margin   <a, r> / ||a||     (what was reported)
  * d' on the raw projection      <a, r>             (no normalisation)
  * how much ||a|| actually moved
  * how much the mean projection actually moved

If the raw-projection d' also stays flat, the normalisation is exonerated and the
refutation of Theorem 1's mechanism stands on its own. If raw d' decays while
normalised d' does not, the reported flatness is an artifact of the readout and
Theorem 1 must be retested with the raw margin.

Usage:
  python scripts/analyse_margin_normalisation.py <run-dir>
  python scripts/analyse_margin_normalisation.py            # newest ladder run
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
from cliffguard.eval.noise_floor import difference_in_means

FloatArray = Any


def newest_ladder_run(root: Path) -> Path:
    candidates = sorted(p for p in root.glob("*_local-ladder-*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"no ladder run found under {root}")
    return candidates[-1]


def held_out_split(
    pos: FloatArray, neg: FloatArray, normalise: bool, n_splits: int, seed: int
) -> tuple[float, float]:
    """Held-out d' with the direction fitted on one half and scored on the other.

    Mirrors cliffguard.eval.discriminability.held_out_d_prime exactly, except that
    the normalisation of the margin is a parameter. Fitting the direction on the
    prompts it then scores is optimistic by roughly 0.13 d' units on this data, so
    an in-sample comparison between the two readouts would not be trustworthy.
    """
    rng = np.random.default_rng(seed)
    n_p, n_n = pos.shape[0], neg.shape[0]
    half_p, half_n = n_p // 2, n_n // 2
    values: list[float] = []
    for _ in range(n_splits):
        perm_p, perm_n = rng.permutation(n_p), rng.permutation(n_n)
        fit_p, score_p = perm_p[:half_p], perm_p[half_p : 2 * half_p]
        fit_n, score_n = perm_n[:half_n], perm_n[half_n : 2 * half_n]
        direction = pos[fit_p].mean(axis=0) - neg[fit_n].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        unit = direction / norm

        def project(acts: FloatArray, u: FloatArray = unit) -> FloatArray:
            raw = acts @ u
            return raw / np.linalg.norm(acts, axis=1) if normalise else raw

        try:
            values.append(
                d_prime(project(pos[score_p]), project(neg[score_n]), fires_high=True)
            )
        except ValueError:
            continue
    if not values:
        raise ValueError("no usable splits")
    return float(np.mean(values)), float(np.std(values))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", type=Path, default=None)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = args.run or newest_ladder_run(Path("artifacts/runs"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    schemes = manifest["schemes"]
    print(f"run    : {run.name}")
    print(f"model  : {manifest.get('model_id')}   layer {manifest.get('layer')}   "
          f"n={manifest.get('n_per_class')}/class")
    print()

    acts = {
        (s, cls): np.load(run / "activations" / f"{s}_{'harmful' if cls == 'h' else 'benign'}.npy")
        for s in schemes
        for cls in ("h", "l")
    }

    baseline_norm = float(np.linalg.norm(acts[("FP16", "l")], axis=1).mean())
    r_fp16 = difference_in_means(acts[("FP16", "h")], acts[("FP16", "l")])
    unit_fp16 = r_fp16 / np.linalg.norm(r_fp16)
    baseline_proj = float((acts[("FP16", "h")] @ unit_fp16).mean())

    header = (f"{'scheme':10s} {'d(norm)':>9s} {'d(raw)':>9s} {'sd(norm)':>9s} {'sd(raw)':>9s} "
              f"{'||a||/fp16':>11s} {'proj/fp16':>10s}")
    print(header)
    print("-" * len(header))

    rows: dict[str, Any] = {}
    for scheme in schemes:
        pos, neg = acts[(scheme, "h")], acts[(scheme, "l")]
        norm_mean, norm_sd = held_out_split(pos, neg, True, args.splits, args.seed)
        raw_mean, raw_sd = held_out_split(pos, neg, False, args.splits, args.seed)

        # Direction of THIS scheme, so the projection tracks the scheme's own readout.
        direction = difference_in_means(pos, neg)
        unit = direction / np.linalg.norm(direction)
        norm_ratio = float(np.linalg.norm(neg, axis=1).mean()) / baseline_norm
        proj_ratio = float((pos @ unit).mean()) / baseline_proj

        rows[scheme] = {"d_prime_normalised": norm_mean, "d_prime_normalised_sd": norm_sd,
                        "d_prime_raw": raw_mean, "d_prime_raw_sd": raw_sd,
                        "activation_norm_ratio_vs_fp16": norm_ratio,
                        "mean_projection_ratio_vs_fp16": proj_ratio}
        print(f"{scheme:10s} {norm_mean:+9.4f} {raw_mean:+9.4f} {norm_sd:9.4f} {raw_sd:9.4f} "
              f"{norm_ratio:11.4f} {proj_ratio:10.4f}")

    fp16_norm = rows["FP16"]["d_prime_normalised"]
    fp16_raw = rows["FP16"]["d_prime_raw"]
    worst = schemes[-1]
    drop_norm = fp16_norm - rows[worst]["d_prime_normalised"]
    drop_raw = fp16_raw - rows[worst]["d_prime_raw"]
    sd_norm = rows[worst]["d_prime_normalised_sd"]
    sd_raw = rows[worst]["d_prime_raw_sd"]

    print()
    print(f"FP16 -> {worst}:")
    print(f"  normalised margin: d' drop {drop_norm:+.4f}  ({drop_norm / sd_norm:+.2f} sd)")
    print(f"  raw projection   : d' drop {drop_raw:+.4f}  ({drop_raw / sd_raw:+.2f} sd)")
    print()
    if abs(drop_raw) > 2.0 * sd_raw and abs(drop_norm) <= 2.0 * sd_norm:
        print("VERDICT: the flat d' curve IS an artifact of the norm-normalised margin.")
        print("         Theorem 1's mechanism must be retested with the raw projection;")
        print("         the reported refutation applies to the readout, not the behaviour.")
    elif abs(drop_raw) <= 2.0 * sd_raw and abs(drop_norm) <= 2.0 * sd_norm:
        print("VERDICT: both readouts are flat. The normalisation is EXONERATED and the")
        print("         refutation of Theorem 1's mechanism stands on its own -- weight-space")
        print("         eta does not reach the behavioural readout by either route.")
    else:
        print("VERDICT: mixed. Read the per-scheme columns before concluding anything.")

    out = run / "results" / "margin_normalisation_check.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
