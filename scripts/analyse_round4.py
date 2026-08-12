"""Round 4: what the whole ladder looks like once the scorer is the corrected one.

Two questions, and they are not the same question.

  Does the scorer change the answer?      The manuscript's primary quantities --
  the drift slope, the fourteen transition cells, the simultaneous bound -- were
  computed under a scorer the manuscript itself shows is defective, because only
  full precision and the 4.5-bit rung had been re-graded when it was written.
  Round 4 re-graded all eight rungs of both behavioural runs, so the same
  quantities can now be computed twice and compared. They are computed here by
  calling `review_reanalysis.py` under each scorer rather than by reimplementing
  it, so the two sides differ in the grading and in nothing else.

  Does the ASSIGNMENT change the answer?  The corrected scorer replaced label
  words with single-token letters, which fixes the tokenization asymmetry and
  says nothing about whether the judge has a letter preference. Round 4 re-graded
  full precision and the 4.5-bit rung under two permuted assignments. The
  comparison that matters is not whether the class counts move -- they do -- but
  whether the PAIRED contrast moves, because the paired contrast is what every
  claim in the paper is made of.

Those two come apart, and that separation is the result. Permuting the option
order moves absolute refusal rates by around ten points, so an absolute rate
from this instrument is not identified. The within-prompt paired difference
between full precision and a quantized rung moves by one to two points and keeps
its sign and its significance, because the assignment effect is close to common
across the pair and differences it out.

Usage:
  python scripts/analyse_round4.py                 # -> docs/paper/round4_stats.json
  python scripts/analyse_round4.py --out other.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cliffguard.eval.scorer_caches import resolve
from scripts.reanalyse_runs import analyse, load_run
from scripts.review_reanalysis import clopper_pearson_upper, exact_mcnemar
from scripts.run_behavioural_ladder import is_degenerate

REPO = Path(__file__).resolve().parents[1]

# The two behavioural runs the manuscript's primary claims are made on.
BEHAVIOURAL = {"Qwen2.5-3B": "*colab-behavioural-qwen3b",
               "Phi-3.5-mini": "*colab-behavioural-phi35"}

# Round 4 permuted only the rungs the manuscript quotes: a letter preference is
# a property of the judge and the prompt and does not become a different
# property at three bits.
PERMUTED_RUNGS = ("FP16", "RTN_4B")
ORDERS: dict[str, list[str] | None] = {
    "canonical": None,
    "COMPLY,UNCLEAR,REFUSE": ["COMPLY", "UNCLEAR", "REFUSE"],
    "UNCLEAR,COMPLY,REFUSE": ["UNCLEAR", "COMPLY", "REFUSE"],
}

SCORERS = {"original": "first-token-legacy", "corrected": "letter"}

# What the round 4 graders were invoked with, repeated here because the cache
# fingerprint depends on it and a resolver asked for the wrong window reports a
# grading that exists as never run.
JUDGE_COMPLETION_CHARS = 2000


def latest(pattern: str, runs: Path) -> Path | None:
    hits = sorted(runs.glob(pattern))
    return hits[-1] if hits else None


def reanalyse(scorer: str, runs: Path) -> dict[str, Any]:
    """review_reanalysis.py under one scorer, as a subprocess.

    A subprocess rather than an import because that script's numbers are the
    manuscript's numbers, and the way to keep them so is to run the same entry
    point a reader would run. GSM8K is omitted: the capability arm shares no
    input with the behavioural one and needs a corpus a clone may not have.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stats.json"
        proc = subprocess.run(
            [sys.executable, "scripts/review_reanalysis.py",
             "--runs", str(runs), "--include", "*colab-behavioural*",
             "--scorer", scorer, "--out", str(out)],
            cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"review_reanalysis.py failed under scorer {scorer!r}:\n"
                + (proc.stdout + proc.stderr)[-2000:])
        payload: dict[str, Any] = json.loads(
            out.read_text(encoding="utf-8"))
        return payload


def labels_for(run: Path, digest: str, schemes: tuple[str, ...]
               ) -> dict[str, np.ndarray] | None:
    """Gated per-prompt labels for one grading, or None if it is not on disk.

    The gate is recomputed rather than inherited. It reads only likelihood and
    surface statistics, so it is identical across scorers and assignments by
    construction -- which is the point: holding it fixed is what makes the
    comparison a comparison of graders.
    """
    loaded = load_run(run, scorer=digest)
    if loaded is None or not loaded["judge_raw"]:
        return None
    threshold = analyse(loaded)["threshold"]
    out: dict[str, np.ndarray] = {}
    for scheme in schemes:
        verdicts = loaded["judge_raw"].get(scheme)
        if verdicts is None:
            return None
        row = []
        for text, value, verdict in zip(loaded["completions"][scheme],
                                        loaded["nll"][scheme], verdicts):
            value = float(value)
            row.append("degenerate" if is_degenerate(text, value, threshold)
                       else {"REFUSE": "refusal",
                             "COMPLY": "compliance"}.get(verdict, "unclear"))
        out[scheme] = np.array(row)
    return out


def paired_cell(base: np.ndarray, cur: np.ndarray) -> dict[str, Any]:
    """One model x rung cell: the transition counts and what they support."""
    to_compliance = int(((base == "refusal") & (cur == "compliance")).sum())
    to_refusal = int(((base == "compliance") & (cur == "refusal")).sum())
    n = len(base)
    return {
        "n": n,
        "to_compliance": to_compliance,
        "to_refusal": to_refusal,
        "discordant": to_compliance + to_refusal,
        "refusal_pct_base": float(100 * (base == "refusal").mean()),
        "refusal_pct_cur": float(100 * (cur == "refusal").mean()),
        "delta_pp": float(100 * ((cur == "refusal").mean()
                                 - (base == "refusal").mean())),
        "to_compliance_rate_itt": to_compliance / n,
        "upper95_itt": clopper_pearson_upper(to_compliance, n),
        "mcnemar_p": exact_mcnemar(to_compliance, to_refusal),
    }


def paired_scorer_bootstrap(runs: Path, n_boot: int = 10_000,
                            seed: int = 20260812) -> dict[str, Any]:
    """A direct interval on $\\kappa_{letter} - \\kappa_{original}$.

    Two separate intervals do not test a difference. Reporting
    $+1.29\\,[0.86, 1.74]$ against $-0.13\\,[-0.78, 0.50]$ and concluding the
    scorer changed the slope is the overlapping-intervals fallacy, and it is
    worse than usual here: the two gradings are of the SAME completions, so the
    estimates are strongly positively dependent and their separate intervals are
    far wider than the interval on their difference. A reader is entitled to the
    quantity the claim is actually about.

    So the difference is bootstrapped directly, resampling PROMPTS once per
    replicate and recomputing both scorers' band slopes on that same resample.
    The pairing is what makes it a test of the grader rather than of the prompt
    sample: every replicate holds the completions, the gate and the band fixed
    and varies only which prompts are drawn.

    The band is fixed at the point estimate's band and not reselected inside the
    loop, for the reason the manuscript already gives about the band: it is
    chosen from observed degeneracy rates and then held, so an interval that
    reselected it would be answering a different question.
    """
    from scripts.review_reanalysis import ols_slope

    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for model, pattern in BEHAVIOURAL.items():
        run = latest(pattern, runs)
        if run is None:
            continue
        by_scorer: dict[str, dict[str, np.ndarray]] = {}
        for name, mode in SCORERS.items():
            digest = resolve(run, completion_chars=JUDGE_COMPLETION_CHARS).get(mode)
            if digest is None:
                break
            labels = labels_for(run, digest, tuple(
                sorted({p.stem.split("_", 2)[-1]
                        for p in run.glob(f"results/judge_{digest}_*.json")})))
            if labels is None:
                break
            by_scorer[name] = labels
        if len(by_scorer) < 2:
            continue

        # The band the point estimate used, per scorer, recomputed here so this
        # function does not depend on the order review_reanalysis ran in.
        bands = {}
        for name, labels in by_scorer.items():
            band = [s for s in labels if s != "FP16"
                    and (labels[s] == "degenerate").mean() < 0.10]
            bands[name] = sorted(band, key=lambda s: float(s.split("_")[1][:-1]) + 0.5)
        if not all(len(b) >= 2 for b in bands.values()):
            continue

        def kappa(labels: dict[str, np.ndarray], band: list[str],
                  idx: np.ndarray) -> float:
            x = [float(s.split("_")[1][:-1]) + 0.5 for s in band]
            y = [100 * float((labels[s][idx] == "refusal").mean()) for s in band]
            slope, _ = ols_slope(x, y)
            return -slope

        n = len(by_scorer["original"]["FP16"])
        full = np.arange(n)
        point = {name: kappa(by_scorer[name], bands[name], full)
                 for name in by_scorer}
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            diffs[b] = (kappa(by_scorer["corrected"], bands["corrected"], idx)
                        - kappa(by_scorer["original"], bands["original"], idx))
        observed = point["corrected"] - point["original"]
        out[model] = {
            "kappa_original": point["original"],
            "kappa_corrected": point["corrected"],
            "difference": observed,
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            # Share of replicates on the other side of zero, doubled: the
            # bootstrap analogue of a two-sided p-value for "no scorer effect".
            "p_two_sided": float(min(1.0, 2 * min(
                (diffs <= 0).mean(), (diffs >= 0).mean()))),
            "excludes_zero": bool(np.percentile(diffs, 2.5) > 0
                                  or np.percentile(diffs, 97.5) < 0),
            "n_bootstrap": n_boot,
            "band": bands["corrected"],
        }
    return out


def option_order_audit(runs: Path) -> dict[str, Any]:
    """The 4.5-bit cell under each assignment, per model.

    Reported as absolute rates AND as the paired delta, because those two
    behave differently and reporting only the first is what makes an
    order-dependent instrument look unusable rather than usable for one thing
    and not another.
    """
    out: dict[str, Any] = {}
    for model, pattern in BEHAVIOURAL.items():
        run = latest(pattern, runs)
        if run is None:
            continue
        per_order: dict[str, Any] = {}
        for name, order in ORDERS.items():
            # 2000: the window the round 4 graders were invoked with. It only
            # enters the fingerprint when it actually truncates, and today's
            # stored completions are far shorter, so this resolves the same
            # files as any larger value -- but it stops being true the moment a
            # 256-token run is graded, and then the truthful value is the one
            # that resolves.
            digest = resolve(run, completion_chars=JUDGE_COMPLETION_CHARS,
                             letter_order=order).get("letter")
            if digest is None:
                continue
            labels = labels_for(run, digest, PERMUTED_RUNGS)
            if labels is None:
                continue
            per_order[name] = {"fingerprint": digest,
                               **paired_cell(labels["FP16"], labels["RTN_4B"])}
        if not per_order:
            continue
        base = [c["refusal_pct_base"] for c in per_order.values()]
        delta = [c["delta_pp"] for c in per_order.values()]
        out[model] = {
            "run": run.name,
            "orders": per_order,
            # The headline of this block: how far each quantity travels when
            # nothing changes but which letter carries which class.
            "spread": {
                "fp16_refusal_pct_range": float(max(base) - min(base)),
                "fp16_refusal_pct_min": float(min(base)),
                "fp16_refusal_pct_max": float(max(base)),
                "paired_delta_pp_range": float(max(delta) - min(delta)),
                "paired_delta_pp_min": float(min(delta)),
                "paired_delta_pp_max": float(max(delta)),
                "sign_stable": bool(min(delta) >= 0) or bool(max(delta) <= 0),
                "all_significant": all(
                    c["mcnemar_p"] < 0.05 for c in per_order.values()),
                "none_significant": all(
                    c["mcnemar_p"] >= 0.05 for c in per_order.values()),
            },
        }
    return out


def letter_order_files(runs: Path) -> dict[str, Any]:
    """The class-agreement audit the round wrote, summarised.

    Kept separate from the paired analysis above: this measures how often a
    completion keeps its CLASS under a permutation, which is the scorer's
    self-consistency, not the experiment's conclusion.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(runs.glob("letter_order_*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        for run_name, run_block in blob.get("runs", {}).items():
            for order, per_scheme in run_block.get("orders", {}).items():
                for scheme, cell in per_scheme.items():
                    rows.append({
                        "source": path.name, "run": run_name,
                        "five_way": bool(blob.get("five_way")),
                        "order": order, "scheme": scheme,
                        "n": cell["n"], "agreement": cell["agreement"],
                        "moved": cell["moved"]})
    if not rows:
        return {}
    three = [r["agreement"] for r in rows if not r["five_way"]]
    five = [r["agreement"] for r in rows if r["five_way"]]
    return {
        "comparisons": len(rows),
        "mean_agreement": float(np.mean([r["agreement"] for r in rows])),
        "worst_agreement": float(min(r["agreement"] for r in rows)),
        # Split, because they are not the same instrument and averaging them
        # hides that the five-way grader is the unstable one.
        "three_way": {"n": len(three),
                      "mean_agreement": float(np.mean(three)) if three else None,
                      "worst_agreement": float(min(three)) if three else None},
        "five_way": {"n": len(five),
                     "mean_agreement": float(np.mean(five)) if five else None,
                     "worst_agreement": float(min(five)) if five else None},
        "rows": rows,
    }


def scorer_comparison(runs: Path) -> dict[str, Any]:
    """Primary quantities under the original and the corrected scorer."""
    stats = {name: reanalyse(mode, runs) for name, mode in SCORERS.items()}
    drift: dict[str, Any] = {}
    for model in list(BEHAVIOURAL) + ["_pooled"]:
        block: dict[str, Any] = {}
        for name, payload in stats.items():
            cell = payload["drift"].get(model)
            if cell is None:
                continue
            # Per-model cells carry the interval under "bootstrap"; the pooled
            # and difference cells ARE the bootstrap and carry it at the top.
            boot = cell.get("bootstrap", cell)
            block[name] = {
                "kappa": cell.get("kappa", cell.get("estimate")),
                "ci_low": boot.get("ci_low"), "ci_high": boot.get("ci_high"),
                "fp16_refusal_pct": cell.get("fp16_refusal_pct"),
                "band_bits": cell.get("band_bits"),
            }
        if block:
            drift[model] = block

    cells: list[dict[str, Any]] = []
    for model in BEHAVIOURAL:
        by_scorer = {
            name: {r["scheme"]: r for r in payload["transitions"][model]["rows"]}
            for name, payload in stats.items()
            if model in payload.get("transitions", {})}
        if len(by_scorer) < 2:
            continue
        # Identical coverage, or nothing. Skipping the cells one scorer is
        # missing would compare two different families: the Holm adjustment
        # each cell carries was computed over that scorer's own full set of
        # cells, and the ladder-wide maxima below are taken over whatever rows
        # survive. A partial re-grade would then be able to manufacture a
        # significance change, or a lower bound, out of a coverage difference
        # -- and it would do it silently, which is the failure mode this whole
        # round is about.
        missing = set(by_scorer["original"]) ^ set(by_scorer["corrected"])
        if missing:
            raise SystemExit(
                f"{model}: the two scorers cover different rungs "
                f"({sorted(missing)} present under one and not the other). "
                "Re-grade the missing rungs; a cross-scorer comparison over "
                "different cell sets is not a comparison.")
        for scheme in by_scorer["original"]:
            row: dict[str, Any] = {"model": model, "scheme": scheme,
                                   "bits": by_scorer["original"][scheme]["bits"]}
            for name, table in by_scorer.items():
                cell = table[scheme]
                row[name] = {
                    "to_compliance": cell["to_compliance"],
                    "to_refusal": cell["to_refusal"],
                    "rate_itt": cell["rate_itt"],
                    "p_holm_all_cells": cell["mcnemar_p_holm_all_cells"],
                    "upper95_simultaneous": cell["upper95_simultaneous"],
                }
            was = row["original"]["p_holm_all_cells"] < 0.05
            now = row["corrected"]["p_holm_all_cells"] < 0.05
            row["significance"] = ("unchanged" if was == now
                                   else "lost" if was else "gained")
            cells.append(row)

    bounds = {}
    for name, payload in stats.items():
        rows = [r for model in BEHAVIOURAL
                if model in payload.get("transitions", {})
                for r in payload["transitions"][model]["rows"]]
        bounds[name] = {
            "max_rate_itt": max(r["rate_itt"] for r in rows),
            "max_upper95_simultaneous": max(r["upper95_simultaneous"]
                                            for r in rows),
            "n_cells": len(rows),
        }

    return {"drift": drift, "cells": cells, "bounds": bounds,
            "n_significance_changes": sum(
                1 for c in cells if c["significance"] != "unchanged")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/paper/round4_stats.json"))
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "scope": (
            "Round 4 re-graded all eight rungs of both behavioural runs and all "
            "eight of the three XSTest runs under the corrected (single-token "
            "multiple-choice) scorer, and re-graded full precision and the "
            "4.5-bit rung under two permuted option assignments. This file "
            "compares the manuscript's primary quantities across scorers, and "
            "reports how far they travel when only the option assignment "
            "changes."),
        "scorer_comparison": scorer_comparison(args.runs),
        "scorer_difference": paired_scorer_bootstrap(args.runs),
        "option_order": option_order_audit(args.runs),
        "class_agreement": letter_order_files(args.runs),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    sc = payload["scorer_comparison"]
    print("=== drift slope kappa (pp/bit) ===")
    for model, block in sc["drift"].items():
        for name, cell in block.items():
            lo, hi = cell.get("ci_low"), cell.get("ci_high")
            span = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else ""
            print(f"  {model:14s} {name:10s} {cell['kappa']:+.3f} {span}")

    print("\n=== paired bootstrap on the scorer difference in kappa ===")
    for model, cell in payload["scorer_difference"].items():
        print(f"  {model:14s} {cell['difference']:+.3f} "
              f"[{cell['ci_low']:+.3f}, {cell['ci_high']:+.3f}]  "
              f"p={cell['p_two_sided']:.4f}  "
              f"{'excludes 0' if cell['excludes_zero'] else 'includes 0'}")

    print("\n=== transition cells whose significance changed with the scorer ===")
    for cell in sc["cells"]:
        if cell["significance"] == "unchanged":
            continue
        print(f"  {cell['model']:14s} {cell['bits']:4.1f} bits: "
              f"{cell['significance']} "
              f"(p {cell['original']['p_holm_all_cells']:.3f} -> "
              f"{cell['corrected']['p_holm_all_cells']:.3f})")
    print(f"  {sc['n_significance_changes']} of {len(sc['cells'])} cells changed")

    print("\n=== option order: absolute rate against paired difference ===")
    for model, block in payload["option_order"].items():
        s = block["spread"]
        print(f"  {model:14s} FP16 refusal spans "
              f"{s['fp16_refusal_pct_min']:.1f}-{s['fp16_refusal_pct_max']:.1f}% "
              f"({s['fp16_refusal_pct_range']:.1f} pp) while the paired delta "
              f"spans {s['paired_delta_pp_min']:+.2f} to "
              f"{s['paired_delta_pp_max']:+.2f} pp "
              f"({s['paired_delta_pp_range']:.2f} pp)")

    ca = payload["class_agreement"]
    if ca:
        print(f"\n=== class agreement across assignments ({ca['comparisons']} "
              f"comparisons) ===")
        print(f"  overall mean {ca['mean_agreement']:.3f}, "
              f"worst {ca['worst_agreement']:.3f}")
        for key in ("three_way", "five_way"):
            b = ca[key]
            if b["n"]:
                print(f"  {key:10s} n={b['n']:2d} mean {b['mean_agreement']:.3f} "
                      f"worst {b['worst_agreement']:.3f}")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
