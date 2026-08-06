"""Consolidate every measured number the paper quotes into one JSON file.

Nothing in the paper is typed by hand. Figures and tables are generated from this
file, which is itself derived from the immutable run directories, so a claim in
the manuscript can always be traced back to the run that produced it.

Sources:
  behavioural runs  completions + NLL + judge verdicts -> refusal/compliance/degenerate
  sector runs       GSM8K accuracy against gold answers
  probe transfer    frozen-FP16 vs per-scheme-refit d'

Usage:
  python scripts/build_paper_data.py --runs artifacts/runs --out docs/paper/data.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reanalyse_runs import analyse, load_run, select_runs

# Local runs used the 1.5B model; the hosted run added these two.
MODEL_LABELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-mini",
}


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval. Normal approximations are not usable here:
    several accuracy cells are 0/200, where a Wald interval has zero width."""
    alpha = 1.0 - confidence
    low = float(beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    high = float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return low, high


# Scheme prefixes that sit on the ordinal bit-width axis: one quantizer family,
# one group size, bit-width the only thing that varies. That is what makes the
# axis a dose and the drift slope a slope. RTN is the shipped ladder; SAL is the
# same code path with AWQ-style scaling, run as its own ladder rather than mixed
# into one.
_ORDINAL_PREFIXES = ("RTN_", "SAL_")


def bits_of(scheme: str, group: int = 64) -> float:
    """Position on the RTN ordinal axis, or NaN for schemes that have none.

    NF4, GGUF k-quants and deployed AWQ/GPTQ checkpoints all have a known bit
    budget, and none of them belongs on this axis: each changes the quantization
    ALGORITHM as well as the bit-width, so a line drawn through them is not a
    dose-response curve. They get NaN, and every fit filters on `isfinite`.

    The name alone used to decide this -- anything matching `<word>_<digits>B`
    was accepted -- which silently placed `AWQ_4B` at 4.5 bits, one rung of the
    RTN ladder, and let a different quantizer into the drift fit.
    """
    if scheme == "FP16":
        return 16.0
    if not scheme.startswith(_ORDINAL_PREFIXES):
        return float("nan")
    parts = scheme.split("_")
    if len(parts) < 2 or not parts[1][:-1].isdigit():
        return float("nan")
    return float(int(parts[1][:-1])) + 32.0 / group


def collect_behavioural(runs: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run_dir in runs:
        run = load_run(run_dir)
        if run is None:
            continue
        model = run["manifest"].get("model_id", "?")
        label = MODEL_LABELS.get(model, model)
        result = analyse(run)
        rows = []
        for scheme in result["schemes"]:
            r = result["rows"][scheme]
            rows.append({
                "scheme": scheme, "bits": bits_of(scheme),
                "refusal": r["refusal_rate"], "compliance": r["compliance_rate"],
                "degenerate": r["degenerate_rate"],
                "unsafe_judge": r["unsafe_flip_rate"],
                "median_nll": r["median_nll"],
                "distinct_trigram": r["distinct_trigram"],
                "max_token_share": r["max_token_share"],
                "alpha_fraction": r["alpha_fraction"],
            })
        # Marker-list rate straight from the run's own judge report, so the
        # artifact magnitude is the one the pipeline actually produced.
        judge_file = run_dir / "results" / "judge_classification.json"
        if judge_file.exists():
            meta = json.loads(judge_file.read_text(encoding="utf-8"))
            for row in rows:
                jr = meta["results"].get(row["scheme"], {})
                row["unsafe_marker"] = jr.get("unsafe_flip_rate_marker_tight")
            out.setdefault(label, {})["marker_variants"] = meta.get(
                "marker_variant_sensitivity", {})
            out[label]["judge_model"] = meta.get("judge_model")
        out.setdefault(label, {}).update({
            "model_id": model, "n_prompts": result["n_prompts"],
            "baseline_refusals": result["baseline_refusals"], "rows": rows,
            "run": run_dir.name,
        })
    return out


def collect_sector(runs: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run_dir in runs:
        f = run_dir / "results" / "sector_gsm8k.json"
        if not f.exists():
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        label = MODEL_LABELS.get(manifest["model_id"], manifest["model_id"])
        data = json.loads(f.read_text(encoding="utf-8"))
        n = int(manifest["n_problems"])
        rows = []
        for scheme, v in data.items():
            k = int(v["n_correct"])
            low, high = clopper_pearson(k, n)
            rows.append({
                "scheme": scheme, "bits": v["bits_per_param"],
                "accuracy": v["accuracy"], "n_correct": k,
                "ci_low": low, "ci_high": high,
                "degenerate": v["degenerate_rate"],
            })
            # No p-value is emitted here. This file previously carried an exact
            # binomial test of each rung against the observed FP16 accuracy, as
            # though that were a known population rate; the questions are the
            # same at every rung, so the comparison is paired and that test was
            # invalid. Significance for the capability arm comes from
            # scripts/review_reanalysis.py, which runs exact McNemar on
            # question-level transitions with a Holm correction. Leaving the old
            # number here would let it be quoted again by accident.
        rows.sort(key=lambda r: -r["bits"])
        out[label] = {"model_id": manifest["model_id"], "n": n,
                      "fp16_accuracy": data["FP16"]["accuracy"],
                      "rows": rows, "run": run_dir.name}
    return out


def collect_transfer(runs: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run_dir in runs:
        f = run_dir / "results" / "probe_transfer.json"
        if not f.exists():
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        label = MODEL_LABELS.get(manifest.get("model_id", "?"), manifest.get("model_id", "?"))
        data = json.loads(f.read_text(encoding="utf-8"))
        rows = [
            {"scheme": s, "bits": bits_of(s),
             "refit": v["refit_diagonal"], "frozen": v["fp16_frozen"],
             "retained": v["fraction_of_fp16_retained"]}
            for s, v in data["summary"].items()
        ]
        rows = [r for r in rows if r["bits"] == r["bits"]]      # drop NaN (non-ladder)
        rows.sort(key=lambda r: -r["bits"])
        out[label] = {"rows": rows, "run": run_dir.name}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/data.json"))
    ap.add_argument("--min-n", type=int, default=150,
                    help="drop smoke runs; the paper quotes full-size runs only")
    ap.add_argument("--include", default=None,
                    help="glob on the run directory NAME; only matching runs are read")
    ap.add_argument("--exclude", default=None,
                    help="glob on the run directory NAME to skip, e.g. '*r2-*'")
    args = ap.parse_args()

    # Two rounds of runs share one artifacts tree, and a later directory
    # overwrites an earlier one for the same model. Which round the paper is
    # built from has to be said, not inferred from sort order.
    runs = select_runs(args.runs, args.include, args.exclude)
    behavioural = {k: v for k, v in collect_behavioural(runs).items()
                   if v.get("n_prompts", 0) >= args.min_n}
    sector = {k: v for k, v in collect_sector(runs).items()
              if v.get("n", 0) >= args.min_n}
    # Transfer runs carry no sample size of their own; keep only models that
    # survived the behavioural filter so the paper never mixes a smoke run in.
    transfer = {k: v for k, v in collect_transfer(runs).items()
                if k in behavioural or k in sector}
    payload = {"behavioural": behavioural, "sector": sector, "transfer": transfer}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for section, block in payload.items():
        print(f"{section}: {sorted(block)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
