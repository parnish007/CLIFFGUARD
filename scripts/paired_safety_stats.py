"""Paired uncertainty for the harmful-compliance rates.

The capability arm reports exact intervals and binomial tests; the safety arm
reported bare percentages. That asymmetry is indefensible -- a 2.2% rate on 500
prompts needs an interval before anyone can say it differs from 0.

Every rung scores the SAME prompts as the FP16 baseline, so the comparison is
paired and the correct test is McNemar's on the discordant pairs, not a
two-sample proportion test. For each rung we report:

  n_unsafe      FP16 refused, this rung complied coherently
  n_safer       FP16 complied, this rung refused  (the reverse discordance)
  rate + CI     Wilson interval on the unsafe-flip rate over all prompts
  McNemar p     exact binomial test on the discordant pairs

Usage:
  python scripts/paired_safety_stats.py --out docs/paper/tables/tab_paired.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import binomtest, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_paper_data import MODEL_LABELS, bits_of
from scripts.reanalyse_runs import analyse, load_run


class Rung(NamedTuple):
    """One ladder rung's paired comparison against the FP16 baseline."""

    scheme: str
    bits: float
    n: int
    unsafe: int          # FP16 refused, this rung complied coherently
    safer: int           # FP16 complied, this rung refused
    rate: float
    ci_low: float
    ci_high: float
    mcnemar_p: float


def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval.

    Preferred over Wald here because several cells are exactly zero, where the
    Wald interval has zero width and would imply impossible certainty.
    """
    if n == 0:
        return 0.0, 0.0
    z = float(norm.ppf(1 - (1 - confidence) / 2))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def labels_for(run: dict[str, Any]) -> dict[str, np.ndarray]:
    result = analyse(run)
    out: dict[str, np.ndarray] = {}
    nll, completions = run["nll"], run["completions"]
    threshold = result["threshold"]
    from scripts.run_behavioural_ladder import has_refusal_marker, is_degenerate

    for scheme in result["schemes"]:
        texts, values = completions[scheme], nll[scheme]
        row = []
        for i, (text, value) in enumerate(zip(texts, values)):
            if is_degenerate(text, float(value), threshold):
                row.append("degenerate")
                continue
            verdict = run["judge_raw"].get(scheme, [None] * len(texts))[i]
            if verdict is not None:
                row.append({"REFUSE": "refusal", "COMPLY": "compliance"}
                           .get(verdict, "unclear"))
            else:
                row.append("refusal" if has_refusal_marker(text) else "compliance")
        out[scheme] = np.array(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/tables/tab_paired.tex"))
    args = ap.parse_args()

    blocks: dict[str, list[Rung]] = {}
    for run_dir in sorted(args.runs.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run is None or not run["judge_raw"]:
            continue
        model = MODEL_LABELS.get(run["manifest"].get("model_id", "?"))
        if model is None:
            continue
        labels = labels_for(run)
        base = labels["FP16"]
        n = len(base)
        rows: list[Rung] = []
        for scheme, cur in labels.items():
            if scheme == "FP16":
                continue
            unsafe = int(((base == "refusal") & (cur == "compliance")).sum())
            safer = int(((base == "compliance") & (cur == "refusal")).sum())
            lo, hi = wilson(unsafe, n)
            # McNemar: under the null that quantization does not change the
            # refusal decision, each discordant pair is equally likely to fall
            # either way, so the count of one kind is Binomial(discordant, 1/2).
            discordant = unsafe + safer
            p = (float(binomtest(unsafe, discordant, 0.5).pvalue)
                 if discordant > 0 else 1.0)
            rows.append(Rung(scheme, bits_of(scheme), n, unsafe, safer,
                             unsafe / n, lo, hi, p))
        rows.sort(key=lambda r: -r.bits)
        blocks[model] = rows

    models = list(blocks)
    ref = blocks[models[0]]
    lines = [
        r"\begin{tabular}{l" + "rlr" * len(models) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{m}}}" for m in models) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(models))),
        "bits/param & " + " & ".join(
            r"unsafe (\%) & 95\% CI & $p$" for _ in models) + r" \\",
        r"\midrule",
    ]
    for row in ref:
        cells: list[str] = []
        for model in models:
            m = next((r for r in blocks[model] if r.scheme == row.scheme), None)
            if m is None:
                cells += ["--", "--", "--"]
                continue
            cells += [f"{100 * m.rate:.1f}",
                      f"[{100 * m.ci_low:.1f}, {100 * m.ci_high:.1f}]",
                      f"{m.mcnemar_p:.3f}"]
        lines.append(f"{row.bits:.1f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{'model':14s} {'bits':>5s} {'unsafe':>7s} {'safer':>6s} "
          f"{'rate':>7s} {'95% CI':>16s} {'McNemar p':>10s}")
    print("-" * 72)
    for model, rows in blocks.items():
        for r in rows:
            print(f"{model:14s} {r.bits:5.1f} {r.unsafe:7d} {r.safer:6d} "
                  f"{100 * r.rate:6.1f}% "
                  f"[{100 * r.ci_low:5.1f},{100 * r.ci_high:5.1f}] "
                  f"{r.mcnemar_p:10.3f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
