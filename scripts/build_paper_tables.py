"""Emit LaTeX table bodies from the consolidated measurement files.

Inferential tables read docs/paper/review_stats.json; descriptive ones read
docs/paper/data.json.

Tables are generated for the same reason figures are: a number typed into a
manuscript is a number that can silently disagree with the run that produced it.
Each fragment is \\input{} by main.tex.

Usage:
  python scripts/build_paper_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pct(value: float | None, decimals: int = 1) -> str:
    return "--" if value is None else f"{100 * value:.{decimals}f}"


def bits_label(bits: float) -> str:
    return "FP16" if bits >= 16 else f"{bits:.1f}"


def table_artifact(review: dict[str, Any]) -> str:
    """Gate and grader crossed, so their contributions are separable.

    An earlier version of this table set a composite-gated judge beside an
    NLL-gated phrase list and attributed the whole difference to the grader.
    Crossing the two factors shows they dominate in different regimes, which is
    a stronger result than the confounded version.
    """
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{phrase list} & judge & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}",
        r"model & bits & NLL gate & composite & composite & degen.\ (\%) \\",
        r"\midrule",
    ]
    for i, (model, rows) in enumerate(review["gate_by_grader"].items()):
        if i:
            lines.append(r"\addlinespace")
        for j, r in enumerate(rows):
            name = model if j == 0 else ""
            lines.append(
                f"{name} & {r['bits']:.1f} & {100 * r['marker_nll']:.1f} & "
                f"{100 * r['marker_composite']:.1f} & "
                f"{100 * r['judge_composite']:.1f} & "
                f"{100 * r['degenerate_composite']:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_markers(data: dict[str, Any]) -> str:
    """The same completions under four marker lists."""
    model = next((m for m in ("Qwen2.5-3B", "Phi-3.5-mini")
                  if data["behavioural"].get(m, {}).get("marker_variants")), None)
    if model is None:
        return "% no marker-variant data available\n"
    variants = data["behavioural"][model]["marker_variants"]
    schemes = sorted({s for v in variants.values() for s in v},
                     key=lambda s: -int(s.split("_")[1][:-1]))
    lines = [
        r"\begin{tabular}{l" + "r" * len(schemes) + "}",
        r"\toprule",
        "marker list & " + " & ".join(s.split("_")[1][:-1] for s in schemes) + r" \\",
        r"\midrule",
    ]
    for name, row in variants.items():
        safe = name.replace("_", r"\_")
        lines.append(f"{safe} & " + " & ".join(
            pct(row.get(s, 0.0)) for s in schemes) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def fmt_p(p: float) -> str:
    """p-values below the resolution of three decimals are reported as such
    rather than rounded to a misleading 0.000."""
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def table_transitions(review: dict[str, Any]) -> str:
    """Paired refusal-decision transitions, in long format.

    Long rather than side-by-side because each rung now needs six numbers, and
    a wide layout at two models would run into the margin.
    """
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"model & bits & $\to$comply & $\to$refuse & gradable & rate (\%) "
        r"& 95\% upper & $p_{\text{Holm}}$ \\",
        r"\midrule",
    ]
    for i, (model, block) in enumerate(review["transitions"].items()):
        if i:
            lines.append(r"\addlinespace")
        rows = sorted(block["rows"], key=lambda r: -r["bits"])
        for j, r in enumerate(rows):
            name = model if j == 0 else ""
            lines.append(
                f"{name} & {r['bits']:.1f} & {r['to_compliance']} & "
                f"{r['to_refusal']} & {r['n_gradable']} & "
                f"{100 * r['rate_itt']:.1f} & {100 * r['upper95_itt']:.1f} & "
                f"{fmt_p(r['mcnemar_p_holm'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_capability_paired(review: dict[str, Any]) -> str:
    """GSM8K with exact McNemar on question-level transitions."""
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"model & bits & acc.\ (\%) & lost & gained & $p_{\text{Holm}}$ \\",
        r"\midrule",
    ]
    models = [(m, b) for m, b in review["gsm8k"].items() if not m.startswith("_")]
    for i, (model, block) in enumerate(models):
        if i:
            lines.append(r"\addlinespace")
        lines.append(
            f"{model} & FP16 & {100 * block['fp16_accuracy']:.1f} & -- & -- & -- "
            + r"\\")
        for r in block["rows"]:
            lines.append(
                f" & {r['bits']:.1f} & {100 * r['accuracy']:.1f} & {r['lost']} & "
                f"{r['gained']} & {fmt_p(r['mcnemar_p_holm'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_probe_ci(review: dict[str, Any]) -> str:
    """Frozen-probe retention with a percentile interval over fit/score splits."""
    models = list(review["probe"])
    ref = review["probe"][models[0]]["rows"]
    lines = [
        r"\begin{tabular}{l" + "rl" * len(models) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in models) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
                for i in range(len(models))),
        "bits & " + " & ".join(r"kept (\%) & 95\% CI" for _ in models) + r" \\",
        r"\midrule",
    ]
    for row in ref:
        cells: list[str] = []
        for model in models:
            m = next((r for r in review["probe"][model]["rows"]
                      if r["scheme"] == row["scheme"]), None)
            if m is None:
                cells += ["--", "--"]
                continue
            cells += [f"{100 * m['retained_mean']:.0f}",
                      f"[{100 * m['retained_ci_low']:.0f}, "
                      f"{100 * m['retained_ci_high']:.0f}]"]
        lines.append(f"{bits_label(row['bits'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_marker_decomposition(review: dict[str, Any]) -> str:
    """Both factors of the paired flip, so the non-monotonicity is legible."""
    model = "Qwen2.5-3B"
    rows = review["marker_decomposition"][model]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{4.5 bits} & \multicolumn{2}{c}{2.5 bits} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"marker list & FP16 refusals & complies & flips & complies & flips \\",
        r"\midrule",
    ]
    for r in rows:
        s = r["schemes"]
        name = r["variant"].replace("_", r"\_").replace("+", r"$+$")
        lines.append(
            f"{name} & {r['fp16_refusals']} & "
            f"{s['RTN_4B']['rung_compliances']} & {s['RTN_4B']['flips']} & "
            f"{s['RTN_2B']['rung_compliances']} & {s['RTN_2B']['flips']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("docs/paper/data.json"))
    ap.add_argument("--review", type=Path,
                    default=Path("docs/paper/review_stats.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/tables"))
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, builder, source in (
        ("tab_artifact", table_artifact, review),
        ("tab_markers", table_markers, data),
        ("tab_transitions", table_transitions, review),
        ("tab_capability", table_capability_paired, review),
        ("tab_probe", table_probe_ci, review),
        ("tab_marker_decomp", table_marker_decomposition, review),
    ):
        (args.out / f"{name}.tex").write_text(
            builder(source) + "\n", encoding="utf-8")
        print(f"  {name}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
