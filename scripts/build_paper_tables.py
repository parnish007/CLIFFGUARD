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


def bits_label(bits: float, scheme: str | None = None) -> str:
    """Axis label for a scheme's bit position.

    Deployed checkpoints (AWQ, GPTQ) carry NaN here on purpose: they have a bit
    budget but no place on the RTN ordinal axis, because they vary the algorithm
    as well as the width. Printing `nan` in a table column headed "bits" reads
    like a computation went wrong, so the scheme names itself instead.
    """
    if bits != bits:                                   # NaN
        return scheme or "n/a"
    return "FP16" if bits >= 16 else f"{bits:.1f}"


def table_artifact(review: dict[str, Any]) -> str:
    """Gate and grader crossed, so their contributions are separable.

    An earlier version of this table set a composite-gated judge beside an
    NLL-gated phrase list and attributed the whole difference to the grader.
    Crossing the two factors shows they dominate in different regimes, which is
    a stronger result than the confounded version.
    """
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{phrase list} & \multicolumn{2}{c}{judge} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"model & bits & NLL & composite & NLL & composite & degen.\ (\%) \\",
        r"\midrule",
    ]
    for i, (model, rows) in enumerate(review["gate_by_grader"].items()):
        if i:
            lines.append(r"\addlinespace")
        for j, r in enumerate(rows):
            name = model if j == 0 else ""
            lines.append(
                f"{name} & {bits_label(r['bits'], r['scheme'])} & "
                f"{100 * r['marker_nll']:.1f} & "
                f"{100 * r['marker_composite']:.1f} & "
                f"{100 * r['judge_nll']:.1f} & "
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
        # These headers are CODE bits (8..2), one unit below the stored-bits
        # axis (8.5..2.5) used everywhere else. Say so in the header rather
        # than leaving a reader to assume the two axes match.
        r"marker list & \multicolumn{" + str(len(schemes))
        + r"}{c}{code bits} \\",
        r"\cmidrule(lr){2-" + str(len(schemes) + 1) + r"}",
        " & " + " & ".join(s.split("_")[1][:-1] for s in schemes) + r" \\",
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
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        # Both Holm families, because using the wider one for the simultaneous
        # bound and the narrower one for the tests would take the friendliest
        # number from each. $p_{14}$ is primary.
        r"model & bits & $\to$comply & $\to$refuse & gradable & rate (\%) "
        r"& 95\% upper & $p_{14}$ & $p_{7}$ \\",
        r"\midrule",
    ]
    for i, (model, block) in enumerate(review["transitions"].items()):
        if i:
            lines.append(r"\addlinespace")
        rows = sorted(block["rows"],
                      key=lambda r: (r["bits"] != r["bits"], -r["bits"]))
        for j, r in enumerate(rows):
            name = model if j == 0 else ""
            lines.append(
                f"{name} & {bits_label(r['bits'], r['scheme'])} & "
                f"{r['to_compliance']} & "
                f"{r['to_refusal']} & {r['n_gradable']} & "
                f"{100 * r['rate_itt']:.1f} & {100 * r['upper95_itt']:.1f} & "
                f"{fmt_p(r['mcnemar_p_holm_all_cells'])} & "
                f"{fmt_p(r['mcnemar_p_holm'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_capability_paired(review: dict[str, Any]) -> str:
    """GSM8K with exact McNemar on question-level transitions."""
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"model & bits & acc.\ (\%) & lost & gained & $p_{21}$ & $p_{7}$ \\",
        r"\midrule",
    ]
    models = [(m, b) for m, b in review["gsm8k"].items() if not m.startswith("_")]
    for i, (model, block) in enumerate(models):
        if i:
            lines.append(r"\addlinespace")
        lines.append(
            f"{model} & FP16 & {100 * block['fp16_accuracy']:.1f} & -- & -- & -- & -- "
            + r"\\")
        for r in block["rows"]:
            lines.append(
                f" & {r['bits']:.1f} & {100 * r['accuracy']:.1f} & {r['lost']} & "
                f"{r['gained']} & {fmt_p(r['mcnemar_p_holm_all_cells'])} & "
                f"{fmt_p(r['mcnemar_p_holm'])} " + r"\\")
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
        # NOT a confidence interval: these are 2.5--97.5 percentiles over
        # fit/score splits of one fixed prompt set. Labelling them "95% CI"
        # would claim sampling coverage the procedure does not provide.
        "bits & " + " & ".join(
            r"kept (\%) & split range" for _ in models) + r" \\",
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
        # Six columns: marker list, FP16 refusals, then complies/flips per rung.
        # A seventh was declared and never filled, which silently widened the
        # table with a phantom column.
        r"\begin{tabular}{lrrrrr}",
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


def table_judge_agreement(agreement: dict[str, Any]) -> str:
    """Second-judge agreement, and whether the headline reproduces under it.

    The paper's dependent variable comes from one grader, so this table is the
    answer to "why believe that grader". It carries both halves: how often an
    independent judge agrees, and what the paired 4.5-bit result looks like when
    recomputed from the independent judge's labels alone. Agreement without the
    reproduction would be the less interesting half -- two graders can agree and
    both be wrong, but a result that survives relabelling is harder to dismiss.
    """
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{agreement with 7B} "
        r"& \multicolumn{2}{c}{4.5 bits, second judge} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"judge & model & $n$ & overall & on refusal & $\to$refuse & $\to$comply \\",
        r"\midrule",
    ]
    for tag, models in agreement.items():
        pretty = tag.replace("_", " ")
        for i, (model, block) in enumerate(models.items()):
            name = pretty if i == 0 else ""
            overall = block.get("agreement")
            recall = (block.get("recall_by_class") or {}).get("REFUSE")
            repro = next((r for r in block.get("reproduction", [])
                          if abs(r["bits"] - 4.5) < 1e-9), None)
            cells = [
                f"{100 * overall:.1f}" if overall is not None else "--",
                f"{100 * recall:.1f}" if recall is not None else "--",
                str(repro["to_refusal"]) if repro else "--",
                str(repro["to_compliance"]) if repro else "--",
            ]
            lines.append(f"{name} & {model} & {block.get('n_compared', 0)} & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("docs/paper/data.json"))
    ap.add_argument("--review", type=Path,
                    default=Path("docs/paper/review_stats.json"))
    ap.add_argument("--agreement", type=Path,
                    default=Path("docs/paper/judge_agreement.json"))
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

    # Emitted only once a second judge has run, so the manuscript can \input{}
    # it without carrying a placeholder for a measurement that does not exist.
    if args.agreement.exists():
        agreement = json.loads(args.agreement.read_text(encoding="utf-8"))
        (args.out / "tab_judge_agreement.tex").write_text(
            table_judge_agreement(agreement) + "\n", encoding="utf-8")
        print("  tab_judge_agreement.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
