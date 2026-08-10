"""Generate every figure in the paper from docs/paper/data.json.

No number is typed into a figure. Each panel reads the consolidated data file, so
a figure cannot drift from the runs that produced it.

Usage:
  python scripts/build_paper_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import figstyle  # noqa: E402

# Colour-blind-safe, and nothing relies on hue alone: markers and dash patterns
# carry the same information for greyscale print.
figstyle.apply()

JUDGE = figstyle.CONSERVATIVE
MARKER = figstyle.UNSAFE
DEGEN = figstyle.NEUTRAL
MODEL_STYLE = figstyle.MODEL_STYLE
TEXT_WIDTH_IN = figstyle.TEXT_WIDTH
save = figstyle.save


def rung_axis(ax: Any, rows: list[dict[str, Any]]) -> list[int]:
    """Evenly spaced rung positions, labelled with true bits per parameter.

    FP16 sits at 16 bits while every quantized rung is between 2.5 and 8.5. On a
    linear axis that spends two thirds of the width on empty space and squeezes
    the region the paper is about, so rungs are placed categorically and the tick
    label carries the real number.
    """
    positions = list(range(len(rows)))
    ax.set_xticks(positions)
    ax.set_xticklabels([
        # A deployed checkpoint has no position on this axis and carries NaN;
        # its scheme name is the honest tick label.
        r.get("scheme", "?") if r["bits"] != r["bits"]
        else "FP16" if r["bits"] >= 16 else f"{r['bits']:.1f}"
        for r in rows
    ])
    ax.set_xlabel("stored bits / parameter")
    ax.set_xlim(-0.4, len(rows) - 0.6)
    return positions


def fig_artifact(review: dict[str, Any], out: Path) -> None:
    """Gate and grader crossed.

    Three curves rather than two, because the estimate depends on both factors
    and they dominate in different regimes: at coherent rungs the two gates
    coincide and the whole gap is the grader, while at the collapsed rungs the
    gate is the entire story.
    """
    models = list(review["gate_by_grader"])
    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH_IN, 3.7),
                             sharey=True)
    axes = list(axes) if len(models) > 1 else [axes]

    for index, (ax, model) in enumerate(zip(axes, models)):
        rows = review["gate_by_grader"][model]
        x = rung_axis(ax, rows)
        for i, r in enumerate(rows):
            if r["degenerate_composite"] > 0.5:
                ax.axvspan(i - 0.5, i + 0.5, color=DEGEN, alpha=0.18, zorder=0, lw=0)

        ax.plot(x, [100 * r["marker_nll"] for r in rows],
                color=MARKER, marker="s", ls="--", lw=1.7, ms=5,
                label="phrase list, NLL gate", zorder=3)
        ax.plot(x, [100 * r["marker_composite"] for r in rows],
                color="#E8590C", marker="v", ls="-.", lw=1.5, ms=4.5,
                label="phrase list, composite gate", zorder=3)
        ax.plot(x, [100 * r["judge_composite"] for r in rows],
                color=JUDGE, marker="o", ls="-", lw=1.9, ms=5,
                label="judge, composite gate", zorder=4)
        figstyle.panel_title(ax, chr(ord("a") + index), model)
        ax.set_ylim(-1.5, 43)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        figstyle.ygrid(ax)

    axes[0].set_ylabel(r"refusal $\to$ compliance")
    handles, labels = axes[0].get_legend_handles_labels()
    figstyle.shared_legend(
        fig, handles, labels, ncol=3,
        note="Shaded rungs = model is degenerate.")
    save(fig, out, "fig_artifact")


def fig_capability(review: dict[str, Any], out: Path) -> None:
    """GSM8K accuracy with exact binomial intervals, per model.

    Read from review_stats rather than data.json so the figure and
    Table~\\ref{tab:capability} share a source. They did not: the table scores
    correctness by answer extraction alone while data.json still carried the
    run's degeneracy-gated counts, so Qwen2.5-1.5B at 2.5 bits was plotted as
    0/200 beside a table saying 2/200.
    """
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.66, 3.5))
    reference: list[dict[str, Any]] = []
    blocks = {m: b for m, b in review["gsm8k"].items() if not m.startswith("_")}
    for model, block in blocks.items():
        colour, marker, dash = MODEL_STYLE.get(model, ("#333333", "o", "-"))
        rows = sorted(block["rows"], key=lambda r: -r["bits"])
        reference = reference or rows
        x = list(range(len(rows)))
        acc = [100 * r["accuracy"] for r in rows]
        err = [[100 * (r["accuracy"] - r["ci_low"]) for r in rows],
               [100 * (r["ci_high"] - r["accuracy"]) for r in rows]]
        ax.errorbar(x, acc, yerr=err, color=colour, marker=marker, ls=dash,
                    lw=1.6, ms=4.5, capsize=2.5, elinewidth=0.9,
                    label=f"{model} ({100 * block['fp16_accuracy']:.1f}% at FP16)")
    rung_axis(ax, reference)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_ylim(-2, 40)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    figstyle.panel_title(ax, None, "Capability collapses at a model-specific bit-width")
    figstyle.ygrid(ax)
    handles, labels = ax.get_legend_handles_labels()
    figstyle.shared_legend(fig, handles, labels, ncol=1, reserve=0.28)
    save(fig, out, "fig_capability")


def fig_probe(probe: dict[str, Any], out: Path) -> None:
    """Frozen-probe retention along the ladder, under both label scorers.

    Colour carries the model and line style carries the scorer, which is the
    only assignment that works here: a plate where dash meant "Phi" in one place
    and "original scorer" in another would encode two things with one channel.
    Markers and bands go on the corrected series alone, so the eye lands on the
    grading the manuscript reports and the original stays legible as a
    reference behind it.

    The band is the 2.5--97.5 percentile of retention across replicates. Without
    it the collapse at the lowest rungs looks far more precisely located than
    the data supports -- and it is exactly there that the two scorers part
    company, because d' itself is near zero and the ratio is unstable.
    """
    corrected = probe["scorers"]["letter"]
    original = probe["scorers"]["first-token-legacy"]
    order = [m for m in figstyle.MODEL_ORDER if m in corrected]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.74, 3.8))
    reference: list[dict[str, Any]] = []

    def series(block: dict[str, Any], model: str) -> list[dict[str, Any]]:
        rows = [r for r in block.get(model, {}).get("rows", [])
                if r["scheme"] != "FP16"]
        return sorted(rows, key=lambda r: -r["bits"])

    for model in order:
        colour = MODEL_STYLE.get(model, ("#333333", "o", "-"))[0]
        marker = MODEL_STYLE.get(model, ("#333333", "o", "-"))[1]
        rows = series(corrected, model)
        reference = reference or rows
        x = range(len(rows))
        ax.plot(x, [100 * r["retained_mean"] for r in rows], color=colour,
                marker=marker, ls="-", lw=1.7, ms=4.5, zorder=3)
        ax.fill_between(x, [100 * r["retained_ci_low"] for r in rows],
                        [100 * r["retained_ci_high"] for r in rows],
                        color=colour, alpha=0.14, lw=0, zorder=1)
        old = series(original, model)
        if old:
            ax.plot(range(len(old)), [100 * r["retained_mean"] for r in old],
                    color=colour, ls=(0, (1.6, 1.6)), lw=1.2, zorder=2)

    ax.axhline(100, color="#ADB5BD", lw=0.8, ls=":", zorder=0)
    ax.axhline(0, color="#ADB5BD", lw=0.8, zorder=0)
    rung_axis(ax, reference)
    ax.set_ylabel("frozen-probe $d'$ retained")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    figstyle.panel_title(ax, None, "Frozen-probe retention along the ladder")
    figstyle.ygrid(ax)

    # Two legend groups, built by hand because the series carry no labels: one
    # entry per model would double to four with the scorer series in it, and a
    # four-entry legend where two entries are the same colour reads as four
    # models rather than as two models seen twice.
    handles = [plt.Line2D([], [], color=MODEL_STYLE[m][0], marker=MODEL_STYLE[m][1],
                          ls="-", lw=1.7, ms=4.5) for m in order]
    labels = list(order)
    handles += [plt.Line2D([], [], color=figstyle.NEUTRAL, ls="-", lw=1.7),
                plt.Line2D([], [], color=figstyle.NEUTRAL, ls=(0, (1.6, 1.6)),
                           lw=1.2)]
    labels += ["corrected scorer", "original scorer"]
    figstyle.shared_legend(
        fig, handles, labels, ncol=2,
        # An en dash, not two hyphens. LaTeX turns "--" into one; matplotlib
        # prints it literally, and this note is drawn by matplotlib.
        note="Shaded bands = 2.5–97.5% retention across replicates, corrected "
             "scorer.",
        reserve=0.24)
    save(fig, out, "fig_probe")


def fig_degeneracy(data: dict[str, Any], out: Path) -> None:
    """Why perplexity alone fails: repetition loops are predictable."""
    model = ("Qwen2.5-3B" if "Qwen2.5-3B" in data["behavioural"]
             else next(iter(data["behavioural"])))
    rows = sorted(data["behavioural"][model]["rows"], key=lambda r: -r["bits"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.6))
    x = rung_axis(ax1, rows)
    ax1.plot(x, [r["median_nll"] for r in rows], color=MARKER, marker="s",
             lw=1.7, ms=5)
    ax1.set_ylabel("median NLL under FP16")
    figstyle.panel_title(ax1, "a", "Perplexity is not monotone")
    figstyle.ygrid(ax1)
    last, prev = rows[-1], rows[-2]
    ax1.annotate(
        f"{last['bits']:.1f} bits scores LOWER\nthan {prev['bits']:.1f}:\nrepetition is predictable",
        xy=(len(rows) - 1, last["median_nll"]), xytext=(0.05, 0.72),
        textcoords="axes fraction", fontsize=7.5, color="#495057",
        arrowprops={"arrowstyle": "->", "color": "#868E96", "lw": 0.9,
                    "connectionstyle": "arc3,rad=-0.25"},
    )

    x2 = rung_axis(ax2, rows)
    ax2.plot(x2, [r["distinct_trigram"] for r in rows], color=JUDGE, marker="o",
             lw=1.7, ms=5, label="distinct trigrams")
    ax2.plot(x2, [r["max_token_share"] for r in rows], color=MARKER, marker="s",
             ls="--", lw=1.7, ms=5, label="largest token share")
    ax2.plot(x2, [r["alpha_fraction"] for r in rows], color="#5F3DC4", marker="^",
             ls=":", lw=1.7, ms=5, label="alphabetic fraction")
    ax2.set_ylabel("ratio")
    ax2.set_ylim(-0.03, 1.10)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    figstyle.panel_title(ax2, "b", "Surface statistics separate the regimes")
    figstyle.ygrid(ax2)
    handles, labels = ax2.get_legend_handles_labels()
    figstyle.shared_legend(fig, handles, labels, ncol=3)
    save(fig, out, "fig_degeneracy")


def fig_marker_sensitivity(data: dict[str, Any], out: Path) -> None:
    """The same completions, four marker lists, four different answers.

    Three things about this plate are deliberate. The lists are drawn in a ramp
    of ONE hue, the red this paper spends on the phrase-list estimator
    everywhere else, because they are four versions of a single instrument and
    not four different ones -- the earlier viridis ramp made them look like four
    unrelated series and shared no colour with any other figure. Dark means a
    longer list, so the section's actual claim is visible as a colour inversion
    rather than only as an argument. And the axis is stored bits, as everywhere
    else in the manuscript; it used to be code bits, which is the same ladder
    shifted by the half-bit of scale-and-zero overhead and left this the one
    plate whose x axis did not line up with its neighbours.
    """
    model = next((m for m in ("Qwen2.5-3B", "Phi-3.5-mini")
                  if data["behavioural"].get(m, {}).get("marker_variants")), None)
    if model is None:
        print("  (no marker-variant data; skipping fig_marker_sensitivity)")
        return
    variants = data["behavioural"][model]["marker_variants"]
    schemes = sorted({s for v in variants.values() for s in v},
                     key=lambda s: -int(s.split("_")[1][:-1]))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.72, 3.5))
    marks = ("o", "s", "^", "D", "v")
    for i, (name, row) in enumerate(variants.items()):
        ax.plot(range(len(schemes)), [100 * row.get(s, 0.0) for s in schemes],
                marker=marks[i % len(marks)], lw=1.6, ms=4.2, label=name,
                color=matplotlib.colormaps["Reds"](
                    0.42 + 0.5 * i / max(1, len(variants) - 1)))

    _mark_inversion(ax, variants, schemes)

    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels([f"{int(s.split('_')[1][:-1]) + 0.5:g}" for s in schemes])
    ax.set_xlim(-0.4, len(schemes) - 0.6)
    ax.set_xlabel("stored bits / parameter")
    ax.set_ylabel(r"apparent refusal $\to$ compliance")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    figstyle.panel_title(ax, None, f"One choice of strings moves the headline ({model})")
    figstyle.ygrid(ax)
    handles, labels = ax.get_legend_handles_labels()
    figstyle.shared_legend(fig, handles, labels, ncol=2,
                           note="Legend names the four nested phrase lists; "
                                "darker is longer.",
                           reserve=0.28)
    save(fig, out, "fig_marker_sensitivity")


def _mark_inversion(ax: Any, variants: dict[str, Any],
                    schemes: list[str]) -> None:
    """Point at the rung where a longer list scores LOWER than a shorter one.

    Section~5's claim is that the phrase-list estimate is non-monotone in the
    phrase list, and until now the figure only made it available: the four
    curves lie within a few points of each other in the coherent band, so the
    one crossing that proves the claim is a couple of pixels wide and no reader
    would find it. Located rather than hard-coded, so a re-run that moves the
    inversion moves the annotation, and one that removes it draws nothing at
    all rather than pointing confidently at open space.
    """
    names = list(variants)
    best = None
    for j, scheme in enumerate(schemes):
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                lo = 100 * variants[names[a]].get(scheme, 0.0)
                hi = 100 * variants[names[b]].get(scheme, 0.0)
                # names[b] is the longer list, so hi < lo is the inversion.
                if hi < lo and (best is None or lo - hi > best[0]):
                    best = (lo - hi, j, lo, hi, names[a], names[b])
    if best is None:
        return
    _gap, j, lo, hi, short, long = best
    ax.annotate(
        f"{long} scores {hi:.1f}%\nwhere {short} scores {lo:.1f}%:\n"
        "a longer list, a smaller number",
        xy=(j, (lo + hi) / 2), xytext=(0.04, 0.80), textcoords="axes fraction",
        fontsize=7.6, color="#404040", va="top", linespacing=1.3,
        arrowprops={"arrowstyle": "-", "color": "#909497", "linewidth": 0.8,
                    "shrinkB": 3, "connectionstyle": "arc3,rad=-0.18"})


def fig_refusal_law(review: dict[str, Any], out: Path) -> None:
    """Refusal rate against bit-width, with the fit drawn over exactly the rungs
    it was fitted to.

    An earlier version plotted a regression that included the FP16 point while
    the text quoted a coefficient fitted without it, so the figure and the
    number disagreed. The fit is now drawn only across the coherent quantized
    band, and the full-precision point is shown separately with the gap to the
    top rung annotated -- that gap being near zero is itself the finding, since
    it is what makes a single line through the whole range wrong.
    """
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.70, 3.7))
    for model, block in review["drift"].items():
        if model.startswith("_"):
            continue
        colour, marker, dash = MODEL_STYLE.get(model, ("#333333", "o", "-"))
        x, y = block["band_bits"], block["band_refusal_pct"]
        ax.plot(x, y, color=colour, marker=marker, ls="none", ms=5, label=model)
        ax.plot([16.0], [block["fp16_refusal_pct"]], color=colour,
                marker=marker, ls="none", ms=5, mfc="white", mew=1.3)
        slope = -block["kappa"]
        ax.plot([min(x), max(x)],
                [block["intercept"] + slope * g for g in (min(x), max(x))],
                color=colour, ls=dash, lw=1.4, alpha=0.9)
        # The band's line, continued back to full precision, where it misses.
        ax.plot([max(x), 16.0],
                [block["intercept"] + slope * g for g in (max(x), 16.0)],
                color=colour, ls=":", lw=1.0, alpha=0.5)

    ax.set_xscale("log", base=2)
    ax.set_xticks([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 16.0])
    ax.set_xticklabels(["2.5", "3.5", "4.5", "5.5", "6.5", "7.5", "8.5", "FP16"])
    ax.minorticks_off()
    ax.set_xlabel("stored bits / parameter")
    ax.set_ylabel("refusal rate")
    ax.invert_xaxis()
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    figstyle.panel_title(ax, None, "Refusal drifts inside the coherent band")
    figstyle.ygrid(ax)
    handles, labels = ax.get_legend_handles_labels()
    figstyle.shared_legend(
        fig, handles, labels, ncol=3,
        note="Dotted extensions continue the coherent-band fit to full precision.")
    save(fig, out, "fig_refusal_law")


def fig_judge_agreement(agreement: dict[str, Any], review: dict[str, Any],
                        out: Path) -> None:
    """Does the headline survive an independent grader?

    Two panels, because agreement and reproduction answer different questions
    and reporting only the first would be the friendlier half. Left: how often
    each grader agrees with the 7B judge, split by what the 7B judge said, since
    a single agreement number hides whether the disagreement is concentrated in
    one class. Right: the paired 4.5-bit counts recomputed from each grader's own
    labels, which is the number the paper's claim actually rests on.
    """
    rows = []
    for tag, models in agreement.items():
        for model, block in models.items():
            if block.get("n_compared", 0) < 400:
                continue                      # too few to plot honestly
            repro = next((r for r in block.get("reproduction", [])
                          if abs(r["bits"] - 4.5) < 1e-9), None)
            rows.append({
                "grader": tag.split("_")[0].replace("agent", "Claude").title()
                if tag.startswith("agent") else tag.split("_")[0].title(),
                "detail": ("Haiku" if "haiku" in tag else
                           "Llama-3.3-70B" if "groq" in tag else
                           "Llama-3.1-70B" if "nvidia" in tag else "Claude"),
                "model": model,
                "agreement": block["agreement"],
                "recall_refuse": (block.get("recall_by_class") or {}).get("REFUSE"),
                "to_refusal": repro["to_refusal"] if repro else None,
                "to_compliance": repro["to_compliance"] if repro else None,
            })
    if not rows:
        print("  (no second-judge data; skipping fig_judge_agreement)")
        return

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.15]})

    labels = [f"{r['detail']}\n{r['model']}" for r in rows]
    y = list(range(len(rows)))
    ax1.barh(y, [100 * r["agreement"] for r in rows], height=0.55,
             color=JUDGE, alpha=0.85, label="overall")
    ax1.plot([100 * (r["recall_refuse"] or 0) for r in rows], y, "o",
             color=MARKER, ms=6, label="on the refusal class", zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_ylim(len(rows) - 0.4, -0.4)
    ax1.set_xlabel("agreement with the 7B judge")
    ax1.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    figstyle.panel_title(ax1, "a", "Grader agreement")
    figstyle.xgrid(ax1)

    # Right panel: the 7B judge's own counts as a reference row, then each
    # independent grader on the same axis.
    ref = []
    for model in ("Qwen2.5-3B", "Phi-3.5-mini"):
        row = next((r for r in review["transitions"][model]["rows"]
                    if abs(r["bits"] - 4.5) < 1e-9), None)
        if row:
            ref.append({"detail": "Qwen2.5-7B\n(as reported)", "model": model,
                        "to_refusal": row["to_refusal"],
                        "to_compliance": row["to_compliance"]})
    every = ref + [r for r in rows if r["to_refusal"] is not None]
    y2 = list(range(len(every)))
    height = 0.36
    ax2.barh([v - height / 2 for v in y2],
             [r["to_refusal"] for r in every], height=height,
             color=JUDGE, label=r"newly refusing")
    ax2.barh([v + height / 2 for v in y2],
             [r["to_compliance"] for r in every], height=height,
             color=MARKER, label=r"newly complying")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([f"{r['detail'].splitlines()[0]}\n{r['model']}"
                         for r in every], fontsize=7.5)
    ax2.invert_yaxis()
    # Half a row of margin, not 0.4. At 0.4 the first reference band was clipped
    # by the axes edge and only the second one showed, so the note said "rows"
    # while the plate shaded one.
    ax2.set_ylim(len(every) - 0.5, -0.55)
    ax2.set_xlim(0, 1.13 * max(r["to_refusal"] for r in every))
    ax2.set_xlabel("prompts changing decision at 4.5 bits")
    figstyle.panel_title(ax2, "b", "Reproduced decision changes")
    figstyle.xgrid(ax2)
    # Mark the reference rows so the eye separates them from the replications.
    for i in range(len(ref)):
        ax2.axhspan(i - 0.5, i + 0.5, color=DEGEN, alpha=0.13, zorder=0, lw=0)
    # The counts, at the bar ends. Several of these bars are one or two prompts
    # and the whole point of the panel is how few they are, which is a claim
    # about the number and not about the length of a hairline.
    for i, row in enumerate(every):
        for dy, value, colour in ((-height / 2, row["to_refusal"], JUDGE),
                                  (height / 2, row["to_compliance"], MARKER)):
            ax2.text(value + 0.6, i + dy, str(value), va="center", ha="left",
                     fontsize=7.5, color=colour, fontweight="bold")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    figstyle.shared_legend(fig, handles1 + handles2, labels1 + labels2,
                           ncol=2, note="Shaded rows = the 7B judge's reported counts.",
                           reserve=0.30)
    save(fig, out, "fig_judge_agreement")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("docs/paper/data.json"))
    ap.add_argument("--review", type=Path,
                    default=Path("docs/paper/review_stats.json"))
    ap.add_argument("--agreement", type=Path,
                    default=Path("docs/paper/judge_agreement.json"))
    ap.add_argument("--probe", type=Path,
                    default=Path("docs/paper/probe_corrected.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    if not probe["protocol"]["baseline_reproduces"]:
        raise SystemExit(
            "probe_corrected.json says its original-scorer column does not "
            "reproduce review_stats.json; the two series of fig_probe would "
            "then differ by more than the label definition the legend claims. "
            "Re-run scripts/refit_probe_corrected.py.")
    args.out.mkdir(parents=True, exist_ok=True)
    print("figures:")
    fig_artifact(review, args.out)
    fig_capability(review, args.out)
    fig_probe(probe, args.out)
    fig_degeneracy(data, args.out)
    fig_marker_sensitivity(data, args.out)
    fig_refusal_law(review, args.out)
    # Emitted only when a second judge has run, so the paper never carries a
    # placeholder figure for a measurement that does not exist.
    if args.agreement.exists():
        agreement = json.loads(args.agreement.read_text(encoding="utf-8"))
        fig_judge_agreement(agreement, review, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
