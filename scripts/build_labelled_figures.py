"""Figures for the labelled-suite runs: what the 2x2 measured, and what it could not.

The result these runs produced is a negative one on the safety arm, and a
negative result is exactly where a figure can mislead most. A line at zero
across seven rungs looks like evidence of safety. It is not: the endpoint never
fired on a harmful prompt at any rung including the full-precision reference, so
the line could not have been anywhere else.

Every figure here is therefore built to show the DENOMINATOR as well as the
number -- how much of the corpus was still producing language, how much of it
reached the token budget mid-sentence, and how large the unadjudicated
population is that the zero rests on.

Usage:
  python scripts/build_labelled_figures.py
  python scripts/build_labelled_figures.py --out docs/paper/figures
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

from scripts.analyse_leakage import pivots  # noqa: E402
from scripts.build_labelled_tables import corrected_by_scheme  # noqa: E402
from scripts import figstyle  # noqa: E402

# Shared with build_paper_figures.py so the two sets of plates look like one
# document. Colour-blind safe; marker and dash carry the same information as
# hue for greyscale print.
figstyle.apply()

MODEL_STYLE = figstyle.MODEL_STYLE
CLASS_COLOUR = figstyle.CLASS_COLOUR
CLASS_SHORT = figstyle.CLASS_SHORT
MARKER_RED = figstyle.UNSAFE
TEXT_WIDTH_IN = figstyle.TEXT_WIDTH
save = figstyle.save

RUNS = {
    "Qwen2.5-3B": "20260808-154229_069d48d_lab-qwen3b-xstest",
    "Phi-3.5-mini": "20260808-162254_069d48d_lab-phi35-xstest",
    "SmolLM2-1.7B": "20260808-164828_069d48d_lab-smol17-xstest",
}
SCHEMES = ["FP16", "RTN_8B", "RTN_7B", "RTN_6B", "RTN_5B", "RTN_4B", "RTN_3B", "RTN_2B"]
# STORED bits per parameter, which is what the rest of the manuscript's axes
# report: an n-bit code plus the group scale and zero point costs n + 0.5 bits
# at group 64. The runner names its schemes by CODE bits, so RTN_8B is 8.5
# stored. Labelling these figures "8" under an axis titled "stored bits" would
# put the same rung at two different positions in one paper.
#
# Full precision is labelled "16" rather than "FP16" for the same reason the
# rungs are labelled by stored bits: the axis is a bit count, and 16 is what
# FP16 stores. It also fits. "FP16" is four characters against three, and on
# the three-panel plates the tick spacing is narrower than that string, so it
# ran into its neighbour and printed "FP168.5".
RUNG_LABEL = {"FP16": "16", "RTN_8B": "8.5", "RTN_7B": "7.5",
              "RTN_6B": "6.5", "RTN_5B": "5.5", "RTN_4B": "4.5",
              "RTN_3B": "3.5", "RTN_2B": "2.5"}
# Stamped on every plate that reads labels at a rung the corrected scorer has
# not reached. Held in one place so the four cannot drift apart, and so that
# deleting it once round 4 lands is one edit rather than four.
LADDER_SCORER = "\u2020 original label scorer at every rung."

def load(repo: Path) -> dict[str, dict[str, Any]]:
    """Everything the figures need, read once from the run directories."""
    data: dict[str, dict[str, Any]] = {}
    for model, run in RUNS.items():
        results = repo / "artifacts/runs" / run / "results"
        tax = json.loads((results / "completion_taxonomy.json").read_text(
            encoding="utf-8"))
        prompts = json.loads((results / "prompts.json").read_text(
            encoding="utf-8"))
        completions = {
            s: json.loads((results / f"completions_{s}.json").read_text(
                encoding="utf-8"))["completions"]
            for s in SCHEMES if (results / f"completions_{s}.json").exists()}
        # Two gradings of the same completions. "resolved" is the original
        # label scorer, which covers every rung; "corrected" is the single-token
        # re-grade, which at the time of writing covers full precision only.
        # Figures about full precision use the corrected one; ladder-wide
        # figures use the original and say so on the plate.
        data[model] = {
            "resolved": tax["resolved"],
            "corrected": corrected_by_scheme(results),
            "harm": prompts["harm_label"],
            "completions": completions,
            "threshold": tax["degeneracy_threshold"],
        }
    return data


def fp16_labels(d: dict[str, Any]) -> tuple[list[str], bool]:
    """Full-precision labels, and whether they are the corrected scorer's.

    Returning the provenance beside the labels rather than resolving it at each
    call site is deliberate: every figure that draws these has to say which
    instrument produced them, and a helper that silently falls back would make
    that caption a guess.
    """
    corrected = d.get("corrected", {}).get("FP16")
    if corrected is not None:
        return corrected, True
    return d["resolved"]["FP16"], False


def counts(labels: list[str], harm: list[str], cls: str) -> dict[str, int]:
    out: dict[str, int] = {k: 0 for k in CLASS_COLOUR}
    for lab, h in zip(labels, harm):
        if h == cls:
            out[lab] = out.get(lab, 0) + 1
    return out


# ---------------------------------------------------------------------------

def fig_saturation(data: dict[str, Any], out: Path) -> None:
    """Why the safety arm reads zero: the compliance class is simply never used.

    Stacked composition of every harmful-prompt verdict along the ladder. The
    endpoint tested by `analyse_matrix.py` is compliance against everything
    else, so the only band that can produce a safety failure is the red one --
    and it has no area anywhere in the plate, at any rung, in any model,
    including the full-precision reference each rung is compared against.
    """
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.5), sharey=True)
    order = ["refusal", "deflection", "disclaimer", "unclear", "degenerate",
             "compliance"]
    for ax, (model, d) in zip(axes, data.items()):
        schemes = [s for s in SCHEMES if s in d["resolved"]]
        bottoms = [0.0] * len(schemes)
        for cls in order:
            vals = [counts(d["resolved"][s], d["harm"], "harmful")[cls] / 150 * 100
                    for s in schemes]
            ax.bar(range(len(schemes)), vals, bottom=bottoms, width=0.78,
                   color=CLASS_COLOUR[cls], edgecolor="white", linewidth=0.4,
                   label=cls if ax is axes[0] else None)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(range(len(schemes)))
        # Eight rungs into a third of the text width leaves each tick about as
        # much room as "8.5" needs, so upright labels printed as one continuous
        # string of digits. Angled, they clear each other with room to spare and
        # the plate stays three panels wide.
        ax.set_xticklabels([RUNG_LABEL[s] for s in schemes], rotation=45,
                           ha="right", fontsize=7.5,
                           rotation_mode="anchor")
        figstyle.panel_title(ax, chr(ord("a") + list(axes).index(ax)), model)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter())
        figstyle.ygrid(ax)
    axes[0].set_ylabel("harmful prompts")
    axes[1].set_xlabel("stored bits / parameter")
    handles, labels = axes[0].get_legend_handles_labels()
    figstyle.shared_legend(fig, handles, labels, ncol=3,
                           note="Stack colours = the judge's completion "
                                f"classes. {LADDER_SCORER}",
                           reserve=0.30)
    save(fig, out, "fig_labelled_saturation")


def fig_truncation(data: dict[str, Any], out: Path) -> None:
    """The measurement window closes before the answers do.

    Left: the share of completions that ran into the 48-token budget rather
    than stopping on their own. Right: the judge's compliance rate on BENIGN
    prompts, the only place compliance appears at all. The budget is what caps
    the second panel -- an answer interrupted mid-sentence has not delivered
    what was asked, which is the judge's definition of deflection.
    """
    # Read from the generated stats file rather than recomputed here. An earlier
    # version used `len(text) >= 150` as a tokenizer-free stand-in and drew a
    # curve that disagreed with the manuscript's own numbers by tens of points
    # at the collapsed rungs -- a figure whose caption described a quantity it
    # was not plotting.
    stats_path = (Path(__file__).resolve().parents[1]
                  / "docs/paper/labelled_paper_stats.json")
    if not stats_path.exists():
        raise SystemExit(
            f"{stats_path} is missing; run scripts/build_labelled_tables.py "
            "first -- it measures truncation under each model's own tokenizer, "
            "which this figure plots.")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.5))
    for model, d in data.items():
        colour, marker, dash = MODEL_STYLE[model]
        schemes = [s for s in SCHEMES if s in d["completions"]]
        xs = range(len(schemes))
        by_scheme = stats[model]["at_cap_by_scheme"]
        at_cap = [100 * by_scheme[s] for s in schemes]
        ax1.plot(xs, at_cap, color=colour, marker=marker, linestyle=dash,
                 markersize=3.4, linewidth=1.3, label=model)
        benign = [counts(d["resolved"][s], d["harm"], "benign")["compliance"]
                  / 150 * 100 for s in schemes]
        ax2.plot(xs, benign, color=colour, marker=marker, linestyle=dash,
                 markersize=3.4, linewidth=1.3, label=model)

    # Titles cut to the same length and capitalised alike. The longer pair ran
    # to the panel edges and met in the middle of the plate, and one of them
    # started lower-case while the other did not.
    for ax, title in ((ax1, "Truncated at the 48-token cap"),
                       (ax2, "Compliance on benign prompts")):
        ax.set_xticks(range(len(SCHEMES)))
        ax.set_xticklabels([RUNG_LABEL[s] for s in SCHEMES], fontsize=8)
        ax.set_xlabel("stored bits / parameter")
        figstyle.panel_title(ax, "a" if ax is ax1 else "b", title)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        figstyle.ygrid(ax)
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 30)
    handles, labels = ax1.get_legend_handles_labels()
    # The dagger belongs to the right panel only. The left one counts tokens
    # under each model's own tokenizer and never consults a grader, so marking
    # it as original-scorer would claim a dependence it does not have.
    figstyle.shared_legend(fig, handles, labels, ncol=3,
                           note="Panel (b) † original label scorer at every "
                                "rung; panel (a) needs no grader.")
    save(fig, out, "fig_labelled_truncation")


def fig_matrix(data: dict[str, Any], out: Path) -> None:
    """The 2x2 at full precision, which is the object the project is named for.

    Rows are the externally supplied prompt label, columns the judge's verdict.
    Printed as counts out of 150 per row, so the reader can see that the cell
    the whole safety question turns on -- harmful x compliance -- is empty
    before any quantization has been applied.
    """
    order = ["refusal", "deflection", "compliance", "disclaimer", "unclear",
             "degenerate"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.9))
    provenance: set[bool] = set()
    for ax, (model, d) in zip(axes, data.items()):
        labels, is_corrected = fp16_labels(d)
        provenance.add(is_corrected)
        grid = [[counts(labels, d["harm"], cls)[c]
                 for c in order] for cls in ("harmful", "benign")]
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=150, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                # The empty safety cell is the point of the figure, so it is
                # marked rather than left as an unremarkable white square.
                emphasis = (i == 0 and order[j] == "compliance")
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5,
                        color=("#C92A2A" if emphasis else
                               "white" if v > 75 else "#212529"),
                        fontweight="bold" if emphasis else "normal")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CLASS_SHORT[c] for c in order], rotation=90,
                           ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["harmful", "benign"] if ax is axes[0] else ["", ""],
                           fontsize=7.5)
        figstyle.panel_title(ax, chr(ord("a") + list(axes).index(ax)), model)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).set_label("prompts (of 150)")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.36, top=0.88, wspace=0.64)
    scorer = ("corrected label scorer" if provenance == {True}
              else "original label scorer" if provenance == {False}
              else "MIXED SCORERS -- do not publish")
    fig.text(0.5, 0.035,
             f"Rows = prompt labels; columns = judge verdicts; {scorer}.",
             ha="center", fontsize=8, color="#404040")
    save(fig, out, "fig_labelled_matrix_fp16")


def fig_unadjudicated(data: dict[str, Any], out: Path) -> None:
    """How much of the zero rests on completions nobody has read.

    Every harmful-prompt completion the endpoint scored as withheld, split by
    whether it also matches a pattern of turning toward an answer ("However, I
    can provide a general outline of how one might commit arson: 1. Choose a
    Target..."). The pattern is a heuristic written by reading this corpus, so
    the dark band is a QUEUE FOR ADJUDICATION and not a count of harm. It is
    plotted because a zero resting on 72 unread completions and a zero resting
    on 4 are not the same zero.
    """
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.4))
    width = 0.26
    for k, (model, d) in enumerate(data.items()):
        colour, _, _ = MODEL_STYLE[model]
        schemes = [s for s in SCHEMES if s in d["completions"]]
        xs = [i + (k - 1) * width for i in range(len(schemes))]
        flagged = []
        for s in schemes:
            lab, comp = d["resolved"][s], d["completions"][s]
            flagged.append(sum(
                1 for i, h in enumerate(d["harm"])
                if h == "harmful"
                and lab[i] in ("refusal", "deflection", "disclaimer")
                and pivots(comp[i])))
        ax.bar(xs, flagged, width=width, color=colour, label=model,
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(SCHEMES)))
    ax.set_xticklabels([RUNG_LABEL[s] for s in SCHEMES])
    ax.set_xlabel("stored bits / parameter")
    ax.set_ylabel("harmful prompts (of 150)")
    ax.set_ylim(0, 88)
    figstyle.panel_title(ax, None, "Withheld verdicts matching a turn toward answering")
    figstyle.ygrid(ax)
    # Without this the empty 3- and 2-bit columns read as safety improving at
    # the hardest rungs, when they are the model having stopped producing
    # language altogether.
    ax.text(0.99, 0.95, "empty at 3-2 bits because output is degenerate,\n"
                        "not because the models became safer",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#495057")
    handles, labels = ax.get_legend_handles_labels()
    figstyle.shared_legend(
        fig, handles, labels, ncol=3,
        note="Counts are unadjudicated completions, not harmful completions. "
             f"{LADDER_SCORER}")
    save(fig, out, "fig_labelled_unadjudicated")


def fig_marker_gap(data: dict[str, Any], out: Path) -> None:
    """One phrase list, three families, three different answers.

    A dumbbell per model: the left dot is how many full-precision completions
    the 25-string refusal-marker list flags, the right dot is how many the
    five-way judge places in a declining class. The bar between them is what the
    marker list does not see, and its length is a property of the MODEL's
    phrasing rather than of its behaviour -- Phi-3.5-mini declines in almost
    exactly as many cases as Qwen and says so in words the list does not carry
    ("I must clarify that...").

    Recall spans 3.6% to 49.3%. Any quantity built by pooling marker rates
    across families is pooling three different instruments.
    """
    from scripts.run_behavioural_ladder import has_refusal_marker

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.3))
    models = list(data)
    # One colour, one meaning. The earlier version drew the judge endpoint in
    # the model's own colour and the phrase-list endpoint in a fixed red, so on
    # the Phi-3.5-mini row -- whose model colour IS that red -- the two ends of
    # the dumbbell were the same colour with opposite meanings, and on the
    # SmolLM2 row a red triangle sat in a legend that showed a red circle.
    # Here colour carries the instrument and nothing else; the model is the row
    # label, which is unambiguous and already on the plate.
    LIST_COLOUR, JUDGE_COLOUR = figstyle.UNSAFE, figstyle.CONSERVATIVE
    provenance: set[bool] = set()
    for row, model in enumerate(models):
        d = data[model]
        labels, is_corrected = fp16_labels(d)
        provenance.add(is_corrected)
        comp = d["completions"]["FP16"]
        declines = sum(1 for lab in labels
                       if lab in ("refusal", "deflection", "disclaimer"))
        marked = sum(1 for c in comp if has_refusal_marker(c))
        ax.plot([marked, declines], [row, row], color="#ADB5BD", linewidth=3,
                solid_capstyle="round", zorder=1)
        ax.scatter([marked], [row], s=52, color=LIST_COLOUR, marker="s",
                   zorder=3, label="refusal-phrase list" if row == 0 else None)
        ax.scatter([declines], [row], s=52, color=JUDGE_COLOUR, marker="o",
                   zorder=3, label="five-way judge" if row == 0 else None)
        ax.annotate(f"{marked / declines:.0%} covered",
                    xy=((marked + declines) / 2, row), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color="#495057")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_ylim(-0.45, len(models) - 0.45)
    ax.set_xlim(0, 310)
    ax.set_xlabel("full-precision completions placed in a declining class "
                  "(of 300)")
    figstyle.panel_title(ax, None, "The same phrase list reads families differently")
    figstyle.xgrid(ax)
    handles, labels = ax.get_legend_handles_labels()
    scorer = ("corrected label scorer" if provenance == {True}
              else "original label scorer" if provenance == {False}
              else "MIXED SCORERS -- do not publish")
    figstyle.shared_legend(
        fig, handles, labels, ncol=2,
        note=f"Bar = declining completions the list does not flag; {scorer}.")
    save(fig, out, "fig_labelled_marker_gap")


def fig_utility(repo: Path, out: Path) -> None:
    """The one arm of this run that moved, with the uncertainty it deserves.

    Usefulness lost on benign prompts, as a rate over the full benign class,
    with one-sided 95% Clopper-Pearson upper bounds. Down to 4 bits the losses
    are matched by recoveries and no rung is distinguishable from noise; at 3
    and 2 bits the loss becomes one-directional.

    The hatched portion is the part of each loss that is DEGENERATE output
    rather than a refusal -- which is what makes this a capability result and
    not a safety one. At 2 bits it is the whole bar.
    """
    stats = json.loads((repo / "docs/paper/matrix_stats.json").read_text(
        encoding="utf-8"))
    order = {"Qwen2.5-3B": "qwen3b", "Phi-3.5-mini": "phi35",
             "SmolLM2-1.7B": "smol17"}
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.5), sharey=True)
    for ax, (model, key) in zip(axes, order.items()):
        block = next((v for k, v in stats.items() if key in k), None)
        if block is None:
            continue
        rows = [r for r in block["paired"]]
        rows.sort(key=lambda r: -float(r["scheme"].split("_")[1].rstrip("B")))
        xs = range(len(rows))
        rates = [100 * (r["utility_rate"] or 0) for r in rows]
        upper = [100 * (r["utility_upper95"] or 0) for r in rows]
        degen = [100 * r["utility_lost_by_class"]["degenerate"] / r["n_benign"]
                 for r in rows]
        # Not the model's colour. The legend is built from panel (a) and shown
        # once for the plate, so a per-model hue put a blue swatch above a red
        # panel and a green one -- and it spent the paper's two loaded colours
        # on an identity the panel title already carries. These bars mean the
        # same thing in all three panels: a benign prompt that stopped being
        # answered, in the blue the rest of the paper uses for withheld, with
        # the degenerate part in the grey it uses for collapsed output.
        ax.bar(xs, rates, width=0.68, color=figstyle.CONSERVATIVE,
               label="usefulness lost")
        ax.bar(xs, degen, width=0.68, color="white",
               edgecolor=CLASS_COLOUR["degenerate"], hatch="////",
               linewidth=0.6, label="…of which degenerate")
        ax.errorbar(xs, rates, yerr=[[0] * len(rows),
                                     [u - r for u, r in zip(upper, rates)]],
                    fmt="none", ecolor="#343A40", elinewidth=0.8, capsize=2.2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([RUNG_LABEL[r["scheme"]] for r in rows])
        figstyle.panel_title(ax, chr(ord("a") + list(axes).index(ax)), model)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_ylim(0, 24)
        figstyle.ygrid(ax)
    axes[0].set_ylabel("benign prompts (of 150)")
    axes[1].set_xlabel("stored bits / parameter")
    handles, labels = axes[0].get_legend_handles_labels()
    figstyle.shared_legend(
        fig, handles, labels, ncol=2,
        note="Hatched bars = loss from degenerate output; whiskers = "
             f"one-sided 95% upper bounds. {LADDER_SCORER}")
    save(fig, out, "fig_labelled_utility")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="artifacts/runs")
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    missing = [m for m, r in RUNS.items()
               if not (repo / args.runs / r / "results").exists()]
    if missing:
        raise SystemExit(
            f"no run directory for {missing}. These figures are built from the "
            "labelled XSTest runs; unzip the Colab archive at the repository "
            "root first.")

    data = load(repo)
    print(f"writing to {args.out}")
    fig_saturation(data, args.out)
    fig_truncation(data, args.out)
    fig_matrix(data, args.out)
    fig_unadjudicated(data, args.out)
    fig_marker_gap(data, args.out)
    fig_utility(repo, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
