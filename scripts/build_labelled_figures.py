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

# Shared with build_paper_figures.py so the two sets of plates look like one
# document. Colour-blind safe; marker and dash carry the same information as
# hue for greyscale print.
MODEL_STYLE = {
    "Qwen2.5-3B": ("#0B7285", "s", "-"),
    "Phi-3.5-mini": ("#E8590C", "^", "--"),
    "SmolLM2-1.7B": ("#5F3DC4", "o", "-."),
}
CLASS_COLOUR = {
    "refusal": "#1864AB",
    "deflection": "#4DABF7",
    "disclaimer": "#B197FC",
    "compliance": "#C92A2A",
    "unclear": "#CED4DA",
    "degenerate": "#868E96",
}
# Written out rather than truncated at plot time: slicing the class names to
# seven characters produced "complia" and "degener" on the axis.
CLASS_SHORT = {
    "refusal": "refuse",
    "deflection": "deflect",
    "disclaimer": "disclaim",
    "compliance": "comply",
    "unclear": "unclear",
    "degenerate": "degen",
}
MARKER_RED = "#C92A2A"   # the phrase-list arm, as in build_paper_figures.py
TEXT_WIDTH_IN = 6.30

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
RUNG_LABEL = {"FP16": "FP16", "RTN_8B": "8.5", "RTN_7B": "7.5",
              "RTN_6B": "6.5", "RTN_5B": "5.5", "RTN_4B": "4.5",
              "RTN_3B": "3.5", "RTN_2B": "2.5"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def save(fig: Any, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


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
        data[model] = {
            "resolved": tax["resolved"],
            "harm": prompts["harm_label"],
            "completions": completions,
            "threshold": tax["degeneracy_threshold"],
        }
    return data


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
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.35), sharey=True)
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
        ax.set_xticklabels([RUNG_LABEL[s] for s in schemes], fontsize=7)
        ax.set_title(model, pad=4)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("harmful prompts")
    axes[1].set_xlabel("stored bits / parameter")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("The safety endpoint never fires: 'compliance' is absent at "
                 "every rung", y=1.02, fontsize=9.5)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.4))
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

    for ax, title in ((ax1, "completions reaching the 48-token cap"),
                      (ax2, "judged 'compliance', benign prompts")):
        ax.set_xticks(range(len(SCHEMES)))
        ax.set_xticklabels([RUNG_LABEL[s] for s in SCHEMES], fontsize=7)
        ax.set_xlabel("stored bits / parameter")
        ax.set_title(title, pad=4, fontsize=8.5)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 30)
    ax1.legend(frameon=False, loc="lower left")
    fig.suptitle("A 48-token budget is the ceiling on the compliance class",
                 y=1.10, fontsize=9.5)
    fig.subplots_adjust(wspace=0.30)
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
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 1.95))
    for ax, (model, d) in zip(axes, data.items()):
        grid = [[counts(d["resolved"]["FP16"], d["harm"], cls)[c]
                 for c in order] for cls in ("harmful", "benign")]
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=150, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                # The empty safety cell is the point of the figure, so it is
                # marked rather than left as an unremarkable white square.
                emphasis = (i == 0 and order[j] == "compliance")
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color=("#C92A2A" if emphasis else
                               "white" if v > 75 else "#212529"),
                        fontweight="bold" if emphasis else "normal")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CLASS_SHORT[c] for c in order], rotation=45,
                           ha="right", fontsize=6.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["harmful", "benign"] if ax is axes[0] else ["", ""],
                           fontsize=7)
        ax.set_title(model, pad=4, fontsize=8.5)
        ax.grid(visible=False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).set_label(
        "prompts (of 150)", fontsize=7)
    fig.suptitle("Full precision, the two axes crossed — the safety cell is "
                 "empty before quantization", y=1.06, fontsize=9.5)
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
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.3))
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
    ax.set_xticklabels([RUNG_LABEL[s] for s in SCHEMES], fontsize=7)
    ax.set_xlabel("stored bits / parameter")
    ax.set_ylabel("harmful prompts (of 150)")
    ax.set_ylim(0, 88)
    ax.set_title("Withheld verdicts that match a turn toward answering — "
                 "unadjudicated, not harmful", pad=6)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    # Without this the empty 3- and 2-bit columns read as safety improving at
    # the hardest rungs, when they are the model having stopped producing
    # language altogether.
    ax.text(0.99, 0.95, "empty at 3-2 bits because output is degenerate,\n"
                        "not because the models became safer",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="#495057")
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

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.15))
    models = list(data)
    for row, model in enumerate(models):
        d = data[model]
        labels = d["resolved"]["FP16"]
        comp = d["completions"]["FP16"]
        declines = sum(1 for lab in labels
                       if lab in ("refusal", "deflection", "disclaimer"))
        marked = sum(1 for c in comp if has_refusal_marker(c))
        colour, marker, _ = MODEL_STYLE[model]
        ax.plot([marked, declines], [row, row], color="#ADB5BD", linewidth=3,
                solid_capstyle="round", zorder=1)
        ax.scatter([marked], [row], s=52, color=MARKER_RED, marker=marker,
                   zorder=3, label="refusal-phrase list" if row == 0 else None)
        ax.scatter([declines], [row], s=52, color=colour, marker="o",
                   zorder=3, label="five-way judge" if row == 0 else None)
        ax.text((marked + declines) / 2, row + 0.22,
                f"{marked / declines:.0%} recall", ha="center", fontsize=7,
                color="#495057")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    # Headroom above the top row for the legend, which otherwise lands on the
    # bottom dumbbell.
    ax.set_ylim(-0.45, len(models) - 0.10)
    ax.set_xlim(0, 310)
    ax.set_xlabel("full-precision completions placed in a declining class "
                  "(of 300)")
    ax.set_title("The same phrase list reads three families three different "
                 "ways", pad=18)
    ax.legend(frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.10))
    ax.grid(axis="y", visible=False)
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
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.3), sharey=True)
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
        colour, _, _ = MODEL_STYLE[model]
        ax.bar(xs, rates, width=0.68, color=colour, label="usefulness lost")
        ax.bar(xs, degen, width=0.68, color="white", edgecolor=colour,
               hatch="////", linewidth=0.6, label="…of which degenerate")
        ax.errorbar(xs, rates, yerr=[[0] * len(rows),
                                     [u - r for u, r in zip(upper, rates)]],
                    fmt="none", ecolor="#343A40", elinewidth=0.8, capsize=2.2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([RUNG_LABEL[r["scheme"]] for r in rows], fontsize=7)
        ax.set_title(model, pad=4, fontsize=8.5)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_ylim(0, 24)
    axes[0].set_ylabel("benign prompts (of 150)")
    axes[1].set_xlabel("stored bits / parameter")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Usefulness lost on benign prompts, with one-sided 95% upper "
                 "bounds", y=1.03, fontsize=9.5)
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
