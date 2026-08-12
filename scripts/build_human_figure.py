"""The human-validation plate: what the merged class is, and who recovers it.

Two panels, because the result has two halves that a single chart would blur.

  (a) The composition of the class every three-way instrument collapses into
      one label. This is the paper's long-standing assertion turned into a
      measurement, and the point of the panel is that no bar dominates: the
      class is three behaviours in near-equal thirds.

  (b) Per-instrument recall on each of those three behaviours, which is where
      the phrase list's failure stops looking like noise. Plotted as recall by
      KIND rather than as an accuracy bar per instrument, because the accuracy
      bar is what hides the finding -- a single number cannot show that one
      instrument is excellent on one sub-behaviour and near-blind on another.

Usage:
  python scripts/build_human_figure.py
  python scripts/build_human_figure.py --stats docs/paper/human_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from scripts import figstyle  # noqa: E402

# The paper's own class colours, so a bar here means what it means everywhere
# else in the document.
KIND_COLOUR = {
    "REFUSE": figstyle.CLASS_COLOUR["refusal"],
    "DEFLECT": figstyle.CLASS_COLOUR["deflection"],
    "DISCLAIM": figstyle.CLASS_COLOUR["disclaimer"],
}
KINDS = ("REFUSE", "DEFLECT", "DISCLAIM")

# Display names, and the order the panel argues in: worst instrument first, so
# the eye travels from the phrase list's gap up to the judges' coverage.
INSTRUMENTS = (
    ("phrase_list", "phrase\nlist"),
    ("judge_mc", "judge\nMC"),
    ("judge_original", "judge\nfirst-token"),
)


def panel_composition(ax, dec: dict) -> None:
    """One stacked bar: what a three-way REFUSE label actually contains."""
    left = 0.0
    total = dec["n_broad_declines"]
    for kind in KINDS:
        share = dec["share"][kind] * 100
        ax.barh(0, share, left=left, height=0.55, color=KIND_COLOUR[kind],
                edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(left + share / 2, 0, f"{share:.1f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=4)
        ax.text(left + share / 2, -0.42, f"{kind.lower()}\nn={dec['counts'][kind]}",
                ha="center", va="top", fontsize=8, color="#404040")
        left += share

    ax.set_xlim(0, 100)
    # Just enough room under the bar for the two-line labels and no more; a
    # taller box leaves a band of blank paper that reads as a missing series.
    ax.set_ylim(-0.92, 0.42)
    ax.set_yticks([])
    ax.set_xlabel("share of the merged declining class (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    figstyle.panel_title(
        ax, "a", f"What one “refuse” label contains (n={total})")


def panel_recall(ax, stats: dict) -> None:
    """Grouped bars: each instrument's recall on each kind of decline."""
    names = [n for n, _ in INSTRUMENTS if n in stats["instruments"]]
    width = 0.26
    x = np.arange(len(names))

    for offset, kind in enumerate(KINDS):
        values = [
            (stats["instruments"][n]["recall_by_underlying"][kind]["recall"] or 0)
            * 100 for n in names]
        pos = x + (offset - 1) * width
        ax.bar(pos, values, width, color=KIND_COLOUR[kind], edgecolor="black",
               linewidth=0.5, zorder=3, label=kind.lower())
        for xi, v in zip(pos, values):
            ax.text(xi, v + 1.8, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color="#404040")

    figstyle.ygrid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([dict(INSTRUMENTS)[n] for n in names])
    ax.set_ylabel("recall (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    figstyle.panel_title(ax, "b", "Recall on declines, by kind")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path,
                    default=Path("docs/paper/human_validation.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    if "decomposition" not in stats:
        raise SystemExit(
            f"{args.stats} predates the label-space correction and carries no "
            "`decomposition`. Re-run scripts/score_human_labels.py.")

    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH, 3.3),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    panel_composition(axes[0], stats["decomposition"])
    panel_recall(axes[1], stats)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=KIND_COLOUR[k],
                             edgecolor="black", linewidth=0.5) for k in KINDS]
    figstyle.shared_legend(
        fig, handles, [k.lower() for k in KINDS], ncol=3,
        note=("Human labels on 300 blinded completions, one annotator. Panel "
              "(b) is recall on the completions a person assigned to each "
              "kind; a three-way instrument scores a hit whenever it calls "
              "them declining at all."),
        reserve=0.30)
    figstyle.save(fig, args.out, "fig_human_validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
