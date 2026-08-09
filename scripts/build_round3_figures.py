"""Figures for round 3: what survives re-measurement, and what does not.

Three plates, one per question the round was run to settle.

The design problem shared by all three is that these are re-measurements, and a
re-measurement figure invites the reader to see a correction as a discovery. So
each plate shows the OLD number and the NEW number in the same frame, at the
same scale, rather than showing the new one and describing the old in a
caption. Where the two agree that is visible; where they disagree, so is the
size of the disagreement.

Usage:
  python scripts/build_round3_figures.py
  python scripts/build_round3_figures.py --out docs/paper/figures
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
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Shared with build_paper_figures.py and build_labelled_figures.py so all three
# sets of plates look like one document.
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
CLASS_SHORT = {
    "refusal": "refuse", "deflection": "deflect", "disclaimer": "disclaim",
    "compliance": "comply", "unclear": "unclear", "degenerate": "degen",
}
UNSAFE_RED, CONSERVATIVE_BLUE = "#C92A2A", "#1864AB"
TEXT_WIDTH_IN = 6.30

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


def fmt_p(p: float) -> str:
    """A p-value a reader can compare at a glance, without false precision."""
    if p < 1e-4:
        return "$p<10^{-4}$"
    if p >= 0.995:
        return "$p\\approx1$"
    return f"$p={p:.3f}$"


# ---------------------------------------------------------------------------
# plate 1: the label scorer
# ---------------------------------------------------------------------------


def fig_scorer(stats: dict[str, Any], out: Path) -> None:
    """The same completions, two scorers, one diverging axis.

    Diverging rather than grouped because the quantity that carries the claim
    is the IMBALANCE between the two directions, not either count. A grouped
    bar makes the reader do that subtraction; a diverging one shows it as
    asymmetry about zero, and the near-symmetry of Phi's corrected row is then
    the whole finding without a sentence of explanation.
    """
    present = stats["scorer_sensitivity"]["models"]
    # Explicit, not dictionary order: the JSON is sorted alphabetically and
    # would put Phi-3.5-mini first, against the ordering every other figure and
    # table in the paper uses.
    models = [m for m in ("Qwen2.5-3B", "Phi-3.5-mini", "SmolLM2-1.7B")
              if m in present]
    rows = [("first-token-legacy", "published scorer"),
            ("letter", "corrected scorer")]

    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH_IN, 2.05),
                             sharex=True, sharey=True)
    axes = list(axes) if len(models) > 1 else [axes]

    limit = max(t["unsafe_flips"] if k == "u" else t["conservative_flips"]
                for m in models for mode, _ in rows
                for t in [present[m]["tables"][mode]] for k in ("u", "c"))
    limit = limit * 1.35 + 3

    for ax, model in zip(axes, models):
        block = present[model]
        for i, (mode, _label) in enumerate(rows):
            table = block["tables"][mode]
            y = len(rows) - 1 - i
            unsafe, conservative = table["unsafe_flips"], table["conservative_flips"]
            ax.barh(y, -unsafe, height=0.46, color=UNSAFE_RED,
                    edgecolor="white", linewidth=0.6)
            ax.barh(y, conservative, height=0.46, color=CONSERVATIVE_BLUE,
                    edgecolor="white", linewidth=0.6)
            ax.text(-unsafe - limit * 0.035, y, str(unsafe), ha="right",
                    va="center", fontsize=8, color=UNSAFE_RED, fontweight="bold")
            ax.text(conservative + limit * 0.035, y, str(conservative), ha="left",
                    va="center", fontsize=8, color=CONSERVATIVE_BLUE,
                    fontweight="bold")
            ax.text(0, y + 0.30, fmt_p(table["mcnemar_p"]), ha="center",
                    va="bottom", fontsize=7, color="#495057")

        ax.axvline(0, color="#343A40", linewidth=0.9)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.55, len(rows) - 0.25)
        ax.set_title(model, color=MODEL_STYLE[model][0])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("prompts")

    # Row labels once, on the shared y axis, so they cannot land on top of a
    # bar in the second panel.
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for _, label in reversed(rows)], fontsize=8)

    fig.legend(handles=[Patch(facecolor=UNSAFE_RED, label="unsafe flip"),
                        Patch(facecolor=CONSERVATIVE_BLUE,
                              label="conservative flip")],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Full precision against 4.5 bits, re-graded on identical text",
                 fontsize=9.5, y=1.03)
    fig.tight_layout()
    save(fig, out, "fig_round3_scorer")


# ---------------------------------------------------------------------------
# plate 2: the generation budget
# ---------------------------------------------------------------------------


def fig_budget(stats: dict[str, Any], out: Path) -> None:
    """Where the extra 208 tokens send the verdict, harmful and benign apart.

    Stacked composition at each budget rather than a single rate, because the
    movement here is almost entirely BETWEEN the withholding classes -- refusal
    into deflection -- and a plot of the compliance rate alone would show a
    flat line and report that nothing happened.
    """
    present = stats["xstest_window"]["models"]
    models = [m for m in ("Qwen2.5-3B", "Phi-3.5-mini", "SmolLM2-1.7B")
              if m in present]
    order = ["refusal", "deflection", "disclaimer", "compliance"]
    fig, axes = plt.subplots(2, len(models),
                             figsize=(TEXT_WIDTH_IN, 3.35), sharey=True)

    for column, model in enumerate(models):
        block = present[model]
        for row, klass in enumerate(("harmful", "benign")):
            ax = axes[row][column]
            key = f"{klass}_counts"
            for x, budget in enumerate(("tokens_48", "tokens_256")):
                counts = block[budget]["letter"][key]
                bottom = 0
                for name in order:
                    value = counts.get(name, 0)
                    if not value:
                        continue
                    ax.bar(x, value, width=0.62, bottom=bottom,
                           color=CLASS_COLOUR[name], edgecolor="white",
                           linewidth=0.7)
                    if value >= 12:
                        ax.text(x, bottom + value / 2, str(value), ha="center",
                                va="center", fontsize=7, color="white",
                                fontweight="bold")
                    bottom += value

                # The harmful-compliance cell is the one number in this plate
                # a reader came for, and at 1 of 150 it is a two-pixel sliver
                # that a stacked bar hides completely. Called out explicitly:
                # a zero here is the paper's claim and a one is the exception
                # to it, and neither should depend on the reader's eyesight.
                if klass == "harmful":
                    complied = counts.get("compliance", 0)
                    ax.annotate(
                        f"comply {complied}",
                        xy=(x, 150), xytext=(x, 178),
                        ha="center", va="bottom", fontsize=7,
                        color=(CLASS_COLOUR["compliance"] if complied
                               else "#868E96"),
                        fontweight="bold" if complied else "normal",
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color=(CLASS_COLOUR["compliance"]
                                               if complied else "#CED4DA")))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["48", "256"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(0, 205 if row == 0 else 158)
            ax.set_yticks([0, 50, 100, 150])
            ax.grid(axis="x", visible=False)
            if column == 0:
                ax.set_ylabel(f"{klass} prompts")
            if row == 0:
                ax.set_title(model, color=MODEL_STYLE[model][0], pad=16)
            if row == 1:
                ax.set_xlabel("generated tokens")

    fig.legend(handles=[Patch(facecolor=CLASS_COLOUR[k], label=CLASS_SHORT[k])
                        for k in order],
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("One generation read at two lengths (XSTest, full precision)",
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, out, "fig_round3_budget")


# ---------------------------------------------------------------------------
# plate 3: reproducibility
# ---------------------------------------------------------------------------


def fig_reproducibility(stats: dict[str, Any], out: Path) -> None:
    """How much text changed, and how little of it reached a verdict.

    Two bars per case, on one axis, because the gap between them IS the result:
    the generated text is not reproducible and the labels very nearly are. A
    figure showing only the text divergence would read as a reason to distrust
    the paper; showing only the label divergence would hide why that is
    surprising.
    """
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, 2.6),
        gridspec_kw={"width_ratios": [2.0, 1.0]})

    ordering = ("Qwen2.5-3B", "Phi-3.5-mini", "SmolLM2-1.7B")
    rows: list[tuple[str, float, float, int]] = []
    drift_models = stats["generation_drift"]["models"]
    for model in (m for m in ordering if m in drift_models):
        block = drift_models[model]
        batch = block["batch_size"]["independent_48"]
        for scheme in ("FP16", "RTN_4B"):
            text, labels = block["text"][scheme], block["labels"][scheme]
            rows.append((f"{model}  {scheme}",
                         100 * text["differ"] / text["n"],
                         100 * labels["disagree"] / labels["n"], batch))
    split = len(rows)
    xstest_models = stats["xstest_window"]["models"]
    for model in (m for m in ordering if m in xstest_models):
        block = xstest_models[model]
        drift = block["decoder_drift_bound"]
        rows.append((f"{model}  FP16", 100 * drift["share_diverged"], 0.0,
                     block["provenance"]["tokens_48"]["batch_size"]))

    y = range(len(rows))
    height = 0.36
    ax.barh([i - height / 2 for i in y], [r[1] for r in rows], height=height,
            color="#ADB5BD", edgecolor="white", linewidth=0.6,
            label="generated text differs")
    ax.barh([i + height / 2 for i in y], [r[2] for r in rows], height=height,
            color=UNSAFE_RED, edgecolor="white", linewidth=0.6,
            label="verdict differs")
    for i, row in enumerate(rows):
        ax.text(row[1] + 0.22, i - height / 2, f"{row[1]:.1f}%", va="center",
                fontsize=7, color="#495057")
        ax.text(max(row[2], 0) + 0.22, i + height / 2, f"{row[2]:.1f}%",
                va="center", fontsize=7, color=UNSAFE_RED)
    # The two arms are different corpora and different budgets; a rule between
    # them stops the eye reading eleven comparable rows.
    ax.axhline(split - 0.5, color="#CED4DA", linewidth=0.8, zorder=0)
    ax.text(14.6, split - 0.62, "HH-RLHF, batch 16 vs 8", ha="right",
            va="bottom", fontsize=7, color="#868E96", style="italic")
    ax.text(14.6, split - 0.38, "XSTest, batch 8 vs 8", ha="right", va="top",
            fontsize=7, color="#868E96", style="italic")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("percent of prompts (500 HH-RLHF, 300 XSTest)")
    ax.set_xlim(0, 15)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("Greedy decoding, generated twice", loc="left")

    # The association, drawn as the two-condition comparison it is. Points are
    # jittered because three XSTest models all sit exactly on zero and would
    # otherwise be one dot standing for three.
    import numpy as np

    rng = np.random.default_rng(0)
    keys = ["batch 16 vs 8", "batch 8 vs 8"]
    for x, key in enumerate(keys):
        values = [r[1] for r in rows
                  if (r[3] == 16) == (key == "batch 16 vs 8")]
        ax2.scatter(x + rng.uniform(-0.11, 0.11, len(values)), values, s=28,
                    zorder=3, color="#343A40", edgecolor="white", linewidth=0.6)
        ax2.text(x, 14.2, f"n={len(values)}", ha="center", fontsize=7,
                 color="#868E96")
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys, fontsize=7.5)
    ax2.set_xlim(-0.5, len(keys) - 0.5)
    ax2.set_ylim(-1.2, 15)
    ax2.set_ylabel("generated text differs (%)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("An association,\nnot an experiment", loc="left", fontsize=8.5)

    fig.tight_layout()
    save(fig, out, "fig_round3_reproducibility")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stats", type=Path,
                    default=repo / "docs" / "paper" / "round3_stats.json")
    ap.add_argument("--out", type=Path,
                    default=repo / "docs" / "paper" / "figures")
    args = ap.parse_args()

    if not args.stats.is_file():
        print(f"no stats at {args.stats}; run scripts/analyse_round3.py first",
              file=sys.stderr)
        return 1
    stats = json.loads(args.stats.read_text(encoding="utf-8"))

    print("round-3 figures:")
    fig_scorer(stats, args.out)
    fig_budget(stats, args.out)
    fig_reproducibility(stats, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
