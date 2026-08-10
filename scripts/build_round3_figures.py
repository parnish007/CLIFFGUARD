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
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import figstyle  # noqa: E402

figstyle.apply()

MODEL_STYLE = figstyle.MODEL_STYLE
CLASS_COLOUR = figstyle.CLASS_COLOUR
CLASS_SHORT = figstyle.CLASS_SHORT
UNSAFE_RED, CONSERVATIVE_BLUE = figstyle.UNSAFE, figstyle.CONSERVATIVE
INK, MUTED, HAIRLINE = "#212529", "#6C757D", "#DEE2E6"
TEXT_WIDTH_IN = figstyle.TEXT_WIDTH
MODEL_ORDER = ("Qwen2.5-3B", "Phi-3.5-mini", "SmolLM2-1.7B")
save = figstyle.save


def fmt_p(p: float) -> str:
    """A p-value a reader can compare at a glance, without false precision."""
    if p < 1e-4:
        return "p < 0.0001"
    if p >= 0.995:
        return "p ≈ 1"
    return f"p = {p:.3f}"


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

    Direct labelling rather than a legend: two colours with fixed meanings,
    named once on the axis, cost the reader nothing to remember and save a
    legend box competing with the bars for the same space.
    """
    present = stats["scorer_sensitivity"]["models"]
    models = [m for m in MODEL_ORDER if m in present]
    # "original", not "published": both gradings are published, in the same
    # document as this figure. What distinguishes them is which came first and
    # which one \S\ref{sec:scorer} shows to be defective.
    rows = [("first-token-legacy", "original"), ("letter", "corrected")]

    # One row per (model, scorer) on a single axis rather than a panel per
    # model. Two panels meant two of everything -- two axis labels, two sets
    # of direction annotations -- competing for the same strip of space under
    # the plot, and they collided. One axis needs one of each.
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.4))

    largest = max(max(present[m]["tables"][mode]["unsafe_flips"],
                      present[m]["tables"][mode]["conservative_flips"])
                  for m in models for mode, _ in rows)
    limit = largest * 1.30 + 4

    entries: list[tuple[str, str, dict[str, Any]]] = []
    for model in models:
        for mode, label in rows:
            entries.append((model, label, present[model]["tables"][mode]))

    for i, (model, label, table) in enumerate(entries):
        y = len(entries) - 1 - i
        unsafe, conservative = table["unsafe_flips"], table["conservative_flips"]
        if label == "corrected":
            ax.axhspan(y - 0.42, y + 0.42, color="#F1F3F5", zorder=0)
        ax.barh(y, -unsafe, height=0.5, color=UNSAFE_RED, zorder=3,
                edgecolor="white", linewidth=0.8)
        ax.barh(y, conservative, height=0.5, color=CONSERVATIVE_BLUE, zorder=3,
                edgecolor="white", linewidth=0.8)
        ax.text(-unsafe - limit * 0.025, y, f"{unsafe}", ha="right",
                va="center", fontsize=9, color=UNSAFE_RED, fontweight="bold",
                zorder=4)
        ax.text(conservative + limit * 0.025, y, f"{conservative}", ha="left",
                va="center", fontsize=9, color=CONSERVATIVE_BLUE,
                fontweight="bold", zorder=4)

    # p-values in their own column outside the data area, where they cannot
    # sit on the bar above. Placed in axes coordinates so the column stays put
    # whatever the counts are.
    for i, (_model, _label, table) in enumerate(entries):
        y = len(entries) - 1 - i
        ax.text(1.015, y, fmt_p(table["mcnemar_p"]),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.5, color=MUTED, clip_on=False)
    ax.text(1.015, len(entries) - 0.35, "McNemar",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8, color=MUTED, style="italic", clip_on=False)

    # Model names once, spanning their two rows. Deliberately in ink and not in
    # the model's own colour: this plate already spends red and blue on the two
    # directions, and a model name in the same red as the compliance bars reads
    # as if Phi-3.5-mini were itself the compliance case. One hue, one meaning.
    for index, model in enumerate(models):
        top = len(entries) - 1 - index * len(rows)
        ax.text(-0.30, top - 0.5, model, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                color=INK, clip_on=False)

    ax.axvline(0, color="#495057", linewidth=1.0, zorder=2)
    figstyle.xgrid(ax)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([label for _model, label, _t in reversed(entries)],
                       fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0)

    ticks = [t for t in (-40, -20, 0, 20, 40) if abs(t) < limit]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(t)) for t in ticks])
    # "toward compliance", not "unsafe": this corpus carries no per-prompt
    # harmfulness label, so a prompt the baseline declined and the rung
    # answered is a change in the model's decision and not an observation of
    # harm. The manuscript retired the word; the plate has to as well, or the
    # figure asserts what the text declines to.
    ax.set_xlabel("← toward compliance        prompts, of 500        "
                  "toward refusal →", labelpad=7)
    figstyle.panel_title(ax, None,
                         "Full precision against 4.5 bits, re-graded on identical text")

    figstyle.note_only(fig, "Blue = transitions toward refusal; red = toward compliance. Neither is an observation of harm.")
    save(fig, out, "fig_round3_scorer")


# ---------------------------------------------------------------------------
# plate 2: the generation budget, drawn as the flow it is
# ---------------------------------------------------------------------------


def _ribbon(ax: Any, x0: float, x1: float, y0: float, y1: float,
            thickness0: float, thickness1: float, colour: str,
            alpha: float) -> None:
    """One class-to-class flow, as a filled cubic band.

    A stacked bar at each budget shows two compositions and leaves the reader
    to difference them. These transitions are measured per prompt, so the
    movement itself can be drawn, and the dominant flow -- refusal into
    deflection -- becomes the visible object rather than an inference.
    """
    control = (x0 + x1) / 2
    # One closed path: along the upper edge as a cubic, straight down the far
    # side, back along the lower edge as a second cubic, closed. Built in a
    # single vertex list because a Path must begin with MOVETO, so the two
    # curves cannot be assembled from separate Path objects.
    vertices = [
        (x0, y0),
        (control, y0), (control, y1), (x1, y1),          # upper edge
        (x1, y1 + thickness1),                            # far side
        (control, y1 + thickness1), (control, y0 + thickness0),
        (x0, y0 + thickness0),                            # lower edge
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=colour,
                           edgecolor="none", alpha=alpha, zorder=1))


def fig_budget(stats: dict[str, Any], out: Path) -> None:
    """Where the extra 208 tokens send the verdict, on harmful prompts.

    Harmful prompts only, because that is the class the safety claim is about
    and because showing both classes at this size would halve every band. The
    benign movement is reported in the text and in the statistics file.
    """
    present = stats["xstest_window"]["models"]
    models = [m for m in MODEL_ORDER if m in present]
    order = ["refusal", "deflection", "disclaimer", "compliance"]

    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH_IN, 4.1))
    axes = list(np.atleast_1d(axes))
    gap, x0, x1 = 3.0, 0.0, 1.0
    total = 150

    for index, (ax, model) in enumerate(zip(axes, models)):
        block = present[model]
        transitions = block.get("budget_transitions", {}).get("harmful_only")
        if transitions is None:
            ax.set_axis_off()
            continue
        flows = transitions["transitions"]

        left_totals = {k: sum(v.values()) for k, v in flows.items()}
        right_totals: dict[str, int] = {}
        for targets in flows.values():
            for name, value in targets.items():
                right_totals[name] = right_totals.get(name, 0) + value

        def stack(totals: dict[str, int]) -> dict[str, tuple[float, float]]:
            out_pos: dict[str, tuple[float, float]] = {}
            cursor = 0.0
            for name in order:
                value = totals.get(name, 0)
                if not value:
                    continue
                out_pos[name] = (cursor, float(value))
                cursor += value + gap
            return out_pos

        left, right = stack(left_totals), stack(right_totals)

        # Ribbons first, so the solid stacks sit on top of their edges.
        left_cursor = {k: v[0] for k, v in left.items()}
        right_cursor = {k: v[0] for k, v in right.items()}
        for source in order:
            for target in order:
                value = flows.get(source, {}).get(target, 0)
                if not value:
                    continue
                y0, y1 = left_cursor[source], right_cursor[target]
                moved = source != target
                _ribbon(ax, x0 + 0.075, x1 - 0.075, y0, y1, value, value,
                        CLASS_COLOUR[source], 0.55 if moved else 0.16)
                left_cursor[source] += value
                right_cursor[target] += value

        for positions, x in ((left, x0), (right, x1)):
            for name, (base, height) in positions.items():
                ax.add_patch(plt.Rectangle((x - 0.075, base), 0.15, height,
                                           facecolor=CLASS_COLOUR[name],
                                           edgecolor="white", linewidth=0.7,
                                           zorder=3))

        # Counts beside each stack, outward, so they never sit on a ribbon.
        # Bands under about six prompts are thinner than their own label, so
        # those are nudged apart rather than allowed to overprint.
        def label_stack(positions: dict[str, tuple[float, float]], x: float,
                        ha: str, dx: float) -> None:
            placed: list[float] = []
            for name, (base, height) in positions.items():
                if height < 2:
                    continue
                y = base + height / 2
                while any(abs(y - other) < 5.0 for other in placed):
                    y += 5.0
                placed.append(y)
                ax.text(x + dx, y, f"{int(height)}", ha=ha, va="center",
                        fontsize=8.5, color=CLASS_COLOUR[name],
                        fontweight="bold", zorder=4)

        label_stack(left, x0, "right", -0.115)
        label_stack(right, x1, "left", 0.115)

        ax.set_xlim(-0.42, 1.42)
        # Headroom above the stacks for the compliance line below. At -8 the
        # line sat on the axes' own top edge and its ascenders printed through
        # the panel title, which is the one place in the plate a reader has to
        # be able to read two things at once.
        ax.set_ylim(total + 3 * gap + 4, -26)
        ax.set_xticks([x0, x1])
        ax.set_xticklabels(["48", "256"], fontsize=9, color=INK)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)
        figstyle.panel_title(ax, chr(ord("a") + index), model)
        if index == 0:
            ax.set_ylabel("150 harmful prompts", labelpad=2)

        # The harmful-compliance cell is the one number a reader came for, and
        # at 1 of 150 it is a hairline the stack label alone will not rescue.
        # A single headline per panel, stated whether the cell is empty or not,
        # because a zero here is the paper's claim and a one is the exception
        # to it -- and neither should depend on finding a two-pixel band.
        complied_48 = left_totals.get("compliance", 0)
        complied_256 = right_totals.get("compliance", 0)
        empty = not complied_48 and not complied_256
        # "compliance", not "harmful compliance": the y-label already says these
        # are 150 harmful prompts, and the longer string was wider than the
        # panel, so the three of them ran into each other across the plate.
        ax.text((x0 + x1) / 2, -13.0,
                f"compliance  {complied_48} → {complied_256}",
                ha="center", va="center", fontsize=8.5,
                fontweight="normal" if empty else "bold",
                color=MUTED if empty else CLASS_COLOUR["compliance"])

    figstyle.shared_legend(
        fig, [Patch(facecolor=CLASS_COLOUR[k]) for k in order],
        [CLASS_SHORT[k] for k in order], ncol=4,
        note="Ribbons show verdict transitions from 48 to 256 generated tokens.",
        reserve=0.28)
    save(fig, out, "fig_round3_budget")


# ---------------------------------------------------------------------------
# plate 3: reproducibility
# ---------------------------------------------------------------------------


def fig_reproducibility(stats: dict[str, Any], out: Path) -> None:
    """How much text changed, and how little of it reached a verdict.

    Paired bars on one full-width axis. An earlier version squeezed a second
    panel beside this one for the batch-size association and put the verdict
    percentages in a side column; between them they left no room, and the
    labels landed on each other. The association is a sentence, not a plate,
    and it is made in the text instead.

    The two bars per row are the result: the generated text is not
    reproducible and the labels very nearly are, and the gap between the pair
    is the thing to look at.
    """
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.3))

    # Group headers occupy their own y positions so they can never sit on a
    # bar or a tick label.
    entries: list[dict[str, Any]] = []
    drift_models = stats["generation_drift"]["models"]
    entries.append({"header": "HH-RLHF, 500 prompts   ·   batch 16 then 8"})
    for model in (m for m in MODEL_ORDER if m in drift_models):
        block = drift_models[model]
        for scheme in ("FP16", "RTN_4B"):
            text, labels = block["text"][scheme], block["labels"][scheme]
            entries.append({
                "model": model,
                "label": f"{model}  {scheme.replace('RTN_4B', '4.5 bits')}",
                "text": 100 * text["differ"] / text["n"],
                "verdict": 100 * labels["disagree"] / labels["n"]})
    entries.append({"spacer": True})
    entries.append({"header": "XSTest, 300 prompts   ·   batch 8 both"})
    xstest_models = stats["xstest_window"]["models"]
    for model in (m for m in MODEL_ORDER if m in xstest_models):
        drift = xstest_models[model]["decoder_drift_bound"]
        entries.append({
            "model": model, "label": f"{model}  FP16",
            "text": 100 * drift["share_diverged"], "verdict": 0.0})

    height, ticks, labels_out = 0.30, [], []
    crowded_pairs: list[tuple[Any, float, float]] = []
    for i, entry in enumerate(entries):
        y = len(entries) - 1 - i
        if entry.get("spacer"):
            continue
        if "header" in entry:
            ax.text(-0.6, y, entry["header"], ha="left", va="center",
                    fontsize=8.5, color=MUTED, style="italic")
            continue
        # Two colours, one meaning each: grey is text, red is verdict. Colouring
        # the text bar by model looked richer and was wrong -- the legend swatch
        # said grey while the bars were blue, red and green, and Phi's text bar
        # was the same red as every verdict bar. The model is already named on
        # the axis, so it does not need a second encoding.
        ax.barh(y + height / 2, entry["text"], height=height, color=MUTED,
                edgecolor="white", linewidth=0.7, zorder=3)
        ax.barh(y - height / 2, entry["verdict"], height=height,
                color=UNSAFE_RED, edgecolor="white", linewidth=0.7, zorder=3)
        # Two labels one bar-height apart collide when the two bars end at the
        # same place, which is exactly what the XSTest rows do: both are 0.0%,
        # both labels land on x = 0.22, and 0.30 axis units is less than a line
        # of 8.5pt type. Where the bars separate, label each at its own end;
        # where they do not, put the pair on one line, in reading order.
        crowded = abs(entry["text"] - entry["verdict"]) < 1.0
        label = ax.text(entry["text"] + 0.22, y if crowded else y + height / 2,
                        f'{entry["text"]:.1f}%', va="center", fontsize=8.5,
                        color="#404040", fontweight="bold")
        if crowded:
            crowded_pairs.append((label, y, entry["verdict"]))
        else:
            ax.text(entry["verdict"] + 0.22, y - height / 2,
                    f'{entry["verdict"]:.1f}%', va="center", fontsize=8.5,
                    color=UNSAFE_RED)
        ticks.append(y)
        labels_out.append(entry["label"])

    # The crowded rows' second label goes immediately after the first, measured
    # rather than guessed: the strings are 0.0% here and need not stay that way,
    # and a hard-coded offset that fits "0.0%" clips "12.4%".
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, y, verdict in crowded_pairs:
        box = label.get_window_extent(renderer)
        end = ax.transData.inverted().transform((box.x1, box.y0))[0]
        ax.text(end + 0.30, y, f"{verdict:.1f}%", va="center", fontsize=8.5,
                color=UNSAFE_RED)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels_out, fontsize=8.5, color=INK)
    ax.set_ylim(-0.8, len(entries) - 0.2)
    ax.set_xlim(0, 15.2)
    ax.set_xlabel("percent of prompts")
    figstyle.xgrid(ax)
    ax.tick_params(axis="y", length=0)
    figstyle.panel_title(ax, None, "Generated twice under greedy decoding")
    figstyle.shared_legend(
        fig, [Patch(facecolor=MUTED), Patch(facecolor=UNSAFE_RED)],
        ["generated text differs", "verdict differs"], ncol=2)
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
