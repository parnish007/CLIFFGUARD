"""The two plates round 4 needs, in the manuscript's house style.

Both answer questions of the same shape -- "how much does this number move when
something that is not the treatment changes?" -- and they are separate plates
because the answers differ, which is the result.

  fig_round4_scorer   The whole ladder graded twice. The drift slope with its
                      bootstrap interval, and the 14 transition cells, original
                      scorer against the single-token multiple-choice one. What
                      it has to show is that the disagreement is not a uniform
                      attenuation: one model holds and one reverses sign, and a
                      pooled coefficient hides exactly that.

  fig_round4_order    The option-order audit, plotted as the comparison that
                      matters. Absolute refusal rates against paired
                      differences, on one axis so the ten-point spread of the
                      first and the one-point spread of the second are read
                      against each other rather than in separate panels at
                      separate scales.

Usage:
  python scripts/build_round4_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import figstyle

# The two graders, and one colour each, held across both plates. The original
# is the neutral grey of a superseded measurement; the replacement takes the
# conservative blue the manuscript already uses for the refusal direction.
ORIGINAL = figstyle.NEUTRAL
CORRECTED = figstyle.CONSERVATIVE
MODELS = ("Qwen2.5-3B", "Phi-3.5-mini")


def scorer_plate(stats: dict[str, Any], out: Path) -> None:
    comparison = stats["scorer_comparison"]
    drift, cells = comparison["drift"], comparison["cells"]

    fig, axes = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH, 3.15))

    # ---- (a) the slope, both scorers, with intervals --------------------
    ax = axes[0]
    rows = [*MODELS, "_pooled"]
    labels = ["Qwen2.5-3B", "Phi-3.5-mini", "pooled"]
    y = np.arange(len(rows))[::-1]
    offset = 0.17
    for shift, key, colour, name in ((+offset, "original", ORIGINAL, "original scorer"),
                                     (-offset, "corrected", CORRECTED,
                                      "single-token MC scorer")):
        est = [drift[r][key]["kappa"] for r in rows]
        lo = [drift[r][key]["ci_low"] for r in rows]
        hi = [drift[r][key]["ci_high"] for r in rows]
        ax.errorbar(est, y + shift,
                    xerr=[np.array(est) - np.array(lo),
                          np.array(hi) - np.array(est)],
                    fmt="o", markersize=4.5, capsize=2.6, linewidth=1.4,
                    color=colour, label=name, zorder=3)
    # Zero is the line the Phi interval crosses, which is the panel's point.
    ax.axvline(0.0, color=figstyle.UNSAFE, linewidth=1.0, linestyle=":",
               zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("refusal drift $\\kappa$ (points per bit removed)")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    figstyle.xgrid(ax)
    figstyle.panel_title(ax, "a", "The slope, graded twice")

    # ---- (b) the 14 cells, paired -----------------------------------------
    ax = axes[1]
    for model in MODELS:
        colour, marker, _ = figstyle.MODEL_STYLE[model]
        rows_m = sorted((c for c in cells if c["model"] == model),
                        key=lambda c: c["bits"])
        x = [100 * c["original"]["rate_itt"] for c in rows_m]
        y2 = [100 * c["corrected"]["rate_itt"] for c in rows_m]
        # A ring marks a cell whose Holm verdict changed; both are Phi's, and
        # marking them is the whole reason this is a scatter and not a table.
        changed = [c["significance"] != "unchanged" for c in rows_m]
        ax.scatter(x, y2, s=34, color=colour, marker=marker,
                   edgecolor="white", linewidth=0.6, zorder=3, label=model)
        ax.scatter([a for a, ch in zip(x, changed) if ch],
                   [b for b, ch in zip(y2, changed) if ch],
                   s=150, facecolor="none", edgecolor=figstyle.UNSAFE,
                   linewidth=1.5, zorder=4)
    top = 5.4
    ax.plot([0, top], [0, top], color=figstyle.RULE, linewidth=1.0,
            linestyle="--", zorder=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    # Bare "%", not the TeX-escaped form: these labels are drawn by matplotlib,
    # which is not in LaTeX mode here, so a backslash renders literally.
    ax.set_xlabel("original scorer (%)")
    ax.set_ylabel("single-token MC scorer (%)")
    figstyle.ygrid(ax)
    figstyle.panel_title(ax, "b", "Refusal$\\rightarrow$compliance, 14 cells")

    handles = [
        plt.Line2D([], [], color=ORIGINAL, marker="o", linestyle="-",
                   markersize=4.5, label="original scorer"),
        plt.Line2D([], [], color=CORRECTED, marker="o", linestyle="-",
                   markersize=4.5, label="single-token MC scorer"),
        plt.Line2D([], [], color=figstyle.MODEL_STYLE["Qwen2.5-3B"][0],
                   marker="o", linestyle="none", label="Qwen2.5-3B"),
        plt.Line2D([], [], color=figstyle.MODEL_STYLE["Phi-3.5-mini"][0],
                   marker="s", linestyle="none", label="Phi-3.5-mini"),
        plt.Line2D([], [], color=figstyle.UNSAFE, marker="o", linestyle="none",
                   markerfacecolor="none", markersize=9, markeredgewidth=1.5,
                   label="Holm verdict changed"),
    ]
    figstyle.shared_legend(
        fig, handles, [h.get_label() for h in handles], ncol=3,
        note="Identical completions, identical gate, identical coherent band; "
             "only the grader's label extraction and prompt format differ. "
             "(a) dotted red line is zero drift. (b) dashed line is equality.",
        reserve=0.30)
    figstyle.save(fig, out, "fig_round4_scorer")


def order_plate(stats: dict[str, Any], out: Path) -> None:
    order = stats["option_order"]
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH, 3.15))

    names = ["canonical", "COMPLY,UNCLEAR,REFUSE", "UNCLEAR,COMPLY,REFUSE"]
    short = ["canonical", "CUR", "UCR"]
    x = np.arange(len(names))

    # ---- (a) absolute rates travel ----------------------------------------
    ax = axes[0]
    for model in MODELS:
        colour, marker, dash = figstyle.MODEL_STYLE[model]
        block = order[model]["orders"]
        vals = [block[n]["refusal_pct_base"] for n in names]
        ax.plot(x, vals, marker=marker, linestyle=dash, color=colour,
                markersize=5, linewidth=1.5, zorder=3, label=model)
        span = order[model]["spread"]["fp16_refusal_pct_range"]
        ax.annotate(f"{span:.1f} pp", xy=(x[-1], vals[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=colour, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    # Right margin for the range annotation at the last assignment, which the
    # axes edge otherwise clips mid-number.
    ax.set_xlim(-0.25, len(names) - 0.45)
    ax.set_ylabel("full-precision refusal (%)")
    ax.set_ylim(66, 90)
    figstyle.ygrid(ax)
    figstyle.panel_title(ax, "a", "Absolute rate: not identified")

    # ---- (b) the paired difference does not ------------------------------
    ax = axes[1]
    for model in MODELS:
        colour, marker, dash = figstyle.MODEL_STYLE[model]
        block = order[model]["orders"]
        vals = [block[n]["delta_pp"] for n in names]
        sig = [block[n]["mcnemar_p"] < 0.05 for n in names]
        ax.plot(x, vals, marker=marker, linestyle=dash, color=colour,
                markersize=5, linewidth=1.5, zorder=3, label=model)
        # Filled = significant under exact McNemar, hollow = not. Qwen is
        # filled at every assignment and Phi at none, which is the panel.
        ax.scatter([xi for xi, s in zip(x, sig) if s],
                   [v for v, s in zip(vals, sig) if s],
                   s=64, color=colour, marker=marker, zorder=4,
                   edgecolor="white", linewidth=0.7)
        ax.scatter([xi for xi, s in zip(x, sig) if not s],
                   [v for v, s in zip(vals, sig) if not s],
                   s=64, facecolor="white", edgecolor=colour, marker=marker,
                   linewidth=1.3, zorder=4)
        span = order[model]["spread"]["paired_delta_pp_range"]
        ax.annotate(f"{span:.1f} pp", xy=(x[-1], vals[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=colour, va="center")
    ax.axhline(0.0, color=figstyle.RULE, linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_xlim(-0.25, len(names) - 0.45)
    ax.set_ylabel("paired FP16 $\\rightarrow$ 4.5-bit $\\Delta$ (pp)")
    # Held to the same 24-point span as panel (a), so the two panels are read
    # at ONE scale. Plotted on its own range the 1.6-point spread would look
    # as dramatic as the 10.4-point one, which is the opposite of the result.
    ax.set_ylim(-9, 15)
    figstyle.ygrid(ax)
    figstyle.panel_title(ax, "b", "Paired difference: largely is")

    handles = [
        plt.Line2D([], [], color=figstyle.MODEL_STYLE["Qwen2.5-3B"][0],
                   marker="o", linestyle="-", label="Qwen2.5-3B"),
        plt.Line2D([], [], color=figstyle.MODEL_STYLE["Phi-3.5-mini"][0],
                   marker="s", linestyle="--", label="Phi-3.5-mini"),
        mpatches.Patch(facecolor="#444444", edgecolor="none",
                       label="filled: $p<0.05$"),
        mpatches.Patch(facecolor="white", edgecolor="#444444",
                       label="hollow: not significant"),
    ]
    figstyle.shared_legend(
        fig, handles, [h.get_label() for h in handles], ncol=4,
        note="Same completions, same gate; only which letter carries which "
             "class changes. Both panels span 24 points, so the spreads are "
             "comparable by eye. Annotations give each model's full range.",
        reserve=0.30)
    figstyle.save(fig, out, "fig_round4_order")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path,
                    default=Path("docs/paper/round4_stats.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    if not args.stats.is_file():
        raise SystemExit(
            f"{args.stats} not found; run scripts/analyse_round4.py first")
    stats = json.loads(args.stats.read_text(encoding="utf-8"))

    figstyle.apply()
    print("round 4 figures:")
    scorer_plate(stats, args.out)
    order_plate(stats, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
