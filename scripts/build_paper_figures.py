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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

# Colour-blind-safe, and nothing relies on hue alone: markers and dash patterns
# carry the same information for greyscale print.
JUDGE = "#0B7285"
MARKER = "#C92A2A"
DEGEN = "#868E96"
MODEL_STYLE = {
    "Qwen2.5-1.5B": ("#5F3DC4", "o", "-"),
    "Qwen2.5-3B": ("#0B7285", "s", "-"),
    "Phi-3.5-mini": ("#E8590C", "^", "--"),
}

# A4 with 2.5 cm margins leaves a 16.0 cm text block = 6.30 in. Figures are sized
# against that constant rather than eyeballed, and \includegraphics uses
# \linewidth, so nothing can overflow into the margin.
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
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def rung_axis(ax: Any, rows: list[dict[str, Any]]) -> list[int]:
    """Evenly spaced rung positions, labelled with true bits per parameter.

    FP16 sits at 16 bits while every quantized rung is between 2.5 and 8.5. On a
    linear axis that spends two thirds of the width on empty space and squeezes
    the region the paper is about, so rungs are placed categorically and the tick
    label carries the real number.
    """
    positions = list(range(len(rows)))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["FP16" if r["bits"] >= 16 else f"{r['bits']:.1f}" for r in rows]
    )
    ax.set_xlabel("stored bits / parameter")
    ax.set_xlim(-0.4, len(rows) - 0.6)
    return positions


def fig_artifact(data: dict[str, Any], out: Path) -> None:
    """The headline: judged harmful compliance against the phrase-list estimate."""
    models = [m for m in ("Qwen2.5-3B", "Phi-3.5-mini") if m in data["behavioural"]]
    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH_IN, 2.75),
                             sharey=True, constrained_layout=True)
    axes = list(axes) if len(models) > 1 else [axes]

    for ax, model in zip(axes, models):
        rows = [r for r in data["behavioural"][model]["rows"] if r["scheme"] != "FP16"]
        rows.sort(key=lambda r: -r["bits"])
        x = rung_axis(ax, rows)

        # Shade rungs where the model has stopped producing language: a
        # compliance rate there is meaningless in either direction.
        for i, r in enumerate(rows):
            if r["degenerate"] > 0.5:
                ax.axvspan(i - 0.5, i + 0.5, color=DEGEN, alpha=0.18, zorder=0, lw=0)

        ax.plot(x, [100 * (r.get("unsafe_marker") or 0.0) for r in rows],
                color=MARKER, marker="s", ls="--", lw=1.7, ms=5,
                label="refusal phrase list", zorder=3)
        ax.plot(x, [100 * r["unsafe_judge"] for r in rows],
                color=JUDGE, marker="o", ls="-", lw=1.9, ms=5,
                label="7B judge", zorder=4)
        ax.set_title(model)
        ax.set_ylim(-1.5, 43)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    axes[0].set_ylabel("harmful compliance")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].text(0.035, 0.52, "shaded rungs:\nmodel is degenerate",
                 transform=axes[0].transAxes, ha="left", va="center",
                 fontsize=7.5, color="#495057")
    fig.suptitle("The safety cliff appears only under phrase-list scoring",
                 fontsize=10.5)
    save(fig, out, "fig_artifact")


def fig_capability(data: dict[str, Any], out: Path) -> None:
    """GSM8K accuracy with exact binomial intervals, per model."""
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.66, 2.9),
                           constrained_layout=True)
    reference: list[dict[str, Any]] = []
    for model, block in data["sector"].items():
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
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Capability collapses at a model-specific bit-width")
    save(fig, out, "fig_capability")


def fig_probe(data: dict[str, Any], out: Path) -> None:
    """Frozen-probe retention along the ladder."""
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.66, 2.9),
                           constrained_layout=True)
    reference: list[dict[str, Any]] = []
    for model, block in data["transfer"].items():
        colour, marker, dash = MODEL_STYLE.get(model, ("#333333", "o", "-"))
        rows = sorted(block["rows"], key=lambda r: -r["bits"])
        reference = reference or rows
        ax.plot(range(len(rows)), [100 * r["retained"] for r in rows],
                color=colour, marker=marker, ls=dash, lw=1.6, ms=4.5, label=model)
    ax.axhline(100, color="#ADB5BD", lw=0.8, ls=":", zorder=0)
    rung_axis(ax, reference)
    ax.set_ylabel("frozen-probe $d'$ retained")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("The probe tracks fluency, not safety")
    save(fig, out, "fig_probe")


def fig_degeneracy(data: dict[str, Any], out: Path) -> None:
    """Why perplexity alone fails: repetition loops are predictable."""
    model = ("Qwen2.5-3B" if "Qwen2.5-3B" in data["behavioural"]
             else next(iter(data["behavioural"])))
    rows = sorted(data["behavioural"][model]["rows"], key=lambda r: -r["bits"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7),
                                   constrained_layout=True)
    x = rung_axis(ax1, rows)
    ax1.plot(x, [r["median_nll"] for r in rows], color=MARKER, marker="s",
             lw=1.7, ms=5)
    ax1.set_ylabel("median NLL under FP16")
    ax1.set_title("Perplexity is not monotone")
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
    ax2.legend(loc="center left", frameon=False)
    ax2.set_title("Surface statistics separate the regimes")
    fig.suptitle("Detecting degeneracy needs more than perplexity", fontsize=10.5)
    save(fig, out, "fig_degeneracy")


def fig_marker_sensitivity(data: dict[str, Any], out: Path) -> None:
    """The same completions, four marker lists, four different answers."""
    model = next((m for m in ("Qwen2.5-3B", "Phi-3.5-mini")
                  if data["behavioural"].get(m, {}).get("marker_variants")), None)
    if model is None:
        print("  (no marker-variant data; skipping fig_marker_sensitivity)")
        return
    variants = data["behavioural"][model]["marker_variants"]
    schemes = sorted({s for v in variants.values() for s in v},
                     key=lambda s: -int(s.split("_")[1][:-1]))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.72, 2.9),
                           constrained_layout=True)
    for i, (name, row) in enumerate(variants.items()):
        ax.plot(range(len(schemes)), [100 * row.get(s, 0.0) for s in schemes],
                marker="o", lw=1.6, ms=4.5, label=name,
                color=matplotlib.colormaps["viridis"](
                    0.08 + 0.72 * i / max(1, len(variants) - 1)))
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels([s.split("_")[1][:-1] for s in schemes])
    ax.set_xlim(-0.4, len(schemes) - 0.6)
    ax.set_xlabel("code bits")
    ax.set_ylabel("apparent harmful compliance")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(frameon=False, title="marker list", title_fontsize=8,
              loc="upper left")
    ax.set_title(f"One choice of strings moves the headline ({model})")
    save(fig, out, "fig_marker_sensitivity")


def fig_refusal_law(data: dict[str, Any], out: Path) -> None:
    """Refusal rate against bit-width in the coherent regime, with fits.

    The regression is restricted to rungs where fewer than 10% of completions are
    degenerate. Beyond that boundary a "refusal rate" is not a measurement of
    refusal, and including those points would fit a line through the collapse.
    """
    from scipy import stats

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.66, 2.9),
                           constrained_layout=True)
    for model, block in data["behavioural"].items():
        colour, marker, dash = MODEL_STYLE.get(model, ("#333333", "o", "-"))
        coherent = [r for r in block["rows"] if r["degenerate"] < 0.10]
        coherent.sort(key=lambda r: r["bits"])
        x = [r["bits"] for r in coherent]
        y = [100 * r["refusal"] for r in coherent]
        ax.plot(x, y, color=colour, marker=marker, ls="none", ms=5, label=model)
        if len(x) >= 3:
            fit = stats.linregress(x, y)
            grid = [min(x), max(x)]
            ax.plot(grid, [fit.intercept + fit.slope * g for g in grid],
                    color=colour, ls=dash, lw=1.3, alpha=0.85)
    ax.set_xlabel("stored bits / parameter")
    ax.set_ylabel("refusal rate")
    ax.invert_xaxis()
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Refusal rises as precision falls, while output stays coherent")
    save(fig, out, "fig_refusal_law")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("docs/paper/data.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    print("figures:")
    fig_artifact(data, args.out)
    fig_capability(data, args.out)
    fig_probe(data, args.out)
    fig_degeneracy(data, args.out)
    fig_marker_sensitivity(data, args.out)
    fig_refusal_law(data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
