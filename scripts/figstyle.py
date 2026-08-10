"""One house style for every figure in the paper.

The three figure builders grew separately and drifted: some plates dropped the
top and right spines, some kept them; panel titles were plain strings in one
file and lettered in another; legends landed inside the axes, outside them, or
in a corner that happened to be empty for that particular data. A reader
notices that before they notice any result.

The style here is the conventional journal one, and conventional is the point.
Boxed axes on all four sides, panel titles lettered "(a) what it shows", one
framed legend centred in a strip reserved beneath the panels, and an optional
note line under that for the sentence a caption should not have to carry. No
spine surgery, no direct-labelling schemes that work for one data range and
collide in the next, no colour used to mean two things in one plate.

Usage:

    from scripts import figstyle

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=figstyle.WIDE)
    ...
    figstyle.panel_title(axes[0], "a", "Resident: co-located win")
    figstyle.shared_legend(fig, handles, labels, ncol=4,
                           note="Dotted line = break-even.")
    figstyle.save(fig, out, "fig_thing")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Column geometry. TEXT_WIDTH is the manuscript's \linewidth in inches, so a
# figure declared at this width is reproduced 1:1 and its 9pt labels really are
# 9pt on the page. WIDE is for three-panel plates, which the manuscript sets at
# \linewidth and which therefore must not be authored wider than it.
TEXT_WIDTH = 6.30
SINGLE = (TEXT_WIDTH, 3.9)
WIDE = (TEXT_WIDTH, 2.9)
TALL = (TEXT_WIDTH, 4.6)

# Model identity, fixed across every plate so a colour never means two things.
# Chosen for colour-blind safety; marker and dash carry the same information
# again, so the figures survive greyscale printing.
MODEL_STYLE: dict[str, tuple[str, str, str]] = {
    "Qwen2.5-3B": ("#1f77b4", "o", "-"),
    "Phi-3.5-mini": ("#d62728", "s", "--"),
    "SmolLM2-1.7B": ("#2ca02c", "^", "-."),
    "Qwen2.5-1.5B": ("#7e3ff2", "D", ":"),
}
MODEL_ORDER = ("Qwen2.5-3B", "Phi-3.5-mini", "SmolLM2-1.7B", "Qwen2.5-1.5B")

# Completion classes. Refusal and compliance are the two the paper's claims are
# about and are the two saturated colours; the intermediate classes are lighter
# so a stack reads as "mostly withheld" or "mostly answered" at a glance.
CLASS_COLOUR: dict[str, str] = {
    "refusal": "#1f4e79",
    "deflection": "#7fb3d5",
    "disclaimer": "#c8b6e2",
    "compliance": "#c0392b",
    "unclear": "#d5d8dc",
    "degenerate": "#909497",
}
CLASS_SHORT: dict[str, str] = {
    "refusal": "refusal", "deflection": "deflection",
    "disclaimer": "disclaimer", "compliance": "compliance",
    "unclear": "unclear", "degenerate": "degenerate",
}

UNSAFE = "#c0392b"
CONSERVATIVE = "#1f4e79"
NEUTRAL = "#909497"
RULE = "#b0b3b8"


def apply() -> None:
    """Install the house style. Call once, before creating any figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        # Boxed on all four sides. This is the single most visible difference
        # from the earlier plates and the one that makes them look like one
        # document rather than three.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.color": "#d5d8dc",
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#b0b3b8",
        "legend.borderpad": 0.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_title(ax: Any, letter: str | None, text: str) -> None:
    """`(a) What this panel shows`, or just the text for a single-panel plate."""
    ax.set_title(f"({letter}) {text}" if letter else text, fontsize=10)


def ygrid(ax: Any) -> None:
    """A horizontal reference grid, behind the data."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d5d8dc", linewidth=0.6)


def xgrid(ax: Any) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d5d8dc", linewidth=0.6)


def shared_legend(fig: Any, handles: Sequence[Any], labels: Sequence[str],
                  ncol: int = 4, note: str | None = None,
                  reserve: float = 0.18) -> None:
    """One framed legend centred under the panels, with an optional note line.

    The panels are laid out in the top `1 - reserve` of the figure and the
    legend goes in the strip below, so it cannot collide with data whatever the
    data happens to be. That is the failure a corner legend has: it is placed
    against one dataset and overlaps the next.

    `note` is the sentence that explains an encoding -- what a dashed line
    means, where a dotted rule sits -- which belongs with the figure rather
    than in a caption a reader may be reading from a different page.
    """
    fig.tight_layout(rect=(0, reserve, 1, 1))
    legend = fig.legend(handles, labels, loc="upper center", ncol=ncol,
                        bbox_to_anchor=(0.5, reserve), frameon=True,
                        fontsize=8.5)
    if note:
        _note_below(fig, legend, note)


def _wrap(fig: Any, note: str, fontsize: float = 8.0,
          fraction: float = 0.96) -> str:
    """Break a note onto as many lines as it needs to fit the figure width.

    Measured, not guessed at a character count: the notes carry percent signs,
    dashes and model names, so a fixed column that fits one fits none of the
    others. An unwrapped note does not overflow visibly in the PNG --- the
    tight bounding box simply grows the canvas --- so the failure shows up
    only on the typeset page, as a figure scaled down to fit a line of small
    print. Wrapping here means no caller can reintroduce it.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    limit = fig.get_size_inches()[0] * fig.dpi * fraction

    def width(text: str) -> float:
        probe = fig.text(0, 0, text, fontsize=fontsize, alpha=0)
        w = probe.get_window_extent(renderer).width
        probe.remove()
        return w

    if width(note) <= limit:
        return note
    lines: list[str] = []
    current = ""
    for word in note.split():
        candidate = f"{current} {word}".strip()
        if current and width(candidate) > limit:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _note_below(fig: Any, anchor: Any, note: str) -> None:
    """Place the note a fixed gap under whatever sits above it.

    Measured rather than positioned at a fixed figure coordinate. A constant
    y works for exactly one figure height, and these range from 2.9 to 4.6
    inches, so a constant left a finger's width of blank paper under the
    two-panel plates and none under the tall ones.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = anchor.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    fig.text(0.5, max(box.y0 - 0.055, 0.005), _wrap(fig, note), ha="center",
             va="top", fontsize=8, color="#404040", linespacing=1.35)


def note_only(fig: Any, note: str, reserve: float = 0.10) -> None:
    """A note line with no legend above it."""
    fig.tight_layout(rect=(0, reserve, 1, 1))
    fig.text(0.5, reserve * 0.35, _wrap(fig, note), ha="center", va="center",
             fontsize=8, color="#404040", linespacing=1.35)


def save(fig: Any, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")
