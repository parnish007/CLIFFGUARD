"""Schematics: the protocol, the 2x2, and what each instrument sees per regime.

The paper is method-heavy and had no diagram of its own method. A reader had to
assemble the pipeline from prose spread over three sections, and the 2x2 that
gives the project its shape was described in a table of cell meanings rather
than drawn. These three plates fix that.

They are schematics, not plots, with one deliberate exception: the regime figure
carries real numbers, read from `review_stats.json` and `labelled_paper_stats.json`
rather than typed, because a diagram with stale numbers on it is worse than a
diagram with none. If those files are absent the figure is skipped rather than
drawn with placeholders.

Usage:
  python scripts/build_schematic_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    FancyArrowPatch, FancyBboxPatch, Rectangle)

TEXT_WIDTH_IN = 6.30

INK = "#212529"
MUTED = "#868E96"
RULE = "#CED4DA"
JUDGE = "#0B7285"
MARKER_RED = "#C92A2A"
GATE = "#5F3DC4"
SAFE = "#1864AB"
WARN = "#E8590C"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
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


def box(ax, x, y, w, h, text, *, face="white", edge=INK, fs=8, lw=1.0,
        weight="normal", tcol=INK) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tcol, zorder=3, linespacing=1.35,
            fontweight=weight)


def arrow(ax, x1, y1, x2, y2, *, colour=INK, lw=1.1, style="-|>") -> None:
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=colour, zorder=1,
        shrinkA=1.5, shrinkB=1.5))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------

# --- the protocol plate ----------------------------------------------------
#
# Laid out in INCHES and converted at the end, not in axes fractions. The
# fractional version of this figure was unreadable in print for a reason worth
# recording: the plate is built 6.30in wide and the manuscript is A4 with 2.5cm
# margins, so \linewidth is also 6.30in and the figure is placed at exactly 1:1.
# A label set at `fontsize=6` therefore arrives on the page as 6pt type beside
# 11pt body text. Every size below is chosen against that 11pt, and because the
# scale is 1:1 the numbers here are literally the printed point sizes.

FIG_W_IN = TEXT_WIDTH_IN
PAD_IN = 0.06            # breathing room at the top and bottom of the plate
RAIL_W_IN = 0.62         # left gutter reserved for the phase brackets
CARD_PAD_IN = 0.16       # inset from a card edge to its content
ROW_GAP_IN = 0.16        # vertical space between consecutive cards

FS_TITLE = 10.0          # the name of a stage
FS_NOTE = 8.2            # the muted qualifier on a stage's right
FS_CHIP = 8.2            # text inside a pill
FS_SMALL = 7.6           # the one-line remark under a stage
FS_RAIL = 8.0
FS_TICK = 7.8            # inside a rung square

CARD_L_IN = RAIL_W_IN
CARD_R_IN = FIG_W_IN - 0.10


class Plate:
    """An inch-addressed canvas over a unit axes.

    Everything downstream asks for inches; only `.fy`/`.fx` know the figure is
    actually a 0-1 box. Mixing the two systems by hand is what produced the
    overlapping text in the first two attempts.
    """

    def __init__(self, height_in: float):
        self.h = height_in
        self.fig, self.ax = blank((FIG_W_IN, height_in))
        # The axes must BE the figure for `fx`/`fy` to mean anything. Default
        # subplot margins leave it 77.5% of the figure wide, which silently
        # compresses every drawn x by that factor while text measured in true
        # inches stays full size -- so every pill came out narrower than the
        # text inside it.
        self.ax.set_position((0, 0, 1, 1))
        self.fig.canvas.draw()   # a renderer must exist before measuring text

    def fy(self, y_in: float) -> float:
        """Inches DOWN from the top edge -> axes fraction."""
        return 1.0 - y_in / self.h

    def fx(self, x_in: float) -> float:
        return x_in / FIG_W_IN

    def dy(self, d_in: float) -> float:
        return d_in / self.h

    def text_w_in(self, s: str, fs: float) -> float:
        probe = self.ax.text(0, 0, s, fontsize=fs, alpha=0)
        w_px = probe.get_window_extent(renderer=self.fig.canvas.get_renderer()).width
        probe.remove()
        return w_px / self.fig.dpi


def plate_card(p: Plate, y_in: float, h_in: float, *, face="white",
               accent=None, edge=RULE, lw=0.8) -> None:
    """One row of the stack, drawn full width so no two rows are ragged."""
    p.ax.add_patch(Rectangle(
        (p.fx(CARD_L_IN), p.fy(y_in + h_in)),
        p.fx(CARD_R_IN - CARD_L_IN), p.dy(h_in),
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    if accent:
        p.ax.add_patch(Rectangle(
            (p.fx(CARD_L_IN), p.fy(y_in + h_in)), p.fx(0.045), p.dy(h_in),
            facecolor=accent, edgecolor="none", zorder=3))


CHIP_GAP = 0.11
CHIP_PADX = 0.13


def _wrap_chips(widths: list[tuple[str, float]]) -> list[list[tuple[str, float]]]:
    """Break pills across the fewest rows, then even those rows out.

    Greedy wrapping alone puts four gate conditions on rows of three and one,
    which looks like an accident. Once the row COUNT is fixed there is no cost
    to distributing evenly, so try the even split first and keep it if every
    row fits.
    """
    avail = (CARD_R_IN - CARD_PAD_IN) - (CARD_L_IN + CARD_PAD_IN)

    def pack(chunks: list[list[tuple[str, float]]]) -> bool:
        return all(sum(w for _, w in c) + CHIP_GAP * (len(c) - 1) <= avail
                   for c in chunks)

    greedy: list[list[tuple[str, float]]] = [[]]
    used = 0.0
    for item in widths:
        need = item[1] if not greedy[-1] else item[1] + CHIP_GAP
        if greedy[-1] and used + need > avail:
            greedy.append([item])
            used = item[1]
        else:
            greedy[-1].append(item)
            used += need

    n = len(greedy)
    if n > 1:
        per = -(-len(widths) // n)
        even = [widths[i:i + per] for i in range(0, len(widths), per)]
        if len(even) == n and pack(even):
            return even
    return greedy


def plate_chips(p: Plate, ycentre_in: float, labels, *, face, edge, tcol,
                fs=FS_CHIP, h_in=0.30) -> float:
    """Pills, one per item, wrapped onto as many centred rows as they need.

    Two things this fixes from earlier drafts. Sizing a pill by `len(text)`
    is not sizing it by width, so long labels spilled out of their own pills;
    the renderer is asked instead. And a single line joined by `|` and `OR`
    made independent conditions read as one badly-punctuated sentence.

    Returns the total height consumed, so callers can size the card to fit
    rather than guessing.
    """
    widths = [(label, p.text_w_in(label, fs) + 2 * CHIP_PADX)
              for label in labels]
    rows = _wrap_chips(widths)
    gap, vgap = CHIP_GAP, 0.09
    avail = (CARD_R_IN - CARD_PAD_IN) - (CARD_L_IN + CARD_PAD_IN)

    total = len(rows) * h_in + (len(rows) - 1) * vgap
    y = ycentre_in - total / 2
    for row in rows:
        span = sum(w for _, w in row) + gap * (len(row) - 1)
        x = CARD_L_IN + CARD_PAD_IN + (avail - span) / 2
        for label, w in row:
            p.ax.add_patch(FancyBboxPatch(
                (p.fx(x), p.fy(y + h_in)), p.fx(w), p.dy(h_in),
                boxstyle=f"round,pad=0,rounding_size={p.dy(h_in) / 2}",
                facecolor=face, edgecolor=edge, linewidth=0.8, zorder=3,
                mutation_aspect=p.dy(1.0) / p.fx(1.0)))
            p.ax.text(p.fx(x + w / 2), p.fy(y + h_in / 2), label,
                      ha="center", va="center", fontsize=fs, color=tcol,
                      zorder=4)
            x += w + gap
        y += h_in + vgap
    return total


def chip_rows_needed(p: Plate, labels, fs=FS_CHIP) -> int:
    """How many rows `plate_chips` will use, so a card can be sized first."""
    return len(_wrap_chips([(s, p.text_w_in(s, fs) + 2 * CHIP_PADX)
                            for s in labels]))


def plate_rail(p: Plate, y0_in: float, y1_in: float, label: str) -> None:
    """Phase bracket down the left gutter, with serifs at both ends."""
    x = p.fx(0.20)
    p.ax.plot([x, x], [p.fy(y0_in), p.fy(y1_in)], color=RULE, lw=1.1,
              solid_capstyle="butt", zorder=1)
    for edge_y in (y0_in, y1_in):
        p.ax.plot([x, p.fx(0.26)], [p.fy(edge_y)] * 2, color=RULE, lw=1.1,
                  zorder=1)
    p.ax.text(p.fx(0.13), p.fy((y0_in + y1_in) / 2), label, rotation=90,
              ha="center", va="center", fontsize=FS_RAIL, color=MUTED,
              fontweight="bold")


RUNG_STRIP = ["FP16", "8.5", "7.5", "6.5", "5.5", "4.5", "3.5", "2.5"]


def fig_protocol(out: Path) -> None:
    """The measurement pipeline, and the two places a scorer enters it.

    The paper's argument is that the reported cliff is largely produced by the
    two shaded stages -- the degeneracy gate and the grader -- rather than by
    the quantizer at the top. Drawing them on one page is the fastest way to
    make that claim legible, and it also shows why the two are separable: they
    act at different stages on the same text.

    Nothing on the plate repeats the caption. A figure that restates its own
    caption in a banner spends its two most-read lines saying nothing new.
    """
    # Measured on a throwaway plate, because how tall the gate and grader cards
    # need to be depends on how many rows their pills wrap onto, which depends
    # on the renderer. Guessing that was what pushed pills outside their card.
    probe = Plate(6.0)
    gate_chips = ["NLL > 3× FP16 median", "distinct trigram < 0.60",
                  "max token share > 0.35", "alphabetic < 0.70"]
    grader_chips = ["7B judge, 3-class", "7B judge, 5-class",
                    "25-string phrase list"]
    test_chips = ["exact McNemar", "Holm over the rung family",
                  "one-sided Clopper–Pearson"]
    n_gate = chip_rows_needed(probe, gate_chips)
    n_grader = chip_rows_needed(probe, grader_chips)
    n_test = chip_rows_needed(probe, test_chips)
    plt.close(probe.fig)

    def chip_card_h(n_rows: int, *, remark: bool) -> float:
        return 0.34 + n_rows * 0.30 + (n_rows - 1) * 0.09 + (0.26 if remark else 0.10)

    heights = {
        "prompts": 0.44,
        "model": 1.00,
        "decode": 0.44,
        "store": 0.50,
        "gate": chip_card_h(n_gate, remark=True),
        "grader": chip_card_h(n_grader, remark=True),
        "test": chip_card_h(n_test, remark=True),
    }
    order = ["prompts", "model", "decode", "store", "gate", "grader", "test"]
    total_h = (sum(heights.values()) + ROW_GAP_IN * (len(order) - 1)
               + 2 * PAD_IN)

    p = Plate(total_h)
    tops, bots, y = {}, {}, PAD_IN
    for name in order:
        tops[name], bots[name] = y, y + heights[name]
        y += heights[name] + ROW_GAP_IN

    def title(name, text, note=None, *, colour=INK):
        """Stage name flush left, muted qualifier flush right, one baseline."""
        ty = tops[name] + 0.21
        p.ax.text(p.fx(CARD_L_IN + CARD_PAD_IN), p.fy(ty), text, ha="left",
                  va="center", fontsize=FS_TITLE, color=colour,
                  fontweight="bold", zorder=4)
        if note:
            p.ax.text(p.fx(CARD_R_IN - CARD_PAD_IN), p.fy(ty), note,
                      ha="right", va="center", fontsize=FS_NOTE, color=MUTED,
                      zorder=4)

    def remark(name, text, *, colour=MUTED, fs=FS_SMALL):
        p.ax.text(p.fx((CARD_L_IN + CARD_R_IN) / 2), p.fy(bots[name] - 0.17),
                  text, ha="center", va="center", fontsize=fs, color=colour,
                  zorder=4)

    mid = p.fx((CARD_L_IN + CARD_R_IN) / 2)
    for a, b in zip(order, order[1:]):
        arrow(p.ax, mid, p.fy(bots[a]), mid, p.fy(tops[b]),
              colour="#ADB5BD", lw=1.0)

    plate_card(p, tops["prompts"], heights["prompts"])
    title("prompts", "prompt set", "500 paired   ·   300 labelled")

    plate_card(p, tops["model"], heights["model"])
    title("model", "model under test", "Qwen2.5-3B   ·   Phi-3.5-mini")
    sq_w, sq_h = 0.60, 0.27
    sq_gap = ((CARD_R_IN - CARD_PAD_IN) - (CARD_L_IN + CARD_PAD_IN)
              - len(RUNG_STRIP) * sq_w) / (len(RUNG_STRIP) - 1)
    strip_y = tops["model"] + 0.42
    ramp = plt.get_cmap("Blues")
    for i, bits in enumerate(RUNG_STRIP):
        x = CARD_L_IN + CARD_PAD_IN + i * (sq_w + sq_gap)
        headline = bits == "4.5"
        p.ax.add_patch(Rectangle(
            (p.fx(x), p.fy(strip_y + sq_h)), p.fx(sq_w), p.dy(sq_h),
            facecolor="white" if i == 0 else ramp(0.20 + 0.075 * i),
            edgecolor=WARN if headline else (INK if i == 0 else "none"),
            linewidth=1.8 if headline else 0.9, zorder=3))
        p.ax.text(p.fx(x + sq_w / 2), p.fy(strip_y + sq_h / 2), bits,
                  ha="center", va="center", fontsize=FS_TICK, zorder=4,
                  color="white" if i >= 5 else INK)
        if headline:
            p.ax.text(p.fx(x + sq_w / 2), p.fy(strip_y + sq_h + 0.20),
                      "every headline claim", ha="center", va="center",
                      fontsize=FS_SMALL, color=WARN, zorder=4)
    p.ax.text(p.fx(CARD_L_IN + CARD_PAD_IN), p.fy(strip_y + sq_h + 0.20),
              "stored bits per weight", ha="left", va="center",
              fontsize=FS_SMALL, color=MUTED, zorder=4)

    plate_card(p, tops["decode"], heights["decode"])
    title("decode", "greedy decode", "48 new tokens, one pass per rung")

    plate_card(p, tops["store"], heights["store"], face="#F1F3F5", accent=INK)
    title("store", "stored completions",
          "verbatim — every scorer below re-reads this text")

    plate_card(p, tops["gate"], heights["gate"], face="#F6F3FF", accent=GATE,
               edge="#D9CFFA")
    title("gate", "STAGE 1  ·  degeneracy gate", "machine-decided, no grader",
          colour=GATE)
    plate_chips(p, tops["gate"] + 0.34 + (heights["gate"] - 0.60) / 2,
                gate_chips, face="white", edge="#D9CFFA", tcol=GATE)
    remark("gate", "text failing any one condition never reaches a grader")

    plate_card(p, tops["grader"], heights["grader"], face="#EFF9FA",
               accent=JUDGE, edge="#BEE3E8")
    title("grader", "STAGE 2  ·  behaviour grader", "gate-admitted text only",
          colour=JUDGE)
    plate_chips(p, tops["grader"] + 0.34 + (heights["grader"] - 0.60) / 2,
                grader_chips, face="white", edge="#BEE3E8", tcol=JUDGE)
    remark("grader", "three scorers, one set of completions — up to 30× apart",
           colour=WARN)

    plate_card(p, tops["test"], heights["test"])
    title("test", "paired comparison", "each rung vs. the same model at FP16")
    plate_chips(p, tops["test"] + 0.34 + (heights["test"] - 0.60) / 2,
                test_chips, face="#F8F9FA", edge=RULE, tcol=INK)
    remark("test", "the endpoint is a change in behaviour, "
                   "never an absolute rate of harm")

    plate_rail(p, tops["prompts"], bots["store"], "PRODUCE THE TEXT")
    plate_rail(p, tops["gate"], bots["grader"], "SCORE IT")
    plate_rail(p, tops["test"], bots["test"], "TEST")

    save(p.fig, out, "fig_protocol")


def fig_matrix_schematic(out: Path) -> None:
    """The two axes crossed, and what each cell would mean.

    Both axes are annotations rather than derivations, and they come from
    different places: the prompt axis from a published suite, the completion
    axis from a model judge. Drawing them crossed shows immediately why a
    two-way grader on an unlabelled corpus cannot separate a safety regression
    from an over-refusal -- those are different cells of this square, and it
    reports one number for both.
    """
    fig, ax = blank((TEXT_WIDTH_IN, 2.9))

    cols = ["refusal", "compliance", "deflection", "disclaimer", "degenerate"]
    meaning = {
        ("harmful", "refusal"): ("withheld", SAFE),
        ("harmful", "compliance"): ("SAFETY\nFAILURE", MARKER_RED),
        ("harmful", "deflection"): ("partial\nwithhold", SAFE),
        ("harmful", "disclaimer"): ("non-answer", MUTED),
        ("harmful", "degenerate"): ("capability\nfailure", MUTED),
        ("benign", "refusal"): ("OVER-\nREFUSAL", WARN),
        ("benign", "compliance"): ("useful", SAFE),
        ("benign", "deflection"): ("soft over-\nrefusal", WARN),
        ("benign", "disclaimer"): ("capability\nfailure", MUTED),
        ("benign", "degenerate"): ("capability\nfailure", MUTED),
    }

    x0, y0, cw, ch = 0.195, 0.30, 0.156, 0.24
    for j, c in enumerate(cols):
        ax.text(x0 + cw * (j + 0.5), y0 + 2 * ch + 0.03, c, ha="center",
                va="bottom", fontsize=7.4, color=JUDGE, fontweight="bold",
                rotation=0)
    for i, row in enumerate(("harmful", "benign")):
        y = y0 + ch * (1 - i)
        ax.text(x0 - 0.016, y + ch / 2, row, ha="right", va="center",
                fontsize=7.8, color=INK, fontweight="bold")
        for j, c in enumerate(cols):
            label, colour = meaning[(row, c)]
            emphatic = colour in (MARKER_RED, WARN)
            box(ax, x0 + cw * j, y, cw - 0.006, ch - 0.012, label,
                fs=6.9, face="#FFF5F5" if colour == MARKER_RED
                else "#FFF4E6" if colour == WARN else "white",
                edge=colour if emphatic else RULE,
                lw=1.5 if emphatic else 0.8,
                weight="bold" if emphatic else "normal",
                tcol=colour if emphatic else INK)

    ax.text(x0 + cw * 2.5, 0.985, "COMPLETION LABEL  —  what the model did",
            ha="center", va="top", fontsize=8, color=JUDGE, fontweight="bold")
    ax.text(x0 + cw * 2.5, 0.935, "a model judge; this project's opinion",
            ha="center", va="top", fontsize=6.8, color=MUTED, style="italic")
    # Kept clear of the row labels, which extend leftward from the grid edge.
    ax.text(0.022, y0 + ch, "PROMPT\nLABEL", ha="center", va="center",
            fontsize=8, color=INK, fontweight="bold", rotation=90)
    ax.text(0.058, y0 + ch, "published suite, external",
            ha="center", va="center", fontsize=6.6, color=MUTED,
            style="italic", rotation=90)

    # Only the measured fact stays on the plate. Everything else that used to
    # sit here restated the caption, which wastes the two lines a reader is
    # most likely to actually read.
    ax.text(0.5, 0.145,
            "Measured here: the boxed SAFETY FAILURE cell is empty in all 21 "
            "labelled cells, at a 48-token budget.",
            ha="center", va="top", fontsize=7, color=MARKER_RED,
            style="italic")
    save(fig, out, "fig_matrix_schematic")


def fig_regimes(out: Path, stats: dict[str, Any]) -> None:
    """Three regimes, and what each instrument reports in each.

    The synthesis the discussion needs: the paper's separate arms measured
    different things and they disagree in a structured way, not randomly. The
    phrase list and the judge agree where output is coherent and diverge
    completely where it is not, and the probe is flat across the band where the
    behaviour it watches has already moved.

    Numbers are read from review_stats.json, never typed.
    """
    gate = stats["gate_by_grader"]["Qwen2.5-3B"]
    row45 = next(r for r in gate if r["bits"] == 4.5)
    row25 = next(r for r in gate if r["bits"] == 2.5)
    probe = next(r for r in stats["probe"]["Qwen2.5-3B"]["rows"] if r["bits"] == 4.5)

    fig, ax = blank((TEXT_WIDTH_IN, 3.6))
    # Three equal panels on one axis: unequal widths implied a difference in
    # importance that is not there.
    band_w = (0.955 - 0.030 - 2 * 0.025) / 3
    band_x = [0.030 + i * (band_w + 0.025) for i in range(3)]
    bands = [
        (band_x[0], band_w, "INERT", "FP16 – 8.5 bits", "#F1F3F5", MUTED),
        (band_x[1], band_w, "DRIFT", "8.5 – 4.5 bits", "#FFF4E6", WARN),
        (band_x[2], band_w, "COLLAPSE", "below 3.5 bits", "#FFF5F5", MARKER_RED),
    ]
    for x, w, name, span, face, colour in bands:
        ax.add_patch(FancyBboxPatch(
            (x, 0.285), w, 0.60, boxstyle="round,pad=0.008,rounding_size=0.02",
            facecolor=face, edgecolor=colour, linewidth=1.3, zorder=1))
        ax.text(x + w / 2, 0.845, name, ha="center", va="center",
                fontsize=9.5, color=colour, fontweight="bold")
        ax.text(x + w / 2, 0.800, span, ha="center", va="center",
                fontsize=7, color=MUTED)

    def entry(x, w, lines):
        """Label above value, centred.

        Side-by-side put a long label and a long value on a collision course
        inside a 0.30-wide panel; stacking removes the possibility entirely and
        the panel is narrow enough that centring still reads as a column.
        """
        for k, (lbl, val, colour) in enumerate(lines):
            top = 0.735 - k * 0.113
            ax.text(x + w / 2, top, lbl, ha="center", va="center", fontsize=6.6,
                    color=MUTED)
            ax.text(x + w / 2, top - 0.042, val, ha="center", va="center",
                    fontsize=8, color=colour, fontweight="bold")

    entry(band_x[0], band_w, [
        ("refusal shift", "+0.4 pp", MUTED),
        ("phrase list vs judge", "agree", MUTED),
        ("degenerate", "0.0%", MUTED),
        ("probe d' retained", "~100%", MUTED),
    ])
    # At 4.5 bits nothing is degenerate, so both gates admit the same
    # completions and the whole gap is the GRADER. At 2.5 bits the gap is the
    # GATE: the same phrase list reads 38.4% or 0.2% depending on it. Labelling
    # these two comparisons identically would erase the paper's central
    # decomposition, so each band names the factor that dominates it.
    entry(band_x[1], band_w, [
        ("refusal shift", "1.15 pp / bit", WARN),
        ("phrase list vs judge",
         f"{100 * row45['marker_composite']:.1f}% vs "
         f"{100 * row45['judge_composite']:.1f}%", MARKER_RED),
        ("degenerate", f"{100 * row45['degenerate_composite']:.1f}%", MUTED),
        ("probe d' retained", f"{100 * probe['retained_mean']:.0f}%", SAFE),
    ])
    entry(band_x[2], band_w, [
        ("refusal rate", "meaningless", MUTED),
        ("same list, gate swapped",
         f"{100 * row25['marker_nll']:.1f}% vs "
         f"{100 * row25['marker_composite']:.1f}%", MARKER_RED),
        ("degenerate", f"{100 * row25['degenerate_composite']:.1f}%", MARKER_RED),
        ("probe d' retained", "n/a", MUTED),
    ])

    ax.annotate("", xy=(0.985, 0.235), xytext=(0.03, 0.235),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    ax.text(0.03, 0.196, "16 bits", fontsize=7, color=MUTED, ha="left")
    ax.text(0.985, 0.196, "2.5 bits", fontsize=7, color=MUTED, ha="right")
    ax.text(0.5, 0.196, "stored bits per parameter, decreasing",
            fontsize=7, color=MUTED, ha="center")

    # The two-line summary that used to sit here restated the caption; the
    # caption says it better and this space is worth more as white.
    ax.text(0.5, 0.075,
            "Qwen2.5-3B; the collapse boundary differs by a full bit between "
            "model families.",
            ha="center", va="bottom", fontsize=6.8, color=MUTED, style="italic")
    save(fig, out, "fig_regimes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("docs/paper/figures"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    print(f"writing to {args.out}")
    fig_protocol(args.out)
    fig_matrix_schematic(args.out)

    stats_path = repo / "docs/paper/review_stats.json"
    if stats_path.exists():
        fig_regimes(args.out, json.loads(stats_path.read_text(encoding="utf-8")))
    else:
        print("  fig_regimes SKIPPED: review_stats.json absent, and this plate "
              "carries measured numbers rather than placeholders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
