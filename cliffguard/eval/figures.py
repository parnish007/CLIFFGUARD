"""Figure generators for CLIFFGUARD paper — see blueprint §13.

Generates the four main paper figures using matplotlib:

  Figure 1 — Safety cliff visualization: refusal margin distribution
    across quantization schemes, with cliff boundary annotated.
  Figure 2 — FPR decoupling: empirical FPR per scheme per primitive,
    showing portability across the quantization axis.
  Figure 3 — Composition gain: ABR per primitive and full stack,
    grouped by scheme, showing H4.
  Figure 4 — Tier C structural weakness: ABR vs baseline for Tier C
    and Tier C+ on A7 prompts, showing H5.

All functions accept pre-computed data dicts and return
matplotlib.figure.Figure objects. No model inference occurs here.
In Phase A, functions accept synthetic data and produce placeholder
figures. Figures are saved to artifacts/figures/ by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any  # noqa: F401

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Phase A
import matplotlib.pyplot as plt
import matplotlib.figure

from cliffguard.types import QuantScheme  # noqa: F401


def figure_cliff_visualization(
    margin_distributions: dict[str, list[float]],
    cliff_boundary: str | None = None,
) -> matplotlib.figure.Figure:
    """Figure 1: box plots of refusal margin distributions per scheme.
    margin_distributions: {scheme_name: [margin_values]}.
    cliff_boundary: scheme name to annotate with a vertical marker.
    Returns a Figure with one subplot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scheme_names = list(margin_distributions.keys())

    if not scheme_names:
        ax.set_title("Figure 1 — Safety cliff: refusal margin distribution")
        fig.tight_layout()
        return fig

    data = [margin_distributions[s] for s in scheme_names]

    ax.boxplot(data, tick_labels=scheme_names, patch_artist=True)
    ax.set_xlabel("Quantization scheme")
    ax.set_ylabel("Refusal margin")
    ax.set_title("Figure 1 — Safety cliff: refusal margin distribution")

    if cliff_boundary is not None and cliff_boundary in scheme_names:
        idx = scheme_names.index(cliff_boundary) + 1  # boxplot is 1-indexed
        ax.axvline(x=idx, color="red", linestyle="--", linewidth=1.5,
                   label=f"Cliff boundary: {cliff_boundary}")
        ax.legend()

    fig.tight_layout()
    return fig


def figure_fpr_decoupling(
    fpr_by_primitive_scheme: dict[str, dict[str, float]],
    fpr_target: float = 0.05,
) -> matplotlib.figure.Figure:
    """Figure 2: line plot of empirical FPR per scheme per primitive.
    fpr_by_primitive_scheme: {primitive_name: {scheme_name: fpr}}.
    fpr_target: horizontal reference line.
    Returns a Figure with one subplot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for primitive, scheme_fpr in fpr_by_primitive_scheme.items():
        scheme_names = list(scheme_fpr.keys())
        fpr_values = [scheme_fpr[s] for s in scheme_names]
        ax.plot(scheme_names, fpr_values, marker="o", label=primitive)

    ax.axhline(y=fpr_target, color="black", linestyle="--", linewidth=1.0,
               label=f"FPR target ({fpr_target:.2f})")
    ax.set_xlabel("Quantization scheme")
    ax.set_ylabel("Empirical FPR")
    ax.set_title("Figure 2 — FPR decoupling across quantization schemes")
    ax.legend()
    fig.tight_layout()
    return fig


def figure_composition_gain(
    abr_by_primitive_scheme: dict[str, dict[str, float]],
    full_stack_key: str = "FULL_STACK",
) -> matplotlib.figure.Figure:
    """Figure 3: grouped bar chart of ABR per primitive and full stack.
    abr_by_primitive_scheme: {primitive_name: {scheme_name: abr}}.
    full_stack_key: key in the dict for the full-stack entry.
    Returns a Figure with one subplot."""
    fig, ax = plt.subplots(figsize=(12, 5))

    primitives = list(abr_by_primitive_scheme.keys())
    if not primitives:
        ax.set_title("Figure 3 — Composition gain (no data)")
        return fig

    # Collect all scheme names from the first primitive.
    all_schemes = list(next(iter(abr_by_primitive_scheme.values())).keys())
    n_primitives = len(primitives)
    n_schemes = len(all_schemes)
    bar_width = 0.8 / max(n_schemes, 1)

    import numpy as np
    x = np.arange(n_primitives)

    for i, scheme in enumerate(all_schemes):
        abr_values = [
            abr_by_primitive_scheme[p].get(scheme, 0.0) for p in primitives
        ]
        offset = (i - n_schemes / 2.0 + 0.5) * bar_width
        bars = ax.bar(x + offset, abr_values, bar_width, label=scheme)
        # Highlight the full-stack bar.
        if full_stack_key in primitives:
            fs_idx = primitives.index(full_stack_key)
            bars[fs_idx].set_edgecolor("red")
            bars[fs_idx].set_linewidth(2.0)

    ax.set_xticks(list(x))
    ax.set_xticklabels(primitives, rotation=30, ha="right")
    ax.set_xlabel("Primitive / full stack")
    ax.set_ylabel("ABR")
    ax.set_title("Figure 3 — Composition gain: ABR per primitive and full stack")
    ax.legend(title="Scheme")
    fig.tight_layout()
    return fig


def figure_tier_c_weakness(
    abr_tier_c: dict[str, float],
    abr_tier_c_plus: dict[str, float],
    abr_baseline: dict[str, float],
) -> matplotlib.figure.Figure:
    """Figure 4: grouped bar chart comparing Tier C, C+, baseline ABR.
    Each dict maps scheme_name -> abr value.
    Returns a Figure with one subplot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    schemes = list(abr_baseline.keys())
    n = len(schemes)

    import numpy as np
    x = np.arange(n)
    bar_width = 0.25

    baseline_vals = [abr_baseline.get(s, 0.0) for s in schemes]
    tier_c_vals = [abr_tier_c.get(s, 0.0) for s in schemes]
    tier_c_plus_vals = [abr_tier_c_plus.get(s, 0.0) for s in schemes]

    ax.bar(x - bar_width, baseline_vals, bar_width, label="Baseline", color="gray")
    ax.bar(x, tier_c_vals, bar_width, label="Tier C", color="steelblue")
    ax.bar(x + bar_width, tier_c_plus_vals, bar_width, label="Tier C+", color="darkorange")

    ax.set_xticks(list(x))
    ax.set_xticklabels(schemes)
    ax.set_xlabel("Quantization scheme")
    ax.set_ylabel("ABR")
    ax.set_title("Figure 4 — Tier C structural weakness: ABR vs baseline")
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(
    fig: matplotlib.figure.Figure,
    path: Path,
    dpi: int = 300,
) -> None:
    """Save figure to path. Creates parent directories.
    Closes the figure after saving to free memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
