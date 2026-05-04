import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import pytest

from cliffguard.eval.figures import (
    figure_cliff_visualization,
    figure_composition_gain,
    figure_fpr_decoupling,
    figure_tier_c_weakness,
    save_figure,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_MARGIN_DISTRIBUTIONS = {
    "FP16": [0.1, 0.2, 0.3, 0.4],
    "NF4": [0.05, 0.1, 0.15],
    "GGUF_Q3_K_M": [-0.1, 0.0, 0.05],
}

_FPR_BY_PRIMITIVE = {
    "PROBE-RM": {"FP16": 0.04, "NF4": 0.05, "GGUF_Q3_K_M": 0.06},
    "PROBE-MT": {"FP16": 0.03, "NF4": 0.04, "GGUF_Q3_K_M": 0.07},
}

_ABR_BY_PRIMITIVE = {
    "PROBE-RM": {"FP16": 0.85, "NF4": 0.80},
    "FULL_STACK": {"FP16": 0.95, "NF4": 0.92},
}

_ABR_BASELINE = {"FP16": 0.50, "NF4": 0.48}
_ABR_TIER_C = {"FP16": 0.50, "NF4": 0.48}
_ABR_TIER_C_PLUS = {"FP16": 0.85, "NF4": 0.82}


# ---------------------------------------------------------------------------
# figure_cliff_visualization
# ---------------------------------------------------------------------------


def test_cliff_visualization_returns_figure() -> None:
    fig = figure_cliff_visualization(_MARGIN_DISTRIBUTIONS)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_cliff_visualization_no_cliff_boundary() -> None:
    fig = figure_cliff_visualization(_MARGIN_DISTRIBUTIONS, cliff_boundary=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_cliff_visualization_with_cliff_boundary_annotates() -> None:
    fig = figure_cliff_visualization(_MARGIN_DISTRIBUTIONS, cliff_boundary="NF4")
    ax = fig.axes[0]
    # axvline adds a Line2D to the axes
    assert len(ax.get_lines()) >= 1
    plt.close(fig)


def test_cliff_visualization_unknown_cliff_boundary_no_legend() -> None:
    # Unknown cliff_boundary → no legend added (legend only appears for valid boundary)
    fig = figure_cliff_visualization(_MARGIN_DISTRIBUTIONS, cliff_boundary="UNKNOWN")
    ax = fig.axes[0]
    assert ax.get_legend() is None
    plt.close(fig)


def test_cliff_visualization_has_one_axes() -> None:
    fig = figure_cliff_visualization(_MARGIN_DISTRIBUTIONS)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_cliff_visualization_empty_distributions() -> None:
    fig = figure_cliff_visualization({})
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure_fpr_decoupling
# ---------------------------------------------------------------------------


def test_fpr_decoupling_returns_figure() -> None:
    fig = figure_fpr_decoupling(_FPR_BY_PRIMITIVE)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_fpr_decoupling_has_one_axes() -> None:
    fig = figure_fpr_decoupling(_FPR_BY_PRIMITIVE)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_fpr_decoupling_custom_fpr_target() -> None:
    fig = figure_fpr_decoupling(_FPR_BY_PRIMITIVE, fpr_target=0.10)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_fpr_decoupling_has_horizontal_reference_line() -> None:
    fig = figure_fpr_decoupling(_FPR_BY_PRIMITIVE, fpr_target=0.05)
    ax = fig.axes[0]
    # axhline adds a Line2D — plus one per primitive line
    assert len(ax.get_lines()) >= 1
    plt.close(fig)


def test_fpr_decoupling_empty_primitives() -> None:
    fig = figure_fpr_decoupling({})
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure_composition_gain
# ---------------------------------------------------------------------------


def test_composition_gain_returns_figure() -> None:
    fig = figure_composition_gain(_ABR_BY_PRIMITIVE)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_composition_gain_has_one_axes() -> None:
    fig = figure_composition_gain(_ABR_BY_PRIMITIVE)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_composition_gain_empty_data() -> None:
    fig = figure_composition_gain({})
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_composition_gain_custom_full_stack_key() -> None:
    data = {
        "PRIM_A": {"FP16": 0.7},
        "TOTAL": {"FP16": 0.9},
    }
    fig = figure_composition_gain(data, full_stack_key="TOTAL")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_composition_gain_full_stack_key_missing() -> None:
    # full_stack_key not in primitives → no red-edge logic, but no error
    fig = figure_composition_gain(_ABR_BY_PRIMITIVE, full_stack_key="MISSING")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure_tier_c_weakness
# ---------------------------------------------------------------------------


def test_tier_c_weakness_returns_figure() -> None:
    fig = figure_tier_c_weakness(_ABR_TIER_C, _ABR_TIER_C_PLUS, _ABR_BASELINE)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_tier_c_weakness_has_one_axes() -> None:
    fig = figure_tier_c_weakness(_ABR_TIER_C, _ABR_TIER_C_PLUS, _ABR_BASELINE)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_tier_c_weakness_empty_dicts() -> None:
    fig = figure_tier_c_weakness({}, {}, {})
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_tier_c_weakness_single_scheme() -> None:
    fig = figure_tier_c_weakness(
        {"FP16": 0.5}, {"FP16": 0.8}, {"FP16": 0.5}
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


def test_save_figure_creates_file(tmp_path: pytest.TempPathFactory) -> None:
    fig = figure_cliff_visualization(_MARGIN_DISTRIBUTIONS)
    out = tmp_path / "test_fig.png"  # type: ignore[operator]
    save_figure(fig, out)
    assert out.exists()


def test_save_figure_closes_figure(tmp_path: pytest.TempPathFactory) -> None:
    fig = figure_cliff_visualization(_MARGIN_DISTRIBUTIONS)
    fig_num = fig.number
    out = tmp_path / "test_fig2.png"  # type: ignore[operator]
    save_figure(fig, out)
    assert fig_num not in plt.get_fignums()


def test_save_figure_creates_parent_dirs(tmp_path: pytest.TempPathFactory) -> None:
    fig = figure_fpr_decoupling(_FPR_BY_PRIMITIVE)
    nested = tmp_path / "a" / "b" / "c" / "fig.png"  # type: ignore[operator]
    save_figure(fig, nested)
    assert nested.exists()


def test_save_figure_custom_dpi(tmp_path: pytest.TempPathFactory) -> None:
    fig = figure_composition_gain(_ABR_BY_PRIMITIVE)
    out = tmp_path / "dpi_fig.png"  # type: ignore[operator]
    save_figure(fig, out, dpi=72)
    assert out.exists()
