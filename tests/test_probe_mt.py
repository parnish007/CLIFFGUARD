import numpy as np
import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.probe.mt import compute_trajectory, evaluate

# ---------------------------------------------------------------------------
# Synthetic margin sequences
# ---------------------------------------------------------------------------

_FLAT = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
_DECREASING = np.array([1.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float64)
_INCREASING = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
_NONLINEAR = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)

_TABLE_ZERO = CalibrationTable(
    primitive="PROBE-MT",
    thresholds={QuantScheme.FP16: 0.0},
)

_TABLE_NEG = CalibrationTable(
    primitive="PROBE-MT",
    thresholds={QuantScheme.FP16: -0.5},
)


# ---------------------------------------------------------------------------
# compute_trajectory
# ---------------------------------------------------------------------------


def test_compute_trajectory_raises_for_length_zero() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        compute_trajectory(np.array([], dtype=np.float64))


def test_compute_trajectory_raises_for_length_one() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        compute_trajectory(np.array([1.0], dtype=np.float64))


def test_compute_trajectory_raises_for_length_two() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        compute_trajectory(np.array([1.0, 2.0], dtype=np.float64))


def test_compute_trajectory_flat_returns_zero_zero() -> None:
    rho_dot, rho_ddot = compute_trajectory(_FLAT)
    assert rho_dot == pytest.approx(0.0, abs=1e-12)
    assert rho_ddot == pytest.approx(0.0, abs=1e-12)


def test_compute_trajectory_decreasing_returns_negative_rho_dot() -> None:
    rho_dot, _ = compute_trajectory(_DECREASING)
    assert rho_dot < 0.0


def test_compute_trajectory_increasing_returns_positive_rho_dot() -> None:
    rho_dot, _ = compute_trajectory(_INCREASING)
    assert rho_dot > 0.0


def test_compute_trajectory_decreasing_exact_value() -> None:
    # _DECREASING steps by -0.5 each time → rho_dot = -0.5, rho_ddot = 0.0
    rho_dot, rho_ddot = compute_trajectory(_DECREASING)
    assert rho_dot == pytest.approx(-0.5)
    assert rho_ddot == pytest.approx(0.0, abs=1e-12)


def test_compute_trajectory_increasing_exact_value() -> None:
    # _INCREASING steps by +0.5 each time → rho_dot = +0.5, rho_ddot = 0.0
    rho_dot, rho_ddot = compute_trajectory(_INCREASING)
    assert rho_dot == pytest.approx(0.5)
    assert rho_ddot == pytest.approx(0.0, abs=1e-12)


def test_compute_trajectory_returns_floats() -> None:
    rho_dot, rho_ddot = compute_trajectory(_FLAT)
    assert isinstance(rho_dot, float)
    assert isinstance(rho_ddot, float)


def test_compute_trajectory_minimum_length_accepted() -> None:
    # Exactly 3 elements must not raise.
    rho_dot, rho_ddot = compute_trajectory(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert rho_dot == pytest.approx(1.0)
    assert rho_ddot == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    _, verdict = evaluate(_FLAT, _TABLE_ZERO, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "PROBE-MT"


def test_evaluate_margin_primitive() -> None:
    margin, _ = evaluate(_FLAT, _TABLE_ZERO, QuantScheme.FP16, Tier.A)
    assert margin.primitive == "PROBE-MT"


def test_evaluate_margin_layer_is_none() -> None:
    margin, _ = evaluate(_FLAT, _TABLE_ZERO, QuantScheme.FP16, Tier.A)
    assert margin.layer is None


def test_evaluate_fires_when_rho_dot_below_threshold() -> None:
    # _DECREASING → rho_dot = -0.5; threshold = 0.0 → -0.5 < 0.0 → fired.
    _, verdict = evaluate(_DECREASING, _TABLE_ZERO, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_rho_dot_above_threshold() -> None:
    # _INCREASING → rho_dot = +0.5; threshold = 0.0 → 0.5 >= 0.0 → not fired.
    _, verdict = evaluate(_INCREASING, _TABLE_ZERO, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_flat_does_not_fire_at_zero_threshold() -> None:
    # rho_dot = 0.0; threshold = 0.0; 0.0 < 0.0 is False → not fired.
    _, verdict = evaluate(_FLAT, _TABLE_ZERO, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_score_matches_rho_dot() -> None:
    rho_dot, _ = compute_trajectory(_NONLINEAR)
    _, verdict = evaluate(_NONLINEAR, _TABLE_ZERO, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(rho_dot)


def test_evaluate_margin_value_matches_rho_dot() -> None:
    rho_dot, _ = compute_trajectory(_DECREASING)
    margin, _ = evaluate(_DECREASING, _TABLE_ZERO, QuantScheme.FP16, Tier.A)
    assert margin.value == pytest.approx(rho_dot)


def test_evaluate_cliff_regime_true_for_q3km() -> None:
    table = CalibrationTable(
        primitive="PROBE-MT",
        thresholds={QuantScheme.GGUF_Q3_K_M: 0.0},
    )
    margin, _ = evaluate(_FLAT, table, QuantScheme.GGUF_Q3_K_M, Tier.B)
    assert margin.is_cliff_regime is True
