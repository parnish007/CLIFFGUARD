# PROBE-HD fires high; PROBE-RM fires low — see blueprint §5.3.
# PROBE-HD: fired = True when harmfulness margin EXCEEDS threshold (score > tau).
# PROBE-RM: fired = True when refusal   margin IS BELOW  threshold (score < tau).

import numpy as np
import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.probe.hd import compute_harmfulness_margin, evaluate

# ---------------------------------------------------------------------------
# Synthetic test vectors (dimension 8 — exact arithmetic)
# ---------------------------------------------------------------------------

_DIM = 8

# Harmfulness direction: all-ones (unnormalised; cosine normalises internally).
_H: np.ndarray = np.ones(_DIM, dtype=np.float64)

# Aligned with _H → cosine = 1.0 (maximally harmful signal)
_Z_SAME: np.ndarray = np.ones(_DIM, dtype=np.float64)

# Anti-aligned with _H → cosine = -1.0 (strongly harmless)
_Z_OPP: np.ndarray = -np.ones(_DIM, dtype=np.float64)

# Orthogonal to _H: dot([1,-1,0,...], [1,1,...,1]) = 0 → cosine = 0.0
_Z_ORTH: np.ndarray = np.array([1.0, -1.0] + [0.0] * (_DIM - 2), dtype=np.float64)

_ZERO: np.ndarray = np.zeros(_DIM, dtype=np.float64)

# Calibration: threshold at 0.5 — fires only for clearly harmful inputs.
_TABLE = CalibrationTable(
    primitive="PROBE-HD",
    thresholds={
        QuantScheme.FP16: 0.5,
        QuantScheme.GGUF_Q3_K_M: 0.3,
    },
)


# ---------------------------------------------------------------------------
# compute_harmfulness_margin
# ---------------------------------------------------------------------------


def test_compute_harmfulness_margin_same_direction_returns_one() -> None:
    assert compute_harmfulness_margin(_Z_SAME, _H) == pytest.approx(1.0)


def test_compute_harmfulness_margin_opposite_direction_returns_minus_one() -> None:
    assert compute_harmfulness_margin(_Z_OPP, _H) == pytest.approx(-1.0)


def test_compute_harmfulness_margin_orthogonal_returns_zero() -> None:
    assert compute_harmfulness_margin(_Z_ORTH, _H) == pytest.approx(0.0, abs=1e-12)


def test_compute_harmfulness_margin_returns_float() -> None:
    assert isinstance(compute_harmfulness_margin(_Z_SAME, _H), float)


def test_compute_harmfulness_margin_raises_on_zero_hidden_state() -> None:
    with pytest.raises(ValueError, match="hidden_state"):
        compute_harmfulness_margin(_ZERO, _H)


def test_compute_harmfulness_margin_raises_on_zero_harmfulness_direction() -> None:
    with pytest.raises(ValueError, match="harmfulness_direction"):
        compute_harmfulness_margin(_Z_SAME, _ZERO)


def test_compute_harmfulness_margin_scale_invariant() -> None:
    assert compute_harmfulness_margin(_Z_SAME * 5.0, _H * 0.2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate — firing direction is HIGH (opposite of PROBE-RM)
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    _, verdict = evaluate(_Z_SAME, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "PROBE-HD"


def test_evaluate_margin_primitive() -> None:
    margin, _ = evaluate(_Z_SAME, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.primitive == "PROBE-HD"


def test_evaluate_margin_layer_is_none() -> None:
    margin, _ = evaluate(_Z_SAME, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.layer is None


def test_evaluate_fires_when_margin_exceeds_threshold() -> None:
    # _Z_SAME → margin = 1.0; threshold = 0.5 → 1.0 > 0.5 → fired.
    # PROBE-HD fires HIGH — contrast with PROBE-RM which fires LOW.
    _, verdict = evaluate(_Z_SAME, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_margin_below_threshold() -> None:
    # _Z_OPP → margin = -1.0; threshold = 0.5 → -1.0 > 0.5 is False → not fired.
    _, verdict = evaluate(_Z_OPP, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_does_not_fire_for_orthogonal_input() -> None:
    # margin = 0.0; threshold = 0.5 → 0.0 > 0.5 is False → not fired.
    _, verdict = evaluate(_Z_ORTH, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_score_matches_harmfulness_margin() -> None:
    expected = compute_harmfulness_margin(_Z_SAME, _H)
    _, verdict = evaluate(_Z_SAME, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(expected)


def test_evaluate_margin_value_matches_harmfulness_margin() -> None:
    expected = compute_harmfulness_margin(_Z_OPP, _H)
    margin, _ = evaluate(_Z_OPP, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.value == pytest.approx(expected)


def test_evaluate_margin_to_threshold_negative_when_fired() -> None:
    # fired when score > threshold → threshold - score < 0 (same sign as VESTIBULE gates).
    _, verdict = evaluate(_Z_SAME, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True
    assert verdict.margin_to_threshold < 0.0


def test_evaluate_margin_to_threshold_positive_when_not_fired() -> None:
    # not fired when score <= threshold → threshold - score >= 0.
    _, verdict = evaluate(_Z_OPP, _H, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False
    assert verdict.margin_to_threshold > 0.0


def test_evaluate_cliff_regime_true_for_q3km() -> None:
    margin, _ = evaluate(_Z_SAME, _H, _TABLE, QuantScheme.GGUF_Q3_K_M, Tier.B)
    assert margin.is_cliff_regime is True
