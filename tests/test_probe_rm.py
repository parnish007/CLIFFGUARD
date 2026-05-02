import numpy as np
import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.probe.rm import compute_margin, evaluate

# ---------------------------------------------------------------------------
# Synthetic test vectors  (dimension 8 — small, exact arithmetic)
# ---------------------------------------------------------------------------

_DIM = 8

# Refusal direction: all-ones (unnormalised — compute_margin normalises internally).
_R: np.ndarray = np.ones(_DIM, dtype=np.float64)

# Same direction as _R → cosine similarity = 1.0
_Z_SAME: np.ndarray = np.ones(_DIM, dtype=np.float64)

# Opposite direction → cosine similarity = -1.0
_Z_OPP: np.ndarray = -np.ones(_DIM, dtype=np.float64)

# Orthogonal to _R: dot([1,-1,0,...], [1,1,...,1]) = 1-1 = 0 → cosine = 0.0
_Z_ORTH: np.ndarray = np.array([1.0, -1.0] + [0.0] * (_DIM - 2), dtype=np.float64)

_ZERO: np.ndarray = np.zeros(_DIM, dtype=np.float64)

# Calibration tables used across tests.
_TABLE = CalibrationTable(
    primitive="PROBE-RM",
    thresholds={
        QuantScheme.FP16: 0.5,
        QuantScheme.GGUF_Q3_K_M: -0.1,
    },
)


# ---------------------------------------------------------------------------
# compute_margin
# ---------------------------------------------------------------------------


def test_compute_margin_same_direction_returns_one() -> None:
    assert compute_margin(_Z_SAME, _R) == pytest.approx(1.0)


def test_compute_margin_opposite_direction_returns_minus_one() -> None:
    assert compute_margin(_Z_OPP, _R) == pytest.approx(-1.0)


def test_compute_margin_orthogonal_returns_zero() -> None:
    assert compute_margin(_Z_ORTH, _R) == pytest.approx(0.0, abs=1e-12)


def test_compute_margin_returns_float() -> None:
    result = compute_margin(_Z_SAME, _R)
    assert isinstance(result, float)


def test_compute_margin_raises_on_zero_hidden_state() -> None:
    with pytest.raises(ValueError, match="hidden_state"):
        compute_margin(_ZERO, _R)


def test_compute_margin_raises_on_zero_refusal_direction() -> None:
    with pytest.raises(ValueError, match="refusal_direction"):
        compute_margin(_Z_SAME, _ZERO)


def test_compute_margin_invariant_to_scaling() -> None:
    # Cosine similarity is scale-invariant; doubling either vector changes nothing.
    assert compute_margin(_Z_SAME * 7.3, _R * 0.1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_margin_primitive() -> None:
    margin, _ = evaluate(_Z_SAME, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.primitive == "PROBE-RM"


def test_evaluate_margin_layer_is_none() -> None:
    margin, _ = evaluate(_Z_SAME, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.layer is None


def test_evaluate_verdict_gate_name() -> None:
    _, verdict = evaluate(_Z_SAME, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "PROBE-RM"


def test_evaluate_fires_when_margin_below_threshold() -> None:
    # _Z_OPP gives margin = -1.0; threshold = 0.5 → -1.0 < 0.5 → fired.
    _, verdict = evaluate(_Z_OPP, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_margin_at_or_above_threshold() -> None:
    # _Z_SAME gives margin = 1.0; threshold = 0.5 → 1.0 >= 0.5 → not fired.
    _, verdict = evaluate(_Z_SAME, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_margin_score_matches_compute_margin() -> None:
    _, verdict = evaluate(_Z_ORTH, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(compute_margin(_Z_ORTH, _R), abs=1e-12)


def test_evaluate_cliff_regime_true_for_q3km() -> None:
    # Margin.is_cliff_regime() should be True for GGUF_Q3_K_M.
    margin, _ = evaluate(_Z_SAME, _R, _TABLE, QuantScheme.GGUF_Q3_K_M, Tier.B)
    assert margin.is_cliff_regime is True


def test_evaluate_cliff_regime_false_for_fp16() -> None:
    margin, _ = evaluate(_Z_SAME, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.is_cliff_regime is False


def test_evaluate_margin_to_threshold_positive_when_fired() -> None:
    # PROBE-RM fires when score < threshold, so threshold - score > 0 when fired.
    # (Opposite sign convention from VESTIBULE gates which fire when score > threshold.)
    _, verdict = evaluate(_Z_OPP, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True
    assert verdict.margin_to_threshold > 0.0


def test_evaluate_margin_to_threshold_negative_when_not_fired() -> None:
    # score=1.0 >= threshold=0.5 → not fired; threshold - score = -0.5 < 0.
    _, verdict = evaluate(_Z_SAME, _R, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False
    assert verdict.margin_to_threshold < 0.0
