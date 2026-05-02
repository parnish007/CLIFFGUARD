import numpy as np
import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.tripwire.r import evaluate, log_likelihood_ratio

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Calibration: threshold at -0.5 — fires when LLR drops below -0.5.
_TABLE = CalibrationTable(
    primitive="TRIPWIRE-R",
    thresholds={QuantScheme.FP16: -0.5},
)

# Synthetic per-token log-probabilities (natural log scale).
_BENIGN = np.array([-1.0, -1.5, -2.0, -1.2, -0.8], dtype=np.float64)

# Input matches benign exactly → LLR = 0.0
_INPUT_SAME = _BENIGN.copy()

# Input is less likely than benign (lower log-probs) → LLR < 0
_INPUT_LOWER = _BENIGN - 2.0

# Input is more likely than benign (higher log-probs) → LLR > 0
_INPUT_HIGHER = _BENIGN + 1.5


# ---------------------------------------------------------------------------
# log_likelihood_ratio
# ---------------------------------------------------------------------------


def test_llr_raises_for_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        log_likelihood_ratio(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )


def test_llr_raises_for_empty_input_only() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        log_likelihood_ratio(np.array([], dtype=np.float64), _BENIGN)


def test_llr_raises_for_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        log_likelihood_ratio(
            np.array([-1.0, -2.0], dtype=np.float64),
            np.array([-1.0, -2.0, -3.0], dtype=np.float64),
        )


def test_llr_returns_zero_when_arrays_equal() -> None:
    assert log_likelihood_ratio(_INPUT_SAME, _BENIGN) == pytest.approx(0.0, abs=1e-12)


def test_llr_returns_negative_when_input_less_likely() -> None:
    llr = log_likelihood_ratio(_INPUT_LOWER, _BENIGN)
    assert llr < 0.0


def test_llr_returns_positive_when_input_more_likely() -> None:
    llr = log_likelihood_ratio(_INPUT_HIGHER, _BENIGN)
    assert llr > 0.0


def test_llr_exact_value_lower() -> None:
    # _INPUT_LOWER = _BENIGN - 2.0 → mean difference = -2.0.
    assert log_likelihood_ratio(_INPUT_LOWER, _BENIGN) == pytest.approx(-2.0)


def test_llr_exact_value_higher() -> None:
    # _INPUT_HIGHER = _BENIGN + 1.5 → mean difference = +1.5.
    assert log_likelihood_ratio(_INPUT_HIGHER, _BENIGN) == pytest.approx(1.5)


def test_llr_returns_float() -> None:
    assert isinstance(log_likelihood_ratio(_INPUT_SAME, _BENIGN), float)


def test_llr_single_token() -> None:
    # Edge case: exactly one token — must work without raising.
    inp = np.array([-3.0], dtype=np.float64)
    ref = np.array([-1.0], dtype=np.float64)
    assert log_likelihood_ratio(inp, ref) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    verdict = evaluate(_INPUT_SAME, _BENIGN, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "TRIPWIRE-R"


def test_evaluate_fires_when_llr_below_threshold() -> None:
    # LLR = -2.0; threshold = -0.5 → -2.0 < -0.5 → fired.
    verdict = evaluate(_INPUT_LOWER, _BENIGN, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_llr_above_threshold() -> None:
    # LLR = +1.5; threshold = -0.5 → 1.5 >= -0.5 → not fired.
    verdict = evaluate(_INPUT_HIGHER, _BENIGN, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_does_not_fire_when_llr_equals_zero() -> None:
    # LLR = 0.0 > threshold -0.5 → not fired.
    verdict = evaluate(_INPUT_SAME, _BENIGN, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_score_matches_llr() -> None:
    expected = log_likelihood_ratio(_INPUT_LOWER, _BENIGN)
    verdict = evaluate(_INPUT_LOWER, _BENIGN, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(expected)


def test_evaluate_threat_model_is_none() -> None:
    verdict = evaluate(_INPUT_SAME, _BENIGN, _TABLE, QuantScheme.FP16, Tier.B)
    assert verdict.threat_model is None


def test_evaluate_raises_on_empty_arrays() -> None:
    with pytest.raises(ValueError):
        evaluate(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            _TABLE,
            QuantScheme.FP16,
            Tier.A,
        )


def test_evaluate_raises_on_mismatched_arrays() -> None:
    with pytest.raises(ValueError):
        evaluate(
            np.array([-1.0, -2.0], dtype=np.float64),
            np.array([-1.0], dtype=np.float64),
            _TABLE,
            QuantScheme.FP16,
            Tier.A,
        )


def test_evaluate_margin_to_threshold_negative_when_fired() -> None:
    # TRIPWIRE-R fires when score < threshold → threshold - score > 0.
    # But score=-2.0, threshold=-0.5 → threshold - score = 1.5 > 0.
    # (Same sign convention as PROBE-RM: fires-low gates have positive margin_to_threshold.)
    verdict = evaluate(_INPUT_LOWER, _BENIGN, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True
    assert verdict.margin_to_threshold > 0.0
