import numpy as np
import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.bprobe.logit import evaluate, logistic_score, sigmoid

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DIM = 5
_WEIGHTS_ZERO = np.zeros(_DIM, dtype=np.float64)
_WEIGHTS_POS = np.ones(_DIM, dtype=np.float64)
_WEIGHTS_NEG = -np.ones(_DIM, dtype=np.float64)
_LOGPROBS = np.array([-0.5, -1.0, -1.5, -2.0, -0.8], dtype=np.float64)

_TABLE = CalibrationTable(
    primitive="B-PROBE-LOGIT",
    thresholds={
        QuantScheme.FP16: 0.5,
        QuantScheme.GGUF_Q3_K_M: 0.4,
    },
)

# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


def test_sigmoid_zero_returns_half() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_large_positive_near_one() -> None:
    assert sigmoid(500.0) == pytest.approx(1.0, abs=1e-9)


def test_sigmoid_large_negative_near_zero() -> None:
    assert sigmoid(-500.0) == pytest.approx(0.0, abs=1e-9)


def test_sigmoid_no_overflow_at_extreme_positive() -> None:
    import math
    assert not math.isnan(sigmoid(1e18))
    assert not math.isinf(sigmoid(1e18))


def test_sigmoid_no_overflow_at_extreme_negative() -> None:
    import math
    assert not math.isnan(sigmoid(-1e18))
    assert not math.isinf(sigmoid(-1e18))


def test_sigmoid_positive_input_above_half() -> None:
    assert sigmoid(2.0) > 0.5


def test_sigmoid_negative_input_below_half() -> None:
    assert sigmoid(-2.0) < 0.5


def test_sigmoid_returns_float() -> None:
    assert isinstance(sigmoid(1.0), float)


# ---------------------------------------------------------------------------
# logistic_score
# ---------------------------------------------------------------------------


def test_logistic_score_raises_for_mismatched_shapes() -> None:
    w = np.ones(3, dtype=np.float64)
    lp = np.ones(5, dtype=np.float64)
    with pytest.raises(ValueError, match="[Ss]hape"):
        logistic_score(lp, w)


def test_logistic_score_in_open_unit_interval() -> None:
    score = logistic_score(_LOGPROBS, _WEIGHTS_POS)
    assert 0.0 < score < 1.0


def test_logistic_score_zero_weights_returns_half() -> None:
    score = logistic_score(_LOGPROBS, _WEIGHTS_ZERO)
    assert score == pytest.approx(0.5)


def test_logistic_score_zero_weights_bias_zero_returns_half() -> None:
    score = logistic_score(_LOGPROBS, _WEIGHTS_ZERO, bias=0.0)
    assert score == pytest.approx(0.5)


def test_logistic_score_positive_weights_negative_logprobs_below_half() -> None:
    # dot(ones, negative_logprobs) < 0 → sigmoid < 0.5
    lp = np.full(_DIM, -2.0, dtype=np.float64)
    score = logistic_score(lp, _WEIGHTS_POS)
    assert score < 0.5


def test_logistic_score_large_positive_bias_near_one() -> None:
    score = logistic_score(_LOGPROBS, _WEIGHTS_ZERO, bias=500.0)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_logistic_score_returns_float() -> None:
    assert isinstance(logistic_score(_LOGPROBS, _WEIGHTS_POS), float)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    _, verdict = evaluate(_LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "B-PROBE-LOGIT"


def test_evaluate_fires_when_score_exceeds_threshold() -> None:
    # bias=500 → score ≈ 1.0 > threshold 0.5 → fired.
    _, verdict = evaluate(
        _LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.FP16, Tier.A, bias=500.0
    )
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_score_below_threshold() -> None:
    # bias=-500 → score ≈ 0.0; 0.0 > 0.5 is False → not fired.
    _, verdict = evaluate(
        _LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.FP16, Tier.A, bias=-500.0
    )
    assert verdict.fired is False


def test_evaluate_score_matches_logistic_score() -> None:
    expected = logistic_score(_LOGPROBS, _WEIGHTS_POS)
    _, verdict = evaluate(_LOGPROBS, _WEIGHTS_POS, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(expected)


def test_evaluate_margin_primitive() -> None:
    margin, _ = evaluate(_LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.primitive == "B-PROBE-LOGIT"


def test_evaluate_margin_layer_is_none() -> None:
    margin, _ = evaluate(_LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.layer is None


def test_evaluate_margin_value_matches_score() -> None:
    expected = logistic_score(_LOGPROBS, _WEIGHTS_POS)
    margin, _ = evaluate(_LOGPROBS, _WEIGHTS_POS, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.value == pytest.approx(expected)


def test_evaluate_margin_is_cliff_regime_for_q3km() -> None:
    margin, _ = evaluate(
        _LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.GGUF_Q3_K_M, Tier.B
    )
    assert margin.is_cliff_regime is True


def test_evaluate_threat_model_is_none() -> None:
    _, verdict = evaluate(_LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.threat_model is None


def test_evaluate_margin_to_threshold_negative_when_fired() -> None:
    # Fires HIGH → threshold - score < 0.
    _, verdict = evaluate(
        _LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.FP16, Tier.A, bias=500.0
    )
    assert verdict.fired is True
    assert verdict.margin_to_threshold < 0.0


def test_evaluate_margin_to_threshold_positive_when_not_fired() -> None:
    _, verdict = evaluate(
        _LOGPROBS, _WEIGHTS_ZERO, _TABLE, QuantScheme.FP16, Tier.A, bias=-500.0
    )
    assert verdict.fired is False
    assert verdict.margin_to_threshold > 0.0
