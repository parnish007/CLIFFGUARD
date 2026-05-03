# B-PROBE-CONSISTENCY fires low (low JSD = danger) — same group as PROBE-RM, PROBE-MT, TRIPWIRE-R.

import numpy as np
import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.bprobe.consistency import evaluate, js_divergence

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Two identical uniform distributions over 4 tokens → JSD = 0.0
_UNIFORM_4 = np.log(np.full(4, 0.25, dtype=np.float64))
_IDENTICAL = np.stack([_UNIFORM_4, _UNIFORM_4])  # shape (2, 4)

# Two maximally different distributions: P1 = [1, 0, 0, 0], P2 = [0, 0, 0, 1]
_DELTA_0 = np.array([0.0, -1e9, -1e9, -1e9], dtype=np.float64)
_DELTA_3 = np.array([-1e9, -1e9, -1e9, 0.0], dtype=np.float64)
_DIFFERENT = np.stack([_DELTA_0, _DELTA_3])  # shape (2, 4)

# Three identical distributions
_IDENTICAL_3 = np.stack([_UNIFORM_4, _UNIFORM_4, _UNIFORM_4])  # shape (3, 4)

# Three different distributions
_DIFFERENT_3 = np.stack([
    np.array([0.0, -1e9, -1e9, -1e9], dtype=np.float64),
    np.array([-1e9, 0.0, -1e9, -1e9], dtype=np.float64),
    np.array([-1e9, -1e9, 0.0, -1e9], dtype=np.float64),
])  # shape (3, 4)

# Calibration: threshold at 0.3 nats — fires when JSD drops below 0.3.
_TABLE = CalibrationTable(
    primitive="B-PROBE-CONSISTENCY",
    thresholds={QuantScheme.FP16: 0.3},
)

# ---------------------------------------------------------------------------
# js_divergence — error cases
# ---------------------------------------------------------------------------


def test_js_divergence_raises_for_n_less_than_2() -> None:
    single = np.log(np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64))
    with pytest.raises(ValueError, match="2"):
        js_divergence(single)


def test_js_divergence_raises_for_zero_rows() -> None:
    empty = np.zeros((0, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        js_divergence(empty)


def test_js_divergence_raises_for_v_less_than_1() -> None:
    no_vocab = np.zeros((2, 0), dtype=np.float64)
    with pytest.raises(ValueError, match="1"):
        js_divergence(no_vocab)


# ---------------------------------------------------------------------------
# js_divergence — correctness
# ---------------------------------------------------------------------------


def test_js_divergence_identical_distributions_returns_zero() -> None:
    assert js_divergence(_IDENTICAL) == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_different_distributions_positive() -> None:
    assert js_divergence(_DIFFERENT) > 0.0


def test_js_divergence_result_in_range_n2() -> None:
    jsd = js_divergence(_IDENTICAL)
    assert 0.0 <= jsd <= np.log(2)


def test_js_divergence_result_in_range_n2_different() -> None:
    jsd = js_divergence(_DIFFERENT)
    assert 0.0 <= jsd <= np.log(2) + 1e-9


def test_js_divergence_result_in_range_n3_identical() -> None:
    jsd = js_divergence(_IDENTICAL_3)
    assert 0.0 <= jsd <= np.log(3) + 1e-9


def test_js_divergence_result_in_range_n3_different() -> None:
    jsd = js_divergence(_DIFFERENT_3)
    assert 0.0 <= jsd <= np.log(3) + 1e-9


def test_js_divergence_identical_3_returns_zero() -> None:
    assert js_divergence(_IDENTICAL_3) == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_different_3_positive() -> None:
    assert js_divergence(_DIFFERENT_3) > 0.0


def test_js_divergence_returns_float() -> None:
    assert isinstance(js_divergence(_IDENTICAL), float)


def test_js_divergence_symmetric() -> None:
    # Swapping rows must not change JSD.
    flipped = np.stack([_DIFFERENT[1], _DIFFERENT[0]])
    assert js_divergence(_DIFFERENT) == pytest.approx(js_divergence(flipped), abs=1e-12)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    _, verdict = evaluate(_IDENTICAL, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "B-PROBE-CONSISTENCY"


def test_evaluate_fires_low_when_jsd_below_threshold() -> None:
    # Identical distributions → JSD = 0.0 < threshold 0.3 → fired.
    # B-PROBE-CONSISTENCY fires LOW: consistent outputs = compliant = dangerous.
    _, verdict = evaluate(_IDENTICAL, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_jsd_above_threshold() -> None:
    # Different distributions → JSD > 0.3 → not fired (high variance = refusing).
    _, verdict = evaluate(_DIFFERENT, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_score_matches_js_divergence() -> None:
    expected = js_divergence(_IDENTICAL)
    _, verdict = evaluate(_IDENTICAL, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(expected)


def test_evaluate_margin_primitive() -> None:
    margin, _ = evaluate(_IDENTICAL, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.primitive == "B-PROBE-CONSISTENCY"


def test_evaluate_margin_layer_is_none() -> None:
    margin, _ = evaluate(_IDENTICAL, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.layer is None


def test_evaluate_margin_value_matches_score() -> None:
    expected = js_divergence(_DIFFERENT)
    margin, _ = evaluate(_DIFFERENT, _TABLE, QuantScheme.FP16, Tier.A)
    assert margin.value == pytest.approx(expected)


def test_evaluate_threat_model_is_none() -> None:
    _, verdict = evaluate(_IDENTICAL, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.threat_model is None


def test_evaluate_raises_for_single_row() -> None:
    single = np.stack([_UNIFORM_4])
    with pytest.raises(ValueError):
        evaluate(single, _TABLE, QuantScheme.FP16, Tier.A)


def test_evaluate_margin_to_threshold_positive_when_fired() -> None:
    # Fires LOW: score < threshold → threshold - score > 0.
    # Same sign convention as PROBE-RM, PROBE-MT, TRIPWIRE-R.
    _, verdict = evaluate(_IDENTICAL, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True
    assert verdict.margin_to_threshold > 0.0


def test_evaluate_margin_to_threshold_negative_when_not_fired() -> None:
    _, verdict = evaluate(_DIFFERENT, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False
    assert verdict.margin_to_threshold < 0.0


def test_evaluate_tier_passed_through() -> None:
    _, verdict = evaluate(_IDENTICAL, _TABLE, QuantScheme.FP16, Tier.C)
    assert verdict.tier == Tier.C
