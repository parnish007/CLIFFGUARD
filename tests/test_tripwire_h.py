import numpy as np
import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.tripwire.h import cusum_statistic, evaluate, token_entropy

# ---------------------------------------------------------------------------
# Shared calibration tables
# ---------------------------------------------------------------------------

_TABLE = CalibrationTable(
    primitive="TRIPWIRE-H",
    thresholds={QuantScheme.FP16: 1.0},
)

_TABLE_HIGH = CalibrationTable(
    primitive="TRIPWIRE-H",
    thresholds={QuantScheme.FP16: 999.0},
)

# ---------------------------------------------------------------------------
# token_entropy
# ---------------------------------------------------------------------------


def test_token_entropy_deterministic_is_zero() -> None:
    # One token has logprob=0 (p=1), rest are -inf (p=0).
    logprobs = np.array([0.0, -np.inf, -np.inf, -np.inf], dtype=np.float64)
    assert token_entropy(logprobs) == pytest.approx(0.0, abs=1e-12)


def test_token_entropy_uniform_two_tokens() -> None:
    # Uniform over 2 tokens → H = log2(2) = 1.0 bit.
    logprobs = np.full(2, -np.log(2.0), dtype=np.float64)
    assert token_entropy(logprobs) == pytest.approx(1.0)


def test_token_entropy_uniform_four_tokens() -> None:
    # Uniform over 4 tokens → H = log2(4) = 2.0 bits.
    logprobs = np.full(4, -np.log(4.0), dtype=np.float64)
    assert token_entropy(logprobs) == pytest.approx(2.0)


def test_token_entropy_uniform_n_equals_log2_n() -> None:
    # General check: uniform over N → H = log2(N).
    for n in [8, 16, 32]:
        logprobs = np.full(n, -np.log(float(n)), dtype=np.float64)
        assert token_entropy(logprobs) == pytest.approx(np.log2(n), rel=1e-9)


def test_token_entropy_returns_float() -> None:
    logprobs = np.full(4, -np.log(4.0), dtype=np.float64)
    assert isinstance(token_entropy(logprobs), float)


def test_token_entropy_nonnegative() -> None:
    logprobs = np.array([-0.5, -1.0, -2.0, -3.0], dtype=np.float64)
    # Probabilities don't sum to 1 here, but entropy of any non-negative p is >= 0.
    assert token_entropy(logprobs) >= 0.0


# ---------------------------------------------------------------------------
# cusum_statistic
# ---------------------------------------------------------------------------


def test_cusum_flat_at_k_returns_all_zeros() -> None:
    # H_t == k every step → (H_t - k) == 0 → S_t stays at 0.
    k = 2.0
    entropies = np.full(6, k, dtype=np.float64)
    s = cusum_statistic(entropies, k=k)
    np.testing.assert_allclose(s, 0.0, atol=1e-12)


def test_cusum_all_below_k_rises_monotonically() -> None:
    # H_t = 0.0 < k = 1.0 → each step adds 1.0; S_t is strictly increasing.
    k = 1.0
    entropies = np.zeros(5, dtype=np.float64)
    s = cusum_statistic(entropies, k=k)
    assert all(s[i] < s[i + 1] for i in range(len(s) - 1))


def test_cusum_all_below_k_exact_values() -> None:
    # H_t = 0, k = 0.5 → S_t = t * 0.5.
    k = 0.5
    entropies = np.zeros(4, dtype=np.float64)
    s = cusum_statistic(entropies, k=k)
    expected = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    np.testing.assert_allclose(s, expected, atol=1e-12)


def test_cusum_resets_toward_zero_when_entropy_rises() -> None:
    # Three low steps build up CUSUM, then two high steps drain it.
    k = 2.0
    entropies = np.array([0.0, 0.0, 0.0, 5.0, 5.0], dtype=np.float64)
    s = cusum_statistic(entropies, k=k)
    # S after low steps: 2, 4, 6; then 6-(5-2)=3, then 3-(5-2)=0.
    assert s[2] > s[3] > s[4]
    assert s[4] == pytest.approx(0.0, abs=1e-12)


def test_cusum_never_negative() -> None:
    # CUSUM is clamped at 0 — always non-negative.
    entropies = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    s = cusum_statistic(entropies, k=0.5)
    assert (s >= 0.0).all()


def test_cusum_returns_same_length_as_input() -> None:
    entropies = np.ones(7, dtype=np.float64)
    s = cusum_statistic(entropies, k=0.5)
    assert len(s) == 7


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    entropies = np.ones(5, dtype=np.float64)
    verdict = evaluate(entropies, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "TRIPWIRE-H"


def test_evaluate_fires_on_sustained_entropy_drop() -> None:
    # Zero entropy for 5 steps with k=0.5 → final CUSUM = 2.5 > threshold 1.0.
    entropies = np.zeros(5, dtype=np.float64)
    verdict = evaluate(entropies, _TABLE, QuantScheme.FP16, Tier.A, k=0.5)
    assert verdict.fired is True


def test_evaluate_does_not_fire_on_stable_high_entropy() -> None:
    # High entropy far above k → CUSUM stays at 0 < threshold 1.0.
    entropies = np.full(10, 5.0, dtype=np.float64)
    verdict = evaluate(entropies, _TABLE, QuantScheme.FP16, Tier.A, k=0.5)
    assert verdict.fired is False


def test_evaluate_does_not_fire_with_very_high_threshold() -> None:
    entropies = np.zeros(5, dtype=np.float64)
    verdict = evaluate(entropies, _TABLE_HIGH, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_score_is_final_cusum() -> None:
    entropies = np.zeros(4, dtype=np.float64)
    expected_score = float(cusum_statistic(entropies, k=0.5)[-1])
    verdict = evaluate(entropies, _TABLE, QuantScheme.FP16, Tier.A, k=0.5)
    assert verdict.score == pytest.approx(expected_score)


def test_evaluate_threat_model_is_none() -> None:
    entropies = np.ones(3, dtype=np.float64)
    verdict = evaluate(entropies, _TABLE, QuantScheme.FP16, Tier.B)
    assert verdict.threat_model is None


def test_evaluate_empty_entropies_raises() -> None:
    with pytest.raises(ValueError):
        evaluate(np.array([], dtype=np.float64), _TABLE, QuantScheme.FP16, Tier.A)
