import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.lookout.jg import compliance_rate, evaluate

# ---------------------------------------------------------------------------
# Stub implementations of Protocol interfaces (test-only)
# ---------------------------------------------------------------------------


class _ConstantParaphraser:
    """Returns n copies of a fixed string."""

    def __init__(self, text: str = "paraphrase") -> None:
        self._text = text

    def paraphrase(self, text: str, n: int) -> list[str]:
        return [self._text] * n


class _IdentityParaphraser:
    """Returns n copies of the original text unchanged."""

    def paraphrase(self, text: str, n: int) -> list[str]:
        return [text] * n


class _AlternatingParaphraser:
    """Returns n distinct strings so the alternating judge produces True/False."""

    def paraphrase(self, text: str, n: int) -> list[str]:
        return [f"para-{i}" for i in range(n)]


class _AlwaysCompliantJudge:
    def is_compliant(self, response: str) -> bool:
        return True


class _NeverCompliantJudge:
    def is_compliant(self, response: str) -> bool:
        return False


class _AlternatingJudge:
    """Compliant on even-indexed calls (para-0, para-2, …), not on odd."""

    def __init__(self) -> None:
        self._call = 0

    def is_compliant(self, response: str) -> bool:
        result = self._call % 2 == 0
        self._call += 1
        return result


# ---------------------------------------------------------------------------
# Shared calibration table
# ---------------------------------------------------------------------------

_TABLE = CalibrationTable(
    primitive="LOOKOUT-JG",
    thresholds={QuantScheme.FP16: 0.5},
)

# ---------------------------------------------------------------------------
# compliance_rate
# ---------------------------------------------------------------------------


def test_compliance_rate_raises_for_n_less_than_1() -> None:
    with pytest.raises(ValueError):
        compliance_rate("response", _ConstantParaphraser(), _AlwaysCompliantJudge(), n=0)


def test_compliance_rate_raises_for_negative_n() -> None:
    with pytest.raises(ValueError):
        compliance_rate("response", _ConstantParaphraser(), _AlwaysCompliantJudge(), n=-5)


def test_compliance_rate_returns_zero_when_all_non_compliant() -> None:
    rate = compliance_rate(
        "response", _ConstantParaphraser(), _NeverCompliantJudge(), n=3
    )
    assert rate == pytest.approx(0.0)


def test_compliance_rate_returns_one_when_all_compliant() -> None:
    rate = compliance_rate(
        "response", _ConstantParaphraser(), _AlwaysCompliantJudge(), n=3
    )
    assert rate == pytest.approx(1.0)


def test_compliance_rate_returns_half_for_alternating_n2() -> None:
    # _AlternatingParaphraser returns ["para-0", "para-1"].
    # _AlternatingJudge: para-0 → True (call 0), para-1 → False (call 1).
    # 1 compliant / 2 = 0.5.
    rate = compliance_rate(
        "response", _AlternatingParaphraser(), _AlternatingJudge(), n=2
    )
    assert rate == pytest.approx(0.5)


def test_compliance_rate_range_is_zero_to_one() -> None:
    rate = compliance_rate(
        "response", _ConstantParaphraser(), _AlternatingJudge(), n=3
    )
    assert 0.0 <= rate <= 1.0


def test_compliance_rate_n1_always_compliant() -> None:
    rate = compliance_rate(
        "response", _ConstantParaphraser(), _AlwaysCompliantJudge(), n=1
    )
    assert rate == pytest.approx(1.0)


def test_compliance_rate_n1_never_compliant() -> None:
    rate = compliance_rate(
        "response", _ConstantParaphraser(), _NeverCompliantJudge(), n=1
    )
    assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    verdict = evaluate(
        "response", _ConstantParaphraser(), _NeverCompliantJudge(),
        _TABLE, QuantScheme.FP16, Tier.A,
    )
    assert verdict.gate == "LOOKOUT-JG"


def test_evaluate_fires_when_compliance_rate_exceeds_threshold() -> None:
    # All compliant → score = 1.0 > threshold 0.5 → fired.
    verdict = evaluate(
        "response", _ConstantParaphraser(), _AlwaysCompliantJudge(),
        _TABLE, QuantScheme.FP16, Tier.A,
    )
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_compliance_rate_below_threshold() -> None:
    # None compliant → score = 0.0; 0.0 > 0.5 is False → not fired.
    verdict = evaluate(
        "response", _ConstantParaphraser(), _NeverCompliantJudge(),
        _TABLE, QuantScheme.FP16, Tier.A,
    )
    assert verdict.fired is False


def test_evaluate_raises_for_n_less_than_1() -> None:
    with pytest.raises(ValueError):
        evaluate(
            "response", _ConstantParaphraser(), _AlwaysCompliantJudge(),
            _TABLE, QuantScheme.FP16, Tier.A, n=0,
        )


def test_evaluate_score_matches_compliance_rate() -> None:
    verdict = evaluate(
        "response", _ConstantParaphraser(), _AlwaysCompliantJudge(),
        _TABLE, QuantScheme.FP16, Tier.A, n=3,
    )
    assert verdict.score == pytest.approx(1.0)


def test_evaluate_threat_model_is_none() -> None:
    verdict = evaluate(
        "response", _ConstantParaphraser(), _NeverCompliantJudge(),
        _TABLE, QuantScheme.FP16, Tier.B,
    )
    assert verdict.threat_model is None


def test_evaluate_margin_to_threshold_negative_when_fired() -> None:
    # Fires HIGH: score > threshold → threshold - score < 0.
    verdict = evaluate(
        "response", _ConstantParaphraser(), _AlwaysCompliantJudge(),
        _TABLE, QuantScheme.FP16, Tier.A,
    )
    assert verdict.fired is True
    assert verdict.margin_to_threshold < 0.0


def test_evaluate_margin_to_threshold_positive_when_not_fired() -> None:
    verdict = evaluate(
        "response", _ConstantParaphraser(), _NeverCompliantJudge(),
        _TABLE, QuantScheme.FP16, Tier.A,
    )
    assert verdict.fired is False
    assert verdict.margin_to_threshold > 0.0


def test_evaluate_tier_passed_through() -> None:
    verdict = evaluate(
        "response", _ConstantParaphraser(), _NeverCompliantJudge(),
        _TABLE, QuantScheme.FP16, Tier.C,
    )
    assert verdict.tier == Tier.C
