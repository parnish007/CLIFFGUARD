import pytest
from pydantic import ValidationError

from cliffguard.types import CalibrationTable, GateVerdict, Margin, QuantScheme, Tier, ThreatModel


def test_threat_model_all_nine_members() -> None:
    assert {m.name for m in ThreatModel} == {f"A{i}" for i in range(1, 10)}


def test_threat_model_descriptions_nonempty() -> None:
    for member in ThreatModel:
        assert member.description(), f"{member.name} has empty description"


def test_tier_all_four_members() -> None:
    assert {m.name for m in Tier} == {"A", "B", "C", "C_PLUS"}


def test_tier_descriptions_nonempty() -> None:
    for member in Tier:
        assert member.description(), f"{member.name} has empty description"


def test_quant_scheme_cliff_candidate_true() -> None:
    assert QuantScheme.GGUF_Q3_K_M.is_cliff_candidate() is True


def test_quant_scheme_cliff_candidate_false() -> None:
    assert QuantScheme.NF4.is_cliff_candidate() is False


def test_quant_scheme_from_string_lowercase() -> None:
    assert QuantScheme.from_string("nf4") is QuantScheme.NF4


def test_quant_scheme_from_string_invalid_raises() -> None:
    with pytest.raises(ValueError):
        QuantScheme.from_string("invalid")


# ---------------------------------------------------------------------------
# Margin
# ---------------------------------------------------------------------------


def test_margin_rejects_empty_primitive() -> None:
    with pytest.raises(ValidationError):
        Margin(value=0.5, scheme=QuantScheme.NF4, primitive="")


def test_margin_is_cliff_regime_true() -> None:
    m = Margin(value=-0.1, scheme=QuantScheme.GGUF_Q3_K_M, primitive="PROBE-RM")
    assert m.is_cliff_regime is True


def test_margin_is_cliff_regime_false() -> None:
    m = Margin(value=0.8, scheme=QuantScheme.NF4, primitive="PROBE-RM")
    assert m.is_cliff_regime is False


# ---------------------------------------------------------------------------
# CalibrationTable
# ---------------------------------------------------------------------------


def test_calibration_table_tau_known_scheme() -> None:
    table = CalibrationTable(
        primitive="PROBE-RM",
        thresholds={QuantScheme.NF4: 0.3, QuantScheme.GGUF_Q4_K_M: 0.25},
    )
    assert table.tau(QuantScheme.NF4) == pytest.approx(0.3)


def test_calibration_table_tau_unknown_scheme_raises() -> None:
    table = CalibrationTable(
        primitive="PROBE-RM",
        thresholds={QuantScheme.NF4: 0.3},
    )
    with pytest.raises(KeyError):
        table.tau(QuantScheme.GGUF_Q3_K_M)


def test_calibration_table_schemes_covered_sorted() -> None:
    table = CalibrationTable(
        primitive="PROBE-RM",
        thresholds={
            QuantScheme.GGUF_Q4_K_M: 0.25,
            QuantScheme.NF4: 0.3,
            QuantScheme.FP16: 0.4,
        },
    )
    covered = table.schemes_covered()
    names = [s.name for s in covered]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# GateVerdict
# ---------------------------------------------------------------------------


def test_gate_verdict_margin_to_threshold() -> None:
    verdict = GateVerdict(
        gate="PROBE-RM",
        fired=False,
        score=0.4,
        threshold=0.8,
        tier=Tier.A,
    )
    assert verdict.margin_to_threshold == pytest.approx(0.4)


def test_gate_verdict_margin_to_threshold_negative_when_fired() -> None:
    verdict = GateVerdict(
        gate="VESTIBULE-LZ",
        fired=True,
        score=0.9,
        threshold=0.85,
        tier=Tier.B,
        threat_model=ThreatModel.A3,
    )
    assert verdict.margin_to_threshold == pytest.approx(-0.05)
