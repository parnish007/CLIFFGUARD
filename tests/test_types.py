import pytest

from cliffguard.types import QuantScheme, Tier, ThreatModel


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
