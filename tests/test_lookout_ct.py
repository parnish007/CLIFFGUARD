import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.lookout.ct import BloomFilter, build_canary_filter, check_output, evaluate

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CANARIES = ["ALPHA-7731", "BETA-0042", "GAMMA-9981"]

_TABLE = CalibrationTable(
    primitive="LOOKOUT-CT",
    thresholds={QuantScheme.FP16: 0.5},
)

_TABLE_STRICT = CalibrationTable(
    primitive="LOOKOUT-CT",
    thresholds={QuantScheme.FP16: -0.5},
)


# ---------------------------------------------------------------------------
# BloomFilter
# ---------------------------------------------------------------------------


def test_bloom_filter_contains_added_item() -> None:
    bf = BloomFilter()
    bf.add("hello")
    assert "hello" in bf


def test_bloom_filter_does_not_contain_unadded_item() -> None:
    bf = BloomFilter()
    bf.add("hello")
    # "world" has extremely low probability of false-positive in m=256, k=3
    assert "world" not in bf


def test_bloom_filter_multiple_items() -> None:
    bf = BloomFilter()
    for item in _CANARIES:
        bf.add(item)
    for item in _CANARIES:
        assert item in bf


def test_bloom_filter_non_string_not_contained() -> None:
    bf = BloomFilter()
    bf.add("hello")
    assert 42 not in bf  # type: ignore[operator]


def test_bloom_filter_bit_positions_length() -> None:
    bf = BloomFilter(m=128, k=5)
    positions = bf._bit_positions("test")
    assert len(positions) == 5


def test_bloom_filter_bit_positions_in_range() -> None:
    bf = BloomFilter(m=64, k=3)
    positions = bf._bit_positions("canary")
    assert all(0 <= p < 64 for p in positions)


def test_bloom_filter_bit_positions_deterministic() -> None:
    bf = BloomFilter()
    assert bf._bit_positions("abc") == bf._bit_positions("abc")


def test_bloom_filter_empty_string() -> None:
    bf = BloomFilter()
    bf.add("")
    assert "" in bf


# ---------------------------------------------------------------------------
# build_canary_filter
# ---------------------------------------------------------------------------


def test_build_canary_filter_all_canaries_present() -> None:
    bf = build_canary_filter(_CANARIES)
    for token in _CANARIES:
        assert token in bf


def test_build_canary_filter_returns_bloom_filter() -> None:
    bf = build_canary_filter(_CANARIES)
    assert isinstance(bf, BloomFilter)


def test_build_canary_filter_empty_list() -> None:
    bf = build_canary_filter([])
    assert isinstance(bf, BloomFilter)


# ---------------------------------------------------------------------------
# check_output
# ---------------------------------------------------------------------------


def test_check_output_finds_present_canary() -> None:
    bf = build_canary_filter(_CANARIES)
    text = f"The secret is {_CANARIES[0]} embedded here."
    matched = check_output(text, bf, _CANARIES)
    assert _CANARIES[0] in matched


def test_check_output_finds_all_present_canaries() -> None:
    bf = build_canary_filter(_CANARIES)
    text = " ".join(_CANARIES)
    matched = check_output(text, bf, _CANARIES)
    assert set(matched) == set(_CANARIES)


def test_check_output_returns_empty_when_none_present() -> None:
    bf = build_canary_filter(_CANARIES)
    matched = check_output("No canary tokens here.", bf, _CANARIES)
    assert matched == []


def test_check_output_partial_match() -> None:
    bf = build_canary_filter(_CANARIES)
    text = f"Only {_CANARIES[1]} is here."
    matched = check_output(text, bf, _CANARIES)
    assert matched == [_CANARIES[1]]


def test_check_output_no_false_positives_on_substrings() -> None:
    # "ALPHA-773" is a prefix of "ALPHA-7731" but is not the full canary.
    bf = build_canary_filter(["ALPHA-7731"])
    matched = check_output("Found ALPHA-773 here.", bf, ["ALPHA-7731"])
    assert matched == []


def test_check_output_empty_output_text() -> None:
    bf = build_canary_filter(_CANARIES)
    matched = check_output("", bf, _CANARIES)
    assert matched == []


def test_check_output_empty_canary_list() -> None:
    bf = build_canary_filter([])
    matched = check_output("some output", bf, [])
    assert matched == []


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    verdict = evaluate("clean output", _CANARIES, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "LOOKOUT-CT"


def test_evaluate_fires_when_canary_found() -> None:
    # threshold = 0.5; score = 1 → 1 > 0.5 → fired.
    text = f"Leaked: {_CANARIES[0]}"
    verdict = evaluate(text, _CANARIES, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_no_canary_found() -> None:
    # threshold = 0.5; score = 0 → 0 > 0.5 is False → not fired.
    verdict = evaluate("clean output", _CANARIES, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_score_equals_count_of_matched_canaries() -> None:
    text = f"{_CANARIES[0]} and {_CANARIES[2]}"
    verdict = evaluate(text, _CANARIES, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(2.0)


def test_evaluate_score_is_zero_for_clean_output() -> None:
    verdict = evaluate("nothing suspicious", _CANARIES, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(0.0)


def test_evaluate_threat_model_is_none() -> None:
    verdict = evaluate("clean", _CANARIES, _TABLE, QuantScheme.FP16, Tier.B)
    assert verdict.threat_model is None


def test_evaluate_fires_high_gate_positive_margin_to_threshold_when_not_fired() -> None:
    # LOOKOUT-CT fires HIGH: score > threshold.
    # Not fired → score <= threshold → threshold - score >= 0 (positive).
    verdict = evaluate("clean", _CANARIES, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False
    assert verdict.margin_to_threshold > 0.0


def test_evaluate_fires_high_gate_negative_margin_to_threshold_when_fired() -> None:
    # Fired → score > threshold → threshold - score < 0 (negative).
    text = " ".join(_CANARIES)
    verdict = evaluate(text, _CANARIES, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True
    assert verdict.margin_to_threshold < 0.0


def test_evaluate_all_canaries_leaked() -> None:
    text = " ".join(_CANARIES)
    verdict = evaluate(text, _CANARIES, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(float(len(_CANARIES)))


def test_evaluate_empty_canary_list_does_not_fire() -> None:
    verdict = evaluate("some output", [], _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False
    assert verdict.score == pytest.approx(0.0)


def test_evaluate_tier_is_passed_through() -> None:
    verdict = evaluate("clean", _CANARIES, _TABLE, QuantScheme.FP16, Tier.C)
    assert verdict.tier == Tier.C
