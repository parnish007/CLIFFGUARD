import pytest

from cliffguard.types import CalibrationTable, QuantScheme, Tier
from cliffguard.vestibule.lz import compression_ratio, evaluate

# ---------------------------------------------------------------------------
# Fixtures / shared test data
# ---------------------------------------------------------------------------

# 500-char English prose — diverse enough to compress to < original size.
_ENGLISH = (
    "Natural language text is highly compressible because words and phrases "
    "recur within and across sentences. Compression algorithms exploit this "
    "statistical redundancy to represent the same information in fewer bytes, "
    "which is why a typical English paragraph shrinks to roughly half its "
    "original byte count when passed through a general-purpose deflate codec. "
    "This property distinguishes human-readable input from adversarial noise."
)

# 500-char run of a single character — maximum repetition, minimum entropy.
_REPETITIVE = "a" * 500

# 300-char pseudo-random printable ASCII — high lexical entropy, compresses poorly.
# Stride of 97 (prime, coprime to 95) over printable range [32, 126] gives
# long non-repeating sequences that DEFLATE cannot back-reference efficiently.
_HIGH_ENTROPY = "".join(chr((i * 97) % 95 + 32) for i in range(300))

# Shared calibration table used across several tests.
_TABLE = CalibrationTable(
    primitive="VESTIBULE-LZ",
    thresholds={QuantScheme.FP16: 0.8},
)


# ---------------------------------------------------------------------------
# compression_ratio unit tests
# ---------------------------------------------------------------------------


def test_compression_ratio_english_in_range() -> None:
    ratio = compression_ratio(_ENGLISH)
    assert 0.0 < ratio <= 1.0


def test_compression_ratio_empty_returns_one() -> None:
    assert compression_ratio("") == 1.0


def test_compression_ratio_repetitive_lower_than_high_entropy() -> None:
    assert compression_ratio(_REPETITIVE) < compression_ratio(_HIGH_ENTROPY)


def test_compression_ratio_returns_float() -> None:
    assert isinstance(compression_ratio(_ENGLISH), float)


# ---------------------------------------------------------------------------
# evaluate unit tests
# ---------------------------------------------------------------------------


def test_evaluate_gate_name() -> None:
    verdict = evaluate(_ENGLISH, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.gate == "VESTIBULE-LZ"


def test_evaluate_score_matches_compression_ratio() -> None:
    verdict = evaluate(_ENGLISH, _TABLE, QuantScheme.FP16, Tier.A)
    assert verdict.score == pytest.approx(compression_ratio(_ENGLISH))


def test_evaluate_fires_when_ratio_exceeds_threshold() -> None:
    # Threshold set well below any real ratio so the gate always fires.
    low_table = CalibrationTable(
        primitive="VESTIBULE-LZ",
        thresholds={QuantScheme.FP16: 0.001},
    )
    verdict = evaluate(_HIGH_ENTROPY, low_table, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True


def test_evaluate_does_not_fire_when_ratio_below_threshold() -> None:
    # Threshold set above 1.0 — no ratio can ever exceed it.
    high_table = CalibrationTable(
        primitive="VESTIBULE-LZ",
        thresholds={QuantScheme.FP16: 2.0},
    )
    verdict = evaluate(_REPETITIVE, high_table, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False


def test_evaluate_margin_negative_when_fired() -> None:
    low_table = CalibrationTable(
        primitive="VESTIBULE-LZ",
        thresholds={QuantScheme.FP16: 0.001},
    )
    verdict = evaluate(_HIGH_ENTROPY, low_table, QuantScheme.FP16, Tier.A)
    assert verdict.fired is True
    assert verdict.margin_to_threshold < 0.0


def test_evaluate_margin_positive_when_not_fired() -> None:
    high_table = CalibrationTable(
        primitive="VESTIBULE-LZ",
        thresholds={QuantScheme.FP16: 2.0},
    )
    verdict = evaluate(_REPETITIVE, high_table, QuantScheme.FP16, Tier.A)
    assert verdict.fired is False
    assert verdict.margin_to_threshold > 0.0


def test_evaluate_threat_model_is_none() -> None:
    verdict = evaluate(_ENGLISH, _TABLE, QuantScheme.FP16, Tier.B)
    assert verdict.threat_model is None


def test_evaluate_tier_is_propagated() -> None:
    verdict = evaluate(_ENGLISH, _TABLE, QuantScheme.FP16, Tier.C)
    assert verdict.tier is Tier.C
