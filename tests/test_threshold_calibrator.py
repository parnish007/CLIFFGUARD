import warnings

import numpy as np
import pytest

from cliffguard.types import QuantScheme
from cliffguard.eval.threshold_calibrator import (
    MIN_RELIABLE_SIZE,
    build_calibration_table,
    calibrate_threshold,
    empirical_fpr,
)

# ---------------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------------


def test_calibrate_raises_for_empty_scores() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        calibrate_threshold(np.array([], dtype=np.float64))


def test_calibrate_raises_for_fpr_target_zero() -> None:
    with pytest.raises(ValueError, match="fpr_target"):
        calibrate_threshold(np.arange(10.0), fpr_target=0.0)


def test_calibrate_raises_for_fpr_target_negative() -> None:
    with pytest.raises(ValueError, match="fpr_target"):
        calibrate_threshold(np.arange(10.0), fpr_target=-0.1)


def test_calibrate_raises_for_fpr_target_one() -> None:
    with pytest.raises(ValueError, match="fpr_target"):
        calibrate_threshold(np.arange(10.0), fpr_target=1.0)


def test_calibrate_raises_for_fpr_target_above_one() -> None:
    with pytest.raises(ValueError, match="fpr_target"):
        calibrate_threshold(np.arange(10.0), fpr_target=1.5)


def test_calibrate_correct_percentile_for_known_scores() -> None:
    # np.arange(100.0): 0,1,...,99 — 95th percentile = 94.05
    scores = np.arange(100.0, dtype=np.float64)
    tau = calibrate_threshold(scores, fpr_target=0.05)
    assert tau == pytest.approx(np.percentile(scores, 95.0))


def test_calibrate_emits_warning_for_small_corpus() -> None:
    scores = np.arange(float(MIN_RELIABLE_SIZE - 1), dtype=np.float64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calibrate_threshold(scores, fpr_target=0.05)
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "§14.4" in str(w[0].message)


def test_calibrate_does_not_warn_for_large_corpus() -> None:
    scores = np.arange(float(MIN_RELIABLE_SIZE), dtype=np.float64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calibrate_threshold(scores, fpr_target=0.05)
    assert len(w) == 0


def test_calibrate_returns_float() -> None:
    scores = np.arange(100.0, dtype=np.float64)
    assert isinstance(calibrate_threshold(scores), float)


def test_calibrate_single_score() -> None:
    scores = np.array([3.14], dtype=np.float64)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tau = calibrate_threshold(scores, fpr_target=0.05)
    assert tau == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# build_calibration_table
# ---------------------------------------------------------------------------


def test_build_table_returns_correct_primitive() -> None:
    scores = np.arange(100.0, dtype=np.float64)
    table = build_calibration_table(
        "VESTIBULE-LZ",
        {QuantScheme.FP16: scores},
    )
    assert table.primitive == "VESTIBULE-LZ"


def test_build_table_populates_all_schemes() -> None:
    scores = np.arange(100.0, dtype=np.float64)
    table = build_calibration_table(
        "PROBE-RM",
        {QuantScheme.FP16: scores, QuantScheme.GGUF_Q4_K_M: scores},
    )
    assert QuantScheme.FP16 in table.thresholds
    assert QuantScheme.GGUF_Q4_K_M in table.thresholds


def test_build_table_threshold_matches_calibrate() -> None:
    scores = np.arange(100.0, dtype=np.float64)
    expected = calibrate_threshold(scores, fpr_target=0.05)
    table = build_calibration_table("PROBE-RM", {QuantScheme.FP16: scores})
    assert table.tau(QuantScheme.FP16) == pytest.approx(expected)


def test_build_table_raises_for_empty_dict() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_calibration_table("PROBE-RM", {})


def test_build_table_stores_fpr_target() -> None:
    scores = np.arange(100.0, dtype=np.float64)
    table = build_calibration_table("X", {QuantScheme.FP16: scores}, fpr_target=0.01)
    assert table.fpr_target == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# empirical_fpr
# ---------------------------------------------------------------------------


def test_empirical_fpr_raises_for_empty_scores() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        empirical_fpr(np.array([], dtype=np.float64), threshold=0.5)


def test_empirical_fpr_fires_high_correct() -> None:
    # scores 0..9; threshold=5 → 6,7,8,9 fire → FPR = 4/10 = 0.4
    scores = np.arange(10.0, dtype=np.float64)
    assert empirical_fpr(scores, threshold=5.0, fires_high=True) == pytest.approx(0.4)


def test_empirical_fpr_fires_low_correct() -> None:
    # scores 0..9; threshold=5 → 0,1,2,3,4 fire → FPR = 5/10 = 0.5
    scores = np.arange(10.0, dtype=np.float64)
    assert empirical_fpr(scores, threshold=5.0, fires_high=False) == pytest.approx(0.5)


def test_empirical_fpr_returns_zero_when_none_exceed_threshold() -> None:
    # threshold above all scores → no fires (fires_high=True)
    scores = np.arange(10.0, dtype=np.float64)
    assert empirical_fpr(scores, threshold=100.0, fires_high=True) == pytest.approx(0.0)


def test_empirical_fpr_returns_one_when_all_exceed_threshold() -> None:
    scores = np.arange(1.0, 11.0, dtype=np.float64)
    assert empirical_fpr(scores, threshold=0.0, fires_high=True) == pytest.approx(1.0)


def test_empirical_fpr_returns_float() -> None:
    scores = np.arange(10.0, dtype=np.float64)
    result = empirical_fpr(scores, threshold=5.0)
    assert isinstance(result, float)


def test_empirical_fpr_in_unit_interval() -> None:
    scores = np.linspace(-1.0, 1.0, 50, dtype=np.float64)
    fpr = empirical_fpr(scores, threshold=0.0)
    assert 0.0 <= fpr <= 1.0
