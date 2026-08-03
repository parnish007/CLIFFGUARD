import warnings

import numpy as np
import pytest

from cliffguard.types import QuantScheme
from cliffguard.eval.threshold_calibrator import (
    fires_high_for,
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
    """Discontinuous order-statistic quantile, not linear interpolation.

    For a strict `score > tau` rule, `method="higher"` guarantees the
    same-sample rate does not exceed the target; linear interpolation does
    not (see test_tiny_n_does_not_overshoot_target)."""
    scores = np.arange(100.0, dtype=np.float64)
    tau = calibrate_threshold(scores, fpr_target=0.05)
    assert tau == pytest.approx(np.percentile(scores, 95.0, method="higher"))
    assert empirical_fpr(scores, tau, fires_high=True) <= 0.05


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
    """A fires-HIGH primitive is calibrated against the upper tail."""
    scores = np.arange(100.0, dtype=np.float64)
    expected = calibrate_threshold(scores, fpr_target=0.05, fires_high=True)
    table = build_calibration_table("VESTIBULE-LZ", {QuantScheme.FP16: scores})
    assert table.tau(QuantScheme.FP16) == pytest.approx(expected)


def test_build_table_uses_lower_tail_for_fires_low_primitive() -> None:
    """Regression test for D0: PROBE-RM fires LOW, so it must be calibrated
    against the LOWER tail. Calibrating it against the upper tail inverts the
    FPR (a 0.05 target becomes a realised 0.95)."""
    scores = np.arange(100.0, dtype=np.float64)
    expected = calibrate_threshold(scores, fpr_target=0.05, fires_high=False)
    table = build_calibration_table("PROBE-RM", {QuantScheme.FP16: scores})
    assert table.tau(QuantScheme.FP16) == pytest.approx(expected)
    # and it must NOT be the upper-tail value
    upper = calibrate_threshold(scores, fpr_target=0.05, fires_high=True)
    assert table.tau(QuantScheme.FP16) != pytest.approx(upper)


def test_build_table_raises_for_empty_dict() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_calibration_table("PROBE-RM", {})


def test_build_table_stores_fpr_target() -> None:
    scores = np.arange(100.0, dtype=np.float64)
    table = build_calibration_table(
        "VESTIBULE-LZ", {QuantScheme.FP16: scores}, fpr_target=0.01
    )
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


# ---------------------------------------------------------------------------
# Semantic round-trip: calibrate -> evaluate -> realised FPR ~= target.
#
# These are the tests whose absence let D0 survive 989 passing tests. Every
# pre-existing test asserted a percentile *value*; none asserted that the
# calibrated threshold actually produces the requested false-positive rate
# under the gate's own firing rule.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fpr_target", [0.01, 0.05, 0.10, 0.25])
def test_realised_fpr_matches_target_for_fires_high(fpr_target: float) -> None:
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 1.0, 20_000)
    tau = calibrate_threshold(benign, fpr_target=fpr_target, fires_high=True)
    realised = empirical_fpr(benign, tau, fires_high=True)
    assert realised == pytest.approx(fpr_target, abs=0.01)


@pytest.mark.parametrize("fpr_target", [0.01, 0.05, 0.10, 0.25])
def test_realised_fpr_matches_target_for_fires_low(fpr_target: float) -> None:
    """D0 regression: a fires-LOW gate calibrated on the lower tail must
    realise the requested FPR, not its complement."""
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 1.0, 20_000)
    tau = calibrate_threshold(benign, fpr_target=fpr_target, fires_high=False)
    realised = empirical_fpr(benign, tau, fires_high=False)
    assert realised == pytest.approx(fpr_target, abs=0.01)


def test_wrong_tail_inverts_the_fpr() -> None:
    """Documents the D0 failure mode explicitly: calibrating a fires-LOW gate
    against the upper tail yields FPR = 1 - target."""
    rng = np.random.default_rng(0)
    benign = rng.normal(0.10, 0.02, 20_000)
    wrong_tau = calibrate_threshold(benign, fpr_target=0.05, fires_high=True)
    realised = empirical_fpr(benign, wrong_tau, fires_high=False)
    assert realised == pytest.approx(0.95, abs=0.01)


def test_build_table_realises_target_fpr_for_probe_rm() -> None:
    """End-to-end: the table built for PROBE-RM must realise 5% FPR under
    PROBE-RM's own fires-LOW rule."""
    rng = np.random.default_rng(1)
    benign = rng.normal(0.10, 0.02, 20_000)
    table = build_calibration_table(
        "PROBE-RM", {QuantScheme.FP16: benign}, fpr_target=0.05
    )
    realised = empirical_fpr(benign, table.tau(QuantScheme.FP16), fires_high=False)
    assert realised == pytest.approx(0.05, abs=0.01)


def test_fires_high_for_resolves_known_primitives() -> None:
    assert fires_high_for("PROBE-RM") is False
    assert fires_high_for("PROBE-MT") is False
    assert fires_high_for("TRIPWIRE-R") is False
    assert fires_high_for("B-PROBE-CONSISTENCY") is False
    assert fires_high_for("VESTIBULE-LZ") is True
    assert fires_high_for("PROBE-HD") is True
    assert fires_high_for("TRIPWIRE-H") is True
    with pytest.raises(KeyError, match="unknown primitive"):
        fires_high_for("UNKNOWN-GATE")


# ---------------------------------------------------------------------------
# Tie / atom / tiny-n behaviour and the conservative same-sample guarantee.
# Defect D-a: linear interpolation can
# land strictly between order statistics and OVERSHOOT the target.
# ---------------------------------------------------------------------------


def test_tiny_n_does_not_overshoot_target() -> None:
    """n=10, alpha=0.25: linear interpolation realises 0.30. The
    discontinuous order-statistic method must not exceed the target."""
    scores = np.arange(10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo = calibrate_threshold(scores, fpr_target=0.25, fires_high=False)
        hi = calibrate_threshold(scores, fpr_target=0.25, fires_high=True)
    assert empirical_fpr(scores, lo, fires_high=False) <= 0.25
    assert empirical_fpr(scores, hi, fires_high=True) <= 0.25


def test_tied_scores_do_not_overshoot_target() -> None:
    scores = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo = calibrate_threshold(scores, fpr_target=0.25, fires_high=False)
        hi = calibrate_threshold(scores, fpr_target=0.25, fires_high=True)
    assert empirical_fpr(scores, lo, fires_high=False) <= 0.25
    assert empirical_fpr(scores, hi, fires_high=True) <= 0.25


def test_discrete_rounded_scores_stay_within_target() -> None:
    rng = np.random.default_rng(0)
    scores = np.round(rng.normal(0.0, 1.0, 20_000), 1)
    for fires in (True, False):
        tau = calibrate_threshold(scores, fpr_target=0.05, fires_high=fires)
        assert empirical_fpr(scores, tau, fires_high=fires) <= 0.05


def test_large_atom_makes_target_unattainable_but_stays_conservative() -> None:
    """A dominant atom can make the target structurally unreachable by any
    deterministic threshold. The requirement is that we UNDERSHOOT, never
    overshoot — silently exceeding the FPR budget is the dangerous direction."""
    scores = np.concatenate([np.zeros(16_000), np.ones(4_000)])
    for fires in (True, False):
        tau = calibrate_threshold(scores, fpr_target=0.05, fires_high=fires)
        assert empirical_fpr(scores, tau, fires_high=fires) <= 0.05


@pytest.mark.parametrize("fpr_target", [0.01, 0.05, 0.10, 0.25])
@pytest.mark.parametrize("fires_high", [True, False])
def test_same_sample_fpr_never_exceeds_target(fpr_target: float, fires_high: bool) -> None:
    rng = np.random.default_rng(0)
    for scores in (
        rng.normal(0.0, 1.0, 5000),
        rng.standard_t(2, 5000),
        np.round(rng.normal(0.0, 1.0, 5000), 1),
    ):
        tau = calibrate_threshold(scores, fpr_target=fpr_target, fires_high=fires_high)
        assert empirical_fpr(scores, tau, fires_high=fires_high) <= fpr_target + 1e-12


def test_held_out_fpr_is_close_but_not_guaranteed() -> None:
    """The conservative bound is SAME-SAMPLE only. On held-out data the rate
    fluctuates around the target; we assert only that it is in a sane band."""
    rng = np.random.default_rng(0)
    calib = rng.normal(0.0, 1.0, 20_000)
    held_out = rng.normal(0.0, 1.0, 20_000)
    tau = calibrate_threshold(calib, fpr_target=0.05, fires_high=False)
    assert 0.03 < empirical_fpr(held_out, tau, fires_high=False) < 0.07


def test_build_table_rejects_unregistered_primitive() -> None:
    """Fail-closed: an unregistered primitive must not silently inherit the
    fires-HIGH default, which would reintroduce defect D0."""
    scores = np.arange(100.0, dtype=np.float64)
    with pytest.raises(KeyError, match="unknown primitive"):
        build_calibration_table("MYSTERY-GATE", {QuantScheme.FP16: scores})


def test_build_table_allows_explicit_override_for_unregistered() -> None:
    scores = np.arange(100.0, dtype=np.float64)
    table = build_calibration_table(
        "MYSTERY-GATE", {QuantScheme.FP16: scores}, fires_high=False
    )
    assert table.tau(QuantScheme.FP16) == pytest.approx(
        calibrate_threshold(scores, fires_high=False)
    )


# ---------------------------------------------------------------------------
# conservative_threshold — population guarantee, not just same-sample
# ---------------------------------------------------------------------------

from cliffguard.eval.threshold_calibrator import conservative_threshold  # noqa: E402


def test_conservative_is_stricter_than_point_calibrator() -> None:
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 1.0, 400)
    point = calibrate_threshold(benign, 0.05, fires_high=True)
    cons = conservative_threshold(benign, 0.05, fires_high=True)
    assert cons >= point          # higher bar for a fires-HIGH gate
    assert empirical_fpr(benign, cons, fires_high=True) <= 0.05


def test_conservative_fires_low_direction() -> None:
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 1.0, 400)
    point = calibrate_threshold(benign, 0.05, fires_high=False)
    cons = conservative_threshold(benign, 0.05, fires_high=False)
    assert cons <= point          # stricter means LOWER for a fires-LOW gate
    assert empirical_fpr(benign, cons, fires_high=False) <= 0.05


def test_conservative_achieves_held_out_coverage() -> None:
    """The property that motivates this function: the point calibrator exceeds
    the FPR target on ~45% of held-out draws; the conservative one must be far
    below that."""
    rng = np.random.default_rng(1)
    exceed_point = exceed_cons = 0
    trials = 60
    for _ in range(trials):
        cal = rng.normal(0.0, 1.0, 400)
        held = rng.normal(0.0, 1.0, 4000)
        tp = calibrate_threshold(cal, 0.05, fires_high=True)
        tc = conservative_threshold(cal, 0.05, fires_high=True)
        exceed_point += empirical_fpr(held, tp, fires_high=True) > 0.05
        exceed_cons += empirical_fpr(held, tc, fires_high=True) > 0.05
    assert exceed_cons < exceed_point
    assert exceed_cons / trials < 0.15


def test_conservative_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        conservative_threshold(np.arange(100.0), 0.05, confidence=1.0)


def test_conservative_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        conservative_threshold(np.array([]), 0.05)
