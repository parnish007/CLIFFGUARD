import math

import numpy as np
import pytest

from cliffguard.eval.noise_floor import (
    NoiseFloorResult,
    angle_between,
    chord_distance,
    difference_in_means,
    split_half_noise_floor,
    theoretical_floor_deg,
    unit,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_unit_normalises() -> None:
    v = np.array([3.0, 4.0])
    assert np.linalg.norm(unit(v)) == pytest.approx(1.0)


def test_unit_raises_on_zero_norm() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        unit(np.zeros(5))


def test_angle_between_orthogonal_is_90() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert angle_between(a, b) == pytest.approx(90.0)


def test_angle_between_identical_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert angle_between(a, a) == pytest.approx(0.0, abs=1e-9)


def test_angle_between_antipodal_is_180() -> None:
    a = np.array([1.0, 0.0])
    assert angle_between(a, -a) == pytest.approx(180.0)


def test_chord_distance_orthogonal_is_sqrt2() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert chord_distance(a, b) == pytest.approx(math.sqrt(2.0))


# ---------------------------------------------------------------------------
# difference_in_means
# ---------------------------------------------------------------------------


def test_difference_in_means_recovers_known_direction() -> None:
    harmful = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    harmless = np.tile(np.array([0.0, 0.0, 0.0]), (10, 1))
    d = difference_in_means(harmful, harmless)
    assert d == pytest.approx(np.array([1.0, 0.0, 0.0]))


def test_difference_in_means_is_unit_norm() -> None:
    rng = np.random.default_rng(0)
    d = difference_in_means(rng.normal(size=(20, 8)) + 1.0, rng.normal(size=(20, 8)))
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_difference_in_means_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        difference_in_means(np.zeros((0, 4)), np.ones((5, 4)))


def test_difference_in_means_raises_on_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        difference_in_means(np.ones((5, 4)), np.ones((5, 6)))


def test_difference_in_means_raises_on_1d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        difference_in_means(np.ones(4), np.ones(4))


# ---------------------------------------------------------------------------
# split_half_noise_floor
# ---------------------------------------------------------------------------


def test_split_half_raises_for_tiny_classes() -> None:
    with pytest.raises(ValueError, match=">= 4 samples"):
        split_half_noise_floor(np.ones((3, 5)), np.ones((10, 5)))


def test_split_half_raises_for_zero_splits() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_splits"):
        split_half_noise_floor(
            rng.normal(size=(20, 5)), rng.normal(size=(20, 5)), n_splits=0
        )


def test_split_half_is_deterministic_under_seed() -> None:
    rng = np.random.default_rng(0)
    h = rng.normal(size=(40, 16)) + 0.5
    benign = rng.normal(size=(40, 16))
    a = split_half_noise_floor(h, benign, n_splits=10, seed=7)
    b = split_half_noise_floor(h, benign, n_splits=10, seed=7)
    assert a.angles_deg == b.angles_deg


def test_split_half_noise_grows_as_snr_falls() -> None:
    """Lower per-class SNR must produce a LARGER estimator noise floor.

    This is the property the Stage 0 gate depends on: the whole point is that a
    weak signal in high dimension produces large apparent rotations for free."""
    rng = np.random.default_rng(0)
    d = 256
    n = 200
    floors = []
    for snr in (8.0, 2.0, 0.5):
        mu = np.zeros(d)
        mu[0] = snr
        h = rng.normal(size=(n, d)) + mu
        benign = rng.normal(size=(n, d))
        floors.append(
            split_half_noise_floor(h, benign, n_splits=15, seed=1).median_deg
        )
    assert floors[0] < floors[1] < floors[2]


def test_split_half_noise_shrinks_with_more_samples() -> None:
    rng = np.random.default_rng(0)
    d = 128
    mu = np.zeros(d)
    mu[0] = 2.0
    small = split_half_noise_floor(
        rng.normal(size=(40, d)) + mu, rng.normal(size=(40, d)), n_splits=15, seed=2
    )
    big = split_half_noise_floor(
        rng.normal(size=(400, d)) + mu, rng.normal(size=(400, d)), n_splits=15, seed=2
    )
    assert big.median_deg < small.median_deg


def test_correction_reduces_the_angle() -> None:
    """The full-n corrected floor must be strictly below the raw split-half
    floor, since the split-half estimate uses half the data."""
    rng = np.random.default_rng(0)
    d = 64
    mu = np.zeros(d)
    mu[0] = 3.0
    res = split_half_noise_floor(
        rng.normal(size=(100, d)) + mu, rng.normal(size=(100, d)), n_splits=20, seed=3
    )
    assert res.corrected_median_deg < res.median_deg


def test_exceeds_floor_true_for_large_observed_angle() -> None:
    res = NoiseFloorResult(angles_deg=[5.0] * 50, n_per_class_used=100, n_splits=50, dimension=64)
    assert res.exceeds_floor(60.0) is True


def test_exceeds_floor_false_for_angle_inside_floor() -> None:
    res = NoiseFloorResult(angles_deg=[20.0] * 50, n_per_class_used=100, n_splits=50, dimension=64)
    assert res.exceeds_floor(3.0) is False


def test_quantile_rejects_out_of_range() -> None:
    res = NoiseFloorResult(angles_deg=[1.0, 2.0], n_per_class_used=2, n_splits=2, dimension=4)
    with pytest.raises(ValueError, match="q must be"):
        res.quantile_deg(1.5)


def test_summary_mentions_split_count() -> None:
    res = NoiseFloorResult(angles_deg=[3.0] * 8, n_per_class_used=50, n_splits=8, dimension=32)
    assert "8 splits" in res.summary()


# ---------------------------------------------------------------------------
# theoretical_floor_deg
# ---------------------------------------------------------------------------


def test_theoretical_floor_decreases_with_snr() -> None:
    lo = theoretical_floor_deg(3072, 200, snr=50.0)
    hi = theoretical_floor_deg(3072, 200, snr=20.0)
    assert lo < hi


def test_theoretical_floor_saturates_at_90() -> None:
    assert theoretical_floor_deg(3072, 10, snr=0.01) == pytest.approx(90.0)


def test_theoretical_floor_rejects_bad_snr() -> None:
    with pytest.raises(ValueError, match="snr"):
        theoretical_floor_deg(100, 10, snr=0.0)


def test_theoretical_floor_matches_empirical_order_of_magnitude() -> None:
    """Sanity cross-check: the closed form and the empirical split-half
    estimate should agree to within a factor of ~2 under the isotropic
    Gaussian model the formula assumes."""
    rng = np.random.default_rng(0)
    d, n, snr = 256, 200, 10.0
    mu = np.zeros(d)
    mu[0] = snr
    empirical = split_half_noise_floor(
        rng.normal(size=(n, d)) + mu, rng.normal(size=(n, d)), n_splits=25, seed=5
    ).median_deg
    predicted = theoretical_floor_deg(d, n // 2, snr)
    assert 0.5 < empirical / predicted < 2.0


# ---------------------------------------------------------------------------
# Paired bootstrap — the correct control for the cross-scheme statistic.
# A within-FP16 split-half floor
# discards the covariance induced by scoring the SAME prompts under both
# schemes, so no fixed rescaling of it is the right comparison.
# ---------------------------------------------------------------------------

from cliffguard.eval.noise_floor import PairedShiftResult, paired_direction_shift  # noqa: E402


def _paired_data(shift: float, n: int = 200, d: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu = np.zeros(d)
    mu[0] = 3.0
    hf = rng.normal(size=(n, d)) + mu
    lf = rng.normal(size=(n, d))
    # quantized = same prompts, plus a deterministic shift along another axis
    perturb = np.zeros(d)
    perturb[1] = shift
    return hf, lf, hf + perturb, lf


def test_bca_can_also_miss_point_estimate_for_measured_style_case() -> None:
    """Neither interval is forced to bracket the point estimate.

    An angle is a non-negative distance statistic: bootstrap resampling only
    adds sampling noise, which can only increase it, so the bootstrap
    distribution is upward-biased and a percentile interval can sit slightly
    off the full-sample value (measured: observed 18.33, CI [16.44, 18.29]).
    On this fixture BCa corrects past the point estimate and also fails to
    bracket it. This negative methodological result must remain visible."""
    hf, lf, hq, lq = _paired_data(shift=1.0)
    r = paired_direction_shift(hf, lf, hq, lq, n_bootstrap=400, seed=0)
    width = r.ci_high_deg - r.ci_low_deg
    assert width < 0.5 * r.observed_angle_deg
    assert abs(r.observed_angle_deg - 0.5 * (r.ci_low_deg + r.ci_high_deg)) < width
    assert r.ci_high_deg < r.observed_angle_deg
    assert r.bca_ci_low_deg > r.observed_angle_deg


def test_bca_brackets_point_when_percentile_interval_does_not() -> None:
    rng = np.random.default_rng(9)
    n, d = 8, 8
    mu = np.zeros(d)
    mu[0] = 2.0
    harmful_fp16 = rng.normal(size=(n, d)) + mu
    harmless_fp16 = rng.normal(size=(n, d))
    perturbation = np.zeros(d)
    perturbation[1] = 1.0

    result = paired_direction_shift(
        harmful_fp16,
        harmless_fp16,
        harmful_fp16 + perturbation,
        harmless_fp16,
        n_bootstrap=30,
        seed=4,
    )

    assert result.ci_high_deg < result.observed_angle_deg
    assert result.bca_ci_low_deg <= result.observed_angle_deg
    assert result.observed_angle_deg <= result.bca_ci_high_deg


def test_stage0_excludes_zero_uses_bca_interval() -> None:
    result = PairedShiftResult(
        observed_angle_deg=1.0,
        ci_low_deg=0.0,
        ci_high_deg=2.0,
        bca_ci_low_deg=0.1,
        bca_ci_high_deg=1.9,
        n_bootstrap=100,
        n_per_class=50,
    )
    assert result.percentile_excludes_zero is False
    assert result.excludes_zero is True


def test_paired_shift_detects_a_real_rotation() -> None:
    hf, lf, hq, lq = _paired_data(shift=2.0)
    r = paired_direction_shift(hf, lf, hq, lq, n_bootstrap=200, seed=0)
    assert r.observed_angle_deg > 5.0
    assert r.excludes_zero


def test_paired_shift_is_near_zero_without_quantization() -> None:
    """Identical inputs under both schemes must give ~0 rotation — the paired
    design cancels the shared sampling noise, which is exactly the point."""
    hf, lf, _, _ = _paired_data(shift=0.0)
    r = paired_direction_shift(hf, lf, hf.copy(), lf.copy(), n_bootstrap=100, seed=0)
    assert r.observed_angle_deg == pytest.approx(0.0, abs=1e-6)


def test_paired_shift_rejects_misaligned_shapes() -> None:
    hf, lf, _, _ = _paired_data(shift=1.0)
    with pytest.raises(ValueError, match="row-aligned"):
        paired_direction_shift(hf, lf, hf[:10], lf, n_bootstrap=10)


def test_paired_shift_rejects_bad_alpha() -> None:
    hf, lf, hq, lq = _paired_data(shift=1.0)
    with pytest.raises(ValueError, match="alpha"):
        paired_direction_shift(hf, lf, hq, lq, n_bootstrap=10, alpha=0.0)


def test_paired_ci_is_tighter_than_unpaired_floor_scaling() -> None:
    """The paired design should not simply reproduce the split-half floor —
    it preserves cross-scheme covariance, so for a shared-noise scenario the
    paired interval is much tighter than the within-FP16 floor suggests."""
    hf, lf, hq, lq = _paired_data(shift=0.0)
    paired = paired_direction_shift(hf, lf, hq, lq, n_bootstrap=100, seed=0)
    floor = split_half_noise_floor(hf, lf, n_splits=20, seed=0)
    assert paired.ci_high_deg < floor.median_deg


def test_two_correction_factors_differ() -> None:
    """corrected_median_deg (two-full-N, /sqrt2) must exceed
    corrected_vs_population_deg (/2)."""
    rng = np.random.default_rng(0)
    d = 64
    mu = np.zeros(d)
    mu[0] = 3.0
    res = split_half_noise_floor(
        rng.normal(size=(100, d)) + mu, rng.normal(size=(100, d)), n_splits=20, seed=1
    )
    assert res.corrected_median_deg > res.corrected_vs_population_deg
    assert res.corrected_vs_population_deg < res.median_deg


# ---------------------------------------------------------------------------
# Stage 0 gate — rotation replication.
#
# This replaced TWO earlier designs that were both wrong:
#   1. exceeds_floor (split-half): wrong sample size; ignores that FP16 and the
#      quantized model score the SAME prompts, so errors are correlated.
#   2. excludes_zero (paired CI vs 0): unfalsifiable. An angle is non-negative,
#      so its CI essentially never contains zero — it fired on 40/40 nulls.
# The tests below exist so neither is reinstated.
# ---------------------------------------------------------------------------

from cliffguard.eval.noise_floor import rotation_replication  # noqa: E402


def _rot_data(shift: float, n: int = 200, d: int = 128, seed: int = 0, noise: float = 0.5):
    rng = np.random.default_rng(seed)
    mu = np.zeros(d)
    mu[0] = 3.0
    hf = rng.normal(size=(n, d)) + mu
    lf = rng.normal(size=(n, d))
    hq = hf + rng.normal(scale=noise, size=hf.shape)
    lq = lf + rng.normal(scale=noise, size=lf.shape)
    if shift:
        p = np.zeros(d)
        p[1] = shift
        hq = hq + p
    return hf, lf, hq, lq


def test_replication_does_not_fire_on_pure_noise() -> None:
    """Type I error control: a rotation made of per-prompt exchangeable noise
    must NOT replicate across disjoint prompt halves."""
    fires = 0
    for t in range(20):
        r = rotation_replication(*_rot_data(0.0, seed=4000 + t), n_splits=30, seed=t)
        fires += int(r.passes())
    assert fires <= 1     # ~0 expected at z>3


def test_replication_fires_on_a_systematic_rotation() -> None:
    r = rotation_replication(*_rot_data(1.0, seed=5000), n_splits=40, seed=0)
    assert r.passes()
    assert r.median_cosine > 0.3


def test_replication_power_increases_with_signal() -> None:
    zs = [
        rotation_replication(*_rot_data(s, seed=6000), n_splits=40, seed=0).z_score
        for s in (0.0, 0.25, 0.5, 1.0)
    ]
    assert zs == sorted(zs)


def test_replication_null_sd_is_one_over_sqrt_d() -> None:
    r = rotation_replication(*_rot_data(0.5, d=256, seed=7000), n_splits=10, seed=0)
    assert r.null_sd == pytest.approx(1.0 / math.sqrt(256))


def test_replication_rejects_misaligned_rows() -> None:
    hf, lf, hq, lq = _rot_data(0.5)
    with pytest.raises(ValueError, match="row-aligned"):
        rotation_replication(hf, lf, hq[:10], lq, n_splits=5)


def test_replication_rejects_too_few_prompts() -> None:
    hf, lf, hq, lq = _rot_data(0.5, n=3, d=16)
    with pytest.raises(ValueError, match=">= 4 prompts"):
        rotation_replication(hf, lf, hq, lq, n_splits=5)


def test_replication_is_deterministic_under_seed() -> None:
    args = _rot_data(0.5, seed=8000)
    a = rotation_replication(*args, n_splits=15, seed=11)
    b = rotation_replication(*args, n_splits=15, seed=11)
    assert a.cosines == b.cosines


def test_replication_summary_states_a_verdict() -> None:
    s = rotation_replication(*_rot_data(1.0, seed=9000), n_splits=10, seed=0).summary()
    assert "Stage 0" in s and ("PASS" in s or "NULL" in s)
