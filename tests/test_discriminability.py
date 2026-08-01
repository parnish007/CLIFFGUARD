import math

import numpy as np
import pytest

from cliffguard.eval.discriminability import (
    d_prime,
    d_prime_with_ci,
    empirical_auc,
    gaussianity_gap,
    implied_eta,
    predict_d_prime,
)


def _two_classes(sep: float, n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(sep, 1.0, n), rng.normal(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# d_prime
# ---------------------------------------------------------------------------


def test_d_prime_recovers_known_separation() -> None:
    pos, neg = _two_classes(sep=2.0, n=20_000)
    assert d_prime(pos, neg) == pytest.approx(2.0, abs=0.08)


def test_d_prime_zero_for_identical_distributions() -> None:
    pos, neg = _two_classes(sep=0.0, n=20_000)
    assert d_prime(pos, neg) == pytest.approx(0.0, abs=0.05)


def test_d_prime_sign_flips_with_firing_direction() -> None:
    pos, neg = _two_classes(sep=2.0, n=5000)
    assert d_prime(pos, neg, fires_high=True) == pytest.approx(
        -d_prime(pos, neg, fires_high=False)
    )


def test_d_prime_positive_for_fires_low_when_pos_scores_lower() -> None:
    """PROBE-RM case: harmful prompts have LOWER refusal margin."""
    rng = np.random.default_rng(0)
    harmful = rng.normal(0.06, 0.02, 5000)
    benign = rng.normal(0.10, 0.02, 5000)
    assert d_prime(harmful, benign, fires_high=False) > 0.0


def test_d_prime_raises_for_single_sample() -> None:
    with pytest.raises(ValueError, match=">= 2 samples"):
        d_prime(np.array([1.0]), np.arange(10.0))


def test_d_prime_raises_for_zero_variance() -> None:
    const = np.ones(10)
    with pytest.raises(ValueError, match="undefined"):
        d_prime(const, const)


def test_d_prime_is_scale_invariant() -> None:
    """d' is invariant under a common affine rescaling of both classes —
    the ROC-invariance property that makes recalibration unable to restore it."""
    pos, neg = _two_classes(sep=1.5, n=4000)
    base = d_prime(pos, neg)
    scaled = d_prime(pos * 7.0 + 3.0, neg * 7.0 + 3.0)
    assert scaled == pytest.approx(base, rel=1e-9)


# ---------------------------------------------------------------------------
# bootstrap CI
# ---------------------------------------------------------------------------


def test_ci_brackets_the_point_estimate() -> None:
    pos, neg = _two_classes(sep=1.5, n=800)
    r = d_prime_with_ci(pos, neg, n_bootstrap=400)
    assert r.ci_low < r.d_prime < r.ci_high


def test_ci_narrows_with_more_data() -> None:
    small = d_prime_with_ci(*_two_classes(1.5, n=200, seed=1), n_bootstrap=400)
    big = d_prime_with_ci(*_two_classes(1.5, n=5000, seed=1), n_bootstrap=400)
    assert (big.ci_high - big.ci_low) < (small.ci_high - small.ci_low)


def test_ci_is_deterministic_under_seed() -> None:
    pos, neg = _two_classes(1.0, n=500)
    a = d_prime_with_ci(pos, neg, n_bootstrap=200, seed=3)
    b = d_prime_with_ci(pos, neg, n_bootstrap=200, seed=3)
    assert a.ci_low == pytest.approx(b.ci_low)


def test_ci_rejects_bad_alpha() -> None:
    pos, neg = _two_classes(1.0, n=100)
    with pytest.raises(ValueError, match="alpha"):
        d_prime_with_ci(pos, neg, alpha=1.5)


def test_auc_property_matches_gaussian_model() -> None:
    pos, neg = _two_classes(sep=2.0, n=10_000)
    r = d_prime_with_ci(pos, neg, n_bootstrap=100)
    assert r.auc == pytest.approx(empirical_auc(pos, neg), abs=0.02)


def test_tpr_at_fpr_increases_with_d_prime() -> None:
    weak = d_prime_with_ci(*_two_classes(0.5, n=3000), n_bootstrap=100)
    strong = d_prime_with_ci(*_two_classes(3.0, n=3000), n_bootstrap=100)
    assert strong.tpr_at_fpr(0.05) > weak.tpr_at_fpr(0.05)


def test_tpr_at_fpr_rejects_out_of_range() -> None:
    r = d_prime_with_ci(*_two_classes(1.0, n=200), n_bootstrap=50)
    with pytest.raises(ValueError, match="fpr"):
        r.tpr_at_fpr(0.0)


def test_summary_contains_d_prime() -> None:
    r = d_prime_with_ci(*_two_classes(1.0, n=300), n_bootstrap=50)
    assert "d' =" in r.summary()


# ---------------------------------------------------------------------------
# empirical AUC / gaussianity
# ---------------------------------------------------------------------------


def test_empirical_auc_is_half_for_identical() -> None:
    pos, neg = _two_classes(0.0, n=3000)
    assert empirical_auc(pos, neg) == pytest.approx(0.5, abs=0.03)


def test_empirical_auc_is_one_for_separated() -> None:
    assert empirical_auc(np.arange(10.0) + 100.0, np.arange(10.0)) == pytest.approx(1.0)


def test_empirical_auc_handles_ties_as_half() -> None:
    same = np.ones(4)
    assert empirical_auc(same, same) == pytest.approx(0.5)


def test_empirical_auc_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        empirical_auc(np.array([]), np.arange(3.0))


def test_gaussianity_gap_small_for_gaussian_data() -> None:
    pos, neg = _two_classes(sep=1.5, n=20_000)
    assert gaussianity_gap(pos, neg) < 0.05


def test_gaussianity_gap_large_for_heavy_tailed_data() -> None:
    """Heavy-tail contamination breaks the equal-variance Gaussian model and
    the gap statistic must surface it.

    This is the failure mode that matters for the project: a few extreme
    outliers inflate the pooled variance, collapsing d' (0.25 -> model AUC
    0.57) while the ranks still separate well (empirical AUC 0.79). If real
    margin distributions look like this, the closed-form TPR predictions of
    the discriminability-decay theorem are not trustworthy and must not be
    used without reporting this gap."""
    rng = np.random.default_rng(0)
    neg = rng.normal(0.0, 1.0, 8000)
    pos = np.concatenate([rng.normal(1.2, 1.0, 7600), rng.normal(0.0, 30.0, 400)])
    assert gaussianity_gap(pos, neg) > 0.1


def test_gaussianity_gap_is_blind_to_unequal_gaussian_variances() -> None:
    """The gap cannot validate the equal-variance part of assumption A2.

    For two Gaussian classes, empirical AUC and Phi(d'/sqrt(2)) both depend
    on the sum of class variances, so a severe variance mismatch can still
    produce a tiny gap.
    """
    rng = np.random.default_rng(17)
    neg = rng.normal(0.0, 1.0, 3_000)
    pos = rng.normal(1.2, 3.0, 3_000)
    assert gaussianity_gap(pos, neg) < 0.02


# ---------------------------------------------------------------------------
# the decay law
# ---------------------------------------------------------------------------


def test_predict_d_prime_matches_closed_form() -> None:
    assert predict_d_prime(2.5, 1.0) == pytest.approx(2.5 / math.sqrt(2.0))


def test_predict_d_prime_is_identity_at_zero_eta() -> None:
    assert predict_d_prime(2.5, 0.0) == pytest.approx(2.5)


def test_predict_d_prime_is_monotone_decreasing_in_eta() -> None:
    vals = [predict_d_prime(2.5, e) for e in (0.0, 0.5, 1.0, 4.0)]
    assert vals == sorted(vals, reverse=True)


def test_predict_d_prime_rejects_negative_eta() -> None:
    with pytest.raises(ValueError, match="eta"):
        predict_d_prime(2.0, -0.1)


def test_implied_eta_inverts_predict_d_prime() -> None:
    """Round-trip: eta -> d' -> eta must return the original value."""
    for eta in (0.1, 0.5, 1.2, 5.0):
        dq = predict_d_prime(2.5, eta)
        assert implied_eta(2.5, dq) == pytest.approx(eta, rel=1e-9)


def test_implied_eta_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="d_prime_q"):
        implied_eta(2.0, 0.0)


# ---------------------------------------------------------------------------
# held-out d' — fitting the direction on the scored prompts is biased upward
# ---------------------------------------------------------------------------

from cliffguard.eval.discriminability import held_out_d_prime  # noqa: E402


def _class_acts(sep: float, n: int = 200, d: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu = np.zeros(d)
    mu[0] = sep
    return rng.normal(size=(n, d)) + mu, rng.normal(size=(n, d))


def test_held_out_is_below_in_sample() -> None:
    """The core reason this function exists: in-sample d' is optimistic."""
    pos, neg = _class_acts(sep=1.0, n=200, d=128)
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    direction = direction / np.linalg.norm(direction)
    mp = (pos @ direction) / np.linalg.norm(pos, axis=1)
    mn = (neg @ direction) / np.linalg.norm(neg, axis=1)
    in_sample = d_prime(mp, mn, fires_high=True)
    held_out, _ = held_out_d_prime(pos, neg, n_splits=20, seed=0)
    assert held_out < in_sample


def test_held_out_increases_with_separation() -> None:
    weak, _ = held_out_d_prime(*_class_acts(0.5, d=64), n_splits=15, seed=0)
    strong, _ = held_out_d_prime(*_class_acts(3.0, d=64), n_splits=15, seed=0)
    assert strong > weak


def test_held_out_near_zero_for_identical_classes() -> None:
    mean, _ = held_out_d_prime(*_class_acts(0.0, n=300, d=64), n_splits=25, seed=0)
    assert abs(mean) < 0.25


def test_held_out_rejects_tiny_classes() -> None:
    with pytest.raises(ValueError, match=">= 4 samples"):
        held_out_d_prime(np.ones((3, 8)), np.ones((10, 8)))


def test_held_out_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        held_out_d_prime(np.ones((10, 8)), np.ones((10, 9)))


def test_held_out_is_deterministic_under_seed() -> None:
    pos, neg = _class_acts(1.0, d=32)
    a = held_out_d_prime(pos, neg, n_splits=10, seed=5)
    b = held_out_d_prime(pos, neg, n_splits=10, seed=5)
    assert a == pytest.approx(b)
