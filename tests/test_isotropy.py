import numpy as np
import pytest

from cliffguard.eval.isotropy import (
    concentration,
    isotropy_test,
    parallel_orthogonal_split,
    random_direction_distance,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# concentration
# ---------------------------------------------------------------------------


def test_concentration_is_one_for_full_fraction() -> None:
    assert concentration(np.array([1.0, 2.0, 3.0]), 1.0) == pytest.approx(1.0)


def test_concentration_detects_a_spike() -> None:
    """One dominant coordinate out of 100 must carry nearly all the energy."""
    d = np.zeros(100)
    d[0] = 1.0
    d[1:] = 1e-6
    assert concentration(d, 0.01) > 0.99


def test_concentration_is_low_for_uniform_energy() -> None:
    d = np.ones(1000)
    assert concentration(d, 0.01) == pytest.approx(0.01, abs=0.005)


def test_concentration_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        concentration(np.ones(10), 0.0)


def test_concentration_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="identically zero"):
        concentration(np.zeros(10), 0.1)


# ---------------------------------------------------------------------------
# parallel / orthogonal split
# ---------------------------------------------------------------------------


def test_split_pure_parallel() -> None:
    ref = np.array([1.0, 0.0, 0.0])
    par, orth = parallel_orthogonal_split(np.array([0.5, 0.0, 0.0]), ref)
    assert par == pytest.approx(0.5)
    assert orth == pytest.approx(0.0, abs=1e-12)


def test_split_pure_orthogonal() -> None:
    ref = np.array([1.0, 0.0, 0.0])
    par, orth = parallel_orthogonal_split(np.array([0.0, 0.3, 0.4]), ref)
    assert par == pytest.approx(0.0, abs=1e-12)
    assert orth == pytest.approx(0.5)


def test_split_preserves_total_energy() -> None:
    rng = np.random.default_rng(0)
    delta = rng.normal(size=50)
    ref = rng.normal(size=50)
    par, orth = parallel_orthogonal_split(delta, ref)
    assert par**2 + orth**2 == pytest.approx(float(delta @ delta))


def test_split_sign_is_meaningful() -> None:
    ref = np.array([1.0, 0.0])
    par, _ = parallel_orthogonal_split(np.array([-0.4, 0.0]), ref)
    assert par < 0.0


# ---------------------------------------------------------------------------
# isotropy_test
# ---------------------------------------------------------------------------


def test_isotropic_perturbation_is_not_flagged() -> None:
    """A genuinely isotropic perturbation must produce small |z| and an
    ISOTROPIC verdict — this is the null-calibration check."""
    rng = np.random.default_rng(0)
    d = 512
    a = _unit(rng.normal(size=d))
    noise = _unit(rng.normal(size=d)) * 0.2
    b = _unit(a + noise)
    res = isotropy_test(a, b, n_null=200, seed=1)
    assert res.is_isotropic()
    assert res.concentration_null_not_rejected()
    assert not res.rejects_concentration_null()
    assert res.max_abs_z < 3.0


def test_targeted_perturbation_is_flagged_anisotropic() -> None:
    """Damage concentrated on a handful of coordinates must be detected."""
    rng = np.random.default_rng(0)
    d = 512
    a = _unit(rng.normal(size=d))
    noise = np.zeros(d)
    noise[:5] = 0.1          # all the damage on 5 of 512 channels
    b = _unit(a + noise)
    res = isotropy_test(a, b, n_null=200, seed=1)
    assert not res.is_isotropic()
    assert res.rejects_concentration_null()
    assert res.concentration_z(0.01) > 3.0


def test_fail_to_reject_does_not_identify_a_dense_target() -> None:
    """A dense Gaussian-looking target cannot be identified from one draw."""
    rng = np.random.default_rng(71)
    d = 512
    a = _unit(rng.normal(size=d))
    fixed_dense_target = _unit(rng.normal(size=d)) * 0.2
    b = _unit(a + fixed_dense_target)
    res = isotropy_test(a, b, n_null=300, seed=72)
    assert res.concentration_null_not_rejected()


def test_isotropy_test_reproduces_repo_artifact_values() -> None:
    """Regression against the real Fold A artifacts is covered in the notebook;
    here we pin the synthetic equivalent so the statistic cannot silently drift."""
    rng = np.random.default_rng(42)
    d = 3072
    a = _unit(rng.normal(size=d))
    b = _unit(a + _unit(rng.normal(size=d)) * 0.2361)
    res = isotropy_test(a, b, n_null=100, seed=0)
    # chord 0.2361 on unit vectors => ~13.6 degrees
    assert res.angle_deg == pytest.approx(13.6, abs=0.6)
    assert res.is_isotropic()


def test_irrecoverable_fraction_dominant_for_orthogonal_damage() -> None:
    rng = np.random.default_rng(0)
    d = 256
    a = _unit(rng.normal(size=d))
    noise = _unit(rng.normal(size=d)) * 0.3
    b = _unit(a + noise)
    res = isotropy_test(a, b, n_null=50, seed=0)
    # renormalisation makes the perturbation almost purely orthogonal
    assert res.irrecoverable_fraction > 0.9


def test_isotropy_test_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        isotropy_test(np.ones(10), np.ones(11))


def test_isotropy_test_raises_on_identical_directions() -> None:
    a = _unit(np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="identical"):
        isotropy_test(a, a.copy())


def test_isotropy_test_raises_for_tiny_null() -> None:
    a = _unit(np.arange(1.0, 11.0))
    b = _unit(np.arange(1.0, 11.0) + 0.1)
    with pytest.raises(ValueError, match="n_null"):
        isotropy_test(a, b, n_null=1)


def test_isotropy_test_is_deterministic_under_seed() -> None:
    rng = np.random.default_rng(0)
    a = _unit(rng.normal(size=64))
    b = _unit(a + _unit(rng.normal(size=64)) * 0.2)
    r1 = isotropy_test(a, b, n_null=50, seed=9)
    r2 = isotropy_test(a, b, n_null=50, seed=9)
    assert r1.concentration_z(0.05) == pytest.approx(r2.concentration_z(0.05))


def test_summary_reports_null_status_without_positive_isotropy_verdict() -> None:
    rng = np.random.default_rng(0)
    a = _unit(rng.normal(size=128))
    b = _unit(a + _unit(rng.normal(size=128)) * 0.2)
    text = isotropy_test(a, b, n_null=50, seed=0).summary()
    assert "concentration null:" in text
    assert "not positive evidence of isotropy" in text
    assert "verdict: ISOTROPIC" not in text


def test_rejection_threshold_must_be_positive_and_finite() -> None:
    rng = np.random.default_rng(0)
    a = _unit(rng.normal(size=64))
    b = _unit(a + _unit(rng.normal(size=64)) * 0.2)
    result = isotropy_test(a, b, n_null=50, seed=0)
    for threshold in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="z_threshold"):
            result.rejects_concentration_null(threshold)


# ---------------------------------------------------------------------------
# random-direction ceiling
# ---------------------------------------------------------------------------


def test_random_direction_distance_approaches_sqrt2() -> None:
    """Independent unit vectors in high dimension are nearly orthogonal, so the
    chord distance saturates at sqrt(2). This is the ceiling any observed
    cross-scheme distance must be read against."""
    assert random_direction_distance(3072, n=300, seed=0) == pytest.approx(
        np.sqrt(2.0), abs=0.02
    )


def test_random_direction_distance_rejects_tiny_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        random_direction_distance(1)


def test_irrecoverable_fraction_is_exactly_cos_squared_half_angle() -> None:
    """It is a function of the angle alone, and nothing else.

    Pinned because a results write-up once reported its decline across a
    quantization ladder as a counterintuitive empirical finding. It is
    trigonometry: any trend in it is the angle trend restated.
    """
    import math

    rng = np.random.default_rng(0)
    for _ in range(12):
        a = rng.normal(size=64)
        b = rng.normal(size=64)
        res = isotropy_test(a, b, n_null=20, seed=0)
        expected = math.cos(math.radians(res.angle_deg) / 2.0) ** 2
        assert res.irrecoverable_fraction == pytest.approx(expected, abs=1e-9)
