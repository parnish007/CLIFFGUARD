import math

import numpy as np
import pytest

from cliffguard.types import QuantScheme
import warnings

from cliffguard.eval.cliff_metrics import (
    behavioral_cliff,
    detect_cliff_boundary,
    detect_cliff_boundary_three_metric,
    geometric_cliff,
    wasserstein_cliff,
)

_SQRT2 = math.sqrt(2.0)
_DIM = 8


def _unit(v: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    return (v / np.linalg.norm(v)).astype(np.float64)


_R_A = _unit(np.ones(_DIM, dtype=np.float64))           # all-ones, normalised
_R_B = _unit(np.array([1.0, -1.0] + [0.0] * (_DIM - 2), dtype=np.float64))  # orthogonal to all-ones in 2-D sense
_R_OPP = -_R_A                                           # antipodal to _R_A


# ---------------------------------------------------------------------------
# geometric_cliff
# ---------------------------------------------------------------------------


def test_geometric_cliff_identical_returns_zero() -> None:
    assert geometric_cliff(_R_A, _R_A) == pytest.approx(0.0, abs=1e-12)


def test_geometric_cliff_antipodal_returns_sqrt2() -> None:
    # ||-r̂ - r̂|| = 2.0 → / sqrt(2) = sqrt(2) ≈ 1.414
    result = geometric_cliff(_R_OPP, _R_A)
    assert result == pytest.approx(_SQRT2, abs=1e-9)


def test_geometric_cliff_value_above_one_for_antipodal() -> None:
    assert geometric_cliff(_R_OPP, _R_A) > 1.0


def test_geometric_cliff_raises_for_shape_mismatch() -> None:
    r_small = _unit(np.ones(4, dtype=np.float64))
    with pytest.raises(ValueError, match="Shape mismatch"):
        geometric_cliff(r_small, _R_A)


def test_geometric_cliff_raises_for_non_unit_q() -> None:
    not_unit = np.ones(_DIM, dtype=np.float64)  # norm = sqrt(DIM) != 1
    with pytest.raises(ValueError, match="not unit norm"):
        geometric_cliff(not_unit, _R_A)


def test_geometric_cliff_raises_for_non_unit_fp16() -> None:
    not_unit = np.ones(_DIM, dtype=np.float64)
    with pytest.raises(ValueError, match="not unit norm"):
        geometric_cliff(_R_A, not_unit)


def test_geometric_cliff_orthogonal_vectors() -> None:
    # ||r̂_B - r̂_A|| where r̂_B ⊥ r̂_A → distance = sqrt(2) → / sqrt(2) = 1.0
    # Only exactly true in 2-D; in higher dims depends on construction.
    # Use explicit orthogonal unit vectors in 2-D:
    r1 = np.array([1.0, 0.0], dtype=np.float64)
    r2 = np.array([0.0, 1.0], dtype=np.float64)
    assert geometric_cliff(r1, r2) == pytest.approx(1.0, abs=1e-9)


def test_geometric_cliff_returns_float() -> None:
    assert isinstance(geometric_cliff(_R_A, _R_A), float)


def test_geometric_cliff_non_negative() -> None:
    assert geometric_cliff(_R_B, _R_A) >= 0.0


# ---------------------------------------------------------------------------
# behavioral_cliff
# ---------------------------------------------------------------------------


def test_behavioral_cliff_equal_asr_returns_zero() -> None:
    assert behavioral_cliff(0.3, 0.3) == pytest.approx(0.0)


def test_behavioral_cliff_positive_when_q_worse() -> None:
    assert behavioral_cliff(0.7, 0.3) == pytest.approx(0.4)


def test_behavioral_cliff_negative_when_q_safer() -> None:
    assert behavioral_cliff(0.1, 0.5) == pytest.approx(-0.4)


def test_behavioral_cliff_raises_for_asr_q_above_one() -> None:
    with pytest.raises(ValueError, match="asr_q"):
        behavioral_cliff(1.1, 0.5)


def test_behavioral_cliff_raises_for_asr_q_below_zero() -> None:
    with pytest.raises(ValueError, match="asr_q"):
        behavioral_cliff(-0.1, 0.5)


def test_behavioral_cliff_raises_for_asr_fp16_above_one() -> None:
    with pytest.raises(ValueError, match="asr_fp16"):
        behavioral_cliff(0.5, 1.1)


def test_behavioral_cliff_raises_for_asr_fp16_below_zero() -> None:
    with pytest.raises(ValueError, match="asr_fp16"):
        behavioral_cliff(0.5, -0.1)


def test_behavioral_cliff_boundary_values_valid() -> None:
    assert behavioral_cliff(0.0, 0.0) == pytest.approx(0.0)
    assert behavioral_cliff(1.0, 1.0) == pytest.approx(0.0)
    assert behavioral_cliff(1.0, 0.0) == pytest.approx(1.0)


def test_behavioral_cliff_returns_float() -> None:
    assert isinstance(behavioral_cliff(0.5, 0.3), float)


# ---------------------------------------------------------------------------
# detect_cliff_boundary
# ---------------------------------------------------------------------------


def test_detect_cliff_returns_none_when_no_scheme_exceeds_kappa() -> None:
    geo = {QuantScheme.GGUF_Q3_K_M: 0.1, QuantScheme.GGUF_Q4_K_M: 0.05}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.1, QuantScheme.GGUF_Q4_K_M: 0.05}
    assert detect_cliff_boundary(geo, beh) is None


def test_detect_cliff_returns_none_when_only_geometric_exceeds() -> None:
    geo = {QuantScheme.GGUF_Q3_K_M: 0.5}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.1}
    assert detect_cliff_boundary(geo, beh) is None


def test_detect_cliff_returns_none_when_only_behavioral_exceeds() -> None:
    geo = {QuantScheme.GGUF_Q3_K_M: 0.1}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.5}
    assert detect_cliff_boundary(geo, beh) is None


def test_detect_cliff_returns_first_qualifying_scheme() -> None:
    # Q4_K_M comes before Q3_K_M in BIT_WIDTH_ORDER; both exceed kappa.
    geo = {QuantScheme.GGUF_Q4_K_M: 0.3, QuantScheme.GGUF_Q3_K_M: 0.4}
    beh = {QuantScheme.GGUF_Q4_K_M: 0.3, QuantScheme.GGUF_Q3_K_M: 0.4}
    assert detect_cliff_boundary(geo, beh) == QuantScheme.GGUF_Q4_K_M


def test_detect_cliff_returns_q3km_when_only_that_exceeds() -> None:
    geo = {QuantScheme.GGUF_Q4_K_M: 0.1, QuantScheme.GGUF_Q3_K_M: 0.5}
    beh = {QuantScheme.GGUF_Q4_K_M: 0.1, QuantScheme.GGUF_Q3_K_M: 0.5}
    assert detect_cliff_boundary(geo, beh) == QuantScheme.GGUF_Q3_K_M


def test_detect_cliff_skips_fp16() -> None:
    # Even if FP16 exceeds kappa in both, it should never be returned.
    geo = {QuantScheme.FP16: 1.0, QuantScheme.GGUF_Q3_K_M: 0.5}
    beh = {QuantScheme.FP16: 1.0, QuantScheme.GGUF_Q3_K_M: 0.5}
    result = detect_cliff_boundary(geo, beh)
    assert result == QuantScheme.GGUF_Q3_K_M


def test_detect_cliff_returns_none_for_empty_dicts() -> None:
    assert detect_cliff_boundary({}, {}) is None


def test_detect_cliff_skips_schemes_absent_from_either_dict() -> None:
    # Q3_K_M only in geo, not in beh → must be skipped.
    geo = {QuantScheme.GGUF_Q3_K_M: 0.9}
    beh = {QuantScheme.GGUF_Q4_K_M: 0.9}
    assert detect_cliff_boundary(geo, beh) is None


def test_detect_cliff_custom_kappa() -> None:
    geo = {QuantScheme.GGUF_Q3_K_M: 0.1}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.1}
    # With kappa=0.05 the scheme qualifies; with default kappa=0.25 it doesn't.
    assert detect_cliff_boundary(geo, beh, kappa=0.05) == QuantScheme.GGUF_Q3_K_M
    assert detect_cliff_boundary(geo, beh, kappa=0.25) is None


# ---------------------------------------------------------------------------
# wasserstein_cliff
# ---------------------------------------------------------------------------


def test_wasserstein_cliff_identical_distributions_returns_zero() -> None:
    a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    assert wasserstein_cliff(a, a) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_cliff_different_distributions_positive() -> None:
    # One centered at 0.0, one at 1.0 — W_1 = 1.0
    a = np.zeros(100, dtype=np.float64)
    b = np.ones(100, dtype=np.float64)
    result = wasserstein_cliff(a, b)
    assert result > 0.0
    assert result == pytest.approx(1.0, abs=1e-9)


def test_wasserstein_cliff_returns_nonnegative() -> None:
    rng = np.random.default_rng(42)
    a = rng.standard_normal(50).astype(np.float64)
    b = rng.standard_normal(50).astype(np.float64) + 0.5
    assert wasserstein_cliff(a, b) >= 0.0


def test_wasserstein_cliff_raises_for_empty_q() -> None:
    with pytest.raises(ValueError):
        wasserstein_cliff(np.array([], dtype=np.float64), np.array([1.0, 2.0]))


def test_wasserstein_cliff_raises_for_empty_fp16() -> None:
    with pytest.raises(ValueError):
        wasserstein_cliff(np.array([1.0, 2.0]), np.array([], dtype=np.float64))


def test_wasserstein_cliff_warns_when_sizes_differ_by_more_than_2x() -> None:
    a = np.ones(10, dtype=np.float64)
    b = np.ones(21, dtype=np.float64) * 2.0  # ratio = 2.1 > 2
    with pytest.warns(UserWarning):
        wasserstein_cliff(a, b)


def test_wasserstein_cliff_no_warning_when_sizes_within_2x() -> None:
    a = np.ones(10, dtype=np.float64)
    b = np.ones(20, dtype=np.float64)  # ratio = 2.0, NOT > 2x (boundary: exactly 2x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        wasserstein_cliff(a, b)  # should not raise


def test_wasserstein_cliff_returns_float() -> None:
    a = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    b = np.array([0.2, 0.6, 0.8], dtype=np.float64)
    assert isinstance(wasserstein_cliff(a, b), float)


# ---------------------------------------------------------------------------
# detect_cliff_boundary_three_metric
# ---------------------------------------------------------------------------


def test_three_metric_returns_none_when_no_scheme_exceeds_all() -> None:
    geo = {QuantScheme.GGUF_Q3_K_M: 0.1}
    wass = {QuantScheme.GGUF_Q3_K_M: 0.5}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.5}
    # geometric < kappa → None
    assert detect_cliff_boundary_three_metric(geo, wass, beh) is None


def test_three_metric_returns_scheme_when_all_three_exceed_kappa() -> None:
    geo = {QuantScheme.GGUF_Q3_K_M: 0.5}
    wass = {QuantScheme.GGUF_Q3_K_M: 0.5}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.5}
    assert detect_cliff_boundary_three_metric(geo, wass, beh) == QuantScheme.GGUF_Q3_K_M


def test_three_metric_requires_all_three_strict_and() -> None:
    # Only geometric and wasserstein exceed kappa, behavioral does not
    geo = {QuantScheme.GGUF_Q3_K_M: 0.5}
    wass = {QuantScheme.GGUF_Q3_K_M: 0.5}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.1}
    assert detect_cliff_boundary_three_metric(geo, wass, beh) is None


def test_three_metric_only_geometric_and_behavioral_not_enough() -> None:
    geo = {QuantScheme.GGUF_Q3_K_M: 0.5}
    wass = {QuantScheme.GGUF_Q3_K_M: 0.1}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.5}
    assert detect_cliff_boundary_three_metric(geo, wass, beh) is None


def test_three_metric_skips_fp16() -> None:
    geo = {QuantScheme.FP16: 1.0, QuantScheme.GGUF_Q3_K_M: 0.5}
    wass = {QuantScheme.FP16: 1.0, QuantScheme.GGUF_Q3_K_M: 0.5}
    beh = {QuantScheme.FP16: 1.0, QuantScheme.GGUF_Q3_K_M: 0.5}
    assert detect_cliff_boundary_three_metric(geo, wass, beh) == QuantScheme.GGUF_Q3_K_M


def test_three_metric_returns_first_in_bit_width_order() -> None:
    # Q4_K_M comes before Q3_K_M; both qualify → returns Q4_K_M
    geo = {QuantScheme.GGUF_Q4_K_M: 0.4, QuantScheme.GGUF_Q3_K_M: 0.5}
    wass = {QuantScheme.GGUF_Q4_K_M: 0.4, QuantScheme.GGUF_Q3_K_M: 0.5}
    beh = {QuantScheme.GGUF_Q4_K_M: 0.4, QuantScheme.GGUF_Q3_K_M: 0.5}
    assert detect_cliff_boundary_three_metric(geo, wass, beh) == QuantScheme.GGUF_Q4_K_M


def test_three_metric_skips_scheme_absent_from_any_dict() -> None:
    # Q3_K_M absent from wasserstein dict → must be skipped
    geo = {QuantScheme.GGUF_Q3_K_M: 0.9}
    wass = {QuantScheme.GGUF_Q4_K_M: 0.9}
    beh = {QuantScheme.GGUF_Q3_K_M: 0.9}
    assert detect_cliff_boundary_three_metric(geo, wass, beh) is None


def test_three_metric_returns_none_for_empty_dicts() -> None:
    assert detect_cliff_boundary_three_metric({}, {}, {}) is None
