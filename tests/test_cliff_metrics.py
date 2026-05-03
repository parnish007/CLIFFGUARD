import math

import numpy as np
import pytest

from cliffguard.types import QuantScheme
from cliffguard.eval.cliff_metrics import (
    behavioral_cliff,
    detect_cliff_boundary,
    geometric_cliff,
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
