import numpy as np
import pytest

from cliffguard.eval.refusal_direction import (
    difference_in_means,
    extract_refusal_direction,
    load_direction,
    save_direction,
)

# ---------------------------------------------------------------------------
# Shared synthetic arrays
# ---------------------------------------------------------------------------

_DIM = 4

# harmful mean = [1, 0, 0, 0], harmless mean = [0, 0, 0, 0] → r = [1, 0, 0, 0]
_HARMFUL = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
_HARMLESS = np.zeros((2, _DIM), dtype=np.float64)

# Identical means → zero-norm direction
_IDENTICAL = np.ones((2, _DIM), dtype=np.float64)

# ---------------------------------------------------------------------------
# difference_in_means
# ---------------------------------------------------------------------------


def test_dim_raises_for_empty_harmful() -> None:
    with pytest.raises(ValueError, match="harmful"):
        difference_in_means(
            np.zeros((0, _DIM), dtype=np.float64),
            _HARMLESS,
        )


def test_dim_raises_for_empty_harmless() -> None:
    with pytest.raises(ValueError, match="harmless"):
        difference_in_means(
            _HARMFUL,
            np.zeros((0, _DIM), dtype=np.float64),
        )


def test_dim_raises_for_hidden_dim_mismatch() -> None:
    wrong_dim = np.zeros((2, _DIM + 2), dtype=np.float64)
    with pytest.raises(ValueError, match="hidden_dim"):
        difference_in_means(_HARMFUL, wrong_dim)


def test_dim_correct_direction() -> None:
    # mean([1,0,0,0],[1,0,0,0]) - mean([0,0,0,0],[0,0,0,0]) = [1,0,0,0]
    r = difference_in_means(_HARMFUL, _HARMLESS)
    np.testing.assert_allclose(r, np.array([1.0, 0.0, 0.0, 0.0]))


def test_dim_returns_float64() -> None:
    r = difference_in_means(_HARMFUL, _HARMLESS)
    assert r.dtype == np.float64


def test_dim_shape_is_hidden_dim() -> None:
    r = difference_in_means(_HARMFUL, _HARMLESS)
    assert r.shape == (_DIM,)


def test_dim_asymmetric_counts() -> None:
    # N_harm != N_safe — should still work
    h1 = np.array([[2.0, 0.0]], dtype=np.float64)
    h2 = np.zeros((5, 2), dtype=np.float64)
    r = difference_in_means(h1, h2)
    np.testing.assert_allclose(r, np.array([2.0, 0.0]))


# ---------------------------------------------------------------------------
# extract_refusal_direction
# ---------------------------------------------------------------------------


def test_extract_returns_unit_norm() -> None:
    r_hat = extract_refusal_direction(_HARMFUL, _HARMLESS)
    assert np.linalg.norm(r_hat) == pytest.approx(1.0, abs=1e-12)


def test_extract_direction_is_correct() -> None:
    # [1,0,0,0] / 1.0 = [1,0,0,0]
    r_hat = extract_refusal_direction(_HARMFUL, _HARMLESS)
    np.testing.assert_allclose(r_hat, np.array([1.0, 0.0, 0.0, 0.0]))


def test_extract_raises_for_zero_norm() -> None:
    # identical means → r = [0,0,...] → zero norm
    with pytest.raises(ValueError, match="zero norm"):
        extract_refusal_direction(_IDENTICAL, _IDENTICAL)


def test_extract_raises_for_empty_arrays() -> None:
    with pytest.raises(ValueError):
        extract_refusal_direction(
            np.zeros((0, _DIM), dtype=np.float64),
            _HARMLESS,
        )


def test_extract_shape_is_hidden_dim() -> None:
    r_hat = extract_refusal_direction(_HARMFUL, _HARMLESS)
    assert r_hat.shape == (_DIM,)


# ---------------------------------------------------------------------------
# save_direction / load_direction
# ---------------------------------------------------------------------------


def test_save_direction_creates_file(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "direction.npy"  # type: ignore[operator]
    direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    save_direction(direction, p)  # type: ignore[arg-type]
    assert p.exists()  # type: ignore[union-attr]


def test_save_direction_creates_parent_dirs(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "subdir" / "nested" / "direction.npy"  # type: ignore[operator]
    save_direction(np.ones(_DIM, dtype=np.float64), p)  # type: ignore[arg-type]
    assert p.exists()  # type: ignore[union-attr]


def test_load_direction_raises_for_nonexistent(tmp_path: pytest.TempPathFactory) -> None:
    missing = tmp_path / "no_file.npy"  # type: ignore[operator]
    with pytest.raises(FileNotFoundError):
        load_direction(missing)  # type: ignore[arg-type]


def test_load_direction_raises_for_2d_array(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "bad.npy"  # type: ignore[operator]
    np.save(str(p), np.ones((4, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="1-D"):
        load_direction(p)  # type: ignore[arg-type]


def test_load_direction_round_trips(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "direction.npy"  # type: ignore[operator]
    original = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    save_direction(original, p)  # type: ignore[arg-type]
    loaded = load_direction(p)  # type: ignore[arg-type]
    np.testing.assert_array_equal(loaded, original)


def test_load_direction_returns_float64(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "direction.npy"  # type: ignore[operator]
    save_direction(np.ones(_DIM, dtype=np.float64), p)  # type: ignore[arg-type]
    loaded = load_direction(p)  # type: ignore[arg-type]
    assert loaded.dtype == np.float64
