import hashlib

import numpy as np
import numpy.typing as npt
import pytest

from cliffguard.eval.refusal_direction import (
    calibrate_refusal_direction,
    collect_hidden_states,
    difference_in_means,
    extract_refusal_direction,
    load_direction,
    save_direction,
)


# ---------------------------------------------------------------------------
# StubHiddenStateAdapter — satisfies HiddenStateAdapter Protocol
# ---------------------------------------------------------------------------


class StubHiddenStateAdapter:
    """Deterministic stub: returns synthetic hidden states without GPU."""

    def __init__(self, hidden_dim: int = 8) -> None:
        self.hidden_dim = hidden_dim

    def get_hidden_states(
        self,
        prompt: str,
        layer: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        z_t_inst: npt.NDArray[np.float64] = rng.standard_normal(self.hidden_dim).astype(np.float64)
        z_t_post_inst: npt.NDArray[np.float64] = rng.standard_normal(self.hidden_dim).astype(np.float64)
        return z_t_inst, z_t_post_inst

    def get_top_k_logprobs(self, prompt: str, k: int = 20) -> npt.NDArray[np.float64]:
        return np.zeros(k, dtype=np.float64)

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


# ---------------------------------------------------------------------------
# save_direction / load_direction — .npz format
# ---------------------------------------------------------------------------


def test_save_direction_npz_creates_file(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "direction.npz"  # type: ignore[operator]
    save_direction(np.array([1.0, 0.0, 0.0], dtype=np.float64), p)  # type: ignore[arg-type]
    assert p.exists()  # type: ignore[union-attr]


def test_save_load_direction_npz_round_trips(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "direction.npz"  # type: ignore[operator]
    original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    save_direction(original, p)  # type: ignore[arg-type]
    loaded = load_direction(p)  # type: ignore[arg-type]
    np.testing.assert_array_equal(loaded, original)


def test_load_direction_npz_returns_float64(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "direction.npz"  # type: ignore[operator]
    save_direction(np.ones(4, dtype=np.float64), p)  # type: ignore[arg-type]
    loaded = load_direction(p)  # type: ignore[arg-type]
    assert loaded.dtype == np.float64


def test_load_direction_raises_for_unsupported_extension(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "direction.txt"  # type: ignore[operator]
    with pytest.raises(ValueError, match=".npy or .npz"):
        load_direction(p)  # type: ignore[arg-type]


def test_save_direction_raises_for_unsupported_extension(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "direction.txt"  # type: ignore[operator]
    with pytest.raises(ValueError, match=".npy or .npz"):
        save_direction(np.ones(4, dtype=np.float64), p)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# collect_hidden_states
# ---------------------------------------------------------------------------

_STUB = StubHiddenStateAdapter(hidden_dim=8)
_PROMPTS = ["Hello world", "What is AI?", "Tell me a story"]


def test_collect_hidden_states_shape() -> None:
    result = collect_hidden_states(_STUB, _PROMPTS, layer=12)
    assert result.shape == (len(_PROMPTS), 8)


def test_collect_hidden_states_dtype() -> None:
    result = collect_hidden_states(_STUB, _PROMPTS, layer=12)
    assert result.dtype == np.float64


def test_collect_hidden_states_t_post_inst_uses_second_element() -> None:
    # t_post_inst should match pair[1] from get_hidden_states
    prompt = "single prompt"
    _, z_post = _STUB.get_hidden_states(prompt, 12)
    result = collect_hidden_states(_STUB, [prompt], layer=12, position="t_post_inst")
    np.testing.assert_array_equal(result[0], z_post)


def test_collect_hidden_states_t_inst_uses_first_element() -> None:
    # t_inst should match pair[0] from get_hidden_states
    prompt = "single prompt"
    z_inst, _ = _STUB.get_hidden_states(prompt, 12)
    result = collect_hidden_states(_STUB, [prompt], layer=12, position="t_inst")
    np.testing.assert_array_equal(result[0], z_inst)


def test_collect_hidden_states_t_inst_differs_from_t_post_inst() -> None:
    result_post = collect_hidden_states(_STUB, ["some prompt"], layer=5, position="t_post_inst")
    result_inst = collect_hidden_states(_STUB, ["some prompt"], layer=5, position="t_inst")
    assert not np.array_equal(result_post, result_inst)



# ---------------------------------------------------------------------------
# calibrate_refusal_direction
# ---------------------------------------------------------------------------

_HARMFUL_PROMPTS = [f"harmful prompt {i}" for i in range(5)]
_HARMLESS_PROMPTS = [f"harmless prompt {i}" for i in range(5)]


def test_calibrate_refusal_direction_returns_unit_norm() -> None:
    r_hat = calibrate_refusal_direction(_STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12)
    assert float(np.linalg.norm(r_hat)) == pytest.approx(1.0, abs=1e-12)


def test_calibrate_refusal_direction_shape() -> None:
    r_hat = calibrate_refusal_direction(_STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12)
    assert r_hat.shape == (8,)


def test_calibrate_refusal_direction_saves_to_npy(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "r_hat.npy"  # type: ignore[operator]
    calibrate_refusal_direction(
        _STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12, save_path=p  # type: ignore[arg-type]
    )
    assert p.exists()  # type: ignore[union-attr]
    loaded = load_direction(p)  # type: ignore[arg-type]
    assert loaded.shape == (8,)


def test_calibrate_refusal_direction_saves_to_npz(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "r_hat.npz"  # type: ignore[operator]
    r_hat = calibrate_refusal_direction(
        _STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12, save_path=p  # type: ignore[arg-type]
    )
    loaded = load_direction(p)  # type: ignore[arg-type]
    np.testing.assert_array_equal(r_hat, loaded)


def test_calibrate_refusal_direction_no_save_path_does_not_write(
    tmp_path: pytest.TempPathFactory,
) -> None:
    calibrate_refusal_direction(_STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12)
    # No file should have been written anywhere in tmp_path
    assert list(tmp_path.iterdir()) == []  # type: ignore[union-attr]
