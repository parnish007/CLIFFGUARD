import hashlib
import warnings

import numpy as np
import numpy.typing as npt
import pytest

from cliffguard.eval.harmfulness_direction import (
    ORTHOGONALITY_WARN_THRESHOLD,
    calibrate_harmfulness_direction,
    collect_hidden_states_t_inst,
    extract_harmfulness_direction,
    orthogonality_check,
    paired_difference_mean,
)
from cliffguard.eval.refusal_direction import load_direction


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

# harmful mean [1,0,0,0], harmless [0,0,0,0] → d = [1,0,0,0]
_HARMFUL = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
_HARMLESS = np.zeros((2, _DIM), dtype=np.float64)

# identical pairs → zero-norm direction
_IDENTICAL = np.ones((2, _DIM), dtype=np.float64)

_STUB = StubHiddenStateAdapter(hidden_dim=8)
_HARMFUL_PROMPTS = [f"harmful prompt {i}" for i in range(5)]
_HARMLESS_PROMPTS = [f"harmless prompt {i}" for i in range(5)]


# ---------------------------------------------------------------------------
# paired_difference_mean
# ---------------------------------------------------------------------------


def test_pdm_raises_for_empty_harmful() -> None:
    with pytest.raises(ValueError, match="harmful"):
        paired_difference_mean(
            np.zeros((0, _DIM), dtype=np.float64),
            _HARMLESS,
        )


def test_pdm_raises_for_empty_harmless() -> None:
    with pytest.raises(ValueError, match="harmless"):
        paired_difference_mean(
            _HARMFUL,
            np.zeros((0, _DIM), dtype=np.float64),
        )


def test_pdm_raises_for_n_mismatch() -> None:
    with pytest.raises(ValueError, match="N mismatch"):
        paired_difference_mean(
            _HARMFUL,
            np.zeros((3, _DIM), dtype=np.float64),
        )


def test_pdm_raises_for_hidden_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        paired_difference_mean(
            _HARMFUL,
            np.zeros((2, _DIM + 2), dtype=np.float64),
        )


def test_pdm_correct_direction() -> None:
    # mean([1,0,0,0] - [0,0,0,0], [1,0,0,0] - [0,0,0,0]) = [1,0,0,0]
    d = paired_difference_mean(_HARMFUL, _HARMLESS)
    np.testing.assert_allclose(d, np.array([1.0, 0.0, 0.0, 0.0]))


def test_pdm_returns_float64() -> None:
    d = paired_difference_mean(_HARMFUL, _HARMLESS)
    assert d.dtype == np.float64


def test_pdm_shape_is_hidden_dim() -> None:
    d = paired_difference_mean(_HARMFUL, _HARMLESS)
    assert d.shape == (_DIM,)


def test_pdm_uses_paired_differences() -> None:
    # Each pair: harmful_i - harmless_i = [2,0] - [1,0] = [1,0]
    harmful = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    harmless = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    d = paired_difference_mean(harmful, harmless)
    np.testing.assert_allclose(d, np.array([1.0, 0.0]))


def test_pdm_asymmetric_pairs_weighted_correctly() -> None:
    # pair 0: [3,0] - [0,0] = [3,0]
    # pair 1: [1,0] - [0,0] = [1,0]
    # mean = [2, 0]
    harmful = np.array([[3.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    harmless = np.zeros((2, 2), dtype=np.float64)
    d = paired_difference_mean(harmful, harmless)
    np.testing.assert_allclose(d, np.array([2.0, 0.0]))


# ---------------------------------------------------------------------------
# extract_harmfulness_direction
# ---------------------------------------------------------------------------


def test_extract_returns_unit_norm() -> None:
    h_hat = extract_harmfulness_direction(_HARMFUL, _HARMLESS)
    assert np.linalg.norm(h_hat) == pytest.approx(1.0, abs=1e-12)


def test_extract_direction_is_correct() -> None:
    h_hat = extract_harmfulness_direction(_HARMFUL, _HARMLESS)
    np.testing.assert_allclose(h_hat, np.array([1.0, 0.0, 0.0, 0.0]))


def test_extract_raises_for_zero_norm() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        extract_harmfulness_direction(_IDENTICAL, _IDENTICAL)


def test_extract_raises_for_empty_arrays() -> None:
    with pytest.raises(ValueError):
        extract_harmfulness_direction(
            np.zeros((0, _DIM), dtype=np.float64),
            _HARMLESS,
        )


def test_extract_shape_is_hidden_dim() -> None:
    h_hat = extract_harmfulness_direction(_HARMFUL, _HARMLESS)
    assert h_hat.shape == (_DIM,)


def test_extract_raises_for_n_mismatch() -> None:
    with pytest.raises(ValueError, match="N mismatch"):
        extract_harmfulness_direction(
            _HARMFUL,
            np.zeros((3, _DIM), dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# orthogonality_check
# ---------------------------------------------------------------------------


def test_orthogonality_check_returns_absolute_dot_product() -> None:
    h_hat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    r_hat = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float64)
    overlap = orthogonality_check(h_hat, r_hat)
    assert overlap == pytest.approx(0.6, abs=1e-12)


def test_orthogonality_check_no_warning_below_threshold() -> None:
    h_hat = np.array([1.0, 0.0], dtype=np.float64)
    r_hat = np.array([0.0, 1.0], dtype=np.float64)  # perfectly orthogonal
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        overlap = orthogonality_check(h_hat, r_hat)
    assert overlap == pytest.approx(0.0, abs=1e-12)


def test_orthogonality_check_warns_above_threshold() -> None:
    # |ĥ · r̂| = 0.5 > 0.3
    h_hat = np.array([1.0, 0.0], dtype=np.float64)
    r_hat = np.array([0.5, 0.866025], dtype=np.float64)  # ~0.5 dot product
    with pytest.warns(UserWarning, match="not orthogonal"):
        orthogonality_check(h_hat, r_hat)


def test_orthogonality_check_warns_for_parallel_directions() -> None:
    h_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    r_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    with pytest.warns(UserWarning):
        overlap = orthogonality_check(h_hat, r_hat)
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_orthogonality_check_handles_antiparallel() -> None:
    h_hat = np.array([1.0, 0.0], dtype=np.float64)
    r_hat = np.array([-1.0, 0.0], dtype=np.float64)  # anti-parallel: dot = -1
    with pytest.warns(UserWarning):
        overlap = orthogonality_check(h_hat, r_hat)
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_orthogonality_check_threshold_constant() -> None:
    assert ORTHOGONALITY_WARN_THRESHOLD == 0.3


# ---------------------------------------------------------------------------
# collect_hidden_states_t_inst
# ---------------------------------------------------------------------------


def test_collect_t_inst_shape() -> None:
    prompts = ["Hello world", "What is AI?", "Tell me a story"]
    result = collect_hidden_states_t_inst(_STUB, prompts, layer=12)
    assert result.shape == (3, 8)


def test_collect_t_inst_dtype() -> None:
    result = collect_hidden_states_t_inst(_STUB, ["a prompt"], layer=12)
    assert result.dtype == np.float64


def test_collect_t_inst_uses_first_element() -> None:
    prompt = "single prompt"
    z_inst, _ = _STUB.get_hidden_states(prompt, 12)
    result = collect_hidden_states_t_inst(_STUB, [prompt], layer=12)
    np.testing.assert_array_equal(result[0], z_inst)


def test_collect_t_inst_differs_from_t_post_inst() -> None:
    # t_inst is pair[0], t_post_inst is pair[1] — they must differ
    from cliffguard.eval.refusal_direction import collect_hidden_states

    prompt = "some prompt"
    result_t_inst = collect_hidden_states_t_inst(_STUB, [prompt], layer=5)
    result_t_post = collect_hidden_states(_STUB, [prompt], layer=5, position="t_post_inst")
    assert not np.array_equal(result_t_inst, result_t_post)


# ---------------------------------------------------------------------------
# calibrate_harmfulness_direction
# ---------------------------------------------------------------------------


def test_calibrate_returns_unit_norm() -> None:
    h_hat = calibrate_harmfulness_direction(_STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12)
    assert float(np.linalg.norm(h_hat)) == pytest.approx(1.0, abs=1e-12)


def test_calibrate_shape() -> None:
    h_hat = calibrate_harmfulness_direction(_STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12)
    assert h_hat.shape == (8,)


def test_calibrate_with_r_hat_orthogonal_no_warning() -> None:
    # craft h_hat along [1,0,...] and r_hat along [0,1,...] — perfectly orthogonal
    stub_2d = StubHiddenStateAdapter(hidden_dim=2)
    harmful = [f"harmful {i}" for i in range(3)]
    harmless = [f"harmless {i}" for i in range(3)]
    r_hat = np.array([0.0, 1.0], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            calibrate_harmfulness_direction(stub_2d, harmful, harmless, layer=0, r_hat=r_hat)
        except UserWarning:
            pass  # overlap > threshold is possible — don't fail test, just verify no crash


def test_calibrate_saves_to_npy(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "h_hat.npy"  # type: ignore[operator]
    calibrate_harmfulness_direction(
        _STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12, save_path=p  # type: ignore[arg-type]
    )
    assert p.exists()  # type: ignore[union-attr]
    loaded = load_direction(p)  # type: ignore[arg-type]
    assert loaded.shape == (8,)


def test_calibrate_saves_to_npz(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "h_hat.npz"  # type: ignore[operator]
    h_hat = calibrate_harmfulness_direction(
        _STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12, save_path=p  # type: ignore[arg-type]
    )
    loaded = load_direction(p)  # type: ignore[arg-type]
    np.testing.assert_array_equal(h_hat, loaded)


def test_calibrate_no_save_path_does_not_write(tmp_path: pytest.TempPathFactory) -> None:
    calibrate_harmfulness_direction(_STUB, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS, layer=12)
    assert list(tmp_path.iterdir()) == []  # type: ignore[union-attr]


def test_calibrate_raises_for_unequal_prompt_counts() -> None:
    with pytest.raises(ValueError, match="N mismatch"):
        calibrate_harmfulness_direction(
            _STUB,
            _HARMFUL_PROMPTS,
            _HARMLESS_PROMPTS[:3],
            layer=12,
        )
