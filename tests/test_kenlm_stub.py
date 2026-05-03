import numpy as np

from cliffguard.eval.kenlm_stub import SyntheticKenLMScorer


_K = 20
_SCORER = SyntheticKenLMScorer(seed=42)


def test_score_tokens_shape_multiword() -> None:
    result = _SCORER.score_tokens("hello world foo bar", k=_K)
    assert result.shape == (4, _K)


def test_score_tokens_shape_single_word() -> None:
    result = _SCORER.score_tokens("hello", k=_K)
    assert result.shape == (1, _K)


def test_score_tokens_shape_empty_string() -> None:
    # empty string → split() = [] → max(1, 0) = 1 token
    result = _SCORER.score_tokens("", k=_K)
    assert result.shape == (1, _K)


def test_score_tokens_values_in_range() -> None:
    result = _SCORER.score_tokens("test prompt with words", k=_K)
    assert (result >= -10.0).all()
    assert (result <= 0.0).all()


def test_score_tokens_deterministic_same_prompt() -> None:
    scorer = SyntheticKenLMScorer(seed=7)
    r1 = scorer.score_tokens("what is the capital of France", k=_K)
    r2 = scorer.score_tokens("what is the capital of France", k=_K)
    np.testing.assert_array_equal(r1, r2)


def test_score_tokens_different_prompts_differ() -> None:
    scorer = SyntheticKenLMScorer(seed=7)
    r1 = scorer.score_tokens("hello world", k=_K)
    r2 = scorer.score_tokens("goodbye world", k=_K)
    assert not np.array_equal(r1, r2)


def test_score_tokens_different_seeds_differ() -> None:
    s1 = SyntheticKenLMScorer(seed=1)
    s2 = SyntheticKenLMScorer(seed=2)
    r1 = s1.score_tokens("same text", k=_K)
    r2 = s2.score_tokens("same text", k=_K)
    assert not np.array_equal(r1, r2)


def test_score_tokens_respects_k() -> None:
    result = _SCORER.score_tokens("hello world", k=5)
    assert result.shape == (2, 5)


def test_score_tokens_returns_float64() -> None:
    result = _SCORER.score_tokens("test", k=_K)
    assert result.dtype == np.float64
