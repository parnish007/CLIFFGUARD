import numpy as np
import pytest

from cliffguard.eval.kenlm_stub import SyntheticKenLMScorer
from cliffguard.eval.tripwire_r_calibration import (
    calibrate_tripwire_r,
    score_corpus_tripwire_r,
)

_SCORER = SyntheticKenLMScorer(seed=0)
_CORPUS = [
    "What is the capital of France?",
    "Tell me about machine learning.",
    "How does photosynthesis work?",
    "Explain the French Revolution.",
    "What is quantum entanglement?",
]


# ---------------------------------------------------------------------------
# score_corpus_tripwire_r
# ---------------------------------------------------------------------------


def test_score_corpus_raises_for_empty_corpus() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        score_corpus_tripwire_r([], _SCORER)


def test_score_corpus_returns_correct_shape() -> None:
    scores = score_corpus_tripwire_r(_CORPUS, _SCORER)
    assert scores.shape == (len(_CORPUS),)


def test_score_corpus_returns_float64() -> None:
    scores = score_corpus_tripwire_r(_CORPUS, _SCORER)
    assert scores.dtype == np.float64


def test_score_corpus_llr_is_zero_for_same_scorer_same_prompt() -> None:
    # SyntheticKenLMScorer is deterministic: same prompt → same logprobs.
    # input_mean == benign_mean → LLR = mean(input - benign) = 0.0 for every prompt.
    scores = score_corpus_tripwire_r(_CORPUS, _SCORER)
    np.testing.assert_allclose(scores, 0.0, atol=1e-12)


def test_score_corpus_single_prompt() -> None:
    scores = score_corpus_tripwire_r(["hello world"], _SCORER)
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# calibrate_tripwire_r
# ---------------------------------------------------------------------------


def test_calibrate_tripwire_r_returns_float() -> None:
    result = calibrate_tripwire_r(_CORPUS, _SCORER, fpr_target=0.05)
    assert isinstance(result, float)


def test_calibrate_tripwire_r_zero_for_all_zero_llrs() -> None:
    # All LLRs are 0.0 → any percentile of all-zeros is 0.0.
    result = calibrate_tripwire_r(_CORPUS, _SCORER, fpr_target=0.05)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_calibrate_tripwire_r_respects_fpr_target() -> None:
    # Both fpr_target values yield 0.0 here since all scores are 0.0,
    # but the function should accept different fpr_target values.
    r1 = calibrate_tripwire_r(_CORPUS, _SCORER, fpr_target=0.01)
    r2 = calibrate_tripwire_r(_CORPUS, _SCORER, fpr_target=0.10)
    # Both are 0.0 (percentile of zeros), but the call must not raise.
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_calibrate_tripwire_r_uses_lower_tail() -> None:
    """D0 regression for the direct calibration path.

    TRIPWIRE-R fires LOW. Codex found (docs/build_log.md Entry 2) that this
    production call site still selected the upper tail after the resolver was
    added to build_calibration_table, because it calls calibrate_threshold
    directly. The realised FPR under the fires-LOW rule must be <= target."""
    import numpy as np
    from cliffguard.eval.threshold_calibrator import empirical_fpr

    class _Scorer:
        def score(self, text: str) -> float:
            return 0.0

    rng = np.random.default_rng(0)
    scores = rng.normal(0.0, 1.0, 5000)

    from cliffguard.eval import tripwire_r_calibration as trc

    original = trc.score_corpus_tripwire_r
    trc.score_corpus_tripwire_r = lambda corpus, scorer, k: scores  # type: ignore[assignment]
    try:
        tau = trc.calibrate_tripwire_r(["x"] * 10, _Scorer(), fpr_target=0.05)  # type: ignore[arg-type]
    finally:
        trc.score_corpus_tripwire_r = original  # type: ignore[assignment]

    assert empirical_fpr(scores, tau, fires_high=False) <= 0.05
    # and it must NOT be the upper-tail value
    assert empirical_fpr(scores, tau, fires_high=False) < 0.5
