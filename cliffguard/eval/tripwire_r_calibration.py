"""TRIPWIRE-R Fold A calibration harness — see blueprint §12.5.

Scores the Fold A benign corpus through TRIPWIRE-R using a KenLMScorer
to obtain the empirical LLR distribution on benign prompts, then calls
the threshold calibrator to produce tau_q for TRIPWIRE-R.
"""

import numpy as np
import numpy.typing as npt

from cliffguard.eval.kenlm_stub import KenLMScorer
from cliffguard.eval.threshold_calibrator import calibrate_threshold
from cliffguard.tripwire.r import log_likelihood_ratio


def score_corpus_tripwire_r(
    corpus: list[str],
    scorer: KenLMScorer,
    k: int = 20,
) -> npt.NDArray[np.float64]:
    """Score each prompt in corpus through TRIPWIRE-R.
    For each prompt:
      1. Get input_logprobs from scorer.score_tokens(prompt, k)
         and compute mean per-token logprob → 1-D vector of length k.
      2. Get benign_logprobs as scorer.score_tokens(prompt, k)
         again (same scorer, same prompt = same benign reference).
      3. Compute LLR via log_likelihood_ratio(input_mean, benign_mean).
    Returns array of shape (len(corpus),) containing per-prompt LLR.
    Raises ValueError if corpus is empty."""
    if not corpus:
        raise ValueError("corpus must be non-empty")
    llrs: list[float] = []
    for prompt in corpus:
        token_logprobs = scorer.score_tokens(prompt, k)
        input_mean: npt.NDArray[np.float64] = np.mean(token_logprobs, axis=0)
        benign_logprobs = scorer.score_tokens(prompt, k)
        benign_mean: npt.NDArray[np.float64] = np.mean(benign_logprobs, axis=0)
        llrs.append(log_likelihood_ratio(input_mean, benign_mean))
    return np.array(llrs, dtype=np.float64)


def calibrate_tripwire_r(
    corpus: list[str],
    scorer: KenLMScorer,
    fpr_target: float = 0.05,
    k: int = 20,
) -> float:
    """Score corpus and return calibrated tau_q for TRIPWIRE-R.
    Calls score_corpus_tripwire_r then calibrate_threshold.

    TRIPWIRE-R fires LOW (`fired = score < tau`), so the threshold must come
    from the LOWER tail. Calibrating it against the upper tail inverts the FPR
    (a 0.05 target becomes a realised 0.95) — defect D0.

    Returns the threshold float."""
    scores = score_corpus_tripwire_r(corpus, scorer, k)
    return calibrate_threshold(scores, fpr_target=fpr_target, fires_high=False)
