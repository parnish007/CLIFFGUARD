"""TRIPWIRE-R perplexity-ratio gate — see blueprint §5.5.

Computes the Neyman-Pearson log-likelihood ratio between a KenLM
language model trained on benign traffic and the empirical token
log-probability sequence of the current input. A high ratio signals
that the input is unlikely under the benign distribution — the
signature of adversarial encodings (A6: ArtPrompt, bijection learning,
low-resource-language jailbreaks).

In scaffolding mode (Phase A), KenLM is replaced by a stub interface
that accepts pre-computed per-token log-probabilities. The real KenLM
integration is implemented in Phase B (Task 25).
"""

import numpy as np
import numpy.typing as npt

from cliffguard.types import CalibrationTable, GateVerdict, QuantScheme, Tier


def log_likelihood_ratio(
    input_logprobs: npt.NDArray[np.float64],
    benign_logprobs: npt.NDArray[np.float64],
) -> float:
    """Compute the mean per-token log-likelihood ratio:
      LLR = mean(input_logprobs - benign_logprobs)

    A positive LLR means the input is MORE likely under the input
    model than under the benign model — not a concern.
    A negative LLR means the input is LESS likely under benign —
    potential adversarial encoding.
    Raises ValueError if arrays have different lengths or are empty.
    """
    if len(input_logprobs) == 0 or len(benign_logprobs) == 0:
        raise ValueError("input_logprobs and benign_logprobs must be non-empty")
    if len(input_logprobs) != len(benign_logprobs):
        raise ValueError(
            f"Array length mismatch: input has {len(input_logprobs)} tokens, "
            f"benign has {len(benign_logprobs)} tokens"
        )
    return float(np.mean(input_logprobs - benign_logprobs))


def evaluate(
    input_logprobs: npt.NDArray[np.float64],
    benign_logprobs: npt.NDArray[np.float64],
    calibration: CalibrationTable,
    scheme: QuantScheme,
    tier: Tier,
) -> GateVerdict:
    """Evaluate the TRIPWIRE-R gate.

    score = log_likelihood_ratio(input_logprobs, benign_logprobs).
    fired = True if score < calibration.tau(scheme).
    (Negative LLR = input unlikely under benign distribution = fire.)
    gate name is "TRIPWIRE-R".
    threat_model is None.
    Raises ValueError on empty or mismatched arrays — consistent with
    TRIPWIRE-H convention established in Task 10.
    """
    score = log_likelihood_ratio(input_logprobs, benign_logprobs)
    threshold = calibration.tau(scheme)
    return GateVerdict(
        gate="TRIPWIRE-R",
        fired=score < threshold,
        score=score,
        threshold=threshold,
        tier=tier,
        threat_model=None,
    )
